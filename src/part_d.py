"""Part D: policy evaluation and the refund agent. Retrieval comes from policy_rag.retrieve."""
import json
import sqlite3
import sys
import time
from datetime import datetime

import pandas as pd

from src.paths import DB_PATH, OUTPUTS, RAW, ROOT, STATE
from src.policy_rag import (
    PolicyIndex,
    build_default_index,
    evaluate,
    evaluation_scores,
    load_chunks,
    retrieve,
)

sys.path.insert(0, str(ROOT))
from mock_llm import chat  # noqa: E402

SYSTEM = (
    "You are a refund AGENT. Look up the order, read the refund policy, calculate the refund, "
    "and either issue the refund or escalate it to a human. Give a final decision when the tools are done."
)


class SimulatedCrash(Exception):
    """Raised on purpose so a run can be resumed from its checkpoint."""


def get_order(order_id):
    """Read one order from the analytical database."""
    with sqlite3.connect(DB_PATH) as con:
        row = con.execute(
            "SELECT order_id, status, order_amount, delivery_minutes FROM orders WHERE order_id = ?",
            (order_id,),
        ).fetchone()
    if row is None:
        return {"error": "order not found"}
    minutes = None if row[3] is None else float(row[3])
    if minutes is not None and minutes.is_integer():
        minutes = int(minutes)
    return {
        "order_id": row[0],
        "status": row[1],
        "order_amount": round(float(row[2]), 2),
        "delivery_minutes": minutes,
    }


def search_policy(query):
    """Search the policy chunks and return the top matches."""
    return {"results": retrieve(query, k=3)}


def calculate_refund(order_amount, delivery_minutes):
    """Refund the full amount when delivery took more than 45 minutes, otherwise INR 0."""
    amount = round(float(order_amount), 2)
    refund = amount if float(delivery_minutes) > 45 else 0.0
    return {"refund_inr": refund}


def _append_ledger(tool, order_id, refund_inr):
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "tool": tool,
        "order_id": order_id,
        "refund_inr": round(float(refund_inr), 2),
    }
    with (OUTPUTS / "refund_ledger.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
    return {"refund_inr": record["refund_inr"]}


def issue_refund(order_id, refund_inr):
    """Record an automatic refund in the ledger."""
    return _append_ledger("issue_refund", order_id, refund_inr)


def escalate_to_human(order_id, refund_inr):
    """Record that a refund above INR 1000 was sent to a human lead."""
    return _append_ledger("escalate_to_human", order_id, refund_inr)


TOOLS_SPEC = [
    {
        "name": "get_order",
        "description": "Read one order from the analytical database.",
        "parameters": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"],
        },
    },
    {
        "name": "search_policy",
        "description": "Search the policy chunks and return the top matches.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": "calculate_refund",
        "description": "Refund the full amount when delivery took more than 45 minutes, otherwise INR 0.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_amount": {"type": "number"},
                "delivery_minutes": {"type": "number"},
            },
            "required": ["order_amount", "delivery_minutes"],
        },
    },
    {
        "name": "issue_refund",
        "description": "Record an automatic refund in the ledger.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "refund_inr": {"type": "number"},
            },
            "required": ["order_id", "refund_inr"],
        },
    },
    {
        "name": "escalate_to_human",
        "description": "Record that a refund above INR 1000 was sent to a human lead.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "refund_inr": {"type": "number"},
            },
            "required": ["order_id", "refund_inr"],
        },
    },
]
TOOL_MAP = {
    "get_order": get_order,
    "search_policy": search_policy,
    "calculate_refund": calculate_refund,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human,
}


def _checkpoint(state):
    STATE.mkdir(parents=True, exist_ok=True)
    (STATE / f"{state['run_id']}.json").write_text(json.dumps(state), encoding="utf-8")


def _fallback(status, summary, state):
    return {
        "status": status,
        "final_answer": summary,
        "steps": state["steps"],
        "llm_calls": state["llm_calls"],
        "tokens_used": state["tokens_used"],
        "tools_executed": state["tools_executed"],
        "refund_inr": 0.0,
    }


def _finish_from_answer(content, state):
    status = content.split(":", 1)[0].strip()
    refund = 0.0
    for step in reversed(state["steps"]):
        if isinstance(step["observation"], dict) and "refund_inr" in step["observation"]:
            refund = float(step["observation"]["refund_inr"])
            break
    if status not in {"REFUND", "ESCALATE"}:
        refund = 0.0
    return {
        "status": status,
        "final_answer": content,
        "steps": state["steps"],
        "llm_calls": state["llm_calls"],
        "tokens_used": state["tokens_used"],
        "tools_executed": list(state["tools_executed"]),
        "refund_inr": refund,
    }


def _loop(state, crash_after_tools=None):
    started = time.monotonic()
    while state["llm_calls"] < state["max_iterations"]:
        if time.monotonic() - started >= state["time_limit_seconds"]:
            return _fallback(
                "FALLBACK_TIMEOUT",
                "Human handoff: the time limit was reached before a refund decision.",
                state,
            )
        response = chat(state["messages"], tools=TOOLS_SPEC)
        state["llm_calls"] += 1
        state["tokens_used"] += int(response.usage.total_tokens)
        tool_calls = response.choices[0].message.tool_calls or []
        if state["tokens_used"] > state["token_limit"]:
            return _fallback(
                "FALLBACK_TOKEN_LIMIT",
                "Human handoff: the token limit was reached before a refund decision.",
                state,
            )
        if not tool_calls:
            return _finish_from_answer(response.choices[0].message.content, state)
        state["messages"].append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": call.id, "name": call.name, "arguments": call.arguments} for call in tool_calls
                ],
            }
        )
        for call in tool_calls:
            observation = TOOL_MAP[call.name](**call.arguments)
            state["messages"].append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "name": call.name,
                    "content": json.dumps(observation),
                }
            )
            state["steps"].append(
                {"tool": call.name, "arguments": call.arguments, "observation": observation}
            )
            state["tools_executed"].append(call.name)
            state["step_number"] = len(state["steps"])
            _checkpoint(state)
            if crash_after_tools is not None and len(state["tools_executed"]) >= crash_after_tools:
                raise SimulatedCrash(state["run_id"])
    return _fallback(
        "FALLBACK_MAX_ITERATIONS",
        "Human handoff: the assistant used the maximum number of steps without a final decision.",
        state,
    )


def _new_state(user_message, scenario_id, run_id, max_iterations, token_limit, time_limit_seconds):
    return {
        "run_id": run_id,
        "scenario_id": scenario_id,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_message},
        ],
        "tokens_used": 0,
        "llm_calls": 0,
        "step_number": 0,
        "tools_executed": [],
        "steps": [],
        "max_iterations": max_iterations,
        "token_limit": token_limit,
        "time_limit_seconds": time_limit_seconds,
    }


def run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000, time_limit_seconds=10, run_id=None, crash_after_tools=None):
    state = _new_state(
        user_message,
        scenario_id,
        run_id or scenario_id,
        max_iterations,
        token_limit,
        time_limit_seconds,
    )
    return _loop(state, crash_after_tools=crash_after_tools)


def resume_agent(run_id):
    state = json.loads((STATE / f"{run_id}.json").read_text(encoding="utf-8"))
    return _loop(state, crash_after_tools=None)


def _print_eval(label, scores, frame):
    print(
        f"{label} doc_hits={scores['doc_hits']}/{scores['answerable']} "
        f"fact_hits={scores['fact_hits']}/{scores['answerable']} "
        f"fallback={scores['fallback_hits']}/{scores['unanswerable']}"
    )
    missed = frame.loc[frame["expected_doc"].ne("NONE") & (~frame["doc_hit"] | ~frame["fact_hit"])]
    for row in missed.itertuples(index=False):
        print(f"{label} missed={row.question} doc_hit={row.doc_hit} fact_hit={row.fact_hit}")


def _chunk_counts(chunks):
    counts = {}
    for chunk in chunks:
        counts[chunk["source"]] = counts.get(chunk["source"], 0) + 1
    return counts


def main():
    if "AGENT" not in SYSTEM or "CLASSIFY" in SYSTEM or "POLICY ASSISTANT" in SYSTEM:
        raise AssertionError("agent system prompt markers")
    chunks = load_chunks(overlap=10)
    counts = _chunk_counts(chunks)
    print(f"chunks_total={len(chunks)} chunk_counts={counts}")
    if len(chunks) != 15 or counts != {
        "delivery_policy.txt": 3,
        "escalation_policy.txt": 3,
        "payment_policy.txt": 3,
        "privacy_policy.txt": 2,
        "refund_policy.txt": 4,
    }:
        raise AssertionError(f"chunk counts {counts}")

    index = build_default_index()
    probe = retrieve("refund policy late delivery", k=3)
    print("probe_sources=" + ",".join(hit["source"] for hit in probe))
    if any(hit["source"] != "refund_policy.txt" for hit in probe):
        raise AssertionError("probe query did not retrieve the refund policy")

    overlap_eval = evaluate(index)
    overlap_scores = evaluation_scores(overlap_eval)
    _print_eval("overlap_10", overlap_scores, overlap_eval)
    if overlap_scores != {"doc_hits": 12, "fact_hits": 12, "answerable": 12, "fallback_hits": 2, "unanswerable": 2}:
        raise AssertionError(f"overlap 10 evaluation {overlap_scores}")

    zero_index = PolicyIndex(load_chunks(overlap=0))
    zero_eval = evaluate(zero_index)
    zero_scores = evaluation_scores(zero_eval)
    _print_eval("overlap_0", zero_scores, zero_eval)
    if int((zero_eval["expected_doc"].ne("NONE") & (~zero_eval["doc_hit"] | ~zero_eval["fact_hit"])).sum()) < 1:
        raise AssertionError("overlap 0 should break at least one answerable question")

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    STATE.mkdir(parents=True, exist_ok=True)
    for path in STATE.glob("*.json"):
        path.unlink()
    ledger = OUTPUTS / "refund_ledger.jsonl"
    runs_path = OUTPUTS / "agent_runs.jsonl"
    ledger.write_text("", encoding="utf-8")
    runs_path.write_text("", encoding="utf-8")

    scenarios = pd.read_csv(RAW / "agent_scenarios.csv")
    print("=== SCENARIOS ===")
    for row in scenarios.itertuples(index=False):
        result = run_agent(row.message, row.scenario_id)
        record = {
            "scenario_id": row.scenario_id,
            "status": result["status"],
            "final_answer": result["final_answer"],
            "llm_calls": result["llm_calls"],
            "tokens_used": result["tokens_used"],
            "refund_inr": result["refund_inr"],
            "steps": result["steps"],
        }
        with runs_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        expected_status = "FALLBACK_MAX_ITERATIONS" if row.expected_outcome == "FALLBACK" else row.expected_outcome
        amount_ok = abs(float(result["refund_inr"]) - float(row.expected_refund_inr)) <= 0.01
        print(
            f"{row.scenario_id} status={result['status']} refund={result['refund_inr']:.2f} "
            f"calls={result['llm_calls']} tokens={result['tokens_used']} tools={result['tools_executed']}"
        )
        if result["status"] != expected_status or not amount_ok:
            raise AssertionError(f"{row.scenario_id} expected {expected_status} {row.expected_refund_inr}")
        if row.scenario_id == "S1" and result["llm_calls"] != 5:
            raise AssertionError("S1 should take 5 model calls")
        if row.scenario_id == "S7" and (result["llm_calls"] != 6 or result["tokens_used"] > 6000):
            raise AssertionError("S7 should finish six calls inside the default token limit")

    s1 = scenarios.loc[scenarios["scenario_id"].eq("S1")].iloc[0]
    token_demo = run_agent(s1.message, "S1", token_limit=200, run_id="S1-token-limit")
    print(f"demo_token_status={token_demo['status']} demo_token_calls={token_demo['llm_calls']}")
    if token_demo["status"] != "FALLBACK_TOKEN_LIMIT":
        raise AssertionError("token limit demo")

    timeout_demo = run_agent(s1.message, "S1", time_limit_seconds=0, run_id="S1-timeout")
    print(f"demo_timeout_status={timeout_demo['status']} demo_timeout_calls={timeout_demo['llm_calls']}")
    if timeout_demo["status"] != "FALLBACK_TIMEOUT" or timeout_demo["llm_calls"] != 0:
        raise AssertionError("timeout demo")

    try:
        run_agent(s1.message, "S1", run_id="S1-crash", crash_after_tools=2)
        raise AssertionError("the crash demonstration did not stop")
    except SimulatedCrash:
        resumed = resume_agent("S1-crash")
    print(
        f"crash_tools={resumed['tools_executed']} crash_status={resumed['status']} "
        f"crash_refund={resumed['refund_inr']:.2f}"
    )
    if resumed["tools_executed"] != ["get_order", "search_policy", "calculate_refund", "issue_refund"]:
        raise AssertionError("crash and resume repeated or skipped a tool")
    if resumed["status"] != "REFUND" or abs(resumed["refund_inr"] - 128.25) > 0.01:
        raise AssertionError("crash and resume final result")
    print("PART D COMPLETE")


if __name__ == "__main__":
    main()
