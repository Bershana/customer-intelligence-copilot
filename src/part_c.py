"""Part C: redact support tickets, classify them with the offline model, and route them."""
import json
import random
import re
import sys
import time

import pandas as pd

from src.paths import OUTPUTS, RAW, ROOT

sys.path.insert(0, str(ROOT))
import mock_llm  # noqa: E402
from mock_llm import TransientLLMError, chat, est_tokens  # noqa: E402

ALLOWED = ("billing", "delivery", "technical", "fraud")
ALLOWED_SET = set(ALLOWED)
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
PHONE_RE = re.compile(r"\b[6-9]\d{9}\b")
ORDER_RE = re.compile(r"ORD-\d{6}")
BASE_DELAY = 0.05
MAX_INPUT_TOKENS = 400
HUMAN_STATUSES = {"skipped_empty", "flagged_long", "flagged_invalid_output", "budget_exhausted"}

ZERO_SHOT = (
    "CLASSIFY the support ticket into exactly one category: billing, delivery, technical, fraud. "
    "Reply with JSON only in the form {\"category\": \"...\", \"reason\": \"...\"}. "
    "Do not add any words outside that JSON object."
)
FEW_SHOT = (
    "CLASSIFY the support ticket into exactly one category: billing, delivery, technical, fraud. "
    "Reply with JSON only in the form {\"category\": \"...\", \"reason\": \"...\"}. "
    "EXAMPLES:\n"
    "A customer reports an unauthorized charge and says the account was hacked. Category: fraud.\n"
    "A customer says the rider has not arrived and the delivery is late. Category: delivery.\n"
)


def redact(text):
    text, emails = EMAIL_RE.subn("[EMAIL]", text)
    text, phones = PHONE_RE.subn("[PHONE]", text)
    text, orders = ORDER_RE.subn("[ORDER_ID]", text)
    return text, emails, phones, orders


def parse_reply(content):
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        words = re.findall(r"[a-z]+", content.lower())
        found = [word for word in words if word in ALLOWED_SET]
        if len(set(found)) == 1:
            return "classified_keyword", found[0]
        return "flagged_invalid_output", ""
    category = payload.get("category") if isinstance(payload, dict) else None
    if category in ALLOWED_SET:
        return "classified_json", category
    return "flagged_invalid_output", ""


def call_with_retry(messages, ticket_id, error_count):
    """Retry a transient failure. The first retry is numbered 0."""
    retry_number = 0
    while True:
        try:
            return chat(messages)
        except TransientLLMError:
            error_count[0] += 1
            if retry_number >= 3:
                raise RuntimeError(f"permanent model error for {ticket_id}")
            wait = BASE_DELAY * (2 ** retry_number) * random.uniform(0.9, 1.15)
            low = BASE_DELAY * (2 ** retry_number) * 0.9
            high = BASE_DELAY * (2 ** retry_number) * 1.15
            if not (low <= wait <= high):
                raise AssertionError(f"retry wait {wait} outside {low}..{high}")
            print(f"retry ticket_id={ticket_id} retry_number={retry_number} wait_seconds={wait:.4f}")
            time.sleep(wait)
            retry_number += 1


def load_segments():
    segments = pd.read_csv(OUTPUTS / "customer_segments.csv")
    segments["is_premium"] = segments["is_premium"].map(lambda value: str(value).strip().lower() == "true")
    return segments.set_index("customer_id")


def route_ticket(status, category, is_premium):
    if status in HUMAN_STATUSES:
        return "NORMAL", "human_review"
    if category == "fraud":
        return "HIGH", "security_team"
    if is_premium:
        return "HIGH", "premium_desk"
    return "NORMAL", "support_queue"


def run_pipeline(tickets, system_prompt, token_budget, segments):
    if "CLASSIFY" not in system_prompt:
        raise AssertionError("system prompt must contain CLASSIFY")
    if "EXAMPLES:" in system_prompt and system_prompt is ZERO_SHOT:
        raise AssertionError("zero-shot prompt must not contain EXAMPLES:")
    mock_llm.reset()
    random.seed(42)
    running_tokens = 0
    error_count = [0]
    rows = []
    for ticket in tickets.itertuples(index=False):
        text = "" if ticket.text is None else str(ticket.text)
        customer_id = ticket.customer_id
        if customer_id in segments.index:
            segment_name = str(segments.at[customer_id, "segment_name"])
            is_premium = bool(segments.at[customer_id, "is_premium"])
        else:
            segment_name = ""
            is_premium = False
        base = {
            "ticket_id": ticket.ticket_id,
            "customer_id": customer_id,
            "true_category": ticket.true_category,
            "segment_name": segment_name,
            "tokens_used": 0,
        }
        if text.strip() == "":
            base.update(status="skipped_empty", category="")
        elif est_tokens(text) > MAX_INPUT_TOKENS:
            base.update(status="flagged_long", category="")
        elif running_tokens >= token_budget:
            base.update(status="budget_exhausted", category="")
        else:
            redacted, _, _, _ = redact(text)
            if EMAIL_RE.search(redacted) or PHONE_RE.search(redacted) or ORDER_RE.search(redacted):
                raise AssertionError(f"personal data remained in the prompt for {ticket.ticket_id}")
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": redacted},
            ]
            if messages[1]["content"] != redacted:
                raise AssertionError("user message must be the redacted ticket text only")
            response = call_with_retry(messages, ticket.ticket_id, error_count)
            used = int(response.usage.total_tokens)
            running_tokens += used
            status, category = parse_reply(response.choices[0].message.content)
            base.update(status=status, category=category, tokens_used=used)
        priority, route = route_ticket(base["status"], base["category"], is_premium)
        if base["status"] in HUMAN_STATUSES:
            base["category"] = ""
        base["priority"] = priority
        base["route"] = route
        rows.append(base)
    return pd.DataFrame(rows), running_tokens, error_count[0]


def redact_report(tickets):
    emails = phones = orders = 0
    redacted_texts = []
    for text in tickets["text"].fillna(""):
        redacted, n_email, n_phone, n_order = redact(str(text))
        emails += n_email
        phones += n_phone
        orders += n_order
        redacted_texts.append(redacted)
    leftover = 0
    for redacted in redacted_texts:
        leftover += len(EMAIL_RE.findall(redacted))
        leftover += len(PHONE_RE.findall(redacted))
        leftover += len(ORDER_RE.findall(redacted))
    print(
        f"redacted_emails={emails} redacted_phones={phones} "
        f"redacted_order_ids={orders} redacted_total={emails + phones + orders} leftover={leftover}"
    )
    if (emails, phones, orders, emails + phones + orders, leftover) != (55, 91, 132, 278, 0):
        raise AssertionError("redaction counts")
    return emails, phones, orders


def summarise(name, frame):
    counts = frame["status"].value_counts().to_dict()
    classified = frame["status"].isin(["classified_json", "classified_keyword"])
    compared = frame.loc[classified]
    correct = int((compared["category"] == compared["true_category"]).sum())
    total = int(classified.sum())
    accuracy = correct / total if total else 0.0
    tokens = int(frame["tokens_used"].sum())
    per_ticket = tokens / total if total else 0.0
    print(
        f"{name} classified={total} correct={correct} accuracy={accuracy:.3%} "
        f"tokens={tokens} tokens_per_classified={per_ticket:.2f}"
    )
    print(f"{name} status_counts={counts}")
    if total:
        confusion = pd.crosstab(compared["true_category"], compared["category"], dropna=False)
        print(f"{name} confusion")
        print(confusion.to_string())
    return {
        "classified": total,
        "correct": correct,
        "accuracy": accuracy,
        "tokens": tokens,
        "tokens_per_classified": per_ticket,
        "counts": counts,
    }


def main():
    if "EXAMPLES:" in ZERO_SHOT:
        raise AssertionError("zero-shot prompt contains EXAMPLES:")
    if "EXAMPLES:" not in FEW_SHOT or "fraud" not in FEW_SHOT.lower():
        raise AssertionError("few-shot prompt needs EXAMPLES: and a fraud example")

    tickets = pd.read_csv(RAW / "tickets.csv", keep_default_na=False)
    redact_report(tickets)
    empty = int(tickets["text"].str.strip().eq("").sum())
    long_mask = tickets["text"].map(est_tokens) > MAX_INPUT_TOKENS
    long_tokens = sorted(est_tokens(text) for text in tickets.loc[long_mask, "text"])
    print(f"empty_tickets={empty} long_tickets={int(long_mask.sum())} long_token_estimates={long_tokens}")
    if empty != 4 or int(long_mask.sum()) != 4 or long_tokens != [734, 734, 734, 734]:
        raise AssertionError("empty or long ticket guardrails")

    segments = load_segments()
    # Count transient errors by wrapping the pipeline. The mock exposes no counter,
    # so the retry log lines are the record. A full eligible run must recover 19.
    zero, _zero_tokens, zero_errors = run_pipeline(tickets, ZERO_SHOT, token_budget=10**9, segments=segments)
    zero_summary = summarise("zero_shot", zero)
    print(f"zero_shot_transient_errors={zero_errors}")
    if zero_errors != 19 or zero_summary["counts"].get("classified_json") != 135:
        raise AssertionError("zero-shot status counts or transient errors")
    if zero_summary["counts"].get("classified_keyword") != 12 or zero_summary["counts"].get("flagged_invalid_output") != 9:
        raise AssertionError(f"zero-shot validation counts {zero_summary['counts']}")
    if zero_summary["classified"] != 147 or abs(zero_summary["accuracy"] - 0.857) > 0.005:
        raise AssertionError("zero-shot accuracy")
    wrong = zero.loc[
        zero["status"].isin(["classified_json", "classified_keyword"])
        & zero["category"].ne(zero["true_category"])
    ]
    bad_pairs = sorted((f"{row.true_category}->{row.category}" for row in wrong.itertuples()))
    print(f"zero_shot_errors={bad_pairs}")
    if bad_pairs != ["fraud->billing"] * 11 + ["technical->billing"] * 10:
        raise AssertionError(f"unexpected zero-shot errors {bad_pairs}")

    few, _few_tokens, few_errors = run_pipeline(tickets, FEW_SHOT, token_budget=10**9, segments=segments)
    print(f"few_shot_transient_errors={few_errors}")
    if few_errors != 19:
        raise AssertionError("few-shot transient errors")
    few_summary = summarise("few_shot", few)
    if few_summary["classified"] != 147 or abs(few_summary["accuracy"] - 1.0) > 0.005:
        raise AssertionError("few-shot accuracy")

    demo_budget = 6000
    demo, _demo_tokens, _demo_errors = run_pipeline(tickets, ZERO_SHOT, token_budget=demo_budget, segments=segments)
    exhausted = int(demo["status"].eq("budget_exhausted").sum())
    if exhausted < 1:
        demo_budget = 5000
        demo, _demo_tokens, _demo_errors = run_pipeline(tickets, ZERO_SHOT, token_budget=demo_budget, segments=segments)
        exhausted = int(demo["status"].eq("budget_exhausted").sum())
    print(f"budget_demo_limit={demo_budget} budget_exhausted={exhausted}")
    if exhausted < 1:
        raise AssertionError("token budget did not stop the run")
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    columns = [
        "ticket_id",
        "customer_id",
        "status",
        "category",
        "segment_name",
        "priority",
        "route",
        "tokens_used",
    ]
    demo[columns].to_csv(OUTPUTS / "triage_budget_demo.csv", index=False)

    final, _final_tokens, _final_errors = run_pipeline(tickets, FEW_SHOT, token_budget=100000, segments=segments)
    final_summary = summarise("few_shot_budget_100000", final)
    if len(final) != 164:
        raise AssertionError("triage result row count")
    if int(final["status"].eq("budget_exhausted").sum()) != 0:
        raise AssertionError("final run should not exhaust a budget of 100000")
    if final_summary["classified"] != 147:
        raise AssertionError("final classified count")
    final[columns].to_csv(OUTPUTS / "triage_results.csv", index=False)
    print("PART C COMPLETE")


if __name__ == "__main__":
    main()
