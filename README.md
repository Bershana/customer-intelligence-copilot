# Instamart Customer Intelligence & Support Copilot

## Overview

This repository is one internal tool for a fictional grocery-delivery business. It cleans the raw order export, segments customers, scores who is likely to stop ordering, triages support tickets, answers policy questions from the policy text, and decides simple refund requests. All data is synthetic. Every amount is in INR. Every language-model call goes through the offline file `mock_llm.py`. No API key and no network call to a model are required.

Part A writes the clean tables and `data/instamart.db`. Part B reads those tables and writes the segment file. Part C reads the raw tickets and that segment file. Part D reads the database and reuses the policy retrieval built in the same part. `generate_data.py` and `mock_llm.py` are the supplied files and are unmodified.

## Reproduce

Python 3.11 or newer is required. These commands were run with Python 3.11.9. Run them from the repository root, in this order.

1. `python -m venv .venv`
2. `.venv\Scripts\Activate.ps1` on Windows PowerShell. On macOS or Linux, run `source .venv/bin/activate` instead.
3. `python -m pip install -r requirements.txt`
4. `python generate_data.py`
5. `python -m src.part_a`
6. `python -m src.validate`
7. `python -m src.part_b`
8. `python -m src.part_c`
9. `python -m src.part_d`

Step 4 ends with this exact line:

`generated: customers=1200 products=40 orders_raw=13084 tickets=164`

Step 6 prints exactly `ALL CHECKPOINTS PASSED`.

## Part A

### Audit

| Table | Rows | Columns |
| --- | ---: | ---: |
| customers | 1,200 | 5 |
| products | 40 | 4 |
| orders | 13,084 | 9 |

Customers dtypes: `customer_id` str, `city` str, `signup_date` str, `age` float64, `gender` str. Products dtypes: `product_id` str, `product_name` str, `category` str, `price_inr` int64. Orders dtypes: `order_id` str, `customer_id` str, `product_id` str, `quantity` int64, `order_ts` str, `status` str, `discount_pct` int64, `order_amount` str, `delivery_minutes` float64.

Missing values. A blank cell is missing. Pandas also reads the token `N/A` as missing.

| Table | Column | Missing | Percent |
| --- | --- | ---: | ---: |
| customers | customer_id | 0 | 0.00 |
| customers | city | 0 | 0.00 |
| customers | signup_date | 0 | 0.00 |
| customers | age | 61 | 5.08 |
| customers | gender | 30 | 2.50 |
| products | every column | 0 | 0.00 |
| orders | every column except the two below | 0 | 0.00 |
| orders | order_amount | 53 | 0.41 |
| orders | delivery_minutes | 2,541 | 19.42 |

The 53 missing amounts are the token `N/A`. Another 67 amounts are the text `error`, which is present but not a number. Those two groups are the 120 amounts that cannot be converted.

| Check | Value |
| --- | ---: |
| Raw order rows | 13,084 |
| Exact duplicate rows | 119 |
| Pending rows | 656 |
| Distinct order_id | 12,309 |
| Amounts that are not numbers | 120 |
| Negative delivery_minutes | 106 |
| Blank delivery_minutes | 2,541 |
| Blank age | 61 |
| Age outside 10–100 | 6 |
| Distinct raw city spellings | 24 |
| Distinct non-blank raw gender spellings | 10 |
| Blank gender | 30 |

### Cleaning rules

Customers, in order:

1. Strip spaces and apply title case to city. This changed 476 city values and left six cities: Pune 227, Delhi 210, Bengaluru 197, Chennai 191, Hyderabad 191, Mumbai 184.
2. Map every gender spelling to `male`, `female`, or `unknown`. Blank becomes `unknown`. This changed 950 values. The result is female 610, male 560, unknown 30.
3. Ages outside 10–100 were set to missing (6 values). All missing ages were then filled with the median age of that customer's city (67 values: the 6 invalid ages plus 61 blanks). After this step there are 0 missing ages, and every age is between 18 and 60.
4. `signup_date` was converted to a datetime. No row was dropped.

Orders, in order:

1. `order_ts` was converted to a datetime.
2. Exact duplicate rows were removed: 119 rows. The table fell from 13,084 to 12,965.
3. For each `order_id`, only the row with the latest `order_ts` was kept. This removed 656 older rows, almost all of them the earlier `pending` copy of an order. The table fell to 12,309 rows, one per order.
4. `order_amount` was converted to a number. Unparseable text became missing, and each missing amount was replaced with `round(price_inr × quantity × (1 − discount_pct / 100), 2)`. This filled 112 amounts. Some of the original 120 bad amounts had already been removed with the duplicate rows.
5. Negative `delivery_minutes` were set to missing (105 values; one of the original 106 sat on a row removed earlier). For completed orders, each remaining missing delivery time was filled with the median delivery time of that product category among completed orders (774 values). Cancelled orders were left missing, because a cancelled order was never delivered.

After cleaning: 12,309 rows; 11,110 completed and 1,199 cancelled; no pending rows; 0 missing amounts; 0 missing delivery times on completed orders and 1,199 missing on cancelled orders. The sum of `order_amount` over completed orders is INR 2,998,668.80.

The three clean tables are in `data/clean/` and in `data/instamart.db` as `customers`, `products`, and `orders`. `order_ts` is stored as text `YYYY-MM-DD HH:MM:SS`.

### SQL

| Query | Result |
| --- | --- |
| Q1 | 30 customers have completed orders above INR 500 whose total of those orders exceeds INR 8,000 |
| Q2 | 30 city-category rows; revenues sum to INR 2,998,668.80; the largest cell is Bengaluru / Electronics at INR 233,909.45 |
| Q3 | 11 products are priced above the average catalogue price |
| Q4 | 25 customers have no completed order |
| Q5 | 550 customers spend more than the average customer total; that average is INR 2,552.06 |
| Q6 | Top customer in each city: Bengaluru C0082 (INR 12,856.20), Chennai C0930 (INR 10,151.40), Delhi C0291 (INR 13,742.60), Hyderabad C0743 (INR 12,730.05), Mumbai C0693 (INR 14,766.55), Pune C0426 (INR 10,377.10); 8 rows have RANK() different from ROW_NUMBER() |
| Q7 | 183 daily rows; the last cumulative completed revenue is INR 2,998,668.80 |

Ties cause the Q6 gap because RANK() gives the same rank to equal totals and then skips, while ROW_NUMBER() still assigns a different integer to each tied row.

### Pandas cross-check

The Q2 pivot table with margins has grand total INR 2,998,668.80, the same as the SQL total.

Completed `order_amount` by city:

| City | Sum (INR) | Mean (INR) | Orders |
| --- | ---: | ---: | ---: |
| Bengaluru | 523,643.40 | 276.47 | 1,894 |
| Chennai | 463,629.15 | 272.72 | 1,700 |
| Delhi | 502,829.80 | 243.62 | 2,064 |
| Hyderabad | 528,589.90 | 284.95 | 1,855 |
| Mumbai | 479,058.50 | 297.00 | 1,613 |
| Pune | 500,918.05 | 252.48 | 1,984 |

Completed `order_amount`: mean INR 269.91, median INR 135.00, skewness 4.61, Q1 INR 64.80, Q3 INR 297.00. The 1.5×IQR fences are INR -283.50 and INR 645.30. There are 897 orders above the upper fence and none below the lower fence.

Correlation on completed orders:

|  | order_amount | discount_pct | quantity | delivery_minutes |
| --- | ---: | ---: | ---: | ---: |
| order_amount | 1.000 | -0.095 | 0.087 | -0.003 |
| discount_pct | -0.095 | 1.000 | -0.000 | -0.010 |
| quantity | 0.087 | -0.000 | 1.000 | -0.013 |
| delivery_minutes | -0.003 | -0.010 | -0.013 | 1.000 |

The completed-order table is synthetic Instamart demand from 1 April 2025 through 30 September 2025 in six cities. Each row is one completed order with an amount in INR, a discount, a quantity and a delivery time, joined to one of five product categories. The amount distribution is right-skewed: the mean is INR 269.91 and the median is INR 135.00, with skewness 4.61. The quartiles are INR 64.80 and INR 297.00, so a typical basket is small, while a long tail of electronics pulls the mean up. The 1.5×IQR fences are INR -283.50 and INR 645.30, with 897 completed orders above the upper fence and none below it. Those high amounts are expensive catalogue items that remain after cleaning. The largest absolute correlation is order_amount with discount_pct at -0.095. order_amount with quantity is +0.087, which is smaller in absolute value. A correlation of this size does not establish causation.

## Part B

### RFM on 1 October 2025

There are 1,175 customers with at least one completed order. Recency has median 19, minimum 0 and maximum 180 days. Frequency has mean 9.46 and maximum 33. Monetary value has mean INR 2,552.06 and maximum INR 14,766.55.

### K-means

RFM was standardised with `StandardScaler`. Each fit used `KMeans(n_clusters=K, n_init=10, random_state=42)`.

| K | Inertia | Silhouette |
| ---: | ---: | ---: |
| 2 | 1,820.8 | 0.465 |
| 3 | 1,268.2 | 0.445 |
| 4 | 743.1 | 0.526 |
| 5 | 617.4 | 0.480 |
| 6 | 503.2 | 0.482 |
| 7 | 408.5 | 0.416 |
| 8 | 366.5 | 0.412 |

K = 4 was chosen. It has the highest silhouette in the table, the inertia drop from K = 3 to K = 4 is the last large one, and four groups are enough for a retention team to act on. The K = 5 silhouette is 0.480, inside 0.05 of the reference 0.481.

Inertia at K = 8 is lower than inertia at K = 2 because each added cluster can sit closer to some of the points, so the within-cluster sum of squares falls. That fall continues until every customer is their own cluster and inertia is zero. The lowest inertia therefore cannot choose K.

### Segment profile

Cluster numbers are labels only. They have no rank order. `is_premium` is true only for the segment with the highest mean monetary value, which is segment 1 (138 customers).

| segment_id | Name | Customers | Mean recency (days) | Mean frequency | Mean monetary (INR) | Action |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 0 | Occasional | 436 | 28.86 | 3.77 | 1,057.62 | Offer a small discount on the next grocery order |
| 1 | Premium | 138 | 24.12 | 6.77 | 7,129.70 | Give this group a dedicated desk and early access to high-value items |
| 2 | Frequent staples | 419 | 8.77 | 19.40 | 3,410.02 | Send refill reminders and grocery-bundle offers |
| 3 | At risk | 182 | 109.04 | 2.20 | 685.99 | Send a win-back offer while the customer can still be reached |

The 138-customer Premium segment has mean monetary value about INR 7,130 and mean frequency about 6.8. The 419-customer Frequent staples segment has mean frequency about 19.4 and mean recency about 8.8 days.

PCA on the scaled RFM matrix explains 0.634 and 0.218 of the variance, 0.852 in total. `outputs/customer_segments.csv` has one row per customer with `customer_id`, `recency`, `frequency`, `monetary`, `segment_id`, `segment_name`, and `is_premium`.

### Lapse definition and split

The cutoff is 1 September 2025. The cohort is every customer with at least one completed order before that cutoff: 1,156 customers, of whom 443 (38.32 percent) placed no completed order from 1 September through 30 September. Features use only orders dated before the cutoff, plus the clean customer table. They are recency, frequency, monetary value, average discount, cancellation rate, average delivery minutes, favourite category, city, age, gender, and tenure in days. No order dated 1 September 2025 or later entered the features. Segment labels from the whole-window clustering were not used as features. No feature column contains a missing value.

`train_test_split(test_size=0.2, stratify=y, random_state=42)` gives 924 training rows and 232 test rows. Always predicting "not lapsed" on the test set has accuracy 61.6 percent and a lapsed-class F1 of 0.

City, gender, and favourite category are one-hot encoded with `OneHotEncoder(drop="first")` inside a pipeline, and the numeric features are standardised. The encoder and the scaler are fitted on training rows only. In the fitted logistic scaler the training mean of frequency is 8.1158, while the full-cohort mean of frequency is 8.0285. They differ because the scaler never saw the 232 test rows.

### Models

The baseline is the "not lapsed" prediction above.

Logistic regression (`max_iter=1000`) was tuned with `GridSearchCV` over C in {0.01, 0.1, 1, 10, 100}, `cv=5`, `scoring="f1"`. The best C is 0.1, with cross-validated F1 0.6392.

The random forest (`n_estimators=100`, `random_state=42`) was tuned over `max_depth` in {3, 5, 8} and `min_samples_leaf` in {5, 10, 20}, the same cross-validation and F1 scoring (45 fits). The best depth is 3 and the best leaf size is 5, with cross-validated F1 0.6748.

Lapsed-class results on the untouched test set. The confusion counts are true negative, false positive, false negative, true positive.

| Model | Precision | Recall | F1 | Confusion |
| --- | ---: | ---: | ---: | --- |
| Logistic regression, C = 0.1 | 0.6545 | 0.8090 | 0.7236 | 105, 38, 17, 72 |
| Random forest, depth 3, leaf 5 | 0.6804 | 0.7416 | 0.7097 | 112, 31, 23, 66 |

Logistic regression is the selected model. Its lapsed-class F1 on the test set is 0.72, against 0.71 for the forest, and both clear the baseline F1 of 0 and the 0.55 bar. The logistic model misses 17 lapsing customers and the forest misses 23. Missing a customer who is about to stop ordering costs later baskets, while contacting a loyal customer only spends a small offer, so the higher recall is the better trade. The saved file is this logistic regression.

### Class balance

The training set was oversampled only. Every original training row was kept, and lapsed training rows were drawn with replacement (`random_state=42`) until both classes had 570 rows. The best logistic regression (C = 0.1) was refit on that balanced training set and scored on the original test set.

| Fit | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| Original training set | 0.6545 | 0.8090 | 0.7236 |
| Balanced training set | 0.5870 | 0.9101 | 0.7137 |

Recall moved the most. It rose by about 0.10, while precision fell from 0.65 to 0.59 and did not rise. For the retention team, the balanced model catches more people who are about to leave and also writes to more loyal customers. The saved model remains the original tuned logistic regression. The balanced fit is the experiment, not a replacement.

### Interpretation and the retention list

No encoder, scaler, or resampler saw the test rows.

The five largest random-forest importances are frequency 0.3522, monetary 0.2887, recency 0.1150, cancellation rate 0.0531, and average discount 0.0407.

On the selected logistic regression the frequency coefficient is negative (-1.2811): more completed orders before September lower the lapse score. The monetary coefficient is positive and small (0.0294): once order count is held fixed, a higher spend slightly raises the lapse score, so spend by itself is not loyalty. The recency coefficient is negative and small (-0.0285): after frequency is in the model, an extra day since the last order does not raise the lapse score.

`outputs/lapse_model.joblib` was reloaded with `joblib.load`. The reloaded model reproduced the original test predictions exactly.

Among test-set customers in the Premium segment, the ten highest lapse probabilities are in `outputs/retention_list.csv`. Two of those ten had actually lapsed. The Premium flag comes from the whole-window RFM, which includes September, so the retention list is illustrative and not a leakage-free evaluation. The model's own features still use only orders before the cutoff.

## Part C

Tickets are processed in file order. Empty or whitespace text is `skipped_empty` and is never sent to the model (4 tickets). Text above 400 estimated tokens is `flagged_long` and is never sent (4 tickets, each 734 tokens).

Before a prompt is built, email, phone, and order id are replaced, in that order, with `[EMAIL]`, `[PHONE]`, and `[ORDER_ID]`. Across all 164 tickets that is 55 emails, 91 phone numbers, and 132 order ids, 278 substitutions in total. Applying the three patterns again finds 0 matches. The user message is exactly the redacted text. The code asserts that no email, phone, or order id remains in that message.

Both prompts contain `CLASSIFY`, name billing, delivery, technical, and fraud, and require JSON of the form `{"category": "...", "reason": "..."}`. The zero-shot prompt does not contain `EXAMPLES:`. The few-shot prompt contains `EXAMPLES:` and two labelled examples, including a fraud example.

Each model call retries up to 3 times. The wait before retry number n is `0.05 × 2^n × random.uniform(0.9, 1.15)`. `mock_llm.reset()` runs at the start of every full pass. A full pass over the 156 eligible tickets raises 19 `TransientLLMError`s. Every one is retry number 0, the logged wait is between 0.045 and 0.0575 seconds, and every ticket recovers. No ticket ends in a permanent error.

Valid JSON with one of the four categories is `classified_json`. If the reply is not JSON, lowercase words are extracted with `re.findall(r"[a-z]+", text.lower())`, and the reply is `classified_keyword` only when exactly one allowed category word is present. Every other reply, including JSON whose category is outside the list, is `flagged_invalid_output` and goes to a human. The model is not asked to guess. One full run produces 135 `classified_json`, 12 `classified_keyword`, and 9 `flagged_invalid_output`.

| Run | Classified | Accuracy | Tokens | Tokens per classified ticket |
| --- | ---: | ---: | ---: | ---: |
| Zero-shot, no budget | 147 | 85.7 percent (126 of 147) | 10,960 | 74.56 |
| Few-shot, no budget | 147 | 100 percent (147 of 147) | 15,172 | 103.21 |

Zero-shot confusion, counted only on the 147 classified tickets:

| True category | Predicted billing | Predicted delivery | Predicted fraud | Predicted technical |
| --- | ---: | ---: | ---: | ---: |
| billing | 38 | 0 | 0 | 0 |
| delivery | 0 | 36 | 0 | 0 |
| fraud | 11 | 0 | 23 | 0 |
| technical | 10 | 0 | 0 | 29 |

The zero-shot run is cheaper, at 74.56 tokens per classified ticket, but 21 of 147 tickets are wrong and every error is a fraud ticket called billing (11) or a technical ticket called billing (10). The few-shot prompt costs 103.21 tokens per classified ticket, about 29 tokens more, and 15,172 tokens in total against 10,960. Those extra tokens are worth paying: the fraud example corrects all 11 fraud misses, and a fraud ticket sent to billing is a worse failure than the extra 4,212 tokens.

The zero-shot budget demonstration uses `TOKEN_BUDGET = 6000` and is saved as `outputs/triage_budget_demo.csv`. 70 tickets end as `budget_exhausted`.

`outputs/triage_results.csv` is the few-shot run with `TOKEN_BUDGET = 100000`. It has 164 rows and does not exhaust the budget. Routing uses `outputs/customer_segments.csv`. A customer with no segment row is not premium. Empty, long, invalid, and budget-exhausted tickets go to `human_review` at `NORMAL` priority (17 tickets). Other fraud tickets go to `security_team` at `HIGH` (34). Other premium customers go to `premium_desk` at `HIGH` (11). Everyone else goes to `support_queue` at `NORMAL` (102). Category is blank on the 17 tickets that were not classified. JSON and keyword replies are the rule-validated outputs. Invalid output, empty text, long text, and a spent budget go to a human without a guessed category.

## Part D

### Retrieval

The five policy files are split into 40-word chunks with an overlap of 10 words. There are 15 chunks: refund 4, delivery 3, escalation 3, payment 3, privacy 2. A `TfidfVectorizer` is fitted on those chunks, and `retrieve(query, k=3)` ranks them by cosine similarity. For the query `refund policy late delivery`, the top three chunks are all from `refund_policy.txt`.

Term frequency is how often a word appears in a chunk, and inverse document frequency gives more weight to a word that appears in fewer chunks. TF-IDF misses two phrasings that mean the same thing but use different words, so a question can fail to find the policy sentence that answers it.

`answer(question)` retrieves three chunks and sends a system message that contains `POLICY ASSISTANT`, telling the model to answer only from the context and to say when it cannot find the information. The user message is `CONTEXT:`, then each chunk prefixed by its file name, then `QUESTION:` and the question.

| Setting | Expected document in top 3 | Answer contains the expected fact | Unanswerable questions get the fallback |
| --- | --- | --- | --- |
| Overlap 10 | 12/12 | 12/12 | 2/2 |
| Overlap 0 | 12/12 | 10/12 | 2/2 |

With overlap 0, the question "What delivery time makes a completed order eligible for a full refund?" still retrieves `refund_policy.txt`, but the answer is the fragment "receives a full refund of the order amount." The sentence that contains "more than 45 minutes" is cut at a chunk boundary, so the returned fragment does not contain that fact. The same boundary cut breaks "How soon is money returned when a payment fails but is deducted?": the answer stops at "returned automatically" and does not contain "48 hours". The 10-word overlap keeps both facts inside a chunk.

### Refund agent

The five tools are `get_order`, `search_policy`, `calculate_refund`, `issue_refund`, and `escalate_to_human`. Each has a one-sentence description and a parameter specification in `TOOLS_SPEC`, and `TOOL_MAP` points the name at the Python function. `get_order` reads `data/instamart.db`. `search_policy` calls the same `retrieve` function as the policy assistant. `calculate_refund` returns the full `order_amount` when delivery took more than 45 minutes, and INR 0 otherwise. `issue_refund` and `escalate_to_human` append one JSON line, with a timestamp and the tool name, to `outputs/refund_ledger.jsonl`.

Only a completed order can be refunded. A refund above INR 1,000 is escalated and is not issued automatically. Every scenario is treated as inside the 7-day window.

| Scenario | Status | Refund (INR) | Model calls | Tokens |
| --- | --- | ---: | ---: | ---: |
| S1 | REFUND | 128.25 | 5 | 1,067 |
| S2 | REFUND | 135.00 | 5 | 1,067 |
| S3 | ESCALATE | 1,999.00 | 5 | 1,054 |
| S4 | DENIED | 0.00 | 4 | 741 |
| S5 | DENIED | 0.00 | 3 | 439 |
| S6 | NOT_FOUND | 0.00 | 2 | 140 |
| S7 | FALLBACK_MAX_ITERATIONS | 0.00 | 6 | 3,204 |

S1 takes 5 model calls. S7 finishes all 6 model calls under the default token limit of 6,000 (it uses 3,204) and then stops with `FALLBACK_MAX_ITERATIONS`. Each fallback includes the steps completed so far and a short human-handoff summary.

The S1 demonstration with `token_limit=200` ends as `FALLBACK_TOKEN_LIMIT`. The demonstration with `time_limit_seconds=0` ends as `FALLBACK_TIMEOUT` before the first model call.

The crash-and-resume run of S1 stops after the second tool, once that checkpoint is on disk, and `resume_agent` continues it. The tools executed across the crash and the resume, in order, are `get_order`, `search_policy`, `calculate_refund`, `issue_refund`. None is repeated. The final status is REFUND and the amount is INR 128.25.

`outputs/agent_runs.jsonl` has one line for each of the seven scenarios, with the status, final answer, model calls, tokens, and every step. The ledger records the issued refunds and the escalation. Checkpoints are in `state/`.

This refund agent is ReAct: each model call sees the latest tool result and then picks the next tool. The seven scenarios took 5, 5, 5, 4, 3, 2 and 6 calls, and 1,067, 1,067, 1,054, 741, 439, 140 and 3,204 tokens. A plan-and-execute design would ask once for the whole step list and replan only after an error. On a normal late delivery the plan could be get order, search policy, calculate, then issue, which is one planning call instead of five. The branch is not known before the first observation. S6 finds no order and stops in 2 calls and 140 tokens. A plan that had already listed a refund would have to be thrown away. S5 is cancelled and stops in 3 calls, after the policy search, because the status is not completed. S3 escalates only after the calculation returns INR 1,999.00, which a plan cannot know in advance. S7 shows the other cost: the mock keeps asking for `search_policy`, and the 6-call cap stops it. A planner that repeats the same search would need that same cap. I would keep ReAct here. The happy path is a handful of short calls, and the cases that matter are exactly the unexpected observations: a missing order, a cancelled order, and a refund above INR 1,000.

## Limitations and Next Steps

The data is synthetic, and `mock_llm.py` is a fixed offline stand-in. The triage accuracy and the policy answers will change if a hosted model replaces it. TF-IDF does not match different words that mean the same thing, which is why a chunk boundary with no overlap already drops two facts. K-means labels depend on scaling and on `random_state=42`, and they are not an ordered ranking. The lapse scores describe this six-month window only. The test F1 values are higher than the brief's illustrative 0.62 and 0.65 because the fitted logistic C and the installed scikit-learn version are not the ones that produced that illustration. Both still clear 0.55, and the clustering checkpoints match.

The Premium flag is a post-hoc whole-window label, so the retention list is illustrative and not a leakage-free evaluation. No encoder, scaler, or resampler saw the test rows, no order from 1 September 2025 onward entered the features, and the segment labels were not model features. A later version should build the Premium flag only from history before the cutoff, calibrate the probabilities, and review the balanced-training recall gain before using it to contact customers. The refund agent should also check the 7-day window against a real delivery date instead of treating every scenario as in time.
