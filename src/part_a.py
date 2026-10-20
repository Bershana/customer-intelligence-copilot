"""Part A: audit the raw exports, clean them, store SQLite tables, run SQL and pandas checks."""
import sqlite3

import numpy as np
import pandas as pd

from src.paths import CLEAN, DB_PATH, RAW, SQL_PATH

GENDER_MAP = {
    "male": "male",
    "m": "male",
    "female": "female",
    "f": "female",
}


def _blank_mask(series):
    text = series.astype("string")
    return series.isna() | text.str.strip().eq("") | text.str.lower().eq("<na>") | text.str.lower().eq("nan")


def missing_report(df):
    rows = []
    n = len(df)
    for col in df.columns:
        miss = int(_blank_mask(df[col]).sum())
        rows.append(
            {
                "column": col,
                "dtype": str(df[col].dtype),
                "missing": miss,
                "missing_pct": round(100.0 * miss / n, 2) if n else 0.0,
            }
        )
    return pd.DataFrame(rows)


def load_raw():
    customers = pd.read_csv(RAW / "customers.csv")
    products = pd.read_csv(RAW / "products.csv")
    orders = pd.read_csv(RAW / "orders.csv")
    return customers, products, orders


def audit(customers, products, orders):
    print("=== AUDIT SHAPES ===")
    for name, df in (("customers", customers), ("products", products), ("orders", orders)):
        print(f"{name}: shape={df.shape}")
        print(missing_report(df).to_string(index=False))

    amount_num = pd.to_numeric(orders["order_amount"], errors="coerce")
    delivery_num = pd.to_numeric(orders["delivery_minutes"], errors="coerce")
    age_num = pd.to_numeric(customers["age"], errors="coerce")
    gender_nonblank = customers.loc[~_blank_mask(customers["gender"]), "gender"]

    figures = {
        "orders_rows": len(orders),
        "exact_duplicate_rows": int(orders.duplicated().sum()),
        "pending_rows": int((orders["status"] == "pending").sum()),
        "distinct_order_id": int(orders["order_id"].nunique()),
        "non_numeric_amounts": int(amount_num.isna().sum()),
        "negative_delivery": int((delivery_num < 0).sum()),
        "blank_delivery": int(delivery_num.isna().sum()),
        "blank_age": int(_blank_mask(customers["age"]).sum()),
        "age_outside_10_100": int(((age_num < 10) | (age_num > 100)).sum()),
        "distinct_city_spellings": int(customers["city"].nunique()),
        "distinct_gender_spellings": int(gender_nonblank.nunique()),
        "blank_gender": int(_blank_mask(customers["gender"]).sum()),
    }
    print("=== AUDIT FIGURES ===")
    for key, value in figures.items():
        print(f"{key}={value}")

    expected = {
        "orders_rows": 13084,
        "exact_duplicate_rows": 119,
        "pending_rows": 656,
        "distinct_order_id": 12309,
        "non_numeric_amounts": 120,
        "negative_delivery": 106,
        "blank_delivery": 2541,
        "blank_age": 61,
        "age_outside_10_100": 6,
        "distinct_city_spellings": 24,
        "distinct_gender_spellings": 10,
        "blank_gender": 30,
    }
    for key, value in expected.items():
        if figures[key] != value:
            raise AssertionError(f"audit {key}: got {figures[key]} expected {value}")
    return figures


def _map_gender(value):
    if pd.isna(value) or str(value).strip() == "":
        return "unknown"
    return GENDER_MAP.get(str(value).strip().lower(), "unknown")


def catalogue_amount(price, quantity, discount_pct):
    return round(float(price) * float(quantity) * (1 - float(discount_pct) / 100), 2)


def clean_customers(customers):
    clean = customers.copy()
    original_city = clean["city"].astype(str)
    clean["city"] = original_city.str.strip().str.title()
    city_changed = int((clean["city"] != original_city).sum())

    original_gender = clean["gender"]
    clean["gender"] = clean["gender"].map(_map_gender)
    gender_changed = int((clean["gender"].astype(str) != original_gender.astype(str)).sum())

    clean["age"] = pd.to_numeric(clean["age"], errors="coerce")
    outside = (clean["age"] < 10) | (clean["age"] > 100)
    ages_set_missing = int(outside.sum())
    clean.loc[outside, "age"] = np.nan
    ages_missing_before_fill = int(clean["age"].isna().sum())
    clean["age"] = clean.groupby("city")["age"].transform(lambda s: s.fillna(s.median()))
    ages_filled = ages_missing_before_fill

    clean["signup_date"] = pd.to_datetime(clean["signup_date"])

    stats = {
        "city_values_changed": city_changed,
        "gender_values_changed": gender_changed,
        "ages_set_missing_outside_range": ages_set_missing,
        "ages_filled_with_city_median": ages_filled,
        "city_counts": clean["city"].value_counts().to_dict(),
        "gender_counts": clean["gender"].value_counts().to_dict(),
        "missing_age": int(clean["age"].isna().sum()),
        "age_min": float(clean["age"].min()),
        "age_max": float(clean["age"].max()),
    }
    print("=== CLEAN CUSTOMERS ===")
    for key, value in stats.items():
        print(f"{key}={value}")

    expected_cities = {
        "Pune": 227,
        "Delhi": 210,
        "Bengaluru": 197,
        "Chennai": 191,
        "Hyderabad": 191,
        "Mumbai": 184,
    }
    expected_genders = {"female": 610, "male": 560, "unknown": 30}
    if stats["city_counts"] != expected_cities:
        raise AssertionError(f"city counts {stats['city_counts']}")
    if stats["gender_counts"] != expected_genders:
        raise AssertionError(f"gender counts {stats['gender_counts']}")
    if stats["missing_age"] != 0 or not (18 <= stats["age_min"] and stats["age_max"] <= 60):
        raise AssertionError(f"age range failed {stats}")
    return clean, stats


def clean_orders(orders, products):
    clean = orders.copy()
    clean["order_ts"] = pd.to_datetime(clean["order_ts"])
    before = len(clean)
    clean = clean.drop_duplicates()
    exact_removed = before - len(clean)

    before = len(clean)
    clean = clean.sort_values(["order_id", "order_ts"]).drop_duplicates("order_id", keep="last")
    older_removed = before - len(clean)

    price = products[["product_id", "price_inr", "category"]]
    clean = clean.merge(price, on="product_id", how="left")
    if clean["price_inr"].isna().any():
        raise AssertionError("an order product_id is missing from the catalogue")

    clean["order_amount"] = pd.to_numeric(clean["order_amount"], errors="coerce")
    missing_amount = clean["order_amount"].isna()
    amounts_imputed = int(missing_amount.sum())
    clean.loc[missing_amount, "order_amount"] = [
        catalogue_amount(price_inr, quantity, discount)
        for price_inr, quantity, discount in zip(
            clean.loc[missing_amount, "price_inr"],
            clean.loc[missing_amount, "quantity"],
            clean.loc[missing_amount, "discount_pct"],
        )
    ]
    clean["order_amount"] = [round(float(v), 2) for v in clean["order_amount"]]

    clean["delivery_minutes"] = pd.to_numeric(clean["delivery_minutes"], errors="coerce")
    negative = clean["delivery_minutes"] < 0
    negatives_cleared = int(negative.sum())
    clean.loc[negative, "delivery_minutes"] = np.nan

    completed = clean["status"].eq("completed")
    category_median = clean.groupby("category")["delivery_minutes"].transform("median")
    need_fill = completed & clean["delivery_minutes"].isna()
    delivery_imputed = int(need_fill.sum())
    clean.loc[need_fill, "delivery_minutes"] = category_median.loc[need_fill]

    clean = clean.drop(columns=["price_inr", "category"])
    clean = clean.sort_values(["order_ts", "order_id"]).reset_index(drop=True)

    completed = clean["status"].eq("completed")
    cancelled = clean["status"].eq("cancelled")
    amount_sum = round(sum(clean.loc[completed, "order_amount"]), 2)
    stats = {
        "exact_duplicate_rows_removed": exact_removed,
        "older_status_rows_removed": older_removed,
        "amounts_imputed": amounts_imputed,
        "negative_delivery_set_missing": negatives_cleared,
        "completed_delivery_imputed": delivery_imputed,
        "rows": len(clean),
        "completed": int(completed.sum()),
        "cancelled": int(cancelled.sum()),
        "pending": int(clean["status"].eq("pending").sum()),
        "missing_amount": int(pd.isna(clean["order_amount"]).sum()),
        "missing_delivery_completed": int(clean.loc[completed, "delivery_minutes"].isna().sum()),
        "missing_delivery_cancelled": int(clean.loc[cancelled, "delivery_minutes"].isna().sum()),
        "completed_amount_sum": amount_sum,
    }
    print("=== CLEAN ORDERS ===")
    for key, value in stats.items():
        print(f"{key}={value}")

    expected = {
        "rows": 12309,
        "completed": 11110,
        "cancelled": 1199,
        "pending": 0,
        "missing_amount": 0,
        "missing_delivery_completed": 0,
        "missing_delivery_cancelled": 1199,
        "completed_amount_sum": 2998668.80,
    }
    for key, value in expected.items():
        if stats[key] != value:
            raise AssertionError(f"orders {key}: got {stats[key]} expected {value}")
    return clean, stats


def store(customers, products, orders):
    CLEAN.mkdir(parents=True, exist_ok=True)
    customers_out = customers.copy()
    customers_out["signup_date"] = customers_out["signup_date"].dt.strftime("%Y-%m-%d")
    orders_out = orders.copy()
    orders_out["order_ts"] = orders_out["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")

    customers_out.to_csv(CLEAN / "customers.csv", index=False)
    products.to_csv(CLEAN / "products.csv", index=False)
    orders_out.to_csv(CLEAN / "orders.csv", index=False)

    if DB_PATH.exists():
        DB_PATH.unlink()
    with sqlite3.connect(DB_PATH) as con:
        customers_out.to_sql("customers", con, index=False, if_exists="replace")
        products.to_sql("products", con, index=False, if_exists="replace")
        orders_out.to_sql("orders", con, index=False, if_exists="replace")
    print(f"stored clean csv files and {DB_PATH.name}")


def load_queries():
    text = SQL_PATH.read_text(encoding="utf-8")
    parts = []
    chunk = []
    for line in text.splitlines():
        if line.startswith("-- Q") and chunk:
            parts.append("\n".join(chunk).strip())
            chunk = [line]
        else:
            chunk.append(line)
    if chunk:
        parts.append("\n".join(chunk).strip())
    queries = []
    for part in parts:
        sql = "\n".join(line for line in part.splitlines() if not line.startswith("--")).strip().rstrip(";")
        queries.append(sql)
    if len(queries) != 7:
        raise AssertionError(f"expected 7 SQL queries, found {len(queries)}")
    return queries


def run_sql():
    queries = load_queries()
    with sqlite3.connect(DB_PATH) as con:
        frames = [pd.read_sql_query(sql, con) for sql in queries]

    q1, q2, q3, q4, q5, q6, q7 = frames
    print("=== SQL ===")
    print(f"Q1_customers={len(q1)}")
    print(f"Q2_rows={len(q2)} Q2_revenue_sum={round(float(q2['revenue'].sum()), 2)}")
    top = q2.loc[q2["revenue"].idxmax()]
    print(f"Q2_largest={top['city']} / {top['category']} / {float(top['revenue']):.2f}")
    print(f"Q3_products={len(q3)}")
    print(f"Q4_customers={len(q4)}")
    avg_total = pd.read_sql_query(
        """
        WITH customer_spend AS (
            SELECT customer_id, SUM(order_amount) AS total_spend
            FROM orders WHERE status = 'completed' GROUP BY customer_id
        )
        SELECT AVG(total_spend) AS avg_total FROM customer_spend
        """,
        sqlite3.connect(DB_PATH),
    ).iloc[0, 0]
    print(f"Q5_customers={len(q5)} Q5_average={round(float(avg_total), 2)}")
    leaders = q6.loc[q6["rank_num"] == 1, ["city", "customer_id", "total_spend"]]
    print("Q6_leaders")
    print(leaders.to_string(index=False))
    rank_gap = int((q6["rank_num"] != q6["row_num"]).sum())
    print(f"Q6_rank_ne_row_number={rank_gap}")
    print(f"Q7_rows={len(q7)} Q7_last_cumulative={float(q7.iloc[-1]['cumulative_revenue']):.2f}")

    if len(q1) != 30:
        raise AssertionError("Q1")
    if len(q2) != 30 or round(float(q2["revenue"].sum()), 2) != 2998668.80:
        raise AssertionError("Q2 sum")
    if not (top["city"] == "Bengaluru" and top["category"] == "Electronics" and abs(float(top["revenue"]) - 233909.45) < 0.01):
        raise AssertionError(f"Q2 largest {top.to_dict()}")
    if len(q3) != 11:
        raise AssertionError("Q3")
    if len(q4) != 25:
        raise AssertionError("Q4")
    if len(q5) != 550 or round(float(avg_total), 2) != 2552.06:
        raise AssertionError(f"Q5 {len(q5)} {avg_total}")
    expected_leaders = {
        "Bengaluru": ("C0082", 12856.20),
        "Chennai": ("C0930", 10151.40),
        "Delhi": ("C0291", 13742.60),
        "Hyderabad": ("C0743", 12730.05),
        "Mumbai": ("C0693", 14766.55),
        "Pune": ("C0426", 10377.10),
    }
    for _, row in leaders.iterrows():
        cid, spend = expected_leaders[row["city"]]
        if row["customer_id"] != cid or abs(float(row["total_spend"]) - spend) > 0.01:
            raise AssertionError(f"Q6 leader {row.to_dict()}")
    if rank_gap != 8:
        raise AssertionError(f"Q6 gaps {rank_gap}")
    if len(q7) != 183 or abs(float(q7.iloc[-1]["cumulative_revenue"]) - 2998668.80) > 0.01:
        raise AssertionError("Q7")
    return frames


def exploratory():
    customers = pd.read_csv(CLEAN / "customers.csv")
    products = pd.read_csv(CLEAN / "products.csv")
    orders = pd.read_csv(CLEAN / "orders.csv")
    completed = orders.loc[orders["status"].eq("completed")].merge(customers, on="customer_id").merge(products, on="product_id")

    pivot = pd.pivot_table(
        completed,
        index="city",
        columns="category",
        values="order_amount",
        aggfunc="sum",
        margins=True,
        margins_name="All",
    )
    grand = round(float(pivot.loc["All", "All"]), 2)
    print("=== PANDAS ===")
    print(f"pivot_grand_total={grand}")
    if grand != 2998668.80:
        raise AssertionError(f"pivot grand total {grand}")

    by_city = (
        completed.groupby("city")
        .agg(
            sum_amount=("order_amount", "sum"),
            mean_amount=("order_amount", "mean"),
            count_orders=("order_amount", "count"),
        )
        .reset_index()
    )
    print(by_city.to_string(index=False))
    bengaluru = by_city.loc[by_city["city"].eq("Bengaluru")].iloc[0]
    mumbai = by_city.loc[by_city["city"].eq("Mumbai")].iloc[0]
    if abs(float(bengaluru["sum_amount"]) - 523643.40) > 0.01 or int(bengaluru["count_orders"]) != 1894:
        raise AssertionError(f"Bengaluru {bengaluru.to_dict()}")
    if abs(float(bengaluru["mean_amount"]) - 276.47) > 0.01:
        raise AssertionError(f"Bengaluru mean {bengaluru['mean_amount']}")
    if abs(float(mumbai["sum_amount"]) - 479058.50) > 0.01 or int(mumbai["count_orders"]) != 1613:
        raise AssertionError(f"Mumbai {mumbai.to_dict()}")
    if abs(float(mumbai["mean_amount"]) - 297.00) > 0.01:
        raise AssertionError(f"Mumbai mean {mumbai['mean_amount']}")

    amount = completed["order_amount"]
    q1 = float(amount.quantile(0.25))
    q3 = float(amount.quantile(0.75))
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    stats = {
        "mean": round(float(amount.mean()), 2),
        "median": round(float(amount.median()), 2),
        "skewness": round(float(amount.skew()), 2),
        "q1": round(q1, 2),
        "q3": round(q3, 2),
        "lower_fence": round(lower, 2),
        "upper_fence": round(upper, 2),
        "outliers_above": int((amount > upper).sum()),
        "outliers_below": int((amount < lower).sum()),
    }
    print("=== AMOUNT DISTRIBUTION ===")
    for key, value in stats.items():
        print(f"{key}={value}")
    if stats["mean"] != 269.91 or stats["median"] != 135.00:
        raise AssertionError(f"mean/median {stats}")
    if abs(stats["skewness"] - 4.61) > 0.05:
        raise AssertionError(f"skew {stats['skewness']}")
    if stats["q1"] != 64.80 or stats["q3"] != 297.00 or stats["upper_fence"] != 645.30:
        raise AssertionError(f"quartiles {stats}")
    if stats["outliers_above"] != 897 or stats["outliers_below"] != 0:
        raise AssertionError(f"outliers {stats}")

    corr = completed[["order_amount", "discount_pct", "quantity", "delivery_minutes"]].corr()
    print("=== CORRELATION ===")
    print(corr.round(3).to_string())
    pairs = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            pairs.append((abs(float(corr.iloc[i, j])), cols[i], cols[j], float(corr.iloc[i, j])))
    pairs.sort(reverse=True)
    print(f"largest_abs_pair={pairs[0][1]}__{pairs[0][2]} value={pairs[0][3]:.3f}")
    if pairs[0][1:] [:2] != ("order_amount", "discount_pct") and set(pairs[0][1:3]) != {"order_amount", "discount_pct"}:
        raise AssertionError(f"unexpected correlation leader {pairs[0]}")
    return stats, corr, by_city


def main():
    customers, products, orders = load_raw()
    audit(customers, products, orders)
    clean_cust, _cust_stats = clean_customers(customers)
    clean_ord, _ord_stats = clean_orders(orders, products)
    products_out = products.copy()
    store(clean_cust, products_out, clean_ord)
    run_sql()
    exploratory()
    print("PART A COMPLETE")


if __name__ == "__main__":
    main()
