"""Assert the Part A cleaning checkpoints. Prints one line when they all hold."""
import sqlite3

import pandas as pd

from src.paths import CLEAN, DB_PATH


def main():
    customers = pd.read_csv(CLEAN / "customers.csv")
    orders = pd.read_csv(CLEAN / "orders.csv")
    with sqlite3.connect(DB_PATH) as con:
        db_orders = pd.read_sql_query("SELECT COUNT(*) AS n FROM orders", con).iloc[0, 0]
        tables = {
            row[0]
            for row in con.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }

    completed = orders["status"].eq("completed")
    cancelled = orders["status"].eq("cancelled")
    amount_sum = round(float(orders.loc[completed, "order_amount"].sum()), 2)
    checks = [
        set(customers["city"]) == {"Chennai", "Mumbai", "Bengaluru", "Delhi", "Hyderabad", "Pune"},
        int((customers["city"] == "Pune").sum()) == 227,
        int((customers["city"] == "Delhi").sum()) == 210,
        int((customers["city"] == "Bengaluru").sum()) == 197,
        int((customers["city"] == "Chennai").sum()) == 191,
        int((customers["city"] == "Hyderabad").sum()) == 191,
        int((customers["city"] == "Mumbai").sum()) == 184,
        customers["gender"].value_counts().to_dict() == {"female": 610, "male": 560, "unknown": 30},
        int(customers["age"].isna().sum()) == 0,
        float(customers["age"].min()) >= 18,
        float(customers["age"].max()) <= 60,
        len(orders) == 12309,
        int(completed.sum()) == 11110,
        int(cancelled.sum()) == 1199,
        int(orders["status"].eq("pending").sum()) == 0,
        int(orders["order_amount"].isna().sum()) == 0,
        int(orders.loc[completed, "delivery_minutes"].isna().sum()) == 0,
        int(orders.loc[cancelled, "delivery_minutes"].isna().sum()) == 1199,
        amount_sum == 2998668.80,
        int(db_orders) == 12309,
        {"customers", "products", "orders"} <= tables,
    ]
    if not all(checks):
        raise AssertionError("one or more Part A checkpoints failed")
    print("ALL CHECKPOINTS PASSED")


if __name__ == "__main__":
    main()
