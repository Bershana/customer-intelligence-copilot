"""Part B: RFM segments and a lapse model. Features use only orders before 2025-09-01."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    silhouette_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib

from src.paths import CLEAN, OUTPUTS

SNAPSHOT = pd.Timestamp("2025-10-01 00:00:00")
CUTOFF = pd.Timestamp("2025-09-01 00:00:00")
CHOSEN_K = 4
NUMERIC = [
    "recency",
    "frequency",
    "monetary",
    "avg_discount",
    "cancel_rate",
    "avg_delivery",
    "age",
    "tenure_days",
]
CATEGORICAL = ["city", "gender", "favourite_category"]


def load_clean():
    customers = pd.read_csv(CLEAN / "customers.csv", parse_dates=["signup_date"])
    products = pd.read_csv(CLEAN / "products.csv")
    orders = pd.read_csv(CLEAN / "orders.csv", parse_dates=["order_ts"])
    return customers, products, orders


def build_rfm(orders):
    completed = orders.loc[orders["status"].eq("completed")].copy()
    grouped = completed.groupby("customer_id", as_index=False).agg(
        last_order=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum"),
    )
    grouped["recency"] = (SNAPSHOT - grouped["last_order"]).dt.days
    rfm = grouped[["customer_id", "recency", "frequency", "monetary"]].copy()
    print("=== RFM ===")
    print(f"customers={len(rfm)}")
    print(
        f"recency_median={rfm['recency'].median()} min={rfm['recency'].min()} max={rfm['recency'].max()}"
    )
    print(f"frequency_mean={rfm['frequency'].mean():.2f} max={rfm['frequency'].max()}")
    print(f"monetary_mean={rfm['monetary'].mean():.2f} max={rfm['monetary'].max():.2f}")
    if len(rfm) != 1175:
        raise AssertionError("RFM customer count")
    if int(rfm["recency"].median()) != 19 or int(rfm["recency"].min()) != 0 or int(rfm["recency"].max()) != 180:
        raise AssertionError("RFM recency")
    if round(float(rfm["frequency"].mean()), 2) != 9.46 or int(rfm["frequency"].max()) != 33:
        raise AssertionError("RFM frequency")
    if round(float(rfm["monetary"].mean()), 2) != 2552.06 or abs(float(rfm["monetary"].max()) - 14766.55) > 0.01:
        raise AssertionError("RFM monetary")
    return rfm


def cluster(rfm):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(rfm[["recency", "frequency", "monetary"]])
    print("=== KMEANS ===")
    print("k inertia silhouette")
    table = []
    for k in range(2, 9):
        model = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = model.fit_predict(scaled)
        inertia = float(model.inertia_)
        silhouette = float(silhouette_score(scaled, labels))
        table.append((k, inertia, silhouette))
        print(f"{k} {inertia:.1f} {silhouette:.3f}")
    reference = {
        2: (1820.8, 0.465),
        3: (1268.2, 0.445),
        4: (743.1, 0.526),
        5: (617.4, 0.481),
        6: (503.2, 0.482),
        7: (408.5, 0.416),
        8: (366.5, 0.412),
    }
    for k, inertia, silhouette in table:
        exp_inertia, exp_sil = reference[k]
        if abs(inertia - exp_inertia) / exp_inertia > 0.05 or abs(silhouette - exp_sil) > 0.05:
            raise AssertionError(f"K={k} inertia={inertia:.1f} silhouette={silhouette:.3f}")

    final = KMeans(n_clusters=CHOSEN_K, n_init=10, random_state=42)
    rfm = rfm.copy()
    rfm["segment_id"] = final.fit_predict(scaled)
    profile = (
        rfm.groupby("segment_id")
        .agg(
            size=("customer_id", "size"),
            recency=("recency", "mean"),
            frequency=("frequency", "mean"),
            monetary=("monetary", "mean"),
        )
        .reset_index()
    )
    print("=== SEGMENT PROFILE ===")
    print(profile.to_string(index=False))
    names, actions, premium_id = name_segments(profile)
    rfm["segment_name"] = rfm["segment_id"].map(names)
    rfm["is_premium"] = rfm["segment_id"].eq(premium_id)
    print(f"premium_segment_id={premium_id} name={names[premium_id]} size={int((rfm['is_premium']).sum())}")
    for sid, action in actions.items():
        print(f"action segment_id={sid} name={names[sid]} action={action}")

    pca = PCA(n_components=2, random_state=42)
    pca.fit(scaled)
    ratios = pca.explained_variance_ratio_
    print(f"pca_ratio_1={ratios[0]:.3f} pca_ratio_2={ratios[1]:.3f} pca_total={ratios.sum():.3f}")
    if abs(ratios[0] - 0.634) > 0.02 or abs(ratios[1] - 0.218) > 0.02 or abs(ratios.sum() - 0.852) > 0.02:
        raise AssertionError(f"PCA ratios {ratios}")

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    out = rfm[["customer_id", "recency", "frequency", "monetary", "segment_id", "segment_name", "is_premium"]]
    out.to_csv(OUTPUTS / "customer_segments.csv", index=False)
    premium_rows = profile.loc[profile["segment_id"].eq(premium_id)].iloc[0]
    if not (120 <= int(premium_rows["size"]) <= 160 and float(premium_rows["monetary"]) > 5000):
        raise AssertionError(f"premium profile {premium_rows.to_dict()}")
    return rfm, profile, names, actions, premium_id


def name_segments(profile):
    """Name clusters from their means. The numeric id has no rank order."""
    remaining = set(profile["segment_id"].tolist())
    names, actions = {}, {}
    premium_id = int(profile.loc[profile["monetary"].idxmax(), "segment_id"])
    names[premium_id] = "Premium"
    actions[premium_id] = "Give this group a dedicated desk and early access to high-value items."
    remaining.remove(premium_id)
    freq_id = int(profile.loc[profile["segment_id"].isin(remaining)].sort_values("frequency").iloc[-1]["segment_id"])
    names[freq_id] = "Frequent staples"
    actions[freq_id] = "Send refill reminders and grocery-bundle offers."
    remaining.remove(freq_id)
    if remaining:
        recency_id = int(profile.loc[profile["segment_id"].isin(remaining)].sort_values("recency").iloc[-1]["segment_id"])
        names[recency_id] = "At risk"
        actions[recency_id] = "Send a win-back offer while the customer can still be reached."
        remaining.remove(recency_id)
    for sid in sorted(remaining):
        names[sid] = "Occasional"
        actions[sid] = "Offer a small discount on the next grocery order."
    return names, actions, premium_id


def favourite_category(series):
    counts = series.value_counts()
    top = counts.max()
    return sorted(counts.loc[counts.eq(top)].index)[0]


def build_lapse_table(orders, customers, products):
    history = orders.loc[orders["order_ts"] < CUTOFF].copy()
    if history["order_ts"].max() >= CUTOFF:
        raise AssertionError("feature orders must stay before the cutoff")
    completed = history.loc[history["status"].eq("completed")].copy()
    completed = completed.merge(products[["product_id", "category"]], on="product_id", how="left")
    cohort_ids = completed["customer_id"].drop_duplicates()

    september = orders.loc[
        orders["order_ts"].ge(CUTOFF)
        & orders["order_ts"].lt(pd.Timestamp("2025-10-01"))
        & orders["status"].eq("completed"),
        "customer_id",
    ]
    stayed = set(september)

    last = completed.groupby("customer_id")["order_ts"].max()
    recency = (CUTOFF - last).dt.days
    frequency = completed.groupby("customer_id")["order_id"].nunique()
    monetary = completed.groupby("customer_id")["order_amount"].sum()
    avg_discount = completed.groupby("customer_id")["discount_pct"].mean()
    avg_delivery = completed.groupby("customer_id")["delivery_minutes"].mean()
    favourite = completed.groupby("customer_id")["category"].agg(favourite_category)

    status_counts = history.groupby("customer_id")["status"].value_counts().unstack(fill_value=0)
    for col in ("completed", "cancelled"):
        if col not in status_counts.columns:
            status_counts[col] = 0
    cancel_rate = status_counts["cancelled"] / (status_counts["cancelled"] + status_counts["completed"])

    features = pd.DataFrame({"customer_id": cohort_ids}).set_index("customer_id")
    features["recency"] = recency
    features["frequency"] = frequency
    features["monetary"] = monetary
    features["avg_discount"] = avg_discount
    features["cancel_rate"] = cancel_rate
    features["avg_delivery"] = avg_delivery
    features["favourite_category"] = favourite
    features["lapsed"] = (~features.index.isin(stayed)).astype(int)
    cust = customers.set_index("customer_id")
    features["city"] = cust["city"]
    features["age"] = cust["age"]
    features["gender"] = cust["gender"]
    features["tenure_days"] = (CUTOFF - cust["signup_date"]).dt.days
    features = features.reset_index()
    if features[NUMERIC + CATEGORICAL].isna().any().any():
        raise AssertionError("a lapse feature contains a missing value")
    lapsed_n = int(features["lapsed"].sum())
    print("=== LAPSE COHORT ===")
    print(f"cohort={len(features)} lapsed={lapsed_n} lapsed_pct={100 * lapsed_n / len(features):.2f}")
    if len(features) != 1156 or lapsed_n != 443:
        raise AssertionError("cohort counts")
    return features


def make_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC),
            (
                "cat",
                OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
                CATEGORICAL,
            ),
        ]
    )


def classification_report(name, y_true, y_pred):
    precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    print(f"{name} precision={precision:.4f} recall={recall:.4f} f1={f1:.4f}")
    print(f"{name} confusion_tn_fp_fn_tp={matrix.ravel().tolist()}")
    return {"precision": precision, "recall": recall, "f1": f1, "matrix": matrix}


def train_models(features):
    x = features[NUMERIC + CATEGORICAL]
    y = features["lapsed"]
    ids = features["customer_id"]
    x_train, x_test, y_train, y_test, id_train, id_test = train_test_split(
        x, y, ids, test_size=0.2, stratify=y, random_state=42
    )
    print("=== SPLIT ===")
    print(f"train={len(x_train)} test={len(x_test)} train_lapsed={int(y_train.sum())} test_lapsed={int(y_test.sum())}")
    if len(x_train) != 924 or len(x_test) != 232:
        raise AssertionError("split sizes")

    baseline_pred = np.zeros(len(y_test), dtype=int)
    baseline_acc = accuracy_score(y_test, baseline_pred)
    baseline_f1 = f1_score(y_test, baseline_pred, pos_label=1, zero_division=0)
    print(f"baseline_accuracy={baseline_acc:.3f} baseline_lapsed_f1={baseline_f1:.3f}")
    if abs(baseline_acc - 0.616) > 0.005 or baseline_f1 != 0:
        raise AssertionError("baseline")

    lr = Pipeline(
        [
            ("prep", make_preprocessor()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    lr_search = GridSearchCV(
        lr,
        {"clf__C": [0.01, 0.1, 1, 10, 100]},
        cv=5,
        scoring="f1",
        n_jobs=1,
    )
    lr_search.fit(x_train, y_train)
    print(f"lr_best_params={lr_search.best_params_} lr_best_cv_f1={lr_search.best_score_:.4f}")

    rf = Pipeline(
        [
            ("prep", make_preprocessor()),
            ("clf", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]
    )
    rf_search = GridSearchCV(
        rf,
        {"clf__max_depth": [3, 5, 8], "clf__min_samples_leaf": [5, 10, 20]},
        cv=5,
        scoring="f1",
        n_jobs=1,
    )
    rf_search.fit(x_train, y_train)
    print(f"rf_best_params={rf_search.best_params_} rf_best_cv_f1={rf_search.best_score_:.4f}")

    lr_metrics = classification_report("logistic", y_test, lr_search.predict(x_test))
    rf_metrics = classification_report("forest", y_test, rf_search.predict(x_test))
    if max(lr_metrics["f1"], rf_metrics["f1"]) < 0.55:
        raise AssertionError("neither model reached lapsed F1 0.55")

    scaler = lr_search.best_estimator_.named_steps["prep"].named_transformers_["num"]
    freq_index = NUMERIC.index("frequency")
    print(
        f"scaler_frequency_mean={scaler.mean_[freq_index]:.4f} "
        f"cohort_frequency_mean={features['frequency'].mean():.4f}"
    )

    # Oversample the training rows only. The test set stays unchanged.
    y_arr = y_train.to_numpy()
    rng = np.random.RandomState(42)
    positive = np.flatnonzero(y_arr == 1)
    negative = np.flatnonzero(y_arr == 0)
    draw = rng.choice(positive, size=len(negative) - len(positive), replace=True)
    balanced_index = np.concatenate([np.arange(len(y_arr)), draw])
    x_balanced = x_train.iloc[balanced_index]
    y_balanced = y_train.iloc[balanced_index]
    print(
        f"balanced_train_rows={len(y_balanced)} "
        f"balanced_lapsed={int(y_balanced.sum())} balanced_not_lapsed={int((y_balanced == 0).sum())}"
    )
    best_c = lr_search.best_params_["clf__C"]
    lr_balanced = Pipeline(
        [
            ("prep", make_preprocessor()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42, C=best_c)),
        ]
    )
    lr_balanced.fit(x_balanced, y_balanced)
    balanced_metrics = classification_report("logistic_balanced", y_test, lr_balanced.predict(x_test))

    selected_name = "random_forest" if rf_metrics["f1"] >= lr_metrics["f1"] else "logistic_regression"
    selected = rf_search.best_estimator_ if selected_name == "random_forest" else lr_search.best_estimator_
    print(f"selected_model={selected_name}")

    report_interpretation(lr_search.best_estimator_, rf_search.best_estimator_)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    model_path = OUTPUTS / "lapse_model.joblib"
    joblib.dump(selected, model_path)
    reloaded = joblib.load(model_path)
    original_pred = selected.predict(x_test)
    reloaded_pred = reloaded.predict(x_test)
    if not np.array_equal(original_pred, reloaded_pred):
        raise AssertionError("reloaded model predictions differ")
    print("reloaded_model_predictions_match=True")

    proba = selected.predict_proba(x_test)[:, 1]
    segments = pd.read_csv(OUTPUTS / "customer_segments.csv")
    premium_flag = segments["is_premium"].map(lambda value: str(value).strip().lower() == "true")
    premium_ids = set(segments.loc[premium_flag, "customer_id"])
    retention = pd.DataFrame(
        {
            "customer_id": id_test.to_numpy(),
            "lapse_probability": proba,
            "actually_lapsed": y_test.to_numpy().astype(int),
        }
    )
    retention = retention.loc[retention["customer_id"].isin(premium_ids)]
    retention = retention.sort_values(["lapse_probability", "customer_id"], ascending=[False, True]).head(10)
    retention.to_csv(OUTPUTS / "retention_list.csv", index=False)
    print(f"retention_rows={len(retention)} retention_actually_lapsed={int(retention['actually_lapsed'].sum())}")
    print(retention.to_string(index=False))
    return {
        "lr": lr_metrics,
        "rf": rf_metrics,
        "balanced": balanced_metrics,
        "selected": selected_name,
    }


def report_interpretation(lr_model, rf_model):
    lr_names = list(lr_model.named_steps["prep"].get_feature_names_out())
    lr_coef = lr_model.named_steps["clf"].coef_.ravel()
    print("=== LOGISTIC COEFFICIENTS ===")
    for feature in ("recency", "frequency", "monetary"):
        name = f"num__{feature}"
        sign = "positive" if lr_coef[lr_names.index(name)] > 0 else "negative"
        print(f"{feature} {sign} coef={lr_coef[lr_names.index(name)]:.4f}")

    rf_names = list(rf_model.named_steps["prep"].get_feature_names_out())
    importances = rf_model.named_steps["clf"].feature_importances_
    order = np.argsort(importances)[::-1][:5]
    print("=== FOREST IMPORTANCE ===")
    for idx in order:
        print(f"{rf_names[idx]} {importances[idx]:.4f}")


def main():
    customers, products, orders = load_clean()
    rfm = build_rfm(orders)
    cluster(rfm)
    features = build_lapse_table(orders, customers, products)
    train_models(features)
    print("PART B COMPLETE")


if __name__ == "__main__":
    main()
