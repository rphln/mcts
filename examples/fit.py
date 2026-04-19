from pathlib import Path

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, log_loss
from sklearn.model_selection import StratifiedGroupKFold

FEATURE_COLS = [
    "health:false",
    "health:true",
    "is_alive:false",
    "is_alive:true",
]


def load_and_prepare(paths: list[Path], cache: Path = Path("dist/data.parquet")):
    try:
        df = pl.read_parquet(cache)
    except FileNotFoundError:
        df = pl.concat(
            pl.read_json(path, schema_overrides={"seed": pl.UInt64}) for path in paths
        )

        df.write_parquet(cache)

    df = df.lazy()

    labels = (
        df.group_by("seed", "turn", "team")
        .agg(pl.max("is_alive"))
        .group_by("seed", "team")
        .agg(pl.min("is_alive").alias("survives"))
    )

    q = (
        df.join(labels, on=("seed", "team"))
        .with_columns(
            health=pl.col("current_health").sqrt(),
            is_active=pl.col("team") == pl.col("active_team"),
        )
        .pivot(
            on="is_active",
            on_columns=("true", "false"),
            index=("seed", "turn"),
            values=("is_alive", "health", "poison", "survives"),
            aggregate_function="sum",
            separator=":",
        )
    )

    print(q.explain())

    return q.collect(engine="streaming")


def make_data(path: Path):
    files = sorted(Path(path).rglob("*.json"))

    df = load_and_prepare(files)
    print(df.describe())

    X = df[FEATURE_COLS]
    y = df["survives:true"] > 0

    groups = df["seed"]

    train_idx, test_idx = next(StratifiedGroupKFold(n_splits=5).split(X, y, groups))
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


@pl.Config(float_precision=3, tbl_cols=-1)
def main():
    np.set_printoptions(formatter={"float": "{: 0.3f}".format})

    X_train, X_test, y_train, y_test = make_data(Path("dist/data"))

    estimator = LogisticRegression(fit_intercept=True, solver="newton-cholesky")
    estimator.fit(X_train, y_train)

    y_pred = estimator.predict(X_test)
    print(classification_report(y_test, y_pred))
    print()

    y_pred = estimator.predict_proba(X_test)
    print("Mean NLL:", log_loss(y_test, y_pred))
    print()

    print(f"{estimator.intercept_=} {estimator.coef_=}")
    print()

    white = estimator.coef_[0, 1::2]
    black = estimator.coef_[0, 0::2]

    weights = (white - black) / 2.0
    weights = weights.reshape(1, -1, order="F")

    print(f"{weights=}")


if __name__ == "__main__":
    main()
