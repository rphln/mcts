#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_turn_samples(paths: list[Path]):
    rows = []

    for path in paths:
        with path.open() as file:
            data = json.load(file)

        hw = data["health_white"]
        hb = data["health_black"]

        match data["match_outcome"]:
            case "Draw":
                continue
            case y:
                pass

        for w, b in zip(hw, hb, strict=True):
            w = sorted(w)
            b = sorted(b)

            w = np.array(w)
            b = np.array(b)

            row = {
                "y": y,
                "b_0": b[0],
                "b_1": b[1],
                "b_2": b[2],
                "b_3": b[3],
                "w_0": w[0],
                "w_1": w[1],
                "w_2": w[2],
                "w_3": w[3],
            }
            rows.append(row)

    df = pd.DataFrame(rows)

    X = df.drop(columns=["y"])
    y = df["y"]

    return X, y


def read_class(path: Path) -> str:
    with path.open() as file:
        data = json.load(file)

    return data["match_outcome"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    files = sorted(Path(args.data_dir).glob("*.json"))
    classes = list(map(read_class, files))

    train_files, test_files = train_test_split(
        files, test_size=args.test_size, random_state=42, stratify=classes
    )

    X_train, y_train = load_turn_samples(train_files)
    X_test, y_test = load_turn_samples(test_files)

    labels = LabelEncoder()

    y_train = labels.fit_transform(y_train)
    y_test = labels.transform(y_test)

    clf = LogisticRegression(
        random_state=42,
        fit_intercept=True,
        class_weight="balanced",
        penalty="l2",
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)

    print(clf.feature_names_in_)
    print(clf.coef_, clf.intercept_)

    print(f"Train samples: {len(X_train)}")
    print(f"Test samples:  {len(y_test)}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.6f}")
    print(f"Balanced accuracy: {balanced_accuracy_score(y_test, y_pred):.6f}")
    print(f"Log loss: {log_loss(y_test, y_proba):.6f}")
    print()

    print("Classification report:")
    print(classification_report(y_test, y_pred, target_names=labels.classes_))
    print("Confusion matrix (rows=true, cols=pred)")
    print("Class order:", list(labels.classes_))
    print(confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
    main()
