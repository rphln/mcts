import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


def _sample_to_features(sample: dict) -> dict:
    w = np.array(sample["white_health"])
    b = np.array(sample["black_health"])

    return {
        "_0": np.count_nonzero(w > 0) - np.count_nonzero(b > 0),
        "_1": np.sum(np.sqrt(w)) - np.sum(np.sqrt(b)),
    }


def load_turn_samples(paths: list[Path], max_seq_len: int = 64):
    rows = []

    for path in paths:
        with path.open() as file:
            data = json.load(file)

        match data["outcome"]:
            case "Draw":
                continue
            case "WhiteWins":
                y = True
            case "BlackWins":
                y = False
            case _:
                raise ValueError()

        for sample in data["samples"][-max_seq_len:]:
            row = _sample_to_features(sample)
            rows.append({"y": y, **row})

    df = pd.DataFrame(rows)

    x = df.drop(columns=["y"])
    y = df["y"]

    return x, y


def main():
    np.set_printoptions(precision=3)

    files = sorted(Path("dist").rglob("*.json"))
    train_files, test_files = train_test_split(files, test_size=0.1, random_state=0)

    x_train, y_train = load_turn_samples(train_files)
    x_test, y_test = load_turn_samples(test_files)

    clf = LogisticRegression(random_state=0, fit_intercept=False)
    clf.fit(x_train, y_train)

    print("Coefficients:", clf.coef_)
    print("Intercept:", clf.intercept_)
    print()

    y_pred = clf.predict(x_test)
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()
