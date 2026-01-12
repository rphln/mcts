import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


def _sample_to_features(sample: dict) -> dict:
    t = sample["tick"]

    w_h = np.array(sample["white_health"])
    b_h = np.array(sample["black_health"])

    w_t = np.array(sample["white_ready_at"])
    b_t = np.array(sample["black_ready_at"])

    w_c = (w_h > 0) * (w_t - t)
    b_c = (b_h > 0) * (b_t - t)

    return {
        "h_0": np.count_nonzero(w_h > 0) - np.count_nonzero(b_h > 0),
        "h_1": np.sum(w_h**0.5) - np.sum(b_h**0.5),
        "c_1": np.sum(w_c) - np.sum(b_c),
    }


def load_turn_samples(paths: list[Path], max_seq_len: int = 1024):
    rows = []

    for path in paths:
        with path.open() as file:
            data = json.load(file)

        match data["outcome"]:
            case "Draw":
                continue
            case "WhiteWins":
                y = 1
            case "BlackWins":
                y = 0
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
    files = sorted(Path("dist").rglob("*.json"))
    train_files, test_files = train_test_split(files, test_size=0.2, random_state=0)

    x_train, y_train = load_turn_samples(train_files)
    x_test, y_test = load_turn_samples(test_files)

    clf = LogisticRegression(random_state=0)
    clf.fit(x_train, y_train)

    print("Coefficients:", clf.coef_)
    print("Intercept:", clf.intercept_)
    print()

    y_pred = clf.predict(x_test)
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()
