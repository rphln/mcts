import json
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm, trange


def load_turn_samples(paths: list[Path]):
    rows = []

    for path in paths:
        with path.open() as file:
            data = json.load(file)

        match data["outcome"]:
            case "Draw":
                y = 0.5
            case "WhiteWins":
                y = 1.0
            case "BlackWins":
                y = 0.0
            case _:
                raise ValueError()

        for _idx, sample in enumerate(data["samples"]):
            row = {
                "y": y,
                "b_h_0": sample["black_healths"][0],
                "b_h_1": sample["black_healths"][1],
                "b_h_2": sample["black_healths"][2],
                "b_h_3": sample["black_healths"][3],
                "w_h_0": sample["white_healths"][0],
                "w_h_1": sample["white_healths"][1],
                "w_h_2": sample["white_healths"][2],
                "w_h_3": sample["white_healths"][3],
                "b_c_0": sample["black_ready_at"][0],
                "b_c_1": sample["black_ready_at"][1],
                "b_c_2": sample["black_ready_at"][2],
                "b_c_3": sample["black_ready_at"][3],
                "w_c_0": sample["white_ready_at"][0],
                "w_c_1": sample["white_ready_at"][1],
                "w_c_2": sample["white_ready_at"][2],
                "w_c_3": sample["white_ready_at"][3],
            }
            rows.append(row)

    df = pd.DataFrame(rows)

    X = df.drop(columns=["y"])
    y = df["y"]

    return X, y


def main():
    device = "cuda"

    lr = 1e-2
    batch_size = 8192
    epochs = 256

    files = sorted(Path("dist/").rglob("*.json"))
    X, y = load_turn_samples(files)

    X = torch.from_numpy(X.to_numpy()).to(device)
    y = torch.from_numpy(y.to_numpy()).to(device)

    A = nn.Parameter(torch.ones(1, device=device))
    B = nn.Parameter(torch.ones(1, device=device))
    C = nn.Parameter(torch.ones(1, device=device))

    optim = AdamW([A, B, C], lr=lr)
    sched = ReduceLROnPlateau(optim, patience=10, factor=0.5)

    dataset = TensorDataset(X, y)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )

    with trange(1, epochs + 1) as pbar:
        for _epoch in pbar:
            epoch_loss = 0.0
            epoch_accuracy = 0.0

            for batch_idx, (x, y) in enumerate(tqdm(loader, leave=False)):
                health = x[:, 0:8]
                _ready_at = x[:, 8:16]

                preds = A * (health > 0) + B * torch.sqrt(health)

                max_sum = preds[:, 4:8].sum(dim=1)
                min_sum = preds[:, 0:4].sum(dim=1)

                logits = max_sum - min_sum

                loss = nn.functional.binary_cross_entropy_with_logits(logits, y)
                loss.backward()

                optim.step()
                optim.zero_grad()

                epoch_loss += (loss.item() - epoch_loss) / (batch_idx + 1)

                accuracy = ((y == 1.0) == (logits >= 0.0)).float()
                accuracy = accuracy[y != 0.5].mean()

                epoch_accuracy += (accuracy.item() - epoch_accuracy) / (batch_idx + 1)

            sched.step(epoch_loss)

            pbar.set_postfix(
                loss=epoch_loss,
                accuracy=epoch_accuracy,
                A=A.item(),
                B=B.item(),
                C=C.item(),
            )


if __name__ == "__main__":
    main()
