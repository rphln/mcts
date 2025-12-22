import json
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm, trange


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
            case "WhiteWins":
                y = 1.0
            case "BlackWins":
                y = 0.0
            case _:
                raise ValueError()

        for w, b in zip(hw, hb, strict=True):
            w = sorted(w)
            b = sorted(b)

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

    optim = Adam([A, B], lr=lr)
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
                preds = A * (x > 0) + B * torch.sqrt(x)

                max_sum = preds[:, 4:8].sum(dim=1)
                min_sum = preds[:, 0:4].sum(dim=1)

                logits = max_sum - min_sum

                loss = nn.functional.binary_cross_entropy_with_logits(logits, y)
                loss.backward()

                optim.step()
                optim.zero_grad()

                epoch_loss += (loss.item() - epoch_loss) / (batch_idx + 1)

                accuracy = ((y == 1.0) == (logits >= 0.0)).float().mean()
                epoch_accuracy += (accuracy.item() - epoch_accuracy) / (batch_idx + 1)

            sched.step(epoch_loss)

            pbar.set_postfix(
                loss=epoch_loss,
                accuracy=epoch_accuracy,
                attack_scale=A.item(),
                health_scale=B.item(),
            )


if __name__ == "__main__":
    main()
