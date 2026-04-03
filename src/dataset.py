from __future__ import annotations

from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset


class TransitionFeatureDataset(Dataset):
    def __init__(self, parquet_path: str) -> None:
        self.df = pd.read_parquet(parquet_path)
        required_cols = ["img_feat", "next_img_feat", "action_id"]
        missing = [c for c in required_cols if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns in {parquet_path}: {missing}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        state_feat = torch.tensor(row["img_feat"], dtype=torch.float32)
        next_state_feat = torch.tensor(row["next_img_feat"], dtype=torch.float32)
        action_id = torch.tensor(int(row["action_id"]), dtype=torch.long)
        return {
            "state_feat": state_feat,
            "action_id": action_id,
            "next_state_feat": next_state_feat,
        }

