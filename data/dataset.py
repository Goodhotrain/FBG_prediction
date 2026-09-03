"""Leakage-safe tabular and longitudinal data pipeline."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

@dataclass
class FBGPreprocessor:
    static_scaler: StandardScaler
    nutrition_scaler: StandardScaler
    static_columns: list[str]
    nutrition_columns: list[str]

    def state_dict(self) -> dict[str, Any]:
        return {"static_columns": self.static_columns, "nutrition_columns": self.nutrition_columns,
                "static_mean": self.static_scaler.mean_.tolist(), "static_scale": self.static_scaler.scale_.tolist(),
                "nutrition_mean": self.nutrition_scaler.mean_.tolist(), "nutrition_scale": self.nutrition_scaler.scale_.tolist()}

def _validate(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing required columns: {sorted(missing)}")

class FBGDataset(Dataset):
    """One sample per label row, with nutrition records available up to that day."""
    def __init__(self, args: Any, subset: str | None, preprocessor: FBGPreprocessor | None = None):
        self.sequence_length = int(args.sequence_length)
        self.nutrition = pd.read_csv(args.nutrition_path)
        self.labels = pd.read_csv(args.label_path)
        self.static = pd.read_csv(args.static_path)
        _validate(self.nutrition, {"UserID", "Day"}, "nutrition data")
        _validate(self.static, {"UserID", "Day"}, "static data")
        if self.labels.shape[1] < 2:
            raise ValueError("label data must contain target and split columns")
        target_col, split_col = self.labels.columns[:2]
        indices = self.labels.index if subset is None else self.labels.index[self.labels[split_col] == subset]
        self.indices = indices.to_numpy(dtype=np.int64)
        self.targets = pd.to_numeric(self.labels.loc[indices, target_col], errors="raise").to_numpy(np.float32)
        if not len(self.indices):
            raise ValueError(f"split {subset!r} contains no samples")
        if len(self.static) != len(self.labels):
            raise ValueError("static and label files must have the same number of rows")
        self.static_columns = [c for c in self.static if c not in {"UserID", "Day"}]
        self.nutrition_columns = [c for c in self.nutrition if c not in {"UserID", "Day"}]
        raw_static = self.static[self.static_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        raw_nutrition = self.nutrition[self.nutrition_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        nutrition_log = np.log1p(np.clip(raw_nutrition.to_numpy(np.float64), 0.0, None))
        if preprocessor is None:
            train_rows = self.static.loc[self.indices, ["UserID", "Day"]]
            last_training_day = train_rows.groupby("UserID")["Day"].max().to_dict()
            fit_rows = [i for i, (user, day) in enumerate(self.nutrition[["UserID", "Day"]].itertuples(index=False, name=None))
                        if user in last_training_day and day <= last_training_day[user]]
            nutrition_fit = nutrition_log[fit_rows] if fit_rows else nutrition_log
            preprocessor = FBGPreprocessor(StandardScaler().fit(raw_static.iloc[self.indices]), StandardScaler().fit(nutrition_fit), self.static_columns, self.nutrition_columns)
        if preprocessor.static_columns != self.static_columns or preprocessor.nutrition_columns != self.nutrition_columns:
            raise ValueError("feature columns differ from the training preprocessor")
        self.preprocessor = preprocessor
        self.static_values = preprocessor.static_scaler.transform(raw_static).astype(np.float32)
        scaled = preprocessor.nutrition_scaler.transform(nutrition_log).astype(np.float32)
        self.nutrition_values = pd.DataFrame(scaled, columns=self.nutrition_columns)
        self.nutrition_values[["UserID", "Day"]] = self.nutrition[["UserID", "Day"]].values

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, ...]:
        row = int(self.indices[item])
        user_id, day = self.static.loc[row, ["UserID", "Day"]]
        history = self.nutrition_values[(self.nutrition_values.UserID == user_id) & (self.nutrition_values.Day <= day)].sort_values("Day")
        values = history[self.nutrition_columns].to_numpy(np.float32)[-self.sequence_length:]
        mask = np.ones(len(values), dtype=np.bool_)
        pad = self.sequence_length - len(values)
        if pad:
            values = np.pad(values, ((pad, 0), (0, 0)))
            mask = np.pad(mask, (pad, 0))
        if not mask.any():
            # Attention kernels cannot operate on a fully masked sequence.
            mask[-1] = True
        return torch.tensor(row), torch.from_numpy(self.static_values[row]), torch.from_numpy(values), torch.from_numpy(mask), torch.tensor(self.targets[item])

def build_dataloaders(args: Any) -> tuple[DataLoader, DataLoader, FBGPreprocessor]:
    train = FBGDataset(args, args.train_split)
    valid = FBGDataset(args, args.valid_split, train.preprocessor)
    common = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    generator = torch.Generator().manual_seed(args.seed)
    return DataLoader(train, shuffle=True, generator=generator, **common), DataLoader(valid, shuffle=False, **common), train.preprocessor
