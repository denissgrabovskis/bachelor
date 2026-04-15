import os
import random
import sqlite3
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")


# ============================================================
# Reproducibility
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# Settings
# ============================================================

DB_PATH = "timber.sqlite"
OUTPUT_PATH = "predictions/LSTM.xlsx"

WINDOW_SIZE = 3
NUMERIC_FEATURE_COUNT = 6

EMBEDDING_SIZE = 4
HIDDEN_SIZE = 16

EPOCHS = 400
LEARNING_RATE = 0.01


# ============================================================
# Query
# ============================================================

MATERIAL_GROUPS_TIME_SERIES_QUERY = """
    with
        eligible_groups as (
            select
                materials.brand as brand,
                materials.category as category,
                count(distinct s.year||s.month) as months_with_sales
            from sales s

            join materials
            on s.material_id = materials.id

            where coalesce(s.sale_m3, 0) > 0

            group by materials.brand, materials.category
            having
                months_with_sales >= 6
                and not (materials.brand = 'KARTEX' and materials.category = 'DAŽĀDAS PRECES')
        ),
        periods as (
            select distinct year, month
            from sales
        ),
        grouped_sales as (
            select
                materials.brand as brand,
                materials.category as category,
                sales.year as year,
                sales.month as month,
                sum(sales.sale_m3) as sales_m3
            from sales

            join materials
            on sales.material_id = materials.id

            group by materials.brand, materials.category, sales.year, sales.month
        ),
        grouped_deliveries as (
            select
                materials.brand as brand,
                materials.category as category,
                deliveries.year as year,
                deliveries.month as month,
                sum(deliveries.received_m3) as delivery_m3
            from deliveries

            join materials
            on deliveries.material_id = materials.id

            group by materials.brand, materials.category, deliveries.year, deliveries.month
        )

    select
        eligible_groups.brand || ': ' || eligible_groups.category as material_group,
        printf('%d-%02d', periods.year, periods.month) as period,
        coalesce(grouped_sales.sales_m3, 0.0) as sales_m3,
        coalesce(grouped_deliveries.delivery_m3, 0.0) as delivery_m3
    from eligible_groups

    cross join periods

    left join grouped_sales
        on grouped_sales.brand = eligible_groups.brand
       and grouped_sales.category = eligible_groups.category
       and grouped_sales.year = periods.year
       and grouped_sales.month = periods.month

    left join grouped_deliveries
        on grouped_deliveries.brand = eligible_groups.brand
       and grouped_deliveries.category = eligible_groups.category
       and grouped_deliveries.year = periods.year
       and grouped_deliveries.month = periods.month

    order by eligible_groups.brand, eligible_groups.category, periods.year, periods.month
"""


# ============================================================
# Data loading
# ============================================================

def load_group_data_frame():
    conn = sqlite3.connect(DB_PATH)
    data_frame = pd.read_sql_query(MATERIAL_GROUPS_TIME_SERIES_QUERY, conn)
    conn.close()

    return data_frame


def build_group_feature_frames(data_frame: pd.DataFrame):
    group_feature_frames = {}

    material_groups = sorted(data_frame["material_group"].unique())
    group_id_map = {
        material_group: index
        for index, material_group in enumerate(material_groups)
    }

    for material_group, group_data_frame in data_frame.groupby("material_group"):
        feature_frame = (
            group_data_frame
            .sort_values("period")
            .copy()
            .reset_index(drop=True)
        )

        feature_frame["delta_sales_m3"] = feature_frame["sales_m3"].diff().fillna(0.0)
        feature_frame["delta_delivery_m3"] = feature_frame["delivery_m3"].diff().fillna(0.0)

        feature_frame["rolling_mean_2"] = (
            feature_frame["sales_m3"]
            .rolling(window=2, min_periods=1)
            .mean()
        )

        feature_frame["rolling_mean_3"] = (
            feature_frame["sales_m3"]
            .rolling(window=3, min_periods=1)
            .mean()
        )

        feature_frame["group_id"] = group_id_map[material_group]

        group_feature_frames[material_group] = feature_frame

    return group_feature_frames, group_id_map


# ============================================================
# Window building
# ============================================================

def build_windows(group_feature_frames, train_start: str, train_end: str):
    x_numeric = []
    x_group_ids = []
    y = []

    for material_group, feature_frame in group_feature_frames.items():
        train_frame = feature_frame[
            (feature_frame["period"] >= train_start) &
            (feature_frame["period"] <= train_end)
        ].reset_index(drop=True)

        numeric_values = train_frame[
            [
                "sales_m3",
                "delta_sales_m3",
                "delivery_m3",
                "delta_delivery_m3",
                "rolling_mean_2",
                "rolling_mean_3",
            ]
        ].to_numpy(dtype=np.float32)

        group_id = int(train_frame["group_id"].iloc[0])

        for i in range(len(train_frame) - WINDOW_SIZE):
            x_numeric.append(numeric_values[i:i + WINDOW_SIZE])
            x_group_ids.append(group_id)
            y.append(float(train_frame["sales_m3"].iloc[i + WINDOW_SIZE]))

    x_numeric = np.array(x_numeric, dtype=np.float32)
    x_group_ids = np.array(x_group_ids, dtype=np.int64)
    y = np.array(y, dtype=np.float32).reshape(-1, 1) # rows number = auto, col number = 1 [[0], [1], [2], ...]

    return x_numeric, x_group_ids, y


# ============================================================
# Scaling
# ============================================================

def fit_feature_scaler(x_numeric: np.ndarray):
    # Compute one mean/std per numeric feature across all windows and all time steps.
    mean = x_numeric.mean(axis=(0, 1), keepdims=True) # 1D array of mean for each feature across all timestamps across all samples
    std = x_numeric.std(axis=(0, 1), keepdims=True)
    std[std == 0] = 1.0

    return mean, std


def transform_numeric_features(x_numeric: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (x_numeric - mean) / std


def fit_target_scaler(y: np.ndarray):
    mean = y.mean(axis=0, keepdims=True)
    std = y.std(axis=0, keepdims=True)
    std[std == 0] = 1.0

    return mean, std


def transform_target(y: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (y - mean) / std


def inverse_transform_target(y_scaled: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return y_scaled * std + mean


# ============================================================
# Model
# ============================================================

class PooledLSTM(nn.Module):
    def __init__(self, group_count: int):
        super().__init__()

        self.group_embedding = nn.Embedding(
            num_embeddings=group_count,
            embedding_dim=EMBEDDING_SIZE,
        )

        self.lstm = nn.LSTM(
            input_size=NUMERIC_FEATURE_COUNT,
            hidden_size=HIDDEN_SIZE,
            batch_first=True,
        )

        self.output = nn.Linear(HIDDEN_SIZE + EMBEDDING_SIZE, 1)

    def forward(self, x_numeric: torch.Tensor, x_group_ids: torch.Tensor):
        # LSTM reads the numeric monthly sequence step by step.
        lstm_output, _ = self.lstm(x_numeric)
        last_hidden = lstm_output[:, -1, :]

        # Group ID is injected through an embedding instead of raw integer value.
        group_embedding = self.group_embedding(x_group_ids)

        combined = torch.cat([last_hidden, group_embedding], dim=1)
        return self.output(combined)


# ============================================================
# Training and prediction
# ============================================================

def train_model(x_numeric, x_group_ids, y, group_count):
    x_numeric_tensor = torch.tensor(x_numeric, dtype=torch.float32)
    x_group_ids_tensor = torch.tensor(x_group_ids, dtype=torch.long)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    model = PooledLSTM(group_count=group_count)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model.train()

    for _ in range(EPOCHS):
        optimizer.zero_grad()

        predictions = model(x_numeric_tensor, x_group_ids_tensor)
        loss = loss_function(predictions, y_tensor)

        loss.backward()
        optimizer.step()

    return model


def predict_next_month(
    model,
    group_feature_frame: pd.DataFrame,
    train_start: str,
    train_end: str,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
):
    train_frame = group_feature_frame[
        (group_feature_frame["period"] >= train_start) &
        (group_feature_frame["period"] <= train_end)
    ].reset_index(drop=True)

    x_numeric = train_frame[
        [
            "sales_m3",
            "delta_sales_m3",
            "delivery_m3",
            "delta_delivery_m3",
            "rolling_mean_2",
            "rolling_mean_3",
        ]
    ].to_numpy(dtype=np.float32)

    x_numeric = x_numeric[-WINDOW_SIZE:]
    x_numeric = x_numeric.reshape(1, WINDOW_SIZE, NUMERIC_FEATURE_COUNT)
    x_numeric = transform_numeric_features(x_numeric, x_mean, x_std)

    x_group_ids = np.array([int(train_frame["group_id"].iloc[0])], dtype=np.int64)

    x_numeric_tensor = torch.tensor(x_numeric, dtype=torch.float32)
    x_group_ids_tensor = torch.tensor(x_group_ids, dtype=torch.long)

    model.eval()

    with torch.no_grad():
        prediction_scaled = model(x_numeric_tensor, x_group_ids_tensor).cpu().numpy()

    prediction = inverse_transform_target(prediction_scaled, y_mean, y_std)[0, 0]
    prediction = max(float(prediction), 0.0)

    return prediction


# ============================================================
# Main
# ============================================================

def main():
    os.makedirs("predictions", exist_ok=True)

    data_frame = load_group_data_frame()
    group_feature_frames, group_id_map = build_group_feature_frames(data_frame)

    splits = [
        ("2025-07", "2025-12", "2026-01"),
        ("2025-07", "2026-01", "2026-02"),
        ("2025-07", "2026-02", "2026-03"),
    ]

    results = []

    for train_start, train_end, test_month in splits:
        x_numeric, x_group_ids, y = build_windows(
            group_feature_frames=group_feature_frames,
            train_start=train_start,
            train_end=train_end,
        )

        x_mean, x_std = fit_feature_scaler(x_numeric)
        y_mean, y_std = fit_target_scaler(y)

        x_numeric_scaled = transform_numeric_features(x_numeric, x_mean, x_std)
        y_scaled = transform_target(y, y_mean, y_std)

        model = train_model(
            x_numeric=x_numeric_scaled,
            x_group_ids=x_group_ids,
            y=y_scaled,
            group_count=len(group_id_map),
        )

        for material_group, group_feature_frame in group_feature_frames.items():
            actual_row = group_feature_frame[group_feature_frame["period"] == test_month].iloc[0]
            actual = float(actual_row["sales_m3"])

            prediction = predict_next_month(
                model=model,
                group_feature_frame=group_feature_frame,
                train_start=train_start,
                train_end=train_end,
                x_mean=x_mean,
                x_std=x_std,
                y_mean=y_mean,
                y_std=y_std,
            )

            abs_error = abs(actual - prediction)
            sq_error = (actual - prediction) ** 2
            ape = abs((actual - prediction) / actual) if actual != 0 else None

            results.append({
                "model": "LSTM",
                "material_group": material_group,
                "test_month": test_month,
                "actual": actual,
                "prediction": prediction,
                "abs_error": abs_error,
                "sq_error": sq_error,
                "ape": ape,
            })

    results_df = (
        pd.DataFrame(results)
        .sort_values(["test_month", "material_group"])
        .reset_index(drop=True)
    )

    print(results_df.to_string(index=False, col_space={"model": 7}), end="\n\n")
    results_df.round(3).to_excel(OUTPUT_PATH, index=False)


if __name__ == "__main__":
    main()