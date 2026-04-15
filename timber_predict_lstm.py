import random
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import timber_sqlite

warnings.filterwarnings("ignore")

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


TIMESTEP_COUNT = 3
GROUP_EMBEDDING_COUNT = 4
HIDDEN_STATE_SIZE = 16

EPOCHS = 400
LEARNING_RATE = 0.01


def build_samples(groups_features_by_periods, train_start, train_end):
    x = []
    group_ids = []
    y = []

    for group_name, group_id, group_periods in groups_features_by_periods:
        train_periods_df = group_periods.loc[train_start:train_end]
        period_features_np = train_periods_df.to_numpy(dtype=np.float32)

        for i in range(len(period_features_np) - TIMESTEP_COUNT):
            x.append(period_features_np[i: i+TIMESTEP_COUNT])
            group_ids.append(group_id)
            y.append(train_periods_df["sale_m3"].iloc[i + TIMESTEP_COUNT])

    x = np.array(x, dtype=np.float32)
    group_ids = np.array(group_ids, dtype=np.int64)
    y = np.array(y, dtype=np.float32).reshape(-1, 1)

    return x, group_ids, y

def standardize(x: np.ndarray, mean = None, std = None):
    features_axis = (0,1)
    mean = x.mean(axis=features_axis, keepdims=True) if mean is None else mean # axis=(0,1) collapse/destruct samples, then timesteps, then calculate mean per feature, but keep the dimensions
    std = x.std(axis=features_axis, keepdims=True) if std is None else std # standard deviation - how spread out are values - 68% fall under in the range mean+-std
    std[std == 0] = 1.0

    return (x - mean) / std, mean, std # after standardization mean becomes 0 and most values (68%) fall in [-1:1] range


def inverse_transform_target(y_scaled: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return y_scaled * std + mean


# 1. X > LSTM > Output
# 2. Output + Group Embeddings > Linear Input
# 3. Linear Input > Linear > Predicted Sales
class SalesPredictor(nn.Module):
    def __init__(self, feature_count: int, group_count: int):
        super().__init__()

        self.groups_embedding = nn.Embedding(
            num_embeddings=group_count,
            embedding_dim=GROUP_EMBEDDING_COUNT,
        )

        self.lstm = nn.LSTM(
            input_size=feature_count,
            hidden_size=HIDDEN_STATE_SIZE,
            batch_first=True, # True - sample>timestep>feature; False - timestep>sample>feature
        )

        self.linear = nn.Linear(HIDDEN_STATE_SIZE + GROUP_EMBEDDING_COUNT, 1)

    def forward(self, x: torch.Tensor, groups_ids: torch.Tensor):
        lstm_output, _ = self.lstm(x) # feed single group
        lstm_last_timestep_output = lstm_output[:, -1, :] # take last each group sample last timestep all features result

        groups_embedding = self.groups_embedding(groups_ids) # retrieve embeddings (when training the groups_ids are constant)

        # concat lstm last timestep output into input with groups embeddings to make it act like input for the linear nn
        linear_input = torch.cat([lstm_last_timestep_output, groups_embedding], dim=1)

        return self.linear(linear_input)

def train_model(x, groups_ids, y):
    x = torch.tensor(x, dtype=torch.float32)
    groups_ids = torch.tensor(groups_ids, dtype=torch.long)
    y = torch.tensor(y, dtype=torch.float32)

    model = SalesPredictor(feature_count=x.shape[-1], group_count=len(np.unique(groups_ids)))

    # reason behind MSELoss and Adam is explained in thesis
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model.train()

    for _ in range(EPOCHS):
        optimizer.zero_grad()

        predictions = model(x, groups_ids)
        loss = loss_function(predictions, y)

        loss.backward()
        optimizer.step()

    return model


def predict_next_month(
    model,
    group_id,
    period: pd.DataFrame,
    x_standardize,
    y_unstandardize,
):
    x = period.to_numpy(dtype=np.float32)

    x = x[-TIMESTEP_COUNT:] # the model only reads last 3 months

    # make it as if there are many timesteps to predict, although there is only one (in other words, wrap the timestep)
    x = x_standardize(x).reshape(1, TIMESTEP_COUNT, x.shape[-1])
    group_id = np.array([group_id], dtype=np.int64)

    x = torch.tensor(x, dtype=torch.float32)
    group_id = torch.tensor(group_id, dtype=torch.long)
    model.eval()
    with torch.no_grad():
        prediction = model(x, group_id).cpu().numpy()
        prediction = y_unstandardize(prediction)
        prediction = max(prediction, 0.0)


    return prediction


# ============================================================
# Main
# ============================================================

train_splits = timber_sqlite.get_train_splits()
groups_features_by_periods = timber_sqlite.get_groups_with_summed(
    'sales.sale_m3',
        'deliveries.received_m3',
        transform = lambda features, group_name, group_id: [
            group_name,
            group_id,
            features
                .assign(
                    delta_sale_m3 = features["sale_m3"].diff().fillna(0.0), # fillna - replace first diff
                    delta_received_m3 = features["received_m3"].diff().fillna(0.0),
                    rolling_sales_mean_2 = features["sale_m3"].rolling(window=2, min_periods=1).mean(),
                    rolling_sales_mean_3 = features["sale_m3"].rolling(window=3, min_periods=1).mean(),
                )
                .select_dtypes(include='number')
        ]
)


results = []
for train_start, train_end, test_month in train_splits:
    x, groups_id, y = build_samples(
        groups_features_by_periods=groups_features_by_periods,
        train_start=train_start,
        train_end=train_end,
    )

    x, x_mean, x_std = standardize(x)
    y, y_mean, y_std = standardize(y)

    # create new model each for each test
    model = train_model(x=x, groups_ids=groups_id, y=y)

    for material_group, group_id, group_periods in groups_features_by_periods:
        actual = group_periods.loc[test_month]['sale_m3']

        prediction = predict_next_month(
            model=model,
            group_id=group_id,
            period=group_periods[train_start:train_end],
            x_standardize = lambda x: standardize(x, x_mean, x_std)[0],
            y_unstandardize = lambda y: (y * y_std + y_mean)[0, 0],
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

results_df = pd.DataFrame(results).sort_values(['test_month', 'material_group'])
print(results_df.to_string(index=False, col_space={'model': 7}), end="\n\n")
results_df.round(3).to_excel(f'predictions/lstm.xlsx', index=False)


