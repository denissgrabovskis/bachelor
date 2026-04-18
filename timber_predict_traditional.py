import warnings
import pandas as pd

from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.arima.model import ARIMA

import timber_sqlite

warnings.filterwarnings("ignore")

def naive_predict_next_month(train_series):
    prediction = float(train_series.iloc[-1])
    return prediction, dict()

def simple_exponential_smoothing(train_series):
    model = SimpleExpSmoothing(
        train_series,
        initialization_method="estimated"
    )
    fitted = model.fit()
    return float(fitted.forecast(1).iloc[0]), dict()

def arima_predict_next_month(train_series):
    p_d_q_combinations = [
        (0, 0, 0),
        (0, 0, 1),
        (0, 1, 0),
        (0, 1, 1),
        (0, 1, 2),
        (1, 0, 0),
        (1, 0, 1),
        (1, 1, 0),
        (1, 1, 1),
        (2, 1, 0),
    ]

    best_aic = float("inf")
    best_fitted = None
    best_order = None

    for combination in p_d_q_combinations:
        try:
            model = ARIMA(
                train_series,
                order=combination,

                # when True (which is default) some p,d,q combinations may be ignored due to automatically set constraints
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            fitted = model.fit()

            if fitted.aic < best_aic:
                best_aic = fitted.aic
                best_fitted = fitted
                best_order = combination
        except Exception:
            continue

    if best_fitted is None:
        raise RuntimeError("ARIMA could not be fitted for the given series.")

    prediction = float(best_fitted.forecast(1).iloc[0])
    prediction = max(prediction, 0.0) # reset possibly negative number

    return prediction, {"order": str(best_order)}


groups = timber_sqlite.get_groups_with_summed("sales.sale_m3", transform= lambda series, group_name, i: [group_name, series['sale_m3'].astype('float64')])
train_splits = timber_sqlite.get_train_splits()
models = {
    'naive': naive_predict_next_month,
    'SES': simple_exponential_smoothing,
    'ARIMA': arima_predict_next_month,
}


for model, predict_next_month in models.items():
    results = []
    for material_group, series in groups:
        for train_start, train_end, test_month in train_splits:
            train = series.loc[train_start:train_end]
            actual = float(series.loc[test_month])

            prediction, meta = predict_next_month(train)

            abs_error = abs(actual - prediction)
            sq_error = (actual - prediction) ** 2
            ape = abs((actual - prediction) / actual) if actual != 0 else None

            results.append({
                "model": model,
                "material_group": material_group,
                "test_month": test_month,
                "actual": actual,
                "prediction": prediction,
                "abs_error": abs_error,
                "sq_error": sq_error,
                "ape": ape,
                **meta,
            })

    results_df = pd.DataFrame(results).sort_values(['test_month', 'material_group']).round(3)
    print(results_df.to_string(index=False, col_space={'model': 7}), end="\n\n")
    results_df.to_excel(f'predictions/{model}.xlsx', index=False)
    results_df.to_csv(f'predictions/{model}.csv', index=False)
