import warnings
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import prediction_config

warnings.filterwarnings("ignore")

groups, splits = prediction_config.get()

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

def arima_predict_next_month(train_series):
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

    return prediction, best_order

results = []

for material_group, series in groups.items():
    for train_start, train_end, test_month in splits:
        train = series.loc[train_start:train_end]
        actual = float(series.loc[test_month])

        prediction, order = arima_predict_next_month(train)

        abs_error = abs(actual - prediction)
        sq_error = (actual - prediction) ** 2
        ape = abs((actual - prediction) / actual) if actual != 0 else None

        results.append({
            "model": "ARIMA",
            "material_group": material_group,
            "test_month": test_month,
            "actual": actual,
            "prediction": prediction,
            "abs_error": abs_error,
            "sq_error": sq_error,
            "ape": ape,
            "arima_order": str(order),
        })

print(pd.DataFrame(results).sort_values(['test_month', 'material_group']).to_string(index=False))
# print(pd.DataFrame(results).sort_values(['test_month', 'material_group']).to_excel('results.xlsx', index=False))