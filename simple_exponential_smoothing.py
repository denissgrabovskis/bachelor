import warnings
import pandas as pd
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
import prediction_config

warnings.filterwarnings("ignore")

groups, splits = prediction_config.get()

def ses_predict_next_month(train_series):
    model = SimpleExpSmoothing(
        train_series,
        initialization_method="estimated"
    )
    fitted = model.fit()
    return float(fitted.forecast(1).iloc[0])

results = []
for material_group, series in groups.items():
    for train_start, train_end, test_month in splits:
        train = series.loc[train_start:train_end]
        actual = float(series.loc[test_month])

        prediction = ses_predict_next_month(train)

        abs_error = abs(actual - prediction)
        sq_error = (actual - prediction) ** 2
        ape = abs((actual - prediction) / actual) if actual != 0 else None

        results.append({
            "model": "SES",
            "material_group": material_group,
            "test_month": test_month,
            "actual": actual,
            "prediction": prediction,
            "abs_error": abs_error,
            "sq_error": sq_error,
            "ape": ape,
        })

print(pd.DataFrame(results).sort_values(['test_month', 'material_group']).to_string(index=False))
# print(pd.DataFrame(results).sort_values(['test_month', 'material_group']).to_excel('results.xlsx', index=False))
