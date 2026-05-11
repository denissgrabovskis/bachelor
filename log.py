from contextlib import contextmanager
import pandas as pd

_prefix = None
_enabled = False
_throttle = False
_buffer = []
@contextmanager
def prefix(prefix):
    global _prefix
    _prefix = prefix
    try:
        yield
    except Exception as e:
        flush_buffer()
        raise e
    finally:
        _prefix = None
        flush_buffer()


def output(prefix, args, kwargs):
    if (prefix is not None):
        print(prefix, end="")
    print(*args, **kwargs)

def flush_buffer():
    global _buffer
    for b in _buffer:
        output(*b)
    _buffer = []

def log(*args, **kwargs):
    global _enabled
    if not _enabled: return


    global _prefix, _throttle, _buffer
    if _throttle:
        _buffer.append([_prefix, args, kwargs])
        if len(_buffer) > 30:
            flush_buffer()
    else:
        output(_prefix, args, kwargs)


def enable():
    global _enabled
    _enabled = True
def throttle():
    global _throttle
    _throttle = True

def save_results(model_name, results):
    results_df = pd.DataFrame(results).sort_values(['test_month', 'material_group'])

    group_actual_sum = results_df.groupby("material_group")["actual"].transform("sum")
    group_abs_error_sum = results_df.groupby("material_group")["abs_error"].transform("sum")

    results_df["custom_ape"] = results_df["abs_error"] / group_actual_sum
    results_df["group_wape"] = group_abs_error_sum / group_actual_sum
    results_df["wape"] = results_df["abs_error"].sum() / results_df["actual"].sum()

    results_df = results_df.round(3)

    print(results_df.to_string(index=False, col_space={'model': 7}), end="\n\n")
    results_df.to_excel(f'predictions/{model_name}.xlsx', index=False)
    results_df.to_csv(f'predictions/{model_name}.csv', index=False)