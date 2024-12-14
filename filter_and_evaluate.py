from features import FINAL_MODEL
from sklearn.metrics import classification_report, r2_score, root_mean_squared_error
import pandas as pd
import pickle

def filter_and_eval_res():
    df_test = pd.read_pickle('validation_2.pkl')
    with open('resolution_rf_all_FINAL_MODEL_FUZZY_time10_model.pkl', 'rb') as file:
        res_model = pickle.load(file)

    conditions = get_conditions(df_test)
    for cond in conditions:
        eval_res(df_test, cond, res_model)

def filter_and_eval_startup():
    df_test = pd.read_pickle('validation_2.pkl')
    with open('startup_time_rf_all_FINAL_MODEL_FUZZY_time10_model.pkl', 'rb') as file:
        start_model = pickle.load(file)

    conditions = get_conditions(df_test)
    for cond in conditions:
        eval_startup(df_test, cond, start_model)

def get_conditions(df):
    conds = [pd.Series([True] * len(df), index=df.index)]
    # conds.append(conds[-1] & (df['10_chunksizes_85'] > 321531.219))
    # conds.append(conds[-1] & (df['allprev_chunksizes_50'] > 17477.5))
    # conds.append(conds[-1] & (df['allprev_max_chunksize'] <= 1935595.5))
    conds.append(conds[-1] & (df['allprev_max_chunksize'] <= 1276703.875))
    # conds.append(conds[-1] & (df['service_Video_throughput_down'] <= 610.264))
    return conds

def eval_res(df_test, cond, model):
    df_test = filter(df_test, cond)
    features_test = df_test[FINAL_MODEL]
    expected_test_res = df_test['resolution']

    y_pred = model.predict(features_test)
    print(df_test.shape)
    print(classification_report(expected_test_res, y_pred, zero_division=0))
    print()

def eval_startup(df_test, cond, model):
    df_test = filter(df_test, cond)
    features_test = df_test[FINAL_MODEL]
    expected_test_start = df_test['startup_time']

    y_pred = model.predict(features_test)
    print(df_test.shape)
    print(r2_score(expected_test_start, y_pred))
    print(root_mean_squared_error(expected_test_start, y_pred))
    print()

def filter(df, cond):
    return df[cond]

if __name__ == '__main__':
    filter_and_eval_res()
    # filter_and_eval_startup()