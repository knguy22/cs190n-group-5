import pandas as pd

df = pd.read_pickle("all_traffic_time_10.pkl")

df_train = df[df['service'] == 'netflix']
df_testing = df[df['service'] != 'netflix']

df_train.to_pickle("train.pkl")
df_testing.to_pickle("test.pkl")