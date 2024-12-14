import pandas as pd

df = pd.read_pickle("all_traffic_time_10.pkl")

df_train = df[df['service'] == 'youtube']
df_testing = df[df['service'] != 'youtube']

df_train.to_pickle("youtube_train.pkl")
df_testing.to_pickle("youtube_test.pkl")