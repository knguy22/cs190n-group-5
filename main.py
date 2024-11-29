import pandas as pd

def main():
    df = pd.read_pickle('all_traffic_time_10.pkl')
    print(df.columns)

if __name__ == '__main__':
    main()