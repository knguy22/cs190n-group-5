import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, make_scorer

from pkl_cols import feature_cols, expected_cols

def main():
    df = pd.read_pickle('all_traffic_time_10.pkl')
    df_cols = df.columns.tolist()
    for col in feature_cols:
        assert col in df_cols
    
    features, expected = extract(df)
    train(features, expected)

def extract(df):
    features = df[feature_cols]
    expected = df[expected_cols]
    return features, expected

def train(features, expected):
    # Define the model
    rf_classifier = RandomForestClassifier(random_state=42)

    # Define the hyperparameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Define the scoring metric
    scorer = make_scorer(f1_score, average='weighted')

    # Perform grid search
    grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, scoring=scorer, cv=5, n_jobs=-1)
    grid_search.fit(features, expected)

if __name__ == '__main__':

    main()
