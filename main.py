import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, make_scorer
import pickle
from pkl_cols import feature_cols, expected_cols

# pd.set_option('display.max_columns', None)  # Show all columns
# pd.set_option('display.max_colwidth', None)  # Show full column contents

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

    print(features.head(5))
    print(expected.head(5))
    # print_features_dtypes(features)
    # print_features_dtypes(expected)
    return features, expected

def train(features, expected):
    # Define the model
    rf_regressor = RandomForestRegressor(random_state=42)

    # Define the hyperparameter grid
    param_grid = {
        'n_estimators': [40],
        'max_depth': [10],
        'min_samples_split': [5, 10],
        'min_samples_leaf': [2, 4],
    }

    # Define the scoring metric
    scorer = make_scorer(f1_score, average='weighted')

    # Perform grid search
    grid_search = GridSearchCV(estimator=rf_regressor, param_grid=param_grid, scoring=scorer, cv=5, n_jobs=-1, error_score='raise')
    grid_search.fit(features, expected)

    # Save the model 
    print(f'Best hyperparameters: {grid_search.best_params_}')
    print(f'F1 score on test data: {grid_search.best_score_:.4f}')

    best_model = grid_search.best_estimator_
    with open('best_random_forest_model.pkl', 'wb') as file:
        pickle.dump(best_model, file)
    
    print('Model saved to best_random_forest_model.pkl')
    
    return best_model

def print_features_dtypes(features):
    # Convert dtypes to a DataFrame for better formatting
    dtypes_df = pd.DataFrame({
        'Feature': features.columns,
        'Data Type': features.dtypes.astype(str)
    })

    # Print the DataFrame as a string without truncation
    print(dtypes_df.to_string(index=False))

if __name__ == '__main__':

    main()
