import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import pickle
from pkl_cols import feature_cols

def main():
    # init
    df = pd.read_pickle('all_traffic_time_10.pkl')
    df_cols = df.columns.tolist()
    for col in feature_cols:
        assert col in df_cols

    features = df[feature_cols]
    expected_startup = df['startup_time']
    expected_resolution = df['resolution']

    train_resolution(features, expected_resolution, 'rf_resolution.pkl')
    train_startup(features, expected_startup, 'rf_startup_time.pkl')

def train_startup(features, expected, filename):
    # sampling
    max_samples = 100000
    sample_indices = np.random.choice(features.index, min(max_samples, len(features)), replace=False)
    f_samples = features.loc[sample_indices]
    e_samples = expected.loc[sample_indices]
    
    # Define the model
    rf_regressor = RandomForestRegressor(random_state=42, n_jobs=-1)
    
    # Perform grid search
    param_grid = {
        'n_estimators': [30, 50, 100],
        'max_depth': [10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', None],
        'bootstrap': [True, False],
        'criterion': ['squared_error'],
    }
    
    grid_search = GridSearchCV(
        estimator=rf_regressor, 
        param_grid=param_grid, 
        cv=5, 
        n_jobs=-1, 
        verbose=0,
        scoring='neg_mean_squared_error'
    )
    
    print('Performing grid search for startup...')
    grid_search.fit(f_samples, e_samples)
    
    # Save the model 
    print(f'Best hyperparameters: {grid_search.best_params_}')
    print(f'Negative Mean Squared Error: {grid_search.best_score_:.4f}')
    
    best_model = grid_search.best_estimator_
    with open(filename, 'wb') as file:
        pickle.dump(best_model, file)
    
    print(f'Model saved to {filename}')
    
    return best_model

def train_resolution(features, expected, filename):
    # sampling
    max_samples = 100000
    sample_indices = np.random.choice(features.index, min(max_samples, len(features)), replace=False)
    f_samples = features.loc[sample_indices]
    e_samples = expected.loc[sample_indices]
    
    # Define the model
    rf_classifier = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    # Perform grid search
    param_grid = {
        'n_estimators': [30, 50, 100],
        'max_depth': [10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', None],
        'bootstrap': [True, False],
        'criterion': ['gini', 'entropy'],
    }
    
    grid_search = GridSearchCV(
        estimator=rf_classifier, 
        param_grid=param_grid, 
        cv=5, 
        n_jobs=-1, 
        verbose=0,
        scoring='f1_weighted'  # Use weighted F1 for multiclass or imbalanced problems
    )
    
    print('Performing grid search for resolution...')
    grid_search.fit(f_samples, e_samples)
    
    # Save the model
    print(f'Best hyperparameters: {grid_search.best_params_}')
    print(f'Best F1 score: {grid_search.best_score_:.4f}')
    
    best_model = grid_search.best_estimator_
    with open(filename, 'wb') as file:
        pickle.dump(best_model, file)
    
    print(f'Model saved to {filename}')
    
    return best_model

if __name__ == '__main__':
    main()
