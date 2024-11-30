import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import classification_report, confusion_matrix
import pickle
from pkl_cols import feature_cols

def main():
    # init
    df = pd.read_pickle('all_traffic_time_10.pkl')
    df_cols = df.columns.tolist()
    for col in feature_cols:
        assert col in df_cols

    samples, validation = sample(df)

    model_res = train_resolution(samples['features'], samples['expected_resolution'], 'rf_resolution.pkl')
    model_start = train_startup(samples['features'], samples['expected_startup'], 'rf_startup_time.pkl')
    
    model_scores = score_model(validation, model_start, model_res)
    print(model_scores)

def sample(df):
    features = df[feature_cols]
    expected_startup = df['startup_time']
    expected_resolution = df['resolution']

    max_samples = 100000
    sample_indices = np.random.choice(features.index, min(max_samples, len(features)), replace=False)
    other_indices = np.setdiff1d(np.arange(len(features)), sample_indices)

    samples = {
        "features": features.loc[sample_indices],
        "expected_startup": expected_startup.loc[sample_indices],
        "expected_resolution": expected_resolution.loc[sample_indices]
    }

    validation = {
        "features": features.loc[other_indices],
        "expected_startup": expected_startup.loc[other_indices],
        "expected_resolution": expected_resolution.loc[other_indices]
    }

    return samples, validation

def score_model(validation, model_startup, model_resolution):
    """
    Score the trained models using validation data.
    
    Parameters:
    - validation: Dictionary containing validation features and expected values
    - model_startup: Trained RandomForestRegressor for startup time prediction
    - model_resolution: Trained RandomForestClassifier for resolution prediction
    
    Returns:
    - Dictionary with model performance metrics
    """
    # Validate startup time model (Regression)
    startup_predictions = model_startup.predict(validation['features'])
    startup_mse = mean_squared_error(validation['expected_startup'], startup_predictions)
    startup_mae = mean_absolute_error(validation['expected_startup'], startup_predictions)
    startup_r2 = r2_score(validation['expected_startup'], startup_predictions)
    
    # Validate resolution model (Classification)
    resolution_predictions = model_resolution.predict(validation['features'])
    resolution_report = classification_report(
        validation['expected_resolution'], 
        resolution_predictions, 
        output_dict=True
    )
    resolution_confusion_matrix = confusion_matrix(
        validation['expected_resolution'], 
        resolution_predictions
    )
    
    # Print and return results
    print("Startup Time Model Performance:")
    print(f"Mean Squared Error: {startup_mse:.4f}")
    print(f"Mean Absolute Error: {startup_mae:.4f}")
    print(f"R-squared: {startup_r2:.4f}")
    
    print("\nResolution Model Performance:")
    print("Classification Report:")
    print(classification_report(
        validation['expected_resolution'], 
        resolution_predictions
    ))
    print("\nConfusion Matrix:")
    print(resolution_confusion_matrix)
    
    return {
        'startup_mse': startup_mse,
        'startup_mae': startup_mae,
        'startup_r2': startup_r2,
        'resolution_report': resolution_report,
        'resolution_confusion_matrix': resolution_confusion_matrix
    }

def train_startup(features, expected, filename):
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
    grid_search.fit(features, expected)
    
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
