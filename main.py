import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import pickle
from pkl_cols import feature_cols

def main():
    df = pd.read_pickle('all_traffic_time_10.pkl')

    samples, validation = sample(df)

    model_start = train_startup(samples['features'], samples['expected_startup'], 'rf_startup_time.pkl')
    model_res = train_resolution(samples['features'], samples['expected_resolution'], 'rf_resolution.pkl')

    model_scores = score_model(validation, model_start, model_res)
    print(model_scores)

def sample(df):
    res_col = 'resolution'
    startup_col = 'startup_mc'

    # correct types
    df[res_col] = df[res_col].astype(int)

    # filter out impossible values
    print("Startup time statistics before filtering:")
    print(df[startup_col].describe())

    df[startup_col] = df[startup_col].where(df[startup_col] > 0, np.nan)
    df = df.dropna(subset=[startup_col])

    print("Startup time statistics after filtering:")
    print(df[startup_col].describe())

    # Reset index to avoid mismatches
    df = df.reset_index(drop=True)

    # check the columns
    df_cols = df.columns.tolist()
    for col in feature_cols:
        assert col in df_cols
    features = df[feature_cols]

    # output of models
    expected_startup = df[startup_col]
    expected_resolution = df[res_col]

    # sampling
    max_samples = 100000
    sample_indices = np.random.choice(df.index, min(max_samples, len(df)), replace=False)
    other_indices = np.setdiff1d(np.arange(len(df)), sample_indices)

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

    df.loc[sample_indices].to_pickle('sample_traffic_time_10.pkl')
    df.loc[other_indices].to_pickle('validation_traffic_time_10.pkl')

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
        output_dict=True,
        zero_division=0, # prevent divisions by zero
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
        resolution_predictions,
        zero_division=0, # prevent divisions by zero
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
        'n_estimators': [30, 50],
        'max_depth': [10, None],
        'min_samples_split': [5],
        'min_samples_leaf': [2],
        'max_features': ['sqrt'],
        'bootstrap': [True],
        'criterion': ['squared_error'],
    }

    grid_search = GridSearchCV(
        estimator=rf_regressor, 
        param_grid=param_grid, 
        cv=3,
        n_jobs=7, 
        verbose=2,
        scoring='neg_mean_squared_error',
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
    # Confirm resolution classes
    unique_resolutions = expected.unique()
    print("Unique resolution classes:", unique_resolutions)

    # Ensure the five expected classes are present
    expected_classes = {240, 360, 480, 720, 1080}
    assert set(unique_resolutions).issuperset(expected_classes), \
        f"Missing expected resolution classes. Found: {unique_resolutions}"
    print(set(unique_resolutions))

    # sampling
    max_samples = 100000
    sample_indices = np.random.choice(features.index, min(max_samples, len(features)), replace=False)
    f_samples = features.loc[sample_indices]
    e_samples = expected.loc[sample_indices]

    # Define the model
    rf_classifier = RandomForestClassifier(random_state=42, n_jobs=-1)
    def custom_f1_weighted(estimator, X, y):
        y_pred = estimator.predict(X)
        return f1_score(y, y_pred, average='weighted', labels=[0, 240, 360, 480, 720, 1080])

    # Perform grid search
    param_grid = {
        'n_estimators': [30, 50],
        'max_depth': [10, None],
        'min_samples_split': [5],
        'min_samples_leaf': [2],
        'max_features': ['sqrt'],
        'bootstrap': [True],
        'criterion': ['gini', 'entropy'],
    }

    grid_search = GridSearchCV(
        estimator=rf_classifier, 
        param_grid=param_grid, 
        cv=3, 
        n_jobs=7, 
        verbose=2,
        scoring=custom_f1_weighted,
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

def graph_col(df, col, filename):
    plt.hist(df[col], bins=100)
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.title(f'{col} Distribution')
    plt.savefig(filename)

if __name__ == '__main__':
    main()
