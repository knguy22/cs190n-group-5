from trustee import ClassificationTrustee, RegressionTrustee
# from pkl_cols import final_filtered_feature_cols as feature_cols
from features import FINAL_MODEL
from sklearn import tree, metrics
from sklearn.metrics import classification_report, r2_score, root_mean_squared_error
import pandas as pd
import pickle
import graphviz

# most of this code is copied from the Trustee documentation

def main():
    print("Importing data...")
    df_train = pd.read_pickle('testing_2.pkl')
    df_test = pd.read_pickle('validation_2.pkl')

    # print("Importing resolution model...")
    # with open('resolution_rf_all_FINAL_MODEL_FUZZY_time10_model.pkl', 'rb') as file:
    #     res_model = pickle.load(file)

    print("Importing startup model...")
    with open('startup_time_rfr_all_FINAL_MODEL_FUZZY_time10_model.pkl', 'rb') as file:
        start_model = pickle.load(file)

    features_train = df_train[FINAL_MODEL]
    features_test = df_test[FINAL_MODEL]

    # expected_train_res = df_train['resolution']
    # expected_test_res = df_test['resolution']

    expected_train_start = df_train['startup_time']
    expected_test_start = df_test['startup_time']

    # print("Resolution on training data:")
    # y_pred = res_model.predict(features_train)
    # print(metrics.classification_report(expected_train_res, y_pred, zero_division=0))

    # print("Resolution on test data:")
    # y_pred = res_model.predict(features_test)
    # print(metrics.classification_report(expected_test_res, y_pred, zero_division=0))

    print("Startup on training data:")
    y_pred = start_model.predict(features_train)
    print(r2_score(expected_train_start, y_pred))
    print(root_mean_squared_error(expected_train_start, y_pred))

    print("Startup on test data:")
    y_pred = start_model.predict(features_test)
    print(r2_score(expected_test_start, y_pred))
    print(root_mean_squared_error(expected_test_start, y_pred))

    print("starting trustee")
    trustee_start(features_train, expected_train_start, features_test, expected_test_start, start_model, 'rf_start')
    # trustee_res(features_train, expected_train_res, features_test, expected_test_res, res_model, 'rf_res')

def trustee_res(X_train, y_train, X_test, y_test, model, filename):
    expected_classes = ['240', '360', '480', '720', '1080']

    # Evaluate model accuracy
    y_pred = model.predict(X_test)
    print(metrics.classification_report(y_test, y_pred, zero_division=0))

    # train trustee
    trustee = ClassificationTrustee(expert=model)
    trustee.fit(X_train, y_train, num_iter=50, num_stability_iter=10, samples_size=0.2, verbose=True)
    dt, pruned_dt, agreement, reward = trustee.explain()
    print(f"Model explanation training (agreement, fidelity): ({agreement}, {reward})")
    print(f"Model Explanation size: {dt.tree_.node_count}")
    print(f"Top-k Pruned Model explanation size: {pruned_dt.tree_.node_count}")

    # Use explanations to make predictions
    dt_y_pred = dt.predict(X_test)
    pruned_dt_y_pred = pruned_dt.predict(X_test)

    # Evaluate accuracy and fidelity of explanations
    print("Model explanation global fidelity report:")
    print(classification_report(y_pred, dt_y_pred, zero_division=0))
    print("Top-k Model explanation global fidelity report:")
    print(classification_report(y_pred, pruned_dt_y_pred, zero_division=0))

    print("Model explanation score report:")
    print(classification_report(y_test, dt_y_pred, zero_division=0))
    print("Top-k Model explanation score report:")
    print(classification_report(y_test, pruned_dt_y_pred, zero_division=0))

    export_dt(dt, pruned_dt, expected_classes, list(X_train.columns), filename)

def trustee_start(X_train, y_train, X_test, y_test, model, filename):
    # Evaluate model accuracy
    y_pred = model.predict(X_test)
    print("Model R2-score:", r2_score(y_test, y_pred))
    print("Root Mean Squared Error:", root_mean_squared_error(y_test, y_pred))

    # Initialize Trustee and fit for classification models
    trustee = RegressionTrustee(expert=model)
    trustee.fit(X_train, y_train, num_iter=50, num_stability_iter=7, samples_size=0.2, verbose=True)

    # Get the best explanation from Trustee
    dt, pruned_dt, agreement, reward = trustee.explain()
    print(f"Model explanation training (agreement, fidelity): ({agreement}, {reward})")
    print(f"Model Explanation size: {dt.tree_.node_count}")
    print(f"Top-k Pruned Model explanation size: {pruned_dt.tree_.node_count}")

    # Use explanations to make predictions
    dt_y_pred = dt.predict(X_test)
    pruned_dt_y_pred = pruned_dt.predict(X_test)

    # Evaluate accuracy and fidelity of explanations
    print("Model explanation global fidelity:")
    print("Model R2-score:",r2_score(y_pred, dt_y_pred))
    print("Root Mean Squared Error:",root_mean_squared_error(y_pred, dt_y_pred))

    print("Top-k Model explanation global fidelity:")
    print("Model R2-score:",r2_score(y_pred, pruned_dt_y_pred))
    print("Root Mean Squared Error:",root_mean_squared_error(y_pred, pruned_dt_y_pred))

    print("Model explanation R2-score:")
    print("Model R2-score:",r2_score(y_test, dt_y_pred))
    print("Root Mean Squared Error:",root_mean_squared_error(y_test, dt_y_pred))

    print("Top-k Model explanation R2-score:")
    print("Model R2-score:",r2_score(y_test, pruned_dt_y_pred))
    print("Root Mean Squared Error:",root_mean_squared_error(y_test, pruned_dt_y_pred))

    export_dt(dt, pruned_dt, None, list(X_train.columns), filename)

def export_dt(dt, pruned_dt, class_names, feature_names, filename):
    # # Output decision tree to pdf (can take a while to render)
    # dot_data = tree.export_graphviz(
    #     dt,
    #     class_names=class_names,
    #     feature_names=feature_names,
    #     filled=True,
    #     rounded=True,
    #     special_characters=True,
    # )
    # graph = graphviz.Source(dot_data)
    # graph.render(f"dt_explanation_{filename}")

    # Output pruned decision tree to pdf
    dot_data = tree.export_graphviz(
        pruned_dt,
        class_names=class_names,
        feature_names=feature_names,
        filled=True,
        rounded=True,
        special_characters=True,
    )
    graph = graphviz.Source(dot_data)
    graph.render(f"pruned_{filename}")

if __name__ == '__main__':
    main()
