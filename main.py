import os

import data
import engineering
import modelling


def task_1(url: str):
    # 1. read from input
    df = data.get_dataframe_from_url(url)

    # 2. split dataset
    X, y = data.split_features_and_target(df, "Adopted")
    X_train, X_val, X_test, y_train, y_val, y_test = data.train_val_test_split(X, y)

    # 3. perform feature engineering
    fe = engineering.FeatureEngineer()
    X_train = fe.process_features(X_train)
    X_val = fe.process_features(X_val)
    X_test = fe.process_features(X_test)
    y_train = fe.process_target(y_train)
    y_val = fe.process_target(y_val)
    y_test = fe.process_target(y_test)

    # 4. train ML model
    modeller = modelling.Modeller()
    n_estimators = modeller.get_best_iteration(X_train, X_val, y_train, y_val)
    modeller.fit(X_train, y_train, n_estimators)
    y_pred = modeller.predict(X_test)

    # 5. log model performance to user
    modeller.evaluate(y_pred, y_test)

    # save for use in task 2
    modeller.save()


def task_2(url: str):
    # 1. load data
    df = data.get_dataframe_from_url(url)

    # 2. use model to create prediction
    modeller = modelling.Modeller()
    modeller.load()
    X, _ = data.split_features_and_target(df, "Adopted")
    fe = engineering.FeatureEngineer()
    X = fe.process_features(X)
    df["Adopted_prediction"] = modeller.predict(X)

    # 3. save predictions to output/results.csv
    folder_path = os.path.join(os.getcwd(), "output")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    file_path = os.path.join(folder_path, "results.csv")
    df.to_csv(file_path, index=False)


if __name__ == "__main__":
    url = "https://storage.googleapis.com/cloud-samples-data/ai-platform-unified/datasets/tabular/petfinder-tabular-classification.csv"
    task_1(url)
    task_2(url)
