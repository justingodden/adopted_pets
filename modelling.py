import os

import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, recall_score
from xgboost import XGBClassifier


class Modeller:
    def __init__(self):
        pass

    @staticmethod
    def get_best_iteration(X_train: pd.DataFrame,
                           X_val: pd.DataFrame,
                           y_train: pd.Series,
                           y_val: pd.Series) -> int:
        xgb = XGBClassifier(use_label_encoder=False)
        model = xgb.fit(X_train, y_train,
                        eval_metric="logloss",
                        eval_set=[(X_val, y_val)],
                        early_stopping_rounds=20
                        )
        return model.get_booster().best_iteration

    def fit(self,
            X_train: pd.DataFrame,
            y_train: pd.Series,
            n_estimators: int) -> None:
        xgb = XGBClassifier(use_label_encoder=False,
                            n_estimators=n_estimators)
        self.model = xgb.fit(X_train, y_train, eval_metric="logloss")

    def predict(self, df: pd.DataFrame) -> pd.Series:
        return self.model.predict(df)

    @staticmethod
    def evaluate(pred: pd.Series, actual: pd.Series) -> None:
        print(f"F1 Score on the test set: {f1_score(pred, actual):.4f}")
        print(f"Accuracy Score on the test set: {accuracy_score(pred, actual):.4f}")
        print(f"Recall Score on the test set: {recall_score(pred, actual):.4f}")

    def save(self) -> None:
        folder_path = os.path.join(os.getcwd(), "artifacts")
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        file_path = os.path.join(folder_path, "model")
        self.model.save_model(file_path)

    def load(self) -> None:
        file_path = os.path.join(os.getcwd(), "artifacts", "model")
        self.model = XGBClassifier()
        self.model.load_model(file_path)
