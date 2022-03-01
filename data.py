import pandas as pd
from sklearn.model_selection import train_test_split


def get_dataframe_from_url(url: str) -> pd.DataFrame:
    return pd.read_csv(url)


def split_features_and_target(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series]:
    return df.drop(target, axis=1), df[target]


def train_val_test_split(X: pd.DataFrame,
                         y: pd.Series,
                         random_state: int = 42) -> tuple[pd.DataFrame,
                                                          pd.DataFrame,
                                                          pd.DataFrame,
                                                          pd.Series,
                                                          pd.Series,
                                                          pd.Series]:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=random_state)
    return X_train, X_val, X_test, y_train, y_val, y_test
