import pandas as pd


class FeatureEngineer:
    def __init__(self):
        pass

    @staticmethod
    def create_count_map(series: pd.Series) -> dict[str, int]:
        count = series.value_counts(dropna=False).reset_index()
        return dict(zip(count["index"], count.index))

    @staticmethod
    def create_binary(series: pd.Series, one_val: str) -> pd.Series:
        return series.apply(lambda val: 1 if val == one_val else 0)

    @staticmethod
    def create_ordinal(series: pd.Series, mapping: dict[str, int]) -> pd.Series:
        return series.apply(lambda val: mapping[val])

    @staticmethod
    def create_one_hot(series: pd.Series, prefix: str = None) -> pd.DataFrame:
        return pd.get_dummies(series, prefix=prefix)

    def process_features(self, features: pd.DataFrame) -> pd.DataFrame:
        df = features.copy()
        df["Type"] = self.create_binary(df["Type"], "Dog")
        df["Breed1"] = self.create_ordinal(df["Breed1"], self.create_count_map(df["Breed1"]))
        df["Gender"] = self.create_binary(df["Gender"], "Male")
        df = df.join(self.create_one_hot(df["Color1"], prefix="Color1")).drop("Color1", axis=1)
        df = df.join(self.create_one_hot(df["Color2"], prefix="Color2")).drop("Color2", axis=1)
        df["MaturitySize"] = self.create_ordinal(df["MaturitySize"], {"Small": 0, "Medium": 1, "Large": 2})
        df["FurLength"] = self.create_ordinal(df["FurLength"], {"Short": 0, "Medium": 1, "Long": 2})
        df["Vaccinated"] = self.create_ordinal(df["Vaccinated"], {"No": -1, "Not Sure": 0, "Yes": 1})
        df["Sterilized"] = self.create_ordinal(df["Sterilized"], {"No": -1, "Not Sure": 0, "Yes": 1})
        df["Health"] = self.create_ordinal(df["Health"], {"Healthy": 0, "Minor Injury": 1, "Serious Injury": 2})
        return df

    def process_target(self, target: pd.Series) -> pd.Series:
        series = target.copy()
        return self.create_binary(series, "Yes")
