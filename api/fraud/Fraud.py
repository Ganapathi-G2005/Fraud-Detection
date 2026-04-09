from __future__ import annotations

from pathlib import Path

import joblib
import inflection
import pandas as pd


class Fraud:
    def __init__(self) -> None:
        base_dir = Path(__file__).resolve().parent
        parameters_dir = base_dir.parent.parent / "parameters"

        self.minmaxscaler = joblib.load(parameters_dir / "minmaxscaler_cycle1.joblib")
        self.onehotencoder = joblib.load(parameters_dir / "onehotencoder_cycle1.joblib")

    def data_cleaning(self, df1: pd.DataFrame) -> pd.DataFrame:
        cols_old = df1.columns.tolist()
        cols_new = [inflection.underscore(col) for col in cols_old]

        df1.columns = cols_new
        return df1

    def feature_engineering(self, df2: pd.DataFrame) -> pd.DataFrame:
        # Step converted to day/week views.
        df2['step_days'] = df2['step'].apply(lambda i: i / 24)
        df2['step_weeks'] = df2['step'].apply(lambda i: i / (24 * 7))

        # Difference between initial and new origin balances.
        df2['diff_new_old_balance'] = df2['newbalance_orig'] - df2['oldbalance_org']

        # Difference between initial and new destination balances.
        df2['diff_new_old_destiny'] = df2['newbalance_dest'] - df2['oldbalance_dest']

        # Capture account prefix.
        df2['name_orig'] = df2['name_orig'].apply(lambda i: i[0])
        df2['name_dest'] = df2['name_dest'].apply(lambda i: i[0])

        return df2.drop(columns=['name_orig', 'name_dest', 'step_weeks', 'step_days'])

    def data_preparation(self, df3: pd.DataFrame) -> pd.DataFrame:
        # OneHotEncoder
        df3 = self.onehotencoder.transform(df3)

        # Rescaling
        num_columns = [
            'amount',
            'oldbalance_org',
            'newbalance_orig',
            'oldbalance_dest',
            'newbalance_dest',
            'diff_new_old_balance',
            'diff_new_old_destiny',
        ]
        df3[num_columns] = self.minmaxscaler.transform(df3[num_columns])

        # Selected columns
        final_columns_selected = [
            'step',
            'oldbalance_org',
            'newbalance_orig',
            'newbalance_dest',
            'diff_new_old_balance',
            'diff_new_old_destiny',
            'type_TRANSFER',
        ]
        return df3[final_columns_selected]

    def get_prediction(
        self,
        model: object,
        original_data: pd.DataFrame,
        test_data: pd.DataFrame,
    ) -> str:
        pred = model.predict(test_data)
        original_data['prediction'] = pred

        return original_data.to_json(orient="records", date_format="iso")