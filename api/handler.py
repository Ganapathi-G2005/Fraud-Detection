from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from flask import Flask, Response, request

from fraud.Fraud import Fraud

# loading model
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.parent / "models" / "model_cycle1.joblib"
model = joblib.load(MODEL_PATH)

# initialize API
app = Flask(__name__)

@app.route('/fraud/predict', methods=['POST'])
def churn_predict():
    test_json: Any = request.get_json(silent=True)
   
    if test_json:  # there is data
        if isinstance(test_json, dict):  # unique example
            test_raw = pd.DataFrame(test_json, index=[0])
            
        else:  # multiple example
            test_raw = pd.DataFrame(test_json, columns=test_json[0].keys())
            
        # Instantiate Fraud class
        pipeline = Fraud()
        
        # data cleaning
        df1 = pipeline.data_cleaning(test_raw)
        
        # feature engineering
        df2 = pipeline.feature_engineering(df1)
        
        # data preparation
        df3 = pipeline.data_preparation(df2)
        
        # prediction
        df_response = pipeline.get_prediction(model, test_raw, df3)
        
        return df_response
        
        
    else:
        return Response("{}", status=200, mimetype='application/json')

if __name__ == '__main__':
    app.run(host='0.0.0.0')
