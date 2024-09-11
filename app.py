from flask import Flask, jsonify
from flask_cors import CORS
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

app = Flask(__name__)
CORS(app)

def prepare_data():
    sp500 = yf.Ticker("^GSPC")
    sp500 = sp500.history(period="max")

    sp500 = sp500.drop(columns=["Dividends", "Stock Splits"])
    sp500["Tomorrow"] = sp500["Close"].shift(-1)
    sp500["Target"] = (sp500["Tomorrow"] > sp500["Close"]).astype(int)
    sp500 = sp500.loc["1990-01-01":].copy()

    horizons = [2, 5, 60, 250, 1000]
    new_predictors = []

    for horizon in horizons:
        rolling_averages = sp500.rolling(horizon).mean()

        ratio_column = f"Close_Ratio_{horizon}"
        sp500[ratio_column] = sp500["Close"] / rolling_averages["Close"]

        trend_column = f"Trend_{horizon}"
        sp500[trend_column] = sp500.shift(1).rolling(horizon).sum()["Target"]

        new_predictors += [ratio_column, trend_column]

    sp500 = sp500.dropna()
    
    return sp500, new_predictors

def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict_proba(test[predictors])[:, 1]
    preds = (preds >= 0.6).astype(int)
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined

def backtest(data, model, predictors, start=2500, step=250):
    all_predictions = []

    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i + step)].copy()
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)
    
    return pd.concat(all_predictions)

@app.route('/predict', methods=['GET'])
def predict_endpoint():
    sp500, new_predictors = prepare_data()
    model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)

    predictions = backtest(sp500, model, new_predictors)

    prediction_counts = predictions["Predictions"].value_counts().to_dict()
    precision = precision_score(predictions["Target"], predictions["Predictions"])

    all_results = predictions.reset_index().to_dict(orient="records")

    last_data = sp500.iloc[-1:]
    tomorrow_prediction = model.predict_proba(last_data[new_predictors])[:, 1]
    tomorrow_prediction = int(tomorrow_prediction >= 0.6)
    
    return jsonify({
        "prediction_counts": prediction_counts,
        "precision_score": precision,
        "all_results": all_results,
        "tomorrow_prediction": "Increase" if tomorrow_prediction == 1 else "Decrease"
    })

if __name__ == '__main__':
    app.run(debug=True)
