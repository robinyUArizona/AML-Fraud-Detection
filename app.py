from flask import Flask, request, render_template
from src.pipeline.prediction_pipeline import CustomData, PredictionPipeline

application = Flask(__name__)
app = application

# Route for a home page
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("home.html")
    else:
        data = CustomData(
            from_bank = request.form.get("from_bank"),
            account = request.form.get("account"),
            to_bank = request.form.get("to_bank"),
            account_1 = request.form.get("account_1"),
            amount_received =  request.form.get("amount_received"),
            receiving_currency = request.form.get("receiving_currency"),
            payment_format = request.form.get("payment_format")
        )
        pred_df = data.get_data_as_DataFrame()
        print(pred_df)

        predict_pipeline = PredictionPipeline()
        results = predict_pipeline.predict(pred_df)
        return render_template("home.html", results=results[0])
    
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)