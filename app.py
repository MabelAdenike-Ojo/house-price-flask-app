from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load model and columns
model = joblib.load("house_price_model.joblib")
model_columns = joblib.load("model_columns.joblib")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()  # JSON input from user
        input_df = pd.DataFrame([data])
        
        # Ensure columns match training
        input_df = input_df.reindex(columns=model_columns, fill_value=0)
        
        prediction = model.predict(input_df)[0]
        return jsonify({"Predicted_House_Price": round(prediction, 2)})
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run()