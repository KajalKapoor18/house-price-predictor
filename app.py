from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Load model and encoders
model = joblib.load("house_price_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        house_size = float(request.form["house_size"])
        location = request.form["location"]
        city = request.form["city"]
        num_bathrooms = int(request.form["bathrooms"])
        num_balconies = int(request.form["balconies"])

        input_df = pd.DataFrame([{
            "house_size": house_size,
            "location": location,
            "city": city,
            "numBathrooms": num_bathrooms,
            "numBalconies": num_balconies
        }])

        for col in ["location", "city"]:
            le = label_encoders[col]
            if input_df[col].iloc[0] not in le.classes_:
                return render_template("index.html", prediction_text=f"Error: '{input_df[col].iloc[0]}' not in {col}.")
            input_df[col] = le.transform(input_df[col])

        prediction = model.predict(input_df)[0]
        return render_template("index.html", prediction_text=f"Predicted Price: ₹{round(prediction, 2)}")

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(debug=True, host="0.0.0.0", port=port)