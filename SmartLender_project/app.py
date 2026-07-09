from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

model = joblib.load("model.pkl")


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict")
def predict():
    return render_template("predict.html")


@app.route("/result", methods=["POST"])
def result():

    features = [
        float(request.form["Gender"]),
        float(request.form["Married"]),
        float(request.form["Dependents"]),
        float(request.form["Education"]),
        float(request.form["Self_Employed"]),
        float(request.form["ApplicantIncome"]),
        float(request.form["CoapplicantIncome"]),
        float(request.form["LoanAmount"]),
        float(request.form["Loan_Amount_Term"]),
        float(request.form["Credit_History"]),
        float(request.form["Property_Area"])
    ]

    prediction = model.predict([features])

    if prediction[0] == 1:
        result = "Loan Approved"
    else:
        result = "Loan Rejected"

    return render_template("result.html", prediction=result)


if __name__ == "__main__":
    app.run(debug=True)