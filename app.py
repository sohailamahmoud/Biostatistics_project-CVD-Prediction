from pyexpat import model
from xml.etree.ElementInclude import include
from click import style
import numpy as np
from flask import Flask, request, render_template
import pickle


submission = []
app = Flask(__name__)

@app.route("/", methods=["GET","POST"])
def home():

    return render_template("index.html")



model = pickle.load(open("model.pkl", "rb"))



@app.route("/predict", methods=["GET","POST"])
def show_predict():
    global submission
    if request.method== "POST":
       
        submission.append(request.form.get("age"))

        gender = request.form.get("gender")
        if gender == "female":
            gender = 1
        else:
            gender = 2
        submission.append(gender)

        submission.append(request.form.get("height"))

        submission.append(request.form.get("weight"))
        
        submission.append(request.form.get("AP-HI"))
        
        submission.append(request.form.get("AP-LO"))
        
        Cholesterol = request.form.get("Cholesterol")
        if Cholesterol == "Normal":
            Cholesterol = 1
        elif Cholesterol == "Above normal":
            Cholesterol = 2
        else:
            Cholesterol = 3
        submission.append(Cholesterol)

        glucose = request.form.get("Glucose")
        if glucose == "Normal":
            glucose = 1
        elif glucose == "Above normal":
            glucose = 2
        else:
            glucose = 3
        submission.append(glucose)

        smoke = request.form.get("smoke")
        if smoke == "No":
            smoke = 0
        else:
            smoke = 1
        submission.append(smoke)

        alcohol = request.form.get("alcohol")
        if alcohol == "No":
            alcohol = 0
        else:
            alcohol = 1
        submission.append(alcohol)

        Active = request.form.get("Active")
        if Active == "No":
            Active = 0
        else:
            Active = 1
        submission.append(Active)  

    final_features = [submission]

    prediction = model.predict(final_features)

    submission = []
    
    if prediction == 1:
        prediction_text = "have a cardiovascular disease. Please contact a doctor for certain results."
    elif prediction == 0:
        prediction_text = "not have a cardiovascular disease."

    return render_template("index.html", prediction_text = " You may {}".format(prediction_text))

if __name__ == "__main__":
    app.run(debug=True)