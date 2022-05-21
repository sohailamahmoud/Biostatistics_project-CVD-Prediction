from pyexpat import model
from xml.etree.ElementInclude import include
from click import style
import numpy as np
from flask import Flask, request, render_template
import pickle


submission = []

def home():
    if request.method== "POST":
        print(request.form)
        submission.append(
                ( 
             request.form.get("age"),
             request.form.get("gender"),
             request.form.get("height"),
             request.form.get("weight"),
             request.form.get("AP-HI"),
             request.form.get("AP-LO"),
             request.form.get("Cholesterol"),
             request.form.get("Glucose"),
             request.form.get("smoke"),
             request.form.get("alcohol"),
             request.form.get("Active"),
                 )
                       )
    return render_template("index.html")

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))

@app.route("/", methods=["GET","POST"])
def home():
    if request.method == "POST":
        print(request.form)
    return render_template("index.html")

@app.route("/predict", methods=["GET","POST"])
def show_predict():

    int_features = [x for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    
    if prediction == 1:
        prediction_text = "cardio"
    elif prediction == 0:
        prediction_text = "not"

    return render_template("index.html", prediction_text = "The prediction is {}".format(prediction_text))

if __name__ == "__main__":
    app.run(debug=True)