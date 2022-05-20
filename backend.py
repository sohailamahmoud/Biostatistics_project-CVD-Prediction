import numpy as np
from crypt import methods
from pyexpat import model
from flask import Flask, app, render_template, url_for, request
import pickle

def pre_process(x):
    
    
    
    return x


def load_model(x):
    model= pickle.load(open('model.py', 'rb'))
    out= model.predict(np.array(x), np.reshape(1, -1))
    return out(0)

app = Flask(__name__, template_folder='D:\Edu\Statistics\Statistics Project\Biostatistics_project-master\Biostatistics_project-master\Biostatistics\index1.html')

@app.route('/', methods=["GET"])
def inder():
    return render_template('index1.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request == "POST":
        f = request.file['file']
        x = pre_process(f)
        out = load_model(x)
        if out == 0 :
            return "Negative"
        else:
            return "Positive"
    return None



if __name__  == '__main__' :
    app.run(debug=True)