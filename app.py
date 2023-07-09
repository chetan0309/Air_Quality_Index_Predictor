import numpy as np
from flask import Flask, render_template, request
import pickle

app=Flask(__name__)
model=pickle.load(open("model.pkl","rb"))


@app.route("/")
def home():
    return render_template('index.html') 

@app.route("/predict",methods=["POST"])

def predict():
        int_features=[float(x) for x in request.form.values()]
        final_features=[np.array(int_features)]
        PM_2_5=model.predict(final_features)
        pm=np.round_(PM_2_5,decimals=3)
        return render_template("index.html",PM_predicted="PM 2.5 should be : {}".format(pm))

    
    
        




if __name__=="__main__":
    app.run(debug=True)
