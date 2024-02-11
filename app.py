from flask import Flask, render_template, request,url_for
import joblib as jb
import pickle as plk
from nltk.stem.porter import PorterStemmer
import re
from nltk.corpus import stopwords 
from utilss import changer

model_name = changer("model.pkl")
vec = changer("vectorizer.pkl")
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("homepage.html")

stemmer = PorterStemmer()


@app.route("/spam_classifier",methods=["GET", "POST"])
def classifier():
   if request.method == "POST":
        data = request.form.get("input")
        clean_message = re.sub("[^a-zA-z]"," ", data)
        clean_message = clean_message.lower()
        clean_message = clean_message.split()
        clean_message = [stemmer.stem(everyword) for everyword in clean_message if not everyword in stopwords.words("english")]
        convt_text = vec.transform(clean_message)
        prediction = model_name.predict(convt_text)
        spam= "Spam"
        notspam= "notSpam"
        return render_template("homepage.html",prediction=prediction[0],spam=spam,notspam=notspam)

if __name__ == "__main__":
    app.run(debug=True)

#Updated today< November 11, 5PM