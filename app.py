from flask import Flask, render_template, request,url_for
import joblib as jb
import pickle as plk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
import re
from nltk.corpus import stopwords 
vec = TfidfVectorizer()

model_name = jb.load("spam_model_byRayco.joblib")
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

stemmer = PorterStemmer()
# def removerr(y):
#         clean_message = re.sub("[^a-zA-z]"," ", y)
#         clean_message = clean_message.lower()
#         clean_message = clean_message.split()
#         clean_message = [stemmer.stem(everyword) for everyword in clean_message if not everyword in stopwords.words("english")]
#         clean_message = " ".join(clean_message)
#         return clean_message


# @app.route("/spam_classifier",methods=["GET", "POST"])
# def classifier():
#    if request.method == "POST":
#         data = request.form.get("input")
#         clean_message = re.sub("[^a-zA-z]"," ", data)
#         clean_message = clean_message.lower()
#         clean_message = clean_message.split()
#         clean_message = [stemmer.stem(everyword) for everyword in clean_message if not everyword in stopwords.words("english")]
#         convt_text = vec.fit_transform(clean_message)
#         prediction = model_name.predict(convt_text)
#         return f"<h1> preprocessed {prediction}</h1>"

# if __name__ == "__main__":
#     app.run(debug=True)


# @app.route("/predict",methods=["POST","GET"])
# def predictor()->str:
#     if request.method=="POST":
#      data1 = request.form.get("input")
#      conv=vec.fit_transform()


x = 1
x+=23
print(x)