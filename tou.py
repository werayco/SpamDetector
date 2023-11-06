from sklearn.linear_model import LogisticRegression
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split,KFold,cross_val_score
from sklearn.metrics import accuracy_score
import joblib as jb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.model_selection import KFold


#checking through the dataframe
def preprocess():
    data_1=pd.read_csv(r"C:\Users\LENOVO-PC\Videos\Price\mail_data.csv")

    #assigning spam as 1 and not spam using customized function
    def conv(x):
        if x == "spam":
            return 1
        elif x == "ham":
            return 0
            
    data_1["Category"] = data_1["Category"].apply(conv)

    #lets remove all the numbers, stopwords and stem the message column
    stemmer = PorterStemmer()

    def removerr(y):
        clean_message = re.sub("[^a-zA-z]"," ", y)
        clean_message = clean_message.lower()
        clean_message = clean_message.split()
        clean_message = [stemmer.stem(everyword) for everyword in clean_message if not everyword in stopwords.words("english")]
        clean_message = " ".join(clean_message)
        return clean_message

    #now lets implement the function on the message column 
    data_1["Message"]=data_1["Message"].apply(removerr)

    #assigning features and labels respectively
    x = data_1["Message"] 
    y = data_1["Category"]

    #converting the data into numerical values using Tfidf(you can use multinominal bayes if you please)
    vec=TfidfVectorizer()
    x_1=vec.fit_transform(x)