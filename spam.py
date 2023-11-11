from sklearn.linear_model import LogisticRegression
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split,KFold,cross_val_score
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score
import joblib as jb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import logging
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import KFold


#checking through the dataframe
data_1=pd.read_csv(".\mail_data.csv")

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
vec=CountVectorizer()
x_1=vec.fit_transform(x)

#splitting the data into test and train portions(you can use cross validation)
x_train,x_test,y_train,y_test = train_test_split(x_1,y,test_size=0.3,random_state=20)

#using the logistic regression module
spammer=MultinomialNB()
fitter = spammer.fit(x_train,y_train)
predicted_spam=spammer.predict(x_test)

#checking the accuracy
acc=accuracy_score(y_test,predicted_spam)
print(acc)
kfolds_1=StratifiedKFold(10)
results = cross_val_score(spammer,x_1,y,cv=kfolds_1,scoring="accuracy")
#lets check the returned values from the cross-validation
print(np.mean(results))

#saving our modelk8.9k
our_model=jb.dump(spammer,"spam_model_byRayco.joblib")
name = 'the name of the "man" is kunle'




