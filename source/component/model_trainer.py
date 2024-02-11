from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostRegressor
from sklearn.neighbors import KNeighborsClassifier
from dataclasses import dataclass
import os
from utils import target_to_num,removerr,BestModelGenerator,pickle_saver
from source.exception import ErrorHandling
import sys
from source.logger import logging
from sklearn.metrics import r2_score
import numpy as np



@dataclass 
class ModelConfig:
    savingpath:str=os.path.join("PickleFiles","Model.pkl")

class ModelTrainerProcess:
    def __init__(self):
        self.transform=ModelConfig()

    def process_three(self,train_arr,test_arr):
        try:
            models_1={"LogisticRegression":LogisticRegression(),"Random_forest":RandomForestClassifier(),
                "KNN":KNeighborsClassifier(n_neighbors=3),"DecisonTree":DecisionTreeClassifier()}
            
            x_train,y_train,x_test,y_test=train_arr[:,:-1],train_arr[:,-1],test_arr[:,:-1],test_arr[:,-1]

            model_report:dict = BestModelGenerator(x_train,y_train,x_test,y_test,models=models_1)

            #now lets generate our best model
            max_score=max(sorted(model_report.values()))
            #the best model
            perf_model=list(model_report.keys())[list(model_report.values()).index(max_score)]
            logging.info(f"the best model is {perf_model}")

            # now lets initiatze that model
            model01=models_1[perf_model]
            y_pred=model01.predict(x_test)
            score=r2_score(y_test,y_pred)
            logging.info(f"Your model {perf_model} has a r2score of {score}")

            pickle_saver(self.transform.savingpath,model01)
            return score
        
        except Exception as e:
            raise ErrorHandling(e,sys)




