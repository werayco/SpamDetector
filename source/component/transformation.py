from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
from source.exception import ErrorHandling
import os
import sys
from utils import target_to_num,removerr
from dataclasses import dataclass

@dataclass
class PathForPickle:
    preprocessor_pickle_file_path=os.path.join("pickle.files","processor.pkl")

class TransformationProcess:
    def __init__(self):
        self.PathForPicklee=PathForPickle()

    def process(self):
        try:
            pipeline_for_target=Pipeline(steps=[("onehot",target_to_num())])
            pipeline_for_feature=Pipeline(steps=[("remover",removerr())])
            preprocessor=ColumnTransformer([("feature-trans",pipeline_for_feature,["Message"]),("Target-trans",pipeline_for_target,["Category"])])
            return preprocessor
        except Exception as e:
            raise ErrorHandling(e,sys)
        
    def process_two(self,train_data,test_data):
        try:
            processor_obj=self.process()
            train_arr=processor_obj.fit_transform(train_data)
            test_arr=processor_obj.transform(test_arr)
            return(train_arr,test_arr)
        except Exception as e:
            raise ErrorHandling(e,sys)


