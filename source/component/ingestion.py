from dataclasses import dataclass
from sklearn.model_selection import train_test_split
import os
import pandas as pd
from source.exception import ErrorHandling
import sys

@dataclass
class PathForIngestion:
    train_csv=os.path.join("spam.detector","train.csv")
    test_csv=os.path.join("spam.detector","test.csv")

class IngestionProcess:
    def __init__(self):
        self.initialize=PathForIngestion()

    def mail_data_split(self):
        try:
            mail_data=pd.read_csv("spam_data.csv")
            train_data,test_data=train_test_split(mail_data,random_state=32,test_size=0.25)
            train_data.to_csv(self.initialize.train_csv,skip_header=False,index=True)
            test_data.to_csv(self.initialize.test_csv,skip_header=False,index=True)

            return(self.initialize.train_csv,self.initialize.test_csv,mail_data)
        except Exception as e:
            raise ErrorHandling(e,sys)
        
if __name__=="__main__":
    ing_pro_obj=IngestionProcess()
    train_csv,test_csv=ing_pro_obj.mail_data_split()



