from dataclasses import dataclass
import pandas as pd
from source.exception import ErrorHandling
import sys

class predictor:
    def __init__(self) -> None:
        pass
    def predictur(self):
        
class DataFrameGen:
    def __init__(self,message):
        self.message = message
    
    def Dataframe(self):
        try:
            data={"messsage":[self.message]}
            return pd.DataFrame(data=data)
        except Exception as e:
            raise ErrorHandling(e,sys)

