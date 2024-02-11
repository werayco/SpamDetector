import logging
from datetime import datetime
import os


name_of_file=f"{datetime.now().strftime('%Y_%M_%D_%h_%m_%s')}.log"
directory_of_the_file=os.join.path("SpamDetectorPro",name_of_file)
os.makedirs(directory_of_the_file,exist_ok=True)

logging.basicConfig(
    filename=os.join.path(directory_of_the_file,name_of_file),
    level=logging.INFO,
    format="[ %(asctime)s ] %(lineo)d %(name)s - %(levelname)s - %(message)s")