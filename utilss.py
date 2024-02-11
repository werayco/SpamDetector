import pickle as plk
def changer(file_path):
    with open(file_path,"rb") as file:
        obj=plk.load(file)
    return obj