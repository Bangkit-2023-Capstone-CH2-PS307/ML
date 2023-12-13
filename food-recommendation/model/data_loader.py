import pandas as pd
path='https://storage.googleapis.com/nutrikita-bucket/dataset%20(2).csv'
def load_data() :
    if path != None :
        data=pd.read_csv(path,compression='gzip')
    return data
