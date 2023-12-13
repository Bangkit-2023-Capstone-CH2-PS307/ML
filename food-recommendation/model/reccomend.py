from model.utils import preprocess
from joblib import dump, load
import pandas as pd
import numpy as np
class prediction :
    def __init__(self,dataset_path,model_path,data,scaler_path,n_return=10):
        self.model_path=load(model_path)
        self.scaler_path=load(scaler_path)
        self.data = np.array([data])
        self.n_return=n_return
        self.dataset_path=dataset_path
        self.dataset=self.extract_data()
    def extract_data(self):
        dataset=pd.read_csv(self.dataset_path)
        return dataset
    def predict(self):
        preprocessing=preprocess(self.model_path,self.scaler_path,self.n_return)
        pipeline=preprocessing.predictor()
        return self.dataset.iloc[pipeline.transform(self.data)[0]]
        
        

