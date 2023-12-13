from model.reccomend import prediction
data=[170.9,   2.5,   1.3,   8. ,  29.8,  37.1,   3.6,  30.2,   3.2]
dataset_path='dataset/recipes_up.csv'
model_path='venv/model/neigh_model.joblib'
scaler_path='venv/model/scaler_model.joblib'
preds=prediction(dataset_path,model_path,data,scaler_path)
print(preds.predict())

