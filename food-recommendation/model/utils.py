import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

class preprocess :
    def __init__(self,model,scaler,n_return=10):
        self.model=model
        self.scaler=scaler
        self.n_return=n_return
    def predictor(self):
        transformer = FunctionTransformer(self.model.kneighbors,kw_args={'return_distance':False})
        pipeline = Pipeline([('std_scaler', self.scaler), ('NN', transformer)])
        params={'n_neighbors':self.n_return,'return_distance':False}
        pipeline.get_params()
        pipeline.set_params(NN__kw_args=params)
        return pipeline
# def scaling(dataframe):
#     scaler=StandardScaler()
#     prep_data=scaler.fit_transform(dataframe.iloc[:,6:15].to_numpy())
#     return prep_data,scaler
#
# def nn_predictor(prep_data):
#     neigh = NearestNeighbors(metric='cosine',algorithm='brute')
#     neigh.fit(prep_data)
#     return neigh
#
# def build_pipeline(neigh,scaler,params):
#     transformer = FunctionTransformer(neigh.kneighbors,kw_args=params)
#     pipeline=Pipeline([('std_scaler',scaler),('NN',transformer)])
#     return pipeline
#
# def extract_data(dataframe,ingredient_filter,max_nutritional_values):
#     extracted_data=dataframe.copy()
#     for column,maximum in zip(extracted_data.columns[6:15],max_nutritional_values):
#         extracted_data=extracted_data[extracted_data[column]<maximum]
#     if ingredient_filter!=None:
#         for ingredient in ingredient_filter:
#             extracted_data=extracted_data[extracted_data['RecipeIngredientParts'].str.contains(ingredient,regex=False)]
#     return extracted_data
#
# def apply_pipeline(pipeline,_input,extracted_data):
#     return extracted_data.iloc[pipeline.transform(_input)[0]]
#
# def recommand(dataframe,_input,max_nutritional_values,ingredient_filter=None,params={'return_distance':False}):
#     extracted_data=extract_data(dataframe,ingredient_filter,max_nutritional_values)
#     prep_data,scaler=scaling(extracted_data)
#     neigh=nn_predictor(prep_data)
#     pipeline=build_pipeline(neigh,scaler,params)
#     return apply_pipeline(pipeline,_input,extracted_data)
