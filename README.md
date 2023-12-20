![chart](https://github.com/Bangkit-2023-Capstone-CH2-PS307/ML/assets/89390323/319bc646-2e56-4d16-90ad-4a5ae66100b0)
# Machine Learning Path

Nutrikita utilizing Tensorflow to develop it's two main feature, Food Reccomendation and Food Nutritions Detection


## 1. Food Reccomendation
### Our food reccomendation utilizing content based filtering. We used dataset collected from kaggle. 
![image](https://github.com/Bangkit-2023-Capstone-CH2-PS307/ML/assets/89390323/982041b1-2f50-430d-9e89-d518a21ea249)

## 2. Food Nutritient Recomendation 
### Dataset
The dataset for this feature are collected manually from google images. We collect 11 food category with the specific detail as below 
| Food Class       | Bread | Bubur | Cheese | Daging Cincang | Gambar agar agar | Kentang | Olahan Ikan | Susu | Telur | Wortel | Yogurt |            
|------------------|-------|-------|--------|-----------------|-------------------|---------|--------------|------|-------|--------|--------|
| Quantity         | 192   | 168   | 136    | 192             | 180               | 194     | 193          | 184  | 190   | 76     | 198    |

![chart](https://github.com/Bangkit-2023-Capstone-CH2-PS307/ML/assets/89390323/fce47716-145c-4d75-9b3a-f7b4459407c3)

The dataset example is shown like below : 

![distibution_per_class](https://github.com/Bangkit-2023-Capstone-CH2-PS307/ML/assets/89390323/df34d019-6efd-48a4-b34b-2672d47ce93e)

We trained our model using 3 pretrained model : **Resnet50**, **MobilenetV2**, and **InceptionV3**. We train using 20 epochs and datasplit of **80 : 20**.

| Model      | Accuracy |  | F1 |          
|------------------|-------|-------|
| InceptionV3         | 0.907   | 0.907   |
| Resnet50         | 0.863   | 0.863   |
| Resnet50         | 0.195   | 0.235   |

**MobileNetV2** outperform other pretrained model with an amazing score of 0.907 for both accuracy and F1 score. So this is the model that we weill use to classify our model.

After we got the model trained, we then connect the output classified image into a database contain information about the food that is predicted. Then the output will be the information gathered from the database.





