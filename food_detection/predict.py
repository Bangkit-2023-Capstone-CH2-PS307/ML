from tensorflow.keras.preprocessing import image
from models.model import CustomModel
import numpy as np

class Prediction:
    def __init__(self, img_path, target_size=(224, 224), pre_trained='MobileNetV2'):
        self.img_path = img_path
        self.target_size = target_size
        self.pre_trained = pre_trained
        self.pre_trained_path = f"venv/results/{pre_trained}.h5"
        self.model = self.load_model()

    def preprocess_image(self):
        img = image.load_img(self.img_path, target_size=self.target_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    def load_model(self):
        custom_model = CustomModel(pre_trained=self.pre_trained)
        compiled_model = custom_model.compile_model(custom_model.model)
        compiled_model.load_weights(self.pre_trained_path)
        return compiled_model

    def make_prediction(self):
        img_array = self.preprocess_image()
        preds = self.model.predict(img_array)
        return preds


