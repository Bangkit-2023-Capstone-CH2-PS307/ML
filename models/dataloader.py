
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pickle
class CustomDataLoader:
    def __init__(self, path, batch_size, target_size=(224, 224), class_mode='categorical'):
        self.path = path
        self.batch_size = batch_size
        self.target_size = target_size
        self.class_mode = class_mode
        self.dataset = self.create_dataset()


    def create_dataset(self):
        datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            fill_mode='nearest',
            horizontal_flip=True,
            vertical_flip=True,
            validation_split=0.2
        )

        train_generator = datagen.flow_from_directory(
            self.path,
            target_size=self.target_size,
            color_mode='rgb',
            batch_size=self.batch_size,
            class_mode=self.class_mode,
            subset='training',
            shuffle=True
        )

        valid_generator = datagen.flow_from_directory(
            self.path,
            target_size=self.target_size,
            batch_size=self.batch_size,
            class_mode=self.class_mode,
            shuffle=False
        )
        return train_generator, valid_generator

    def save_class_indices(self, filename):
            # Get class indices from the training generator
            class_indices = self.dataset[0].class_indices
            # Save the class indices using pickle
            with open(f"results/{filename}.pkl", 'wb') as file:
                pickle.dump(class_indices, file)



