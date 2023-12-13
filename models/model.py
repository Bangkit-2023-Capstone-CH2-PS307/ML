import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2, InceptionV3, ResNet50
from tensorflow_addons.metrics import F1Score

class CustomModel :
    def __init__(self,input_shape=(224,224,3),pre_trained='None',num_classes=11):
        self.input_shape=input_shape
        self.pre_trained=pre_trained
        self.num_classes=num_classes
        self.model=self.create_model()
        self.f1_score= F1Score(num_classes=num_classes ,average='weighted', name='f1_score')
    def create_model(self):
        if self.pre_trained == 'MobileNetV2':
            base_model = MobileNetV2(input_shape=self.input_shape,
                                     include_top=False,
                                     weights='imagenet')
        elif self.pre_trained == 'InceptionV3':
            base_model = InceptionV3(input_shape=self.input_shape,
                                      include_top=False,
                                      weights='imagenet')
        elif self.pre_trained == 'ResNet50':
            base_model = ResNet50(input_shape=self.input_shape,
                                  include_top=False,
                                  weights='imagenet')
        else:
            raise ValueError("Unsupported pre-trained model. Choose from 'MobileNetV2', 'InceptionV3', or 'ResNet50'.")
        base_model.trainable=False
        x=tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
        x = tf.keras.layers.Dense(1024, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(self.num_classes, activation='softmax')(x)
        model = tf.keras.models.Model(inputs=base_model.input, outputs=x)
        return model
    def compile_model(self,model):
        
        self.model=model
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                           loss='categorical_crossentropy',metrics=['accuracy',self.f1_score])
        return self.model