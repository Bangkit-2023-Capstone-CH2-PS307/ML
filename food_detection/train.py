from models.dataloader import CustomDataLoader
from models.model import CustomModel
from tensorflow.keras.callbacks import ModelCheckpoint
from predict import Prediction
import pickle

def training(pre_trained):
    # Use a breakpoint in the code line below to debug your script.
    batch_size = 64
    path = r'venv/dataset/final_data'
    custom_dataloader = CustomDataLoader(path, batch_size)
    train_gen, val_gen = custom_dataloader.dataset
    custom_dataloader.save_class_indices(pre_trained)
    custom_dataloader.save_class_indices(f"{pre_trained}/class_indices_{pre_trained}_")
    pre_trained=pre_trained
    custom_model = CustomModel(pre_trained=pre_trained)
    checkpoint_filepath = f'results/{pre_trained}/model_{pre_trained}_.h5'
    checkpoint_callback = ModelCheckpoint(
        checkpoint_filepath,
        monitor='val_f1_score',
        save_best_only=True,
        mode='max',
        verbose=1,
        period=1
    )
    compiled_model = custom_model.compile_model(custom_model.model)

    # Define F1 score as a custom metric
    # Fit the model with the ModelCheckpoint and F1 score metric
    history = compiled_model.fit(
        train_gen,
        epochs=20,
        validation_data=val_gen,
        batch_size=batch_size,
        callbacks=[checkpoint_callback]  # Include F1 score in metrics
    )
    with open(f'results/{pre_trained}/history_{pre_trained}_.pkl','wb') as file :
            pickle.dump(history.history,file)
    return history

trained=training('ResNet50')
