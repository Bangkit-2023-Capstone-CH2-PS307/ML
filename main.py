# # This is a sample Python script.
#
# # Press Shift+F10 to execute it or replace it with your code.
# # Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
# from models.dataloader import CustomDataLoader
# from models.model import CustomModel
# from tensorflow.keras.callbacks import ModelCheckpoint
# from predict import Prediction
# def print_hi(name):
#     # Use a breakpoint in the code line below to debug your script.
#     # batch_size = 16
#     # path = r'venv/dataset/final_data'
#     # custom_dataloader = CustomDataLoader(path, batch_size)
#     # train_gen, val_gen = custom_dataloader.dataset
#     # pre_trained='MobileNetV2'
#     # custom_model = CustomModel(pre_trained=pre_trained)
#     # checkpoint_filepath = f'venv/results/{pre_trained}.h5'
#     # checkpoint_callback = ModelCheckpoint(
#     #     checkpoint_filepath,
#     #     monitor='val_f1_score',
#     #     save_best_only=True,
#     #     mode='max',
#     #     verbose=1,
#     #     period=1
#     # )
#     # compiled_model = custom_model.compile_model(custom_model.model)
#     #
#     # # Define F1 score as a custom metric
#     #
#     #
#     # # Fit the model with the ModelCheckpoint and F1 score metric
#     # history = compiled_model.fit(
#     #     train_gen,
#     #     epochs=10,
#     #     validation_data=val_gen,
#     #     batch_size=batch_size,
#     #     callbacks=[checkpoint_callback]  # Include F1 score in metrics
#     # )
#     image_path=r'venv/dataset/final_data/bread/Image_1.jpg'
#     # Example usage:
#     img_path = r'venv/dataset/final_data/bread/Image_1.jpg'  # Update with the path to your image
#     prediction_instance = Prediction(img_path, pre_trained='MobileNetV2')
#     result = prediction_instance.make_prediction()
#     print(result)
#     return result
# # Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#     print_hi('PyCharm')
#
# # See PyCharm help at https://www.jetbrains.com/help/pycharm/
