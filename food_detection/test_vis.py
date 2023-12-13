from models.visualization import ClassDistributionVisualizer
from models.dataloader import CustomDataLoader

batch_size = 16
path = r'venv/dataset/final_data'
custom_dataloader = CustomDataLoader(path, batch_size)
vis=ClassDistributionVisualizer(custom_dataloader.dataset[0],"venv/results")
vis.display_one_image_per_class()
vis.visualize_and_save()

