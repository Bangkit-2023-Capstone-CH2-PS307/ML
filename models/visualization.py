import os
import matplotlib.pyplot as plt
import numpy as np
import random
from tensorflow.keras.preprocessing import image

class ClassDistributionVisualizer:
    def __init__(self, generator, save_path=None):
        self.generator = generator
        self.save_path = save_path

    def visualize_and_save(self):
        class_names = sorted(self.generator.class_indices.keys())

        # Get class counts from the generator
        class_counts = self.generator.classes
        unique_classes, counts = np.unique(class_counts, return_counts=True)

        # Generate random colors for the plots
        bar_colors = [plt.cm.Paired(i / len(class_names)) for i in range(len(class_names))]
        pie_colors = [plt.cm.Paired(i / len(class_names)) for i in range(len(class_names))]

        # Plot the distribution using a colorful bar plot
        plt.figure(figsize=(15, 6))
        plt.subplot(1, 2, 1)
        plt.bar(unique_classes, counts, color=bar_colors)
        plt.xticks(unique_classes, class_names, rotation=45, ha='right')
        plt.xlabel('Classes')
        plt.ylabel('Count')
        plt.title('Class Distribution (Bar Plot)')

        # Plot the distribution using a colorful pie chart
        plt.subplot(1, 2, 2)
        plt.pie(counts, labels=class_names, autopct='%1.1f%%', colors=pie_colors, startangle=140)
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.title('Class Distribution (Pie Chart)')

        plt.tight_layout()

        # Save the figure if a save_path is provided
        if self.save_path:
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            plt.savefig(f"{self.save_path}/chart.jpg")
            print(f"Figure saved at {self.save_path}")

        # Show the plots
        plt.show()

    def display_one_image_per_class(self):
        # Display one image per class
        class_names = sorted(self.generator.class_indices.keys())
        plt.figure(figsize=(15, 12))
        for i, class_name in enumerate(class_names, 1):
            # Get all filenames belonging to the current class
            filenames_for_class = [filename for filename in self.generator.filenames if class_name in filename]

            # Randomly select one filename
            selected_filename = random.choice(filenames_for_class)

            # Construct the full path to the image
            image_path = os.path.join(self.generator.directory, selected_filename)

            # Load and display the image
            img = image.load_img(image_path, target_size=(224, 224))
            plt.subplot(3, 4, i)  # Assuming 11 classes, adjust if needed
            plt.imshow(img)
            plt.title(class_name)
            plt.axis('off')

        plt.suptitle('Dataset Example', fontsize=16)
        if self.save_path:
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            plt.savefig(f"{self.save_path}/distibution_per_class.jpg")
            print(f"Figure saved at {self.save_path}")

            # Show the plots

        plt.show()