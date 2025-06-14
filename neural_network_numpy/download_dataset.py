import kagglehub
import shutil
import os

# Download latest version
path = kagglehub.dataset_download("oddrationale/mnist-in-csv")

print("Path to dataset files:", path)

# desired destination
target_path = "./mnist_dataset"

# Create destination if it doesn't exist
os.makedirs(target_path, exist_ok=True)

# Move all files to destination
for file_name in os.listdir(path):
    shutil.move(os.path.join(path, file_name), os.path.join(target_path, file_name))

print("Dataset moved to:", target_path)