import kagglehub

# Download latest version
path = kagglehub.dataset_download("towardsentropy/oil-storage-tanks")

print("Path to dataset files:", path)
