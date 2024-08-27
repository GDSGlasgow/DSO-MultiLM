from huggingface_hub import snapshot_download
import os
import zipfile
import torch
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
from tqdm import tqdm
import pickle
import numpy as np
import pandas as pd
import faiss

# Step 1: Download dataset from Hugging Face Hub
snapshot_download(repo_id="osv5m/osv5m", local_dir="datasets/osv5m", repo_type='dataset')

# Step 2: Unzip any .zip files found in the downloaded directory
for root, dirs, files in os.walk("datasets/osv5m"):
    for file in files:
        if file.endswith(".zip"):
            zip_path = os.path.join(root, file)
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(root)
                os.remove(zip_path)
            except zipfile.BadZipFile:
                print(f"Warning: Failed to unzip file {zip_path}")

# Step 3: Load model and processor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CLIPModel.from_pretrained("geolocal/StreetCLIP").eval().to(device)
processor = CLIPProcessor.from_pretrained("geolocal/StreetCLIP")


# Function to get image embeddings
def get_image_embedding(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            embedding = model.get_image_features(**inputs).cpu().numpy()
        return embedding
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None


# Step 4: Read and combine train.csv and test.csv
train_csv_path = os.path.join("datasets", "osv5m", "train.csv")
test_csv_path = os.path.join("datasets", "osv5m", "test.csv")

df_train = pd.read_csv(train_csv_path)
df_test = pd.read_csv(test_csv_path)

# Combine train and test DataFrames
df_combined = pd.concat([df_train, df_test], ignore_index=True)

# Step 5: Initialize variables for processing
image_column = 'id'
latitude_column = 'latitude'
longitude_column = 'longitude'
embeddings_data = {}
skipped_files = []

# Define paths and ensure folders exist
image_base_path_train = os.path.join("datasets", "osv5m", "images", "train")
image_base_path_test = os.path.join("datasets", "osv5m", "images", "test")

# Assuming both train and test image folders have subfolders
train_folders = os.listdir(image_base_path_train)
test_folders = os.listdir(image_base_path_test)

# Step 6: Process images and generate embeddings
for idx, row in tqdm(df_combined.iterrows(), total=df_combined.shape[0], desc="Processing images"):
    image_found = False

    # Check in training image folders first
    for folder in train_folders:
        image_path = os.path.join(image_base_path_train, folder, str(row[image_column]) + '.jpg')
        if os.path.isfile(image_path):
            latitude = row[latitude_column]
            longitude = row[longitude_column]
            embedding = get_image_embedding(image_path)
            if embedding is not None:
                embeddings_data[str(row[image_column])] = {
                    'embedding': embedding,
                    'location': (latitude, longitude)
                }
            image_found = True
            break

    # If not found, check in test image folders
    if not image_found:
        for folder in test_folders:
            image_path = os.path.join(image_base_path_test, folder, str(row[image_column]) + '.jpg')
            if os.path.isfile(image_path):
                latitude = row[latitude_column]
                longitude = row[longitude_column]
                embedding = get_image_embedding(image_path)
                if embedding is not None:
                    embeddings_data[str(row[image_column])] = {
                        'embedding': embedding,
                        'location': (latitude, longitude)
                    }
                image_found = True
                break

    if not image_found:
        skipped_files.append(str(row[image_column]))

# Step 7: Save embeddings to pickle file
output_dir = 'outputs'
os.makedirs(output_dir, exist_ok=True)
train_pkl_path = os.path.join(output_dir, 'embeddings.pkl')
with open(train_pkl_path, 'wb') as f:
    pickle.dump(embeddings_data, f)

# Step 8: Create FAISS index and save it
embedding_matrix = np.vstack([data['embedding'] for data in embeddings_data.values()])
index = faiss.IndexFlatL2(embedding_matrix.shape[1])
index.add(embedding_matrix)
faiss.write_index(index, os.path.join(output_dir, 'embeddings.bin'))

print(f"Processing completed. Embeddings saved to {train_pkl_path} and FAISS index saved to {output_dir}.")