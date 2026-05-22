import os

DATASET_DIR = os.getenv("DATASET_DIR", "./storage/datasets")

def list_datasets():
    if not os.path.exists(DATASET_DIR):
        return []
    return [f for f in os.listdir(DATASET_DIR) if os.path.isfile(os.path.join(DATASET_DIR, f))]

def delete_dataset(filename: str):
    path = os.path.join(DATASET_DIR, filename)
    if os.path.exists(path):
        os.remove(path)
        return True
    return False
