import os
import shutil

OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./outputs")

def list_adapters():
    if not os.path.exists(OUTPUT_DIR):
        return []
    return [d for d in os.listdir(OUTPUT_DIR) if os.path.isdir(os.path.join(OUTPUT_DIR, d))]

def delete_adapter(job_id: str):
    path = os.path.join(OUTPUT_DIR, job_id)
    if os.path.exists(path):
        shutil.rmtree(path)
        return True
    return False
