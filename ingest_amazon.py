import kaggle
import requests
import csv
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

COLLECTION_NAME = "amazon"
DIM = 384
BATCH_SIZE = 16
DOWNLOAD_PATH = '.data/amazon'
CREATE_ENDPOINT = 'http://localhost:8000/collections'
UPSERT_ENDPOINT = 'http://localhost:8000/upsert?collection=amazon'
kaggle.api.authenticate()

# Download latest version
#kaggle.api.dataset_download_files("piyushjain16/amazon-product-data", path=DOWNLOAD_PATH, unzip=True)

def create_collection(name: str, dim: int, metric: str = "cosine"):
    payload = {"name": name, "dim": dim, "metric": metric}
    r = requests.post(CREATE_ENDPOINT, json=payload, timeout=30)
    if r.status_code == 200:
        print(f"Created collection '{name}' (dim={dim}, metric={metric}).")
    elif r.status_code == 400 and "exists" in r.text.lower():
        print(f"Collection '{name}' already exists; continuing.")
    else:
        raise RuntimeError(f"Create collection failed: {r.status_code} {r.text}")


def upsert_batch(points):
    if not points:
        return
    payload = {"points": points}
    r = requests.post(UPSERT_ENDPOINT, json=payload, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"Upsert failed: {r.status_code} {r.text}")

def generate_embeddings(csv_path):
    data = []
    create_collection(COLLECTION_NAME, DIM)
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            text = row["TITLE"]
            emb = model.encode(text, normalize_embeddings=True).tolist()
            metadata = {"category": row.get("BULLET_POINTS", ""), "description": row.get("DESCRIPTION", "")}
            data.append({"id": f"prod_{i}", "vector": emb, "metadata": metadata})
    
    for batch_start in range(0, len(data), BATCH_SIZE):
        batch = data[batch_start:min(len(data), batch_start+BATCH_SIZE)]
        upsert_batch(batch)
        print(f"Finished upserting batch of {len(batch)} into collection")

generate_embeddings(f"{DOWNLOAD_PATH}/dataset/train.csv")