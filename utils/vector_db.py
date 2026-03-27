import os
import torch
import torch.nn.functional as F
import numpy as np
import faiss
from PIL import Image
from tqdm import tqdm
# from pymilvus import MilvusClient, DataType #TODO install milvus 

class VectorDBManager:
    def __init__(self, config, features_dict):
        self.config = config
        self.features_dict = features_dict
        
        # Mapping for image IDs and their corresponding indices in the vector database
        self.img_ids = list(features_dict.keys())
        self.id_to_index = {img_id: i for i, img_id in enumerate(self.img_ids)}
        self.index_to_id = {i: img_id for i, img_id in enumerate(self.img_ids)}
        
        # Convert features to a single numpy array for efficient indexing
        self.embeddings = np.vstack([features_dict[img_id].numpy() for img_id in self.img_ids]).astype('float32')
        
        # If config is a class instance, use . notation. If it's a dict, use dict access.
        # Support both just in case:
        vector_db_type = getattr(config, 'VECTOR_DB', config.get('VECTOR_DB', 'faiss')) if hasattr(config, 'get') or hasattr(config, 'VECTOR_DB') else config['VECTOR_DB']

        if vector_db_type == "faiss":
            self.db = self._init_faiss()
        elif vector_db_type == "milvus":
            self.db = self._init_milvus()
        else:
            raise ValueError(f"Unknown Vector DB: {vector_db_type}")

    def _init_faiss(self):
        metric = getattr(self.config, 'METRIC', self.config.get('METRIC', 'cosine')) if hasattr(self.config, 'get') or hasattr(self.config, 'METRIC') else self.config['METRIC']
        embed_dim = getattr(self.config, 'EMBED_DIM', self.config.get('EMBED_DIM', 512)) if hasattr(self.config, 'get') or hasattr(self.config, 'EMBED_DIM') else self.config['EMBED_DIM']

        print(f"[*] Starting FAISS... (Metric: {metric.upper()})")
        d = embed_dim
        
        if metric == "cosine":
            index = faiss.IndexFlatIP(d)
        else:
            index = faiss.IndexFlatL2(d)
            
        index.add(self.embeddings)
        print(f"[*] {index.ntotal} vector added.")
        return index

    def _init_milvus(self):
        metric = getattr(self.config, 'METRIC', self.config.get('METRIC', 'cosine')) if hasattr(self.config, 'get') or hasattr(self.config, 'METRIC') else self.config['METRIC']
        embed_dim = getattr(self.config, 'EMBED_DIM', self.config.get('EMBED_DIM', 512)) if hasattr(self.config, 'get') or hasattr(self.config, 'EMBED_DIM') else self.config['EMBED_DIM']

        print(f"[*] Starting Milvus Lite... (Metric: {metric.upper()})")
        client = MilvusClient("fashion_iq_local.db") 
        collection_name = "fashion_items"
        
        if client.has_collection(collection_name):
            client.drop_collection(collection_name)
            
        client.create_collection(
            collection_name=collection_name,
            dimension=embed_dim,
            metric_type="COSINE" if metric == "cosine" else "L2",
            auto_id=False
        )
        
        # Prepare data for insertion (Milvus expects a list of dicts with "id" and "vector" keys)
        data = [
            {"id": i, "vector": self.embeddings[i].tolist()} 
            for i in range(len(self.embeddings))
        ]
        
        print("[*] Loading vectors into Milvus... (This might take a while)...")
        client.insert(collection_name=collection_name, data=data)
        
        # Create index for faster search
        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="vector", 
            index_type="FLAT", 
            metric_type="COSINE" if metric == "cosine" else "L2"
        )
        client.create_index(collection_name=collection_name, index_params=index_params)
        client.load_collection(collection_name)
        
        self.collection_name = collection_name
        return client

    def search(self, query_vector, top_k=5):
        query_np = query_vector.cpu().detach().numpy().astype('float32')
        vector_db_type = getattr(self.config, 'VECTOR_DB', self.config.get('VECTOR_DB', 'faiss')) if hasattr(self.config, 'get') or hasattr(self.config, 'VECTOR_DB') else self.config['VECTOR_DB']
        metric = getattr(self.config, 'METRIC', self.config.get('METRIC', 'cosine')) if hasattr(self.config, 'get') or hasattr(self.config, 'METRIC') else self.config['METRIC']
        
        if vector_db_type == "faiss":
            distances, indices = self.db.search(query_np, top_k)
            results = [(self.index_to_id[idx], dist) for idx, dist in zip(indices[0], distances[0])]
            
        elif vector_db_type == "milvus":
            res = self.db.search(
                collection_name=self.collection_name,
                data=query_np.tolist(),
                limit=top_k,
                search_params={"metric_type": "COSINE" if metric == "cosine" else "L2"}
            )

            results = [(self.index_to_id[hit["id"]], hit["distance"]) for hit in res[0]]
            
        return results

    @staticmethod
    def build_custom_db(folder_path, clip_model, preprocess, device):
        """Extract features from a custom folder of images and return a features dictionary."""
        print(f"\n[*] Proccessing images in {folder_path} to build vector database...")
        features_dict = {}
        valid_exts = ('.png', '.jpg', '.jpeg', '.webp')
        
        for filename in tqdm(os.listdir(folder_path), desc="Extracting features..."):
            if filename.lower().endswith(valid_exts):
                img_path = os.path.join(folder_path, filename)
                try:
                    # Preprocess the image and extract features
                    img_tensor = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        feat = clip_model.encode_image(img_tensor).float()
                        feat = F.normalize(feat, dim=-1).squeeze(0).cpu() # L2 Norm
                        
                    features_dict[filename] = feat # Use filename as ID (or you can choose a different scheme)
                except Exception as e:
                    print(f"[!] Error: {filename} could not be read. Details: {e}")
                    
        print(f"[*] Total {len(features_dict)} images processed and added to the database.")
        return features_dict
