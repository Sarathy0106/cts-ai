import os
import pickle
import numpy as np
import pandas as pd
from pypdf import PdfReader
import faiss
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")


class VectorStoreManager:
    def __init__(self, index_path="../vector_db"):
        if index_path == "../vector_db":
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.index_path = os.path.abspath(os.path.join(current_dir, "..", "vector_db"))
        else:
            self.index_path = index_path
        self.index = None
        self.metadata = []

    def build_index(self, items):
        if not items:
            return
        texts = [item["content"] for item in items]
        embeddings = EMBEDDING_MODEL.encode(texts, show_progress_bar=True)
        embeddings = np.array(embeddings).astype("float32")
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        self.metadata = items
        os.makedirs(self.index_path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(self.index_path, "faiss.index"))
        with open(os.path.join(self.index_path, "metadata.pkl"), "wb") as f:
            pickle.dump(self.metadata, f)
        print("Impact Agent vector database built and saved.")

    def load_index(self):
        index_file = os.path.join(self.index_path, "faiss.index")
        metadata_file = os.path.join(self.index_path, "metadata.pkl")
        if os.path.exists(index_file) and os.path.exists(metadata_file):
            self.index = faiss.read_index(index_file)
            with open(metadata_file, "rb") as f:
                self.metadata = pickle.load(f)
            return True
        return False

    def search(self, query, top_k=5):
        if self.index is None:
            if not self.load_index():
                raise RuntimeError("Vector database is empty. Build the index first.")
        query_vector = EMBEDDING_MODEL.encode([query]).astype("float32")
        distances, indices = self.index.search(query_vector, top_k)
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(self.metadata):
                item = self.metadata[idx].copy()
                item["score"] = float(distances[0][i])
                results.append(item)
        return results


def _resolve_kb_path(relative_path):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_dir, "..", relative_path))


def inspect_and_load_datasets(dataset_dir=None):
    if dataset_dir is None:
        dataset_dir = _resolve_kb_path("knowledge_base/datasets")
    items = []
    if not os.path.exists(dataset_dir):
        return items
    for filename in os.listdir(dataset_dir):
        if filename.endswith(".csv"):
            filepath = os.path.join(dataset_dir, filename)
            try:
                df = pd.read_csv(filepath, dtype=str).fillna("")
                for idx, row in df.iterrows():
                    row_dict = row.to_dict()
                    content_parts = [f"{k}: {v}" for k, v in row_dict.items() if v.strip()]
                    content = f"Record from {filename} -> " + ", ".join(content_parts)
                    items.append({
                        "source_type": "dataset",
                        "source_name": filename,
                        "content": content,
                        "metadata": {"row_index": idx, "raw_data": row_dict, "file_path": filepath}
                    })
            except Exception as e:
                print(f"[Impact RAG] Error loading {filename}: {e}")
    return items


def inspect_and_load_documentation(doc_dir=None):
    documents = []
    if doc_dir is None:
        doc_dir = _resolve_kb_path("knowledge_base/documentation")
    if not os.path.exists(doc_dir):
        return documents
    for filename in os.listdir(doc_dir):
        if filename.endswith(".pdf"):
            filepath = os.path.join(doc_dir, filename)
            try:
                reader = PdfReader(filepath)
                for page_num, page in enumerate(reader.pages, start=1):
                    text = page.extract_text()
                    if text and text.strip():
                        documents.append({
                            "source_type": "documentation",
                            "source_name": filename,
                            "content": text.strip(),
                            "metadata": {"page_number": page_num, "file_path": filepath}
                        })
            except Exception as e:
                print(f"[Impact RAG] Error loading PDF {filename}: {e}")
    return documents


def retrieve_evidence(query, top_k_docs=3, top_k_data=3):
    store = VectorStoreManager()
    if store.index is None and not store.load_index():
        print("Impact Agent: Vector database not found. Building now...")
        docs = inspect_and_load_documentation()
        datasets = inspect_and_load_datasets()
        all_items = docs + datasets
        store.build_index(all_items)

    all_matches = store.search(query, top_k=top_k_docs + top_k_data)

    documentation_evidence = []
    dataset_evidence = []
    for match in all_matches:
        if match["source_type"] == "documentation" and len(documentation_evidence) < top_k_docs:
            documentation_evidence.append(match)
        elif match["source_type"] == "dataset" and len(dataset_evidence) < top_k_data:
            dataset_evidence.append(match)

    return documentation_evidence, dataset_evidence
