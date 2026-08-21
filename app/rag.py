import os
import pickle
import numpy as np
import pandas as pd
from pypdf import PdfReader
import faiss
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

class VectorStoreManager:
    """
    Handles FAISS Vector Index operations with support for local saving/loading
    and abstraction for future DB replacement.
    """
    def __init__(self, index_path="../vector_db"):
        self.index_path = index_path
        self.index = None
        self.metadata = []

    def build_index(self, items):
        """
        Takes a list of document/record dictionaries, builds embeddings,
        and saves them into the FAISS index.
        """
        if not items:
            return

        texts = [item["content"] for item in items]
        embeddings = EMBEDDING_MODEL.encode(texts, show_progress_bar=True)
        embeddings = np.array(embeddings).astype("float32")

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        self.metadata = items

        # Save index locally
        os.makedirs(self.index_path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(self.index_path, "faiss.index"))
        with open(os.path.join(self.index_path, "metadata.pkl"), "wb") as f:
            pickle.dump(self.metadata, f)
        print("Vector database built and saved locally.")

    def load_index(self):
        """Loads FAISS index and metadata from disk."""
        index_file = os.path.join(self.index_path, "faiss.index")
        metadata_file = os.path.join(self.index_path, "metadata.pkl")

        if os.path.exists(index_file) and os.path.exists(metadata_file):
            self.index = faiss.read_index(index_file)
            with open(metadata_file, "rb") as f:
                self.metadata = pickle.load(f)
            return True
        return False

    def search(self, query, top_k=5):
        """Searches the index and returns a list of matching items with similarity scores."""
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


def inspect_and_load_documentation(doc_dir="../knowledge_base/documentation"):
    documents = []
    if not os.path.exists(doc_dir):
        return documents
    for filename in os.listdir(doc_dir):
        if filename.endswith(".pdf"):
            filepath = os.path.join(doc_dir, filename)
            try:
                reader = PdfReader(filepath)
                for page_idx, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text and text.strip():
                        documents.append({
                            "source_type": "documentation",
                            "source_name": filename,
                            "content": text.strip(),
                            "metadata": {
                                "page_number": page_idx + 1,
                                "file_path": filepath
                            }
                        })
            except Exception as e:
                print(f"Failed to extract {filename}: {e}")
    return documents


def inspect_and_load_datasets(datasets_dir="../knowledge_base/datasets"):
    dataset_records = []
    if not os.path.exists(datasets_dir):
        return dataset_records
    for filename in os.listdir(datasets_dir):
        if filename.endswith(".csv"):
            filepath = os.path.join(datasets_dir, filename)
            try:
                df = pd.read_csv(filepath)
                for idx, row in df.iterrows():
                    row_dict = row.to_dict()
                    row_dict_clean = {k: v for k, v in row_dict.items() if pd.notna(v)}
                    items_str = ", ".join([f"{k}: {v}" for k, v in row_dict_clean.items()])
                    text_content = f"Record from {filename} -> {items_str}"
                    dataset_records.append({
                        "source_type": "dataset",
                        "source_name": filename,
                        "content": text_content,
                        "metadata": {
                            "row_index": idx,
                            "raw_data": row_dict_clean,
                            "file_path": filepath
                        }
                    })
            except Exception as e:
                print(f"Failed to read dataset {filename}: {e}")
    return dataset_records


def retrieve_evidence(query, top_k_docs=3, top_k_data=3):
    """
    Dual-source retriever. Queries the vector store and separates
    evidence from documentation and datasets.
    """
    store = VectorStoreManager()
    if store.index is None and not store.load_index():
        print("Vector database index not found. Building it now...")
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


if __name__ == "__main__":
    # Test building the vector store and query dual-source RAG
    print("=== Phase 4: Building and testing Vector Store ===")
    docs = inspect_and_load_documentation()
    datasets = inspect_and_load_datasets()
    all_items = docs + datasets
    print(f"Total chunks to index: {len(all_items)}")

    store = VectorStoreManager()
    store.build_index(all_items)

    # Test retrieval query
    query = "uncontrolled BP blood pressure and low medication adherence mpr"
    print(f"\nTesting Dual-Source RAG Query: '{query}'")
    doc_matches, data_matches = retrieve_evidence(query)

    print("\n--- Documentation Evidence Matches ---")
    for d in doc_matches:
        print(f"- Source: {d['source_name']}, Page: {d['metadata']['page_number']}, Score: {d['score']:.4f}")
        print(f"  Snippet: {d['content'][:150]}...")

    print("\n--- Dataset Evidence Matches ---")
    for d in data_matches:
        print(f"- Source: {d['source_name']}, Index: {d['metadata']['row_index']}, Score: {d['score']:.4f}")
        print(f"  Content: {d['content']}")
