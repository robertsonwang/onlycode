# RAG Mini-Pipeline Drill (NumPy/Sklearn)
Difficulty: Medium

Build a minimal retrieval pipeline over about five tiny text files using TF-IDF as your embedding model and K-means as your query engine.

Pipeline stages:
1. Ingest file contents
2. Chunk text
3. Embed chunks
4. Retrieve top-k chunks for a query
5. Stub the generation step

Use only:
- `numpy`
- `scikit-learn`

Implement the functions in `starter.py`:

```python
def ingest_files(file_map):
    # file_map: dict[str, str]
    # returns list[dict]

def chunk_documents(documents, chunk_size=40, overlap=10):
    # returns list[dict]

def fit_embedder(chunks):
    # returns (vectorizer, matrix)

def retrieve_top_k(query, vectorizer, matrix, chunks, k=3):
    # returns list[dict]

def generate_stub_answer(query, retrieved):
    # returns str

def run_pipeline(file_map, query, k=3, chunk_size=40, overlap=10):
    # returns {"retrieved": [...], "answer": str}
```

Notes:
- Keep output deterministic.
- For equal retrieval scores, break ties by original chunk order.
- Return scores as rounded floats (6 decimals).
- The generation step is a stub: synthesize a short answer string from retrieved context.
