import numpy as np
from google import genai
from google.genai import types
import config

def get_embeddings(client: genai.Client, texts: list[str]) -> np.ndarray:
    """Call the embedding model and return a (N, DIM) float32 array."""
    if client is None:
        # Mock mode: return pseudo-random unit-normalized embeddings
        print(f"Mocking embeddings for {len(texts)} texts...")
        # Seed based on the text hash to keep it deterministic for the same run
        rng = np.random.default_rng(hash(texts[0]) & 0xffffffff if texts else None)
        vectors = rng.standard_normal((len(texts), config.OUTPUT_DIM)).astype(np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.where(norms > 0, norms, 1.0)

    result = client.models.embed_content(
        model=config.MODEL,
        contents=texts,
        config=types.EmbedContentConfig(
            task_type=config.TASK_TYPE,
            output_dimensionality=config.OUTPUT_DIM,
        ),
    )
    vectors = [e.values for e in (result.embeddings or [])]
    return np.array(vectors, dtype=np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    dot = float(np.dot(a, b))
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    return dot / norm if norm > 0 else 0.0


def generate_embeddings_and_matrix(client: genai.Client, input_queries: list[str], question_list: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    print("Generating embeddings...")
    query_embeddings = get_embeddings(client, input_queries)
    question_embeddings = get_embeddings(client, question_list)
    print(f"Embedding matrix shape: {question_embeddings.shape}")

    # Normalize
    query_embeddings = np.asarray(query_embeddings, dtype=np.float32)
    question_embeddings = np.asarray(question_embeddings, dtype=np.float32)

    similarity_matrix = query_embeddings @ question_embeddings.T
    print(similarity_matrix.shape)
    return query_embeddings, question_embeddings, similarity_matrix
