import os
import sys
from google import genai

import config
from utils.auth import _configure_vertex_ai, _configure_api_key
from utils.embeddings import generate_embeddings_and_matrix
from utils.data_loader import load_dataset, load_queries
from utils.visualization import build_and_save_results, plot_and_save_heatmap
from utils.analysis import run_analysis_pipeline

def main() -> None:
    # Choose auth strategy
    has_gcp = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") and os.environ.get("GOOGLE_CLOUD_PROJECT")
    has_api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_GENAI_API_KEY")
    
    if not has_gcp and not has_api_key:
        print("WARNING: No Google AI or Vertex AI credentials found in environment. Running in MOCK mode.")
        client = None
    else:
        use_vertex = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in ("1", "true")
        if use_vertex:
            _configure_vertex_ai()
        else:
            _configure_api_key()

        # Build client (picks up env vars set above)
        client = genai.Client()

    print(f"\nModel : {config.MODEL}")
    print(f"Dim   : {config.OUTPUT_DIM}")

    low_score_df = load_dataset(config.SCORE_THRESHOLD)
    input_queries = load_queries(config.INPUT_QUERIES_PATH)

    question_list = low_score_df["question_text"].tolist()
    query_embeddings, question_embeddings, similarity_matrix = generate_embeddings_and_matrix(client, input_queries, question_list)

    build_and_save_results(similarity_matrix, input_queries, low_score_df, config.TOP_K, config.OUTPUT_CSV)
    plot_and_save_heatmap(similarity_matrix, input_queries, low_score_df, config.OUTPUT_FIG)

    # Call the response evaluation analysis agent (judge agent)
    run_analysis_pipeline(client, similarity_matrix, input_queries, low_score_df)

    print("Task completed")


if __name__ == "__main__":
    main()
