import os
import json
import pandas as pd
import config

def extract_question_text(question_json: str) -> str:
    try:
        data = json.loads(question_json)
        if isinstance(data, dict):
            return data.get("text", "")
        elif isinstance(data, list):
            texts = []
            for item in data:
                if isinstance(item, dict):
                    t = item.get("text")
                    if t:
                        texts.append(t)
                elif isinstance(item, str):
                    texts.append(item)
            return " ".join(texts)
    except Exception:
        pass
    if isinstance(question_json, str) and not question_json.startswith(("{", "[")):
        return question_json.strip()
    return ""


def load_dataset(score_threshold: float) -> pd.DataFrame:
    """Loads dataset from BigQuery and filters it."""
    # Check if we should fall back to local dataset (for mock / offline validation)
    if not (os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") and os.environ.get("GOOGLE_CLOUD_PROJECT")):
        if os.path.exists(config.DATASET_PATH):
            print(f"Loading local dataset from {config.DATASET_PATH} (mock/local fallback)...")
            df = pd.read_csv(config.DATASET_PATH)
            if "question_text" not in df.columns:
                if "question" in df.columns:
                    df["question_text"] = df["question"].apply(extract_question_text)
                elif "ama_query" in df.columns:
                    df["question_text"] = df["ama_query"].apply(extract_question_text)
            low_score_df = df[
                (df["overall_score"] < score_threshold)
                & (df["question_text"].str.len() > 0)
            ].copy()
            print(f"Loaded {len(low_score_df)} queries from local dataset")
            return low_score_df

    print("Loading dataset from BQ...")
    from google.cloud import bigquery

    bq_client = bigquery.Client(project="ss-aiaa-calevntsub-uat-bc96")

    # Read the SQL query from the bq_query.sql file
    with open(config.BQ_QUERY_PATH, "r", encoding="utf-8") as f:
        query_template = f.read()

    query = query_template.format(score_threshold=score_threshold)

    df = bq_client.query(query).to_dataframe()
    df["question_text"] = df["question"].apply(extract_question_text)
    low_score_df = df[
        (df["overall_score"] < score_threshold)
        & (df["question_text"].str.len() > 0)
    ].copy()

    if low_score_df.empty:
        print("No rows found")
    print(f"Loaded {len(low_score_df)} queries")
    return low_score_df


def load_queries(input_queries_path: str) -> list[str]:
    print("Loading queries...")
    queries_df = pd.read_csv(input_queries_path)
    input_queries = (
        queries_df["query"]
        .dropna()
        .astype(str)
        .tolist()
    )
    print(f"Loaded {len(input_queries)} queries")
    return input_queries
