import numpy as np
import pandas as pd

def build_and_save_results(similarity_matrix: np.ndarray, input_queries: list[str], low_score_df: pd.DataFrame, top_k: int, output_csv: str) -> None:
    print("Building results...")
    results = []

    for query_idx, query in enumerate(input_queries):
        similarities = similarity_matrix[query_idx]

        top_indicies = np.argsort(similarities)[::-1][:top_k]

        for rank, idx in enumerate(top_indicies, start=1):
            results.append({
                "input_query": query,
                "rank": rank,
                "similarity": float(similarities[idx]),
                "task_id": low_score_df.iloc[idx]["task_id"],
                "overall_score": low_score_df.iloc[idx]["overall_score"],
                "matched_question": low_score_df.iloc[idx]["question_text"],
                "agent_response": low_score_df.iloc[idx]["agent_response"],
            })

    result_df = pd.DataFrame(results)

    print(f"Saving results in {output_csv}...")
    result_df.to_csv(
        output_csv,
        index=False
    )


def plot_and_save_heatmap(similarity_matrix: np.ndarray, input_queries: list[str], low_score_df: pd.DataFrame, output_fig: str) -> None:
    print(f"Saving plot in {output_fig}...")
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    question_labels = (
        low_score_df["question_text"]
        .str.replace("\n", " ", regex=False)
        .str.slice(0, 50)
    )

    heatmap_df = pd.DataFrame(
        similarity_matrix,
        index=input_queries,
        columns=question_labels
    )

    plt.figure(figsize=(14, max(4, len(input_queries) * 0.6)))
    sns.heatmap(
        heatmap_df,
        annot=True,
        fmt=".2f",
        cmap="viridis",
    )
    plt.xlabel("Matched task_id")
    plt.ylabel("Input query")
    plt.title("Semantic similarity heatmap")
    plt.tight_layout()

    plt.savefig(
        output_fig,
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()
