# BigQuery Semantic Search & LLM Judge Pipeline

This project implements a modular, concurrent semantic search and LLM-based response evaluation pipeline. It matches user queries against a dataset (loaded from BigQuery or a local CSV file), visualizes similarities with a heatmap, and uses Gemini to audit the quality of assistant responses.

---

## 🌟 Key Features

1. **Semantic Search**: Uses `gemini-embedding-001` to calculate cosine similarities between master queries and real conversation logs.
2. **LLM Judge Agent**: Employs `gemini-2.5-flash` to evaluate the assistant's answers against the user's questions, providing a `judge_score` (0.0 to 1.0) and detailed reasoning.
3. **De-duplicated Evaluations**: Grouping matches by `task_id` allows the pipeline to query Gemini exactly once per unique QA pair. This prevents redundant LLM calls, saves API costs, and speeds up execution.
4. **Discrepancy Analysis**: Automatically calculates `score_diff = abs(overall_score - judge_score)` and sorts the results in descending order to quickly highlight where the human/system score differs most from the judge agent's score.
5. **Parallel Processing**: Employs a concurrent `ThreadPoolExecutor` for API calls to scale efficiently for 500+ items.
6. **Automatic Mock Fallback**: Dynamically detects missing API credentials and runs in an offline `MOCK` mode for testing.

---

## 📁 Directory Structure

```text
c:\ScudHome\playground\
├── input/
│   ├── bquxjob_da3d2902c_19eb205ea6da.csv  # Conversation dataset
│   ├── master_questions.csv                # List of queries to match
│   └── bq_query.sql                        # BigQuery SQL template
├── output/
│   ├── semantic_search_results.csv         # Top-K matching results
│   ├── semantic_search_heatmap.png         # Heatmap of similarity scores
│   └── evaluation_analysis.csv             # Judge evaluations sorted by score difference
├── utils/
│   ├── auth.py                             # Google Cloud & API credentials setup
│   ├── data_loader.py                      # JSON parser and BigQuery/CSV loader
│   ├── embeddings.py                       # Emdedding generation and cosine math
│   ├── visualization.py                    # Heatmap plotting and CSV generation
│   └── analysis.py                         # De-duplicated parallel evaluation pipeline
├── config.py                               # Central configuration variables
├── requirements.txt                        # Project dependencies
└── semantic_search.py                      # Main entrypoint script
```

---

## 🚀 Getting Started

### 1. Set environment variables (PowerShell)

To run the pipeline with live APIs and Vertex AI, set the credentials:

```powershell
$env:GOOGLE_APPLICATION_CREDENTIALS = "SA_Creds_uat.json"
$env:GOOGLE_CLOUD_PROJECT           = "ss-aiaa-calevntsub-uat-bc96"
$env:GOOGLE_CLOUD_LOCATION          = "us-central1"
$env:GOOGLE_GENAI_USE_VERTEXAI       = "true"
```

*Note: If no credentials are found in the environment, the script will automatically fallback to offline **MOCK mode**.*

### 2. Dependencies installation

Ensure Python is installed, then run:

```bash
pip install -r requirements.txt
```

### 3. Application Default Credentials (ADC)
For local development, follow the Google Cloud instruction to set up ADC:
👉 [Google Cloud ADC Documentation](https://docs.cloud.google.com/docs/authentication/set-up-adc-local-dev-environment)

---

## 💻 Usage

### ⚙️ Configure the Pipeline
* **`config.py`**: Open and adjust the thresholds (`SIMILARITY_THRESHOLD = 0.8`), worker count (`MAX_WORKERS = 10`), and target models if needed.
* **`input/bq_query.sql`**: Modify the SQL query to adjust how the raw logs are selected from BigQuery.

### 🏃 Run the Pipeline
Run the main script using standard Python or `uv`:

```bash
python semantic_search.py
# OR
uv run python semantic_search.py
```

The script will:
1. Load the dataset (from BQ if connected, or fallback to the local CSV in `input/`).
2. Generate embeddings and find matches.
3. Save results to `output/semantic_search_results.csv` and generate the heatmap `output/semantic_search_heatmap.png`.
4. Run de-duplicated judge evaluations in parallel and output the final audit spreadsheet `output/evaluation_analysis.csv` sorted descending by `score_diff`.

---

## 📊 Understanding the Output

### Heatmap (`semantic_search_heatmap.png`)
* **X-Axis**: Matched questions from the dataset (truncated to 50 characters).
* **Y-Axis**: Master questions.
* **Interpretation**: Bright cells represent highly similar semantic pairs. Fully dark rows highlight master questions that lack representation in the dataset. Vertical bright columns indicate highly generic database questions that match multiple queries.

### Evaluation CSV (`evaluation_analysis.csv`)
This file is sorted descending by the absolute difference between `overall_score` (human/system score) and `judge_score` (Gemini evaluation). Focus on the top rows to investigate where the agent assistant behaved unexpectedly or where the original scoring system was incorrect.
