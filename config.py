import os

# ---------------------------------------------------------------------------
# Directories and Paths Configuration
# ---------------------------------------------------------------------------

INPUT_DIR = "input"
OUTPUT_DIR = "output"

DATASET_PATH = os.path.join(INPUT_DIR, "bquxjob_da3d2902c_19eb205ea6da.csv")
INPUT_QUERIES_PATH = os.path.join(INPUT_DIR, "master_questions.csv")
BQ_QUERY_PATH = os.path.join(INPUT_DIR, "bq_query.sql")

OUTPUT_CSV = os.path.join(OUTPUT_DIR, "semantic_search_results.csv")
OUTPUT_FIG = os.path.join(OUTPUT_DIR, "semantic_search_heatmap.png")
EVALUATION_CSV = os.path.join(OUTPUT_DIR, "evaluation_analysis.csv")

# ---------------------------------------------------------------------------
# Semantic Search Configuration
# ---------------------------------------------------------------------------

TOP_K = 10                                           # top_k similarities
SCORE_THRESHOLD = 0.7
SIMILARITY_THRESHOLD = 0.8
INPUT_QUERY = "Member plan"                         # your query for semantic search
MODEL = "gemini-embedding-001"                       # same default as KEF's AppSettings
TASK_TYPE = "SEMANTIC_SIMILARITY"
OUTPUT_DIM = 768                                     # same default as KEF's AppSettings.EMBEDDING_DIMENSION
PRINT_N_RESULTS = 10

# ---------------------------------------------------------------------------
# Response Analysis (Judge Agent) Configuration
# ---------------------------------------------------------------------------

JUDGE_MODEL = "gemini-2.5-flash"
MAX_WORKERS = 10                                     # number of concurrent API requests for evaluation

# Auto-detect dataset if the default one is missing in the input directory
if not os.path.exists(DATASET_PATH):
    import glob
    _matches = glob.glob(os.path.join(INPUT_DIR, "bquxjob_*.csv"))
    if _matches:
        DATASET_PATH = _matches[0]
