import os

def _configure_vertex_ai() -> None:
    """Point the SDK at Vertex AI using the env vars KEF expects."""
    sa_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")

    if not sa_path or not os.path.isfile(sa_path):
        raise FileNotFoundError(
            f"Service account file not found: {sa_path!r}\n"
            "Set GOOGLE_APPLICATION_CREDENTIALS to the path of your SA JSON."
        )
    if not project:
        raise ValueError("GOOGLE_CLOUD_PROJECT is not set.")

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.abspath(sa_path)
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "true"
    os.environ["GOOGLE_CLOUD_PROJECT"] = project
    os.environ["GOOGLE_CLOUD_LOCATION"] = location
    print(f"Auth: Vertex AI  project={project}  location={location}")


def _configure_api_key() -> None:
    """Use a plain API key (Google AI Studio)."""
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_GENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "No credentials found.\n"
            "Either set GOOGLE_APPLICATION_CREDENTIALS + GOOGLE_CLOUD_PROJECT + "
            "GOOGLE_GENAI_USE_VERTEXAI=true  (Vertex AI)\n"
            "or set GOOGLE_API_KEY  (AI Studio)."
        )
    os.environ["GOOGLE_API_KEY"] = api_key
    print("Auth: API Key (Google AI Studio)")
