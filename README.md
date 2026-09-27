# Zepto Data & AI Platform - Capstone Project

One connected platform with three modules:
- **data_pipeline**: scrapes, cleans, converts, and stores book catalog data in a normalized SQLite database, queried with SQL and pandas.
- **analytics**: profiles, cleans, visualizes, and models the Titanic dataset end-to-end (EDA + classification + regression).
- **support_assistant**: a RAG-based GenAI support assistant answering Zepto policy questions, built with LangGraph, ChromaDB, and FastAPI.

## Setup

Each module has its own `requirements.txt`. Install per module before running it:

```bash
cd data_pipeline && pip install -r requirements.txt
cd ../analytics && pip install -r requirements.txt
cd ../support_assistant && pip install -r requirements.txt
```

## How to run each module

### data_pipeline
Run the scraping/cleaning/database notebook or script inside `data_pipeline/`. This scrapes books.toscrape.com, cleans and converts prices (1 GBP = 105.50 INR fixed rate), and loads the data into a normalized SQLite database with categories/books tables. SQL queries and their outputs are included in the notebook.

### analytics
Run `01_eda.ipynb` first (loads the Titanic dataset via `sns.load_dataset('titanic')`, profiles it, cleans it, saves `titanic.csv`), then `02_modeling.ipynb` (reads the same `titanic.csv`, builds the classification/regression pipeline, evaluates models, and saves the final pipeline with `joblib`).

### support_assistant
```bash
cd support_assistant
python ingest.py          # embeds the 8 policy docs into ChromaDB
uvicorn main:app --host 0.0.0.0 --port 7860
```
Or with Docker:
```bash
docker build -t zepto-support-assistant .
docker run -p 7860:7860 zepto-support-assistant
```
The API is then available at `POST http://localhost:7860/ask`. See `support_assistant/README.md` for the full architecture description and example API calls.

## Design decisions summary

- **data_pipeline**: books.toscrape.com was scraped instead of a live grocery site since it explicitly permits scraping practice; a fixed conversion rate (not a live FX API) was used per the assignment's baseline requirement.
- **analytics**: the dataset is loaded once via `sns.load_dataset('titanic')` and cached as `titanic.csv`; all later steps (EDA, modeling, tuning) reuse this single cleaned copy to avoid redundant loads or inconsistent cleaning.
- **support_assistant**: runs in a fully offline, deterministic `MOCK_LLM=1` mode by default (no API key required) using a keyword-based intent router and ChromaDB retrieval, with an optional real-LLM extension gated behind `MOCK_LLM=0`.

## Git workflow

This repository's history includes feature branches created, committed to multiple times, and merged back into `main` (see `git log --graph --all`).
