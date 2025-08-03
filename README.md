# Project: MTA Customer Feedback Analysis

This repository contains a pipeline for processing, exploring, and modeling customer feedback data from the MTA. 
A reduced form of the MTA-Dataset (about 223.000) is found at data/MTA_Customer_Feedback_Data.csv
It includes:

* **Configuration** constants (`config.py`)
* **Data loading & cleaning** (`data.py`, `preprocessing.py`)
* **Vectorization** methods (`vectorization.py`) including TF‑IDF, Word2Vec, and transformer embeddings
* ** Full Training-Programme including topic modeling & clustering** (`main.py`) with LDA, NMF, KMeans, GMM, and BERTopic


## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/AndiPfluegl/mta-feedback-analysis.git
   cd mta-feedback-analysis
   ```
2. Create and activate a Conda environment:

   ```bash
   conda env create -f environment.yml
   conda activate NLP
   ```

   *or* using `venv`:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate   # on Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

## Usage

1. **Preprocess & clean data**

   ```bash
   python main.py --step preprocess
   ```
2. **Run hyperparameter sweep & plotting**

   ```bash
   python main.py --step sweep
   ```
3. **Train final models and extract topics**

   ```bash
   python main.py --step train
   ```
4. **Save artifacts**

   ```bash
   python main.py --step save
   ```

> You can also run `main.py` without flags to execute the full pipeline in sequence.

## File structure

```
├── cache/                # Cached cleaned data
├── models/               # Saved models and vectorizers
├── data.py               # Data loading and quality reports
├── preprocessing.py      # Text cleaning and language filtering
├── vectorization.py      # TF‑IDF, Word2Vec, transformer embeddings
├── config.py             # Path and parameter constants
├── main.py               # Main workflow: sweep, train, evaluate
├── environment.yml       # Conda environment specification
```


