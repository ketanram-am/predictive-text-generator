Predictive Text Generator

This project builds a predictive text generator that suggests the next most probable word as you type. It uses a classic N-gram language model trained on multiple NLTK corpora plus a local dataset, and it ships with a mobile-style Tkinter UI for demo purposes.

Overview

- N-gram language model (unigrams, bigrams, trigrams)
- Backoff prediction (trigram -> bigram -> unigram)
- Prefix-aware suggestions (autocomplete while typing)
- Mobile-style Tkinter interface with suggestion chips
- Add your own data file at runtime for a richer demo

Features

- Next-word prediction with context awareness
- Real-time suggestions with top-3 candidates
- Auto-suggest toggle
- On-screen keyboard with shift, backspace, and digits
- Dataset expansion via "Add Data File" button

Dataset Sources

- Local file: data.txt (if present)
- NLTK corpora:
  - Gutenberg (literature)
  - Brown (news/editorial/reviews)
  - Reuters (business/news)

Model Details

- Tokenization: NLTK word_tokenize + regex filtering for words [a-z']+
- Training: frequency counts for n-grams (n=2..3)
- Prediction: longest matching context, then fallback to shorter contexts

Requirements

- Python 3.9+ (3.12 recommended)
- NLTK
- Tkinter (included with Python on Windows)

Setup

1. Create and activate a virtual environment (optional but recommended)
   - Windows (PowerShell):
     python -m venv venv
     .\venv\Scripts\Activate.ps1

2. Install dependencies
   python -m pip install nltk

3. Download required NLTK data
   python -m nltk.downloader punkt punkt_tab gutenberg brown reuters

Run the App
python predictive_text.py

Run in Jupyter
Open predictive_text.ipynb and run the single code cell.

Troubleshooting

- Error: "NLTK data missing"
  Run: python -m nltk.downloader punkt punkt_tab gutenberg brown reuters
- Error: "No module named 'nltk'"
  Run: python -m pip install nltk
- If downloads succeed in terminal but notebook still fails, make sure the
  notebook kernel is using the same Python environment as your terminal.

Project Structure

- predictive_text.py Main application (model + UI)
- predictive_text.ipynb Notebook version of the same code
- data.txt Optional local dataset

Demo Link
[LinkedIn Post Link](https://www.linkedin.com/posts/umme-kulsum-6885a0310_python-machinelearning-nlp-activity-7363957994952245249-OlfJ?utm_source=share&utm_medium=member_android&rcm=ACoAAE8b6cgBsjmmqnSqm2bsNAzYr5xYHBDZIOM)

Author
Ketan Ram
Intern at Algonive
