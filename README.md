# Predictive Text Generator (NLP)

A simple NLP-based predictive text system that suggests the next word using an N-gram language model.

---

## Features

* Next-word prediction using unigram, bigram, trigram
* Backoff strategy (trigram → bigram → unigram)
* Prefix-based autocomplete
* Simple Tkinter UI

---

## Requirements

* Python 3.9+
* nltk

---

## Setup

```bash
pip install nltk
python -m nltk.downloader punkt gutenberg brown reuters
```

---

## Run

```bash
python predictive_text.py
```

---

## Project Structure

```bash
predictive_text.py
predictive_text.ipynb
data.txt
```

---

## Author

Ketan Ram
