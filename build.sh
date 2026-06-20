#!/usr/bin/env bash
pip install -r requirements.txt
python -m spacy download es_core_news_sm
python -m spacy download es_core_news_lg