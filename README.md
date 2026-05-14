# RentRight Prototype

A simple Phase 3 MVP for the CECS 551 term project.

## Features
- Classifies rental disputes with TF-IDF + Logistic Regression
- Extracts dates, money amounts, deadlines, and responsibilities from lease/dispute text
- Generates recommended next steps and a professional draft message

## Setup
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
streamlit run app.py
```

## Demo categories
- repair_request
- security_deposit
- late_payment
- notice_issue
