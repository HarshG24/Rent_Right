# RentRight — AI Lease & Rental Dispute Assistant

## Project Overview

RentRight is an AI-powered lease and rental dispute assistant designed to help renters and small landlords understand lease clauses and respond to common rental disputes. The system analyzes lease text and a user’s dispute description, then provides a structured output containing the predicted dispute type, extracted lease facts, recommended next steps, and a professional draft message.

This project was developed for **CECS 551 — Advanced Artificial Intelligence** as an AI-powered business innovation prototype.

## Team Information

**Team Number:** 11

**Team Members:**
- Amruta Milind Chaudhary
- Harsh Dharmeshkumar Gandhi
- Mansi Khand
- Rishi Prakashkumar Panchal

## Problem Statement

Rental disputes are common in areas such as security deposits, repairs, late payments, and notice periods. Many renters and small landlords struggle to understand lease language because lease agreements are often written in legal or technical wording.

Existing legal information websites provide general advice, and property management platforms focus mainly on rent collection and property operations. However, there is a gap for a focused AI tool that can analyze lease-related text, classify dispute types, extract important details, and generate practical guidance.

RentRight addresses this gap by combining machine learning, NLP extraction, and LLM-based or fallback guidance generation.

## Main Features

- Classifies rental disputes into predefined categories
- Extracts important lease-related facts such as dates, deadlines, money amounts, and responsibilities
- Generates structured guidance for users
- Creates a professional draft message for landlord or tenant communication
- Uses an LLM layer when API access is available
- Uses a fallback template-based guidance system when the LLM API is unavailable
- Provides a simple Streamlit-based user interface

## Supported Dispute Categories

The current prototype supports four dispute categories:

| Category | Description |
|---|---|
| `repair_request` | Maintenance or repair-related issues |
| `security_deposit` | Security deposit return, deduction, or refund disputes |
| `late_payment` | Late rent payment or late-fee disputes |
| `notice_issue` | Lease termination, move-out, or notice-period disputes |

## AI Techniques Used

RentRight uses a hybrid AI pipeline:

### 1. Dispute Classification

The dispute classification module uses:

- **TF-IDF Vectorization** for text feature extraction
- **Logistic Regression** for supervised classification

The classifier predicts the dispute category based on the user’s dispute description.

### 2. Information Extraction

The extraction module uses:

- **spaCy Named Entity Recognition** to detect dates and money amounts
- **Regex/rule-based patterns** to extract deadline and responsibility phrases

Examples of extracted information include:

- `within 21 days`
- `$900`
- `30 days notice`
- `landlord must`
- `tenant is responsible`

### 3. LLM-Based Guidance Generation

When API access is available, RentRight sends structured information to an LLM to generate:

- Issue summary
- Important lease information
- Recommended next steps
- Professional draft message

### 4. Fallback Guidance Module

If the LLM API is unavailable due to quota, billing, or connectivity issues, the system automatically uses a fallback template-based guidance module. This ensures the prototype remains functional during testing and demonstration.

## System Architecture

```text
User Input
   |
   |-- Lease Clause Text
   |-- Dispute Description
   |
   v
Text Classification
TF-IDF + Logistic Regression
   |
   v
Information Extraction
spaCy NER + Regex Rules
   |
   v
Guidance Generation
LLM API OR Fallback Template Module
   |
   v
Structured Output
Issue Summary + Lease Facts + Next Steps + Draft Message
```

## Technology Stack

- Python
- Streamlit
- scikit-learn
- spaCy
- pandas
- OpenAI API
- python-dotenv
- regex

## Project Structure

```text
rentright_prototype/
  app.py
  requirements.txt
  README.md
  .gitignore
  data/
    train.csv
    sample_lease.txt
  screenshots/
    prototype_input.png
    prototype_output.png
  docs/
    RentRight_Phase4_Final_Report.pdf
```

## Installation and Setup

### 1. Clone the Repository

```bash
git clone <your-github-repository-link>
cd rentright_prototype
```

### 2. Create a Virtual Environment

For macOS/Linux:

```bash
python -m venv venv
source venv/bin/activate
```

For Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Required Packages

```bash
pip install -r requirements.txt
```

If you do not have a `requirements.txt` file, install manually:

```bash
pip install streamlit pandas scikit-learn spacy openai python-dotenv
```

### 4. Download spaCy Model

```bash
python -m spacy download en_core_web_sm
```

### 5. Add OpenAI API Key Optional

Create a `.env` file in the project root:

```text
OPENAI_API_KEY=your_api_key_here
```

If API access is unavailable, the system will automatically use fallback guidance.

### 6. Run the Application

```bash
streamlit run app.py
```

Open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

## Example Demo Input

### Lease Clause

```text
The landlord must return the security deposit within 21 days.
```

### Dispute Description

```text
My landlord has not returned my deposit after 25 days.
```

## Example Output

### Predicted Category

```text
security_deposit
```

### Extracted Facts

```json
{
  "dates": ["21 days", "25 days"],
  "money": [],
  "deadlines": ["within 21 days", "25 days"],
  "responsibilities": ["landlord must"]
}
```

### Generated Guidance

The system generates:

1. Issue summary
2. Important lease information
3. Recommended next steps
4. Professional draft message

## Evaluation Summary

The prototype was evaluated using synthetic rental dispute examples. Evaluation focused on:

- Classification accuracy
- Correct extraction of dates, deadlines, money amounts, and responsibilities
- Quality of generated guidance
- Reliability of fallback mode

The system performed well on clear dispute examples, especially security deposit and repair-related cases. Extraction worked best for simple money and deadline patterns. Responsibility extraction was more challenging because lease language can vary significantly.

## Limitations

- The dataset is synthetic and limited in size
- The system does not provide formal legal advice
- Lease language can vary by jurisdiction
- Rule-based extraction may miss complex legal wording
- LLM output depends on API availability
- Fallback mode is reliable but less flexible than LLM output

## Future Improvements

- Expand the labeled training dataset
- Add more dispute categories
- Support PDF lease uploads
- Improve responsibility and obligation extraction
- Add jurisdiction-aware guidance
- Compare Logistic Regression with transformer-based models
- Improve UI design for production use
- Add multilingual support
- Integrate legal-aid referral suggestions

## Disclaimer

RentRight is an academic prototype and is not a substitute for professional legal advice. The system is designed to provide general guidance and help users organize rental dispute information. Users should consult qualified legal professionals for legal decisions.

## How to Cite or Reference This Project

RentRight — AI Lease & Rental Dispute Assistant  
CECS 551 Advanced Artificial Intelligence  
Team 11
