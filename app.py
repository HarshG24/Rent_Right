import re
import pandas as pd
import streamlit as st
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="RentRight Prototype", layout="wide")

client = OpenAI(api_key=api_key) if api_key else None


@st.cache_resource
def load_nlp():
    return spacy.load("en_core_web_sm")


@st.cache_resource
def train_demo_model():
    train_df = pd.read_csv("data/train.csv")
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    X = vectorizer.fit_transform(train_df["text"])
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, train_df["label"])
    return vectorizer, clf


nlp = load_nlp()
vectorizer, clf = train_demo_model()


CATEGORY_STEPS = {
    "repair_request": [
        "Review the repair and maintenance clause in the lease.",
        "Document the issue with dates, photos, and prior notices.",
        "Send a written follow-up request asking for a repair timeline.",
    ],
    "security_deposit": [
        "Review the lease for deposit return conditions and deductions.",
        "Check the expected timeline for returning the deposit.",
        "Send a written follow-up request with your move-out date and requested amount.",
    ],
    "late_payment": [
        "Review the due date, grace period, and late-fee clause.",
        "Collect proof of payment or payment attempt.",
        "Request clarification of how the late fee was calculated.",
    ],
    "notice_issue": [
        "Review the notice period requirement in the lease.",
        "Check whether notice was delivered in the required form.",
        "Respond in writing and keep a record of all communication.",
    ],
}


def classify_dispute(text: str) -> str:
    X = vectorizer.transform([text])
    return clf.predict(X)[0]


def extract_info(text: str):
    doc = nlp(text)
    dates = [ent.text for ent in doc.ents if ent.label_ == "DATE"]
    money = [ent.text for ent in doc.ents if ent.label_ == "MONEY"]
    deadlines = re.findall(r"within\s+\d+\s+days|\d+\s+days\s+notice|\d+\s+days", text.lower())
    responsibilities = re.findall(
        r"(?:tenant|landlord)[^.\n]{0,60}(?:responsible|must|shall)",
        text.lower(),
    )
    return {
        "dates": dates,
        "money": money,
        "deadlines": deadlines,
        "responsibilities": responsibilities,
    }


def generate_guidance_fallback(category: str, extracted: dict) -> str:
    steps = CATEGORY_STEPS.get(category, [])
    amount = extracted["money"][0] if extracted["money"] else "the relevant amount"

    facts = []
    if extracted["dates"]:
        facts.append(f"Dates: {', '.join(extracted['dates'])}")
    if extracted["money"]:
        facts.append(f"Money: {', '.join(extracted['money'])}")
    if extracted["deadlines"]:
        facts.append(f"Deadlines: {', '.join(extracted['deadlines'])}")
    if extracted["responsibilities"]:
        facts.append(f"Responsibilities: {', '.join(extracted['responsibilities'])}")

    facts_text = "\n".join(f"- {item}" for item in facts) if facts else "- No major structured facts were extracted."

    steps_text = "\n".join(f"{i}. {step}" for i, step in enumerate(steps, start=1)) if steps else "1. Review the lease carefully.\n2. Document the issue in writing.\n3. Preserve all communication."

    draft = (
        f"Hello, I am writing regarding a {category.replace('_', ' ')} issue under our lease agreement. "
        f"Based on my review, I would like clarification and a written response regarding {amount}. "
        "Please let me know the next steps and any supporting details I should provide. Thank you."
    )

    return f"""
### 1. Issue Summary
This dispute appears to be related to **{category.replace('_', ' ')}**. The system analyzed the dispute description and extracted relevant lease-related facts.

### 2. Important Lease Information
{facts_text}

### 3. Recommended Next Steps
{steps_text}

### 4. Professional Draft Message
{draft}
"""


def generate_guidance_llm(category: str, extracted: dict, dispute_text: str, lease_text: str) -> str:
    if not client:
        raise ValueError("Missing API key")

    prompt = f"""
You are RentRight, an AI assistant for lease and rental disputes.

You help renters and small landlords understand disputes clearly.
Do not provide formal legal advice. Give practical, cautious, user-friendly guidance.

Dispute category:
{category}

Dispute description:
{dispute_text}

Lease text:
{lease_text}

Extracted information:
- Dates: {extracted.get('dates', [])}
- Money: {extracted.get('money', [])}
- Deadlines: {extracted.get('deadlines', [])}
- Responsibilities: {extracted.get('responsibilities', [])}

Generate the response in this exact structure:

1. Issue Summary
Write 2-3 sentences in simple language.

2. Important Lease Information
List the most relevant dates, deadlines, money amounts, and responsibilities.

3. Recommended Next Steps
Give 3-5 practical action steps.

4. Professional Draft Message
Write a polite message the user can send to the landlord or tenant.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful rental dispute assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.4,
    )

    return response.choices[0].message.content


st.title("RentRight")
st.caption("AI Lease & Rental Dispute Assistant")

left, right = st.columns(2)

with left:
    lease_text = st.text_area("Paste lease clause text", height=220)

with right:
    dispute_text = st.text_area("Describe the dispute", height=220)

if st.button("Analyze dispute", use_container_width=True):
    if not dispute_text.strip():
        st.error("Please enter a dispute description.")
    else:
        category = classify_dispute(dispute_text)
        extracted = extract_info((lease_text or "") + "\n" + dispute_text)

        used_llm = True
        try:
            output = generate_guidance_llm(
                category=category,
                extracted=extracted,
                dispute_text=dispute_text,
                lease_text=lease_text or ""
            )
        except Exception as e:
            used_llm = False
            output = generate_guidance_fallback(category, extracted)
            
            st.warning("LLM output is currently unavailable, so fallback guidance was used.")
        a, b = st.columns([1, 2])

        with a:
            st.subheader("Predicted category")
            st.success(category)

            st.subheader("Extracted facts")
            st.json(extracted)

            st.subheader("Generation mode")
            st.info("LLM" if used_llm else "Fallback template")

        with b:
            st.subheader("Guidance")
            st.markdown(output)