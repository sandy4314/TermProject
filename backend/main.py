from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from utils.preprocessing import preprocess_text
from utils.prediction import SentimentClassifier, ConditionPredictor, ADRExtractor
import logging
from datetime import datetime
from pymongo import MongoClient
import os
from typing import Optional
import pandas as pd
from bson import ObjectId  


app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB setup
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
mongo_client = MongoClient(MONGO_URI)
mongo_db = mongo_client["drug_review_app"]
reports_collection = mongo_db["analysis_reports"]

# Load medicine names from dataset
try:
    df_drugs = pd.read_csv("data/drug_reviews.csv")
    MEDICINE_NAMES = sorted(df_drugs["drugName"].dropna().unique().tolist())
    logger.info(f"Loaded {len(MEDICINE_NAMES)} unique medicine names.")
except Exception as e:
    MEDICINE_NAMES = []
    logger.error(f"Failed to load medicine names: {e}")


NEGATIVE_CUE_WORDS = {
    "severe", "sevre", "problem", "worse", "worsened", "bad", "worst",
    "pain", "headache", "itch", "itching", "itchy",
    "nausea", "vomiting", "diarrhea",
    "cramps", "cramp", "rash", "burning", "swelling", "faint", "fainting"
}

POSITIVE_CUE_WORDS = {
    "good", "great", "very good", "excellent", "amazing",
    "perfect", "worked well", "helped a lot", "cured", "relief", "better"
}

NEUTRAL_CUE_WORDS = {
    "okay", "ok", "nothing special", "average", "fine",
    "not bad", "decent"
}

def adjust_sentiment_with_adrs(original_sentiment: str, original_text: str, adrs: list) -> str:
    """
    Post-process model sentiment using ADRs + positive/neutral/negative cue words.
    """
    text_l = original_text.lower()

    neg_hits = sum(1 for w in NEGATIVE_CUE_WORDS if w in text_l)
    pos_hits = sum(1 for w in POSITIVE_CUE_WORDS if w in text_l)
    neu_hits = sum(1 for w in NEUTRAL_CUE_WORDS if w in text_l)

    no_adrs = (
        not adrs
        or (len(adrs) == 1 and adrs[0] == "No ADRs detected")
    )

    # 1️⃣ ADR-driven: if there ARE ADRs and negative cues, and no strong positive cues → negative
    if not no_adrs and neg_hits >= 1 and pos_hits == 0:
        return "negative"

    # 2️⃣ No ADRs: refine borderline cases
    if no_adrs:
        # 2a. Model says negative, but it sounds just "meh" (okay, nothing special) → neutral
        if original_sentiment == "negative" and neg_hits == 0 and neu_hits >= 1 and pos_hits == 0:
            return "neutral"

        # 2b. Model says neutral, clearly positive wording, no negatives → positive
        if original_sentiment == "neutral" and pos_hits >= 1 and neg_hits == 0:
            return "positive"

    # Otherwise, trust the model
    return original_sentiment

def adjust_condition_with_context(condition: str, sentiment: str, adrs: list) -> str:
    """
    Simple safety rule for condition:
    - If the review is overall positive and has no ADRs, don't show a scary condition.
      Return 'unknown' instead.
    """
    no_adrs = (
        not adrs
        or (len(adrs) == 1 and adrs[0] == "No ADRs detected")
    )

    if sentiment == "positive" and no_adrs:
        return "unknown"

    return condition



class ReviewInput(BaseModel):
    review: str
    medicine: str | None = None


class BatchReviewInput(BaseModel):
    reviews: List[str]


def save_report_to_db(source: str,
                      review: str,
                      medicine: str | None,
                      sentiment_raw: str,
                      sentiment: str,
                      condition_raw: str,
                      condition: str,
                      adrs: list):
    doc = {
        "source": source,  # "predict", "analyze", "batch_predict"
        "review": review,
        "medicine": medicine,
        "sentiment_raw": sentiment_raw,
        "sentiment": sentiment,
        "condition_raw": condition_raw,
        "condition": condition,
        "adrs": adrs,
        "created_at": datetime.utcnow()
    }
    reports_collection.insert_one(doc)


# Initialize models
sentiment_classifier = SentimentClassifier()
condition_predictor = ConditionPredictor()
adr_extractor = ADRExtractor()

@app.post("/analyze")
async def analyze_review(review_input: ReviewInput):
    try:
        logger.info("Processing single review")
        processed_text = preprocess_text(review_input.review)
        sentiment_raw = sentiment_classifier.predict(processed_text)
        condition_raw = condition_predictor.predict(processed_text)
        adrs = adr_extractor.extract(review_input.review)

        sentiment = adjust_sentiment_with_adrs(sentiment_raw, review_input.review, adrs)
        condition = adjust_condition_with_context(condition_raw, sentiment, adrs)

        return {
            "sentiment": sentiment,
            "condition": condition,
            "adrs": adrs
        }
    except Exception as e:
        logger.error(f"Error processing review: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict")
async def predict_review(review_input: ReviewInput):
    try:
        logger.info("Processing single review")
        processed_text = preprocess_text(review_input.review)
        sentiment_raw = sentiment_classifier.predict(processed_text)
        condition_raw = condition_predictor.predict(processed_text)
        adrs = adr_extractor.extract(review_input.review)

        sentiment = adjust_sentiment_with_adrs(sentiment_raw, review_input.review, adrs)
        condition = adjust_condition_with_context(condition_raw, sentiment, adrs)

        # 💾 Save to MongoDB
        save_report_to_db(
            source="predict",
            review=review_input.review,
            medicine=review_input.medicine,
            sentiment_raw=sentiment_raw,
            sentiment=sentiment,
            condition_raw=condition_raw,
            condition=condition,
            adrs=adrs
        )

        return {
            "sentiment": sentiment,
            "condition": condition,
            "adrs": adrs,
            "original_review": review_input.review
        }
    except Exception as e:
        logger.error(f"Error processing review: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/batch_predict")
async def batch_predict_reviews(batch_input: BatchReviewInput):
    try:
        logger.info(f"Processing batch of {len(batch_input.reviews)} reviews")
        results = []
        for review in batch_input.reviews:
            processed_text = preprocess_text(review)
            sentiment_raw = sentiment_classifier.predict(processed_text)
            condition_raw = condition_predictor.predict(processed_text)
            adrs = adr_extractor.extract(review)

            sentiment = adjust_sentiment_with_adrs(sentiment_raw, review, adrs)
            condition = adjust_condition_with_context(condition_raw, sentiment, adrs)

            results.append({
                "sentiment": sentiment,
                "condition": condition,
                "adrs": adrs,
                "original_review": review
            })

        return results
    except Exception as e:
        logger.error(f"Error processing batch reviews: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
    

@app.get("/medicines")
async def get_medicines(q: Optional[str] = None):
    """
    Return a list of medicine names.
    If q is provided, return names containing q (case-insensitive).
    """
    if not MEDICINE_NAMES:
        raise HTTPException(status_code=500, detail="Medicine list not available")

    if q:
        q_l = q.lower()
        suggestions = [m for m in MEDICINE_NAMES if q_l in m.lower()][:20]
    else:
        suggestions = MEDICINE_NAMES  # limit for initial load

    return suggestions


def serialize_report(doc):
    """Convert MongoDB document to JSON-serializable dict."""
    return {
        "id": str(doc.get("_id")),
        "review": doc.get("review"),
        "medicine": doc.get("medicine"),
        "sentiment": doc.get("sentiment"),
        "sentiment_raw": doc.get("sentiment_raw"),
        "condition": doc.get("condition"),
        "condition_raw": doc.get("condition_raw"),
        "adrs": doc.get("adrs", []),
        "created_at": doc.get("created_at").isoformat() if doc.get("created_at") else None,
        "source": doc.get("source"),
    }


@app.get("/admin/reports")
async def get_admin_reports(
    page: int = 1,
    page_size: int = 20,
    sentiment: Optional[str] = None,
    medicine: Optional[str] = None,
):
    """
    Admin endpoint to list stored analysis reports with pagination and basic filters.
    """
    if page < 1:
        page = 1
    if page_size < 1 or page_size > 100:
        page_size = 20

    query = {}
    if sentiment:
        query["sentiment"] = sentiment
    if medicine:
        # case-insensitive contains
        query["medicine"] = {"$regex": medicine, "$options": "i"}

    skip = (page - 1) * page_size

    cursor = (
        reports_collection
        .find(query)
        .sort("created_at", -1)
        .skip(skip)
        .limit(page_size)
    )

    items = [serialize_report(doc) for doc in cursor]
    total = reports_collection.count_documents(query)

    return {
        "items": items,
        "total": total,
        "page": page,
        "page_size": page_size,
    }

