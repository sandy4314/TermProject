from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
import scispacy
import spacy
import joblib
import os
import numpy as np

class SentimentClassifier:
    def __init__(self):
        model_dir = 'models'
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model directory {model_dir} not found. Run train.py first.")
        self.vectorizer = joblib.load(os.path.join(model_dir, 'tfidf_vectorizer_sentiment.joblib'))
        self.model_nb = joblib.load(os.path.join(model_dir, 'sentiment_nb.joblib'))
        self.model_svm = joblib.load(os.path.join(model_dir, 'sentiment_svm.joblib'))
        self.model_nn = joblib.load(os.path.join(model_dir, 'sentiment_nn.joblib'))
        
    def predict(self, text: str) -> str:
        features = self.vectorizer.transform([text])
        # Weighted ensemble (SVM and NN are more reliable)
        predictions = [
            (self.model_nb.predict(features)[0], 0.2),  # Lower weight for NB
            (self.model_svm.predict(features)[0], 0.4),
            (self.model_nn.predict(features)[0], 0.4)
        ]
        # Compute weighted majority vote
        vote_counts = {0: 0, 1: 0, 2: 0}  # 0: negative, 1: neutral, 2: positive
        for pred, weight in predictions:
            vote_counts[pred] += weight
        max_vote = max(vote_counts, key=vote_counts.get)
        return ["negative", "neutral", "positive"][max_vote]

class ConditionPredictor:
    def __init__(self, confidence_margin: float = 0.2):
        """
        confidence_margin: minimum difference between best and second-best class score.
        If margin is smaller than this, we treat prediction as unreliable.
        """
        model_dir = 'models'
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model directory {model_dir} not found. Run train.py first.")

        self.vectorizer = joblib.load(os.path.join(model_dir, 'tfidf_vectorizer_condition.joblib'))
        self.model = joblib.load(os.path.join(model_dir, 'condition_svm.joblib'))
        self.confidence_margin = confidence_margin

        # All known condition labels (from training), except generic "other" if present
        self.condition_labels = [
            str(c) for c in self.model.classes_
            if str(c).lower() not in {"other", "unknown"}
        ]

    def _mentions_known_condition(self, text: str) -> bool:
        """
        Simple check: does the text explicitly contain any of the known condition names?
        """
        if not isinstance(text, str):
            return False
        text_l = text.lower()
        for label in self.condition_labels:
            if label.lower() in text_l:
                return True
        return False

    def predict(self, text: str) -> str:
        # Very short or empty text → don't try to guess condition
        if not isinstance(text, str) or len(text.strip()) < 5:
            return "unknown"

        # 🔹 KEY RULE: only predict a condition if the text explicitly mentions one
        if not self._mentions_known_condition(text):
            return "unknown"

        # Continue with model-based prediction
        features = self.vectorizer.transform([text])

        # Use decision_function for confidence
        scores = self.model.decision_function(features)

        # Binary classification case
        if scores.ndim == 1:
            max_score = float(np.max(scores))
            if abs(max_score) < self.confidence_margin:
                return "unknown"
            return self.model.predict(features)[0]

        # Multiclass case
        scores = scores[0]  # shape: (n_classes,)
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])

        # Second best
        sorted_scores = np.sort(scores)
        second_best_score = float(sorted_scores[-2])
        margin = best_score - second_best_score

        # If best and second-best too close → model not confident
        if margin < self.confidence_margin:
            return "unknown"

        return self.model.classes_[best_idx]

class ADRExtractor:
    def __init__(self):
        self.nlp = spacy.load("en_ner_bc5cdr_md", disable=["parser", "textcat"])  # Medical NER model
        self.side_effect_keywords = {
            'nausea', 'headache', 'dizziness', 'fatigue', 'pain', 'rash', 'vomiting',
            'diarrhea', 'insomnia', 'constipation', 'anxiety', 'dry mouth', 'sweating',
            'tremor', 'weight gain', 'weight loss', 'drowsiness', 'palpitations',
            # 🔹 added:
            'itch', 'itching', 'itchy'
        }
        
    def extract(self, text: str) -> list:
        if not text:
            return ["No ADRs detected"]
        doc = self.nlp(text)
        adrs = set()

        # Extract medical entities labeled as DISEASE or CHEMICAL (potential ADRs)
        for ent in doc.ents:
            ent_clean = ent.text.lower().strip(".,!? ")
            if ent.label_ == "DISEASE" and any(keyword in ent_clean for keyword in self.side_effect_keywords):
                adrs.add(ent_clean)

        # Extract keywords from tokens
        for token in doc:
            token_clean = token.text.lower().strip(".,!? ")
            if token_clean in self.side_effect_keywords:
                adrs.add(token_clean)

        return list(adrs) if adrs else ["No ADRs detected"]
