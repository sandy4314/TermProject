import nltk
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

nlp = spacy.load("en_core_web_sm", disable=["parser"])  # Disable unused components for efficiency
stop_words = set(stopwords.words('english')) - {'not', 'no', 'never'}  # Retain negation words
lemmatizer = WordNetLemmatizer()

def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    # Convert to lowercase
    text = text.lower()
    # Remove special characters, keep punctuation for sentiment cues
    text = re.sub(r'[^a-zA-Z\s.!?]', '', text)
    # Tokenization
    tokens = word_tokenize(text)
    # Remove stopwords except negation words
    tokens = [token for token in tokens if token not in stop_words or token in {'not', 'no', 'never'}]
    # Lemmatization
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(tokens)
