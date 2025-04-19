import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
import re
import logging

# Initialize logger
logger = logging.getLogger(__name__)

# Try to download NLTK resources - only needed first time
try:
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except Exception as e:
    logger.warning(f"Failed to download NLTK resources: {e}")

# Initialize VADER sentiment analyzer
try:
    sid = SentimentIntensityAnalyzer()
except Exception as e:
    logger.error(f"Failed to initialize SentimentIntensityAnalyzer: {e}")
    sid = None

def clean_tweet(tweet_text):
    """
    Clean the tweet text by removing URLs, mentions, hashtags, and special characters.
    
    Args:
        tweet_text (str): The raw tweet text
        
    Returns:
        str: Cleaned tweet text
    """
    # Convert to lowercase
    text = tweet_text.lower()
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove mentions and hashtags
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    
    # Remove RT indicator
    text = re.sub(r'^rt\s+', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove special characters
    text = re.sub(r'[^\w\s]', '', text)
    
    return text

def analyze_sentiment(text, method="combined"):
    """
    Analyze the sentiment of a text using specified method.
    
    Args:
        text (str): Text to analyze
        method (str): Method to use - 'vader', 'textblob', or 'combined' (default)
        
    Returns:
        tuple: (sentiment_label, sentiment_score) 
               sentiment_label is one of 'positive', 'negative', 'neutral'
               sentiment_score is a float between -1 and 1
    """
    if not text:
        return "neutral", 0.0
    
    # Clean the text
    cleaned_text = clean_tweet(text)
    
    if method == "vader":
        return _vader_sentiment(cleaned_text)
    elif method == "textblob":
        return _textblob_sentiment(cleaned_text)
    else:  # combined approach
        return _combined_sentiment(cleaned_text)

def _vader_sentiment(text):
    """Use VADER sentiment analyzer to determine sentiment."""
    if not sid:
        logger.warning("VADER SentimentIntensityAnalyzer not initialized, falling back to TextBlob")
        return _textblob_sentiment(text)
    
    try:
        sentiment_scores = sid.polarity_scores(text)
        compound_score = sentiment_scores['compound']
        
        # Determine sentiment label based on compound score
        if compound_score >= 0.05:
            return "positive", compound_score
        elif compound_score <= -0.05:
            return "negative", compound_score
        else:
            return "neutral", compound_score
    except Exception as e:
        logger.error(f"Error in VADER sentiment analysis: {e}")
        return "neutral", 0.0

def _textblob_sentiment(text):
    """Use TextBlob to determine sentiment."""
    try:
        analysis = TextBlob(text)
        polarity = analysis.sentiment.polarity
        
        # Determine sentiment label based on polarity
        if polarity > 0.1:
            return "positive", polarity
        elif polarity < -0.1:
            return "negative", polarity
        else:
            return "neutral", polarity
    except Exception as e:
        logger.error(f"Error in TextBlob sentiment analysis: {e}")
        return "neutral", 0.0

def _combined_sentiment(text):
    """
    Combined approach using both VADER and TextBlob.
    This provides more robust sentiment analysis by considering both methods.
    """
    vader_label, vader_score = _vader_sentiment(text)
    textblob_label, textblob_score = _textblob_sentiment(text)
    
    # If both agree, use that sentiment
    if vader_label == textblob_label:
        return vader_label, (vader_score + textblob_score) / 2
    
    # If they disagree, use the stronger signal
    if abs(vader_score) > abs(textblob_score):
        return vader_label, vader_score
    else:
        return textblob_label, textblob_score

def analyze_disaster_impact(text):
    """
    Analyze the text to determine the severity of disaster impact mentioned.
    
    Args:
        text (str): Text to analyze
        
    Returns:
        str: Impact level - 'severe', 'moderate', 'minor', or 'unknown'
    """
    # Keywords indicating severity levels
    severe_keywords = [
        'catastrophic', 'devastat', 'fatal', 'death', 'killed', 'casualties', 
        'destroyed', 'emergency', 'evacuat', 'crisis', 'danger', 'severe', 
        'tragedy', 'disaster', 'critical', 'massive'
    ]
    
    moderate_keywords = [
        'damage', 'injured', 'wound', 'affected', 'impact', 'hit', 'threat',
        'loss', 'moderate', 'concern', 'worried', 'warning'
    ]
    
    minor_keywords = [
        'minor', 'small', 'limited', 'contained', 'controlled', 'restored',
        'recovery', 'stable', 'manageable', 'relief'
    ]
    
    # Check for keyword presence
    text_lower = text.lower()
    
    # Count keyword occurrences
    severe_count = sum(1 for keyword in severe_keywords if keyword in text_lower)
    moderate_count = sum(1 for keyword in moderate_keywords if keyword in text_lower)
    minor_count = sum(1 for keyword in minor_keywords if keyword in text_lower)
    
    # Determine impact level based on keyword counts
    if severe_count > 0:
        return "severe"
    elif moderate_count > 0:
        return "moderate"
    elif minor_count > 0:
        return "minor"
    else:
        return "unknown"
