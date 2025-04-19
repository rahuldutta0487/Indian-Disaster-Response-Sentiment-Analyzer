import pandas as pd
import re
from collections import Counter
import nltk
from nltk.util import ngrams
import logging

# Initialize logger
logger = logging.getLogger(__name__)

# Try to download NLTK resources if needed
try:
    nltk.download('punkt', quiet=True)
except Exception as e:
    logger.warning(f"Failed to download NLTK punkt: {e}")

def analyze_trends(df, top_n=10):
    """
    Analyze tweet data to identify trending topics, hashtags, and more.
    
    Args:
        df (pandas.DataFrame): DataFrame containing tweet data
        top_n (int): Number of top items to return in each category
        
    Returns:
        dict: Dictionary containing various trend analysis results
    """
    if df.empty:
        return {
            'hashtags': {},
            'mentions': {},
            'terms': {},
            'phrases': {},
            'domains': {}
        }
    
    try:
        # Extract hashtags
        hashtags = extract_hashtags(df)
        
        # Extract mentions
        mentions = extract_mentions(df)
        
        # Extract trending terms (excluding common stopwords)
        terms = extract_terms(df)
        
        # Extract common phrases (bigrams and trigrams)
        phrases = extract_phrases(df)
        
        # Extract shared domains/URLs
        domains = extract_domains(df)
        
        # Create results dictionary with top N items in each category
        results = {
            'hashtags': dict(Counter(hashtags).most_common(top_n)),
            'mentions': dict(Counter(mentions).most_common(top_n)),
            'terms': dict(Counter(terms).most_common(top_n)),
            'phrases': dict(Counter(phrases).most_common(top_n)),
            'domains': dict(Counter(domains).most_common(top_n))
        }
        
        return results
    
    except Exception as e:
        logger.error(f"Error analyzing trends: {e}")
        return {
            'hashtags': {},
            'mentions': {},
            'terms': {},
            'phrases': {},
            'domains': {}
        }

def extract_hashtags(df):
    """Extract hashtags from tweets."""
    hashtags = []
    
    # First check if we have pre-parsed hashtags
    if 'hashtags' in df.columns:
        for tags in df['hashtags'].dropna():
            if isinstance(tags, list):
                hashtags.extend([tag.lower() for tag in tags if tag])
            elif isinstance(tags, str):
                # In case hashtags were stored as string representation of list
                try:
                    # Try to evaluate if it's a string representation of a list
                    eval_tags = eval(tags)
                    if isinstance(eval_tags, list):
                        hashtags.extend([tag.lower() for tag in eval_tags if tag])
                except:
                    # If evaluation fails, add as is
                    hashtags.append(tags.lower())
    
    # If no pre-parsed hashtags or empty list, extract from text
    if not hashtags and 'text' in df.columns:
        for text in df['text'].dropna():
            # Extract hashtags using regex
            tags = re.findall(r'#(\w+)', text)
            hashtags.extend([tag.lower() for tag in tags])
    
    return hashtags

def extract_mentions(df):
    """Extract user mentions from tweets."""
    mentions = []
    
    # First check if we have pre-parsed mentions
    if 'mentions' in df.columns:
        for users in df['mentions'].dropna():
            if isinstance(users, list):
                mentions.extend([user.lower() for user in users if user])
            elif isinstance(users, str):
                # In case mentions were stored as string representation of list
                try:
                    # Try to evaluate if it's a string representation of a list
                    eval_users = eval(users)
                    if isinstance(eval_users, list):
                        mentions.extend([user.lower() for user in eval_users if user])
                except:
                    # If evaluation fails, add as is
                    mentions.append(users.lower())
    
    # If no pre-parsed mentions or empty list, extract from text
    if not mentions and 'text' in df.columns:
        for text in df['text'].dropna():
            # Extract mentions using regex
            users = re.findall(r'@(\w+)', text)
            mentions.extend([user.lower() for user in users])
    
    return mentions

def extract_terms(df):
    """Extract significant terms from tweets, excluding common stopwords."""
    terms = []
    
    # Use 'clean_text' if available, otherwise use 'text'
    text_col = 'clean_text' if 'clean_text' in df.columns else 'text'
    
    if text_col in df.columns:
        # Define stopwords to filter out common terms
        stopwords = set([
            'a', 'an', 'the', 'and', 'but', 'or', 'for', 'nor', 'on', 'at', 'to', 'by',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'in', 'that', 'this', 
            'these', 'those', 'it', 'its', 'rt', 'via', 'i', 'you', 'he', 'she', 'we',
            'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'our', 'their',
            'from', 'with', 'as', 'of', 'have', 'has', 'had', 'do', 'does', 'did',
            'just', 'more', 'most', 'some', 'such', 'no', 'not', 'only', 'than',
            'then', 'so', 'very', 'can', 'will', 'would', 'should', 'now', 'about',
            'amp', 'http', 'https', 'co', 't.co'
        ])
        
        for text in df[text_col].dropna():
            # Tokenize and convert to lowercase
            words = nltk.word_tokenize(text.lower())
            
            # Filter out stopwords and short words
            filtered_words = [
                word for word in words 
                if word.isalpha() and word not in stopwords and len(word) > 2
            ]
            
            terms.extend(filtered_words)
    
    return terms

def extract_phrases(df):
    """Extract common phrases (bigrams and trigrams) from tweets."""
    phrases = []
    
    # Use 'clean_text' if available, otherwise use 'text'
    text_col = 'clean_text' if 'clean_text' in df.columns else 'text'
    
    if text_col in df.columns:
        for text in df[text_col].dropna():
            # Tokenize and convert to lowercase
            words = nltk.word_tokenize(text.lower())
            
            # Filter out non-alpha and short words
            filtered_words = [word for word in words if word.isalpha() and len(word) > 2]
            
            # Generate bigrams and trigrams
            if len(filtered_words) >= 2:
                bigrams_list = list(ngrams(filtered_words, 2))
                bigram_phrases = [' '.join(bigram) for bigram in bigrams_list]
                phrases.extend(bigram_phrases)
            
            if len(filtered_words) >= 3:
                trigrams_list = list(ngrams(filtered_words, 3))
                trigram_phrases = [' '.join(trigram) for trigram in trigrams_list]
                phrases.extend(trigram_phrases)
    
    return phrases

def extract_domains(df):
    """Extract shared domains/URLs from tweets."""
    domains = []
    
    if 'text' in df.columns:
        for text in df['text'].dropna():
            # Extract URLs using regex
            urls = re.findall(r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', text)
            
            for url in urls:
                # Extract domain from URL
                domain_match = re.search(r'https?://(?:www\.)?([^/]+)', url)
                if domain_match:
                    domain = domain_match.group(1).lower()
                    domains.append(domain)
    
    return domains

def detect_emerging_topics(df, time_window=3600):
    """
    Detect emerging topics within a recent time window.
    
    Args:
        df (pandas.DataFrame): DataFrame containing tweet data
        time_window (int): Time window in seconds (default: 1 hour)
        
    Returns:
        list: List of emerging topics with scores
    """
    if df.empty or 'created_at' not in df.columns or 'text' in df.columns:
        return []
    
    try:
        # Copy the dataframe
        df_copy = df.copy()
        
        # Calculate current time and time window
        now = pd.Timestamp.now()
        window_start = now - pd.Timedelta(seconds=time_window)
        
        # Split into recent and older tweets
        recent_tweets = df_copy[df_copy['created_at'] >= window_start]
        older_tweets = df_copy[df_copy['created_at'] < window_start]
        
        if recent_tweets.empty:
            return []
        
        # Analyze terms in recent tweets
        recent_terms = extract_terms(recent_tweets)
        recent_term_counts = Counter(recent_terms)
        
        # If we have older tweets, compare frequencies
        if not older_tweets.empty:
            older_terms = extract_terms(older_tweets)
            older_term_counts = Counter(older_terms)
            
            # Calculate term frequency change
            emerging_topics = []
            
            for term, recent_count in recent_term_counts.items():
                older_count = older_term_counts.get(term, 0)
                
                # Calculate rate of change
                if older_count == 0:
                    # New term that wasn't in older tweets
                    change_rate = 1.0
                else:
                    # Calculate normalized change
                    older_freq = older_count / len(older_tweets)
                    recent_freq = recent_count / len(recent_tweets)
                    change_rate = (recent_freq - older_freq) / older_freq if older_freq > 0 else 1.0
                
                # Include only terms with significant increase
                if change_rate > 0.5 and recent_count >= 3:
                    emerging_topics.append({
                        'term': term,
                        'score': change_rate,
                        'count': recent_count
                    })
            
            # Sort by score descending
            emerging_topics.sort(key=lambda x: x['score'], reverse=True)
            return emerging_topics[:10]  # Return top 10
        
        else:
            # If no older tweets, just return top recent terms
            top_terms = recent_term_counts.most_common(10)
            return [{'term': term, 'score': 1.0, 'count': count} for term, count in top_terms]
    
    except Exception as e:
        logger.error(f"Error detecting emerging topics: {e}")
        return []
