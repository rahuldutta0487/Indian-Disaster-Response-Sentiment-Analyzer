import pandas as pd
import re
from datetime import datetime
import logging
from sentiment_analyzer import analyze_sentiment, analyze_disaster_impact

# Initialize logger
logger = logging.getLogger(__name__)

def process_tweets(tweets):
    """
    Process raw tweets into a structured DataFrame with sentiment analysis.
    
    Args:
        tweets (list): List of tweet objects from Twitter API
        
    Returns:
        pandas.DataFrame: Processed DataFrame with sentiment analysis
    """
    if not tweets:
        return pd.DataFrame()
    
    processed_data = []
    
    for tweet in tweets:
        try:
            # Extract basic tweet information
            tweet_id = tweet.get('id', '')
            text = tweet.get('text', '')
            created_at = tweet.get('created_at', None)
            
            # Extract user information
            user = tweet.get('user', {})
            username = user.get('username', '')
            display_name = user.get('name', '')
            location = user.get('location', '')
            
            # Extract engagement metrics
            metrics = tweet.get('public_metrics', {})
            retweet_count = metrics.get('retweet_count', 0)
            like_count = metrics.get('like_count', 0)
            reply_count = metrics.get('reply_count', 0)
            
            # Extract hashtags and mentions from entities
            hashtags = []
            mentions = []
            
            entities = tweet.get('entities', {})
            if 'hashtags' in entities:
                hashtags = [tag.get('tag', '') for tag in entities['hashtags']]
            
            if 'mentions' in entities:
                mentions = [mention.get('username', '') for mention in entities['mentions']]
            
            # Perform sentiment analysis
            sentiment_label, sentiment_score = analyze_sentiment(text)
            
            # Analyze disaster impact
            disaster_impact = analyze_disaster_impact(text)
            
            # Add to processed data
            processed_data.append({
                'id': tweet_id,
                'text': text,
                'clean_text': clean_text(text),
                'created_at': created_at,
                'username': username,
                'display_name': display_name,
                'location': location,
                'retweet_count': retweet_count,
                'like_count': like_count,
                'reply_count': reply_count,
                'hashtags': hashtags,
                'mentions': mentions,
                'sentiment': sentiment_label,
                'sentiment_score': sentiment_score,
                'disaster_impact': disaster_impact
            })
        
        except Exception as e:
            logger.error(f"Error processing tweet: {e}")
            continue
    
    # Create DataFrame
    df = pd.DataFrame(processed_data)
    
    # Convert created_at to datetime if it exists
    if 'created_at' in df.columns and not df.empty:
        df['created_at'] = pd.to_datetime(df['created_at'])
    
    return df

def clean_text(text):
    """Clean tweet text for analysis purposes."""
    # Lower case
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove mentions and hashtags
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    
    # Remove RT indicator
    text = re.sub(r'^rt\s+', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def extract_locations(df):
    """
    Extract and geocode locations from the dataframe.
    
    Args:
        df (pandas.DataFrame): DataFrame containing tweet data
        
    Returns:
        pandas.DataFrame: DataFrame with geocoded locations
    """
    if df.empty:
        return df
    
    # Make a copy to avoid modifying the original
    result_df = df.copy()
    
    # Initialize geocoded columns
    result_df['lat'] = None
    result_df['lon'] = None
    
    # For simplicity, we're not doing actual geocoding in this function
    # as it would require external APIs. In a real implementation,
    # you would use a geocoding service to convert location strings to coordinates.
    
    # For demonstration, we'll use a simple random assignment for tweets with locations
    import random
    
    has_location = result_df['location'].notna() & (result_df['location'] != '')
    
    if has_location.any():
        # Assign random coordinates within reasonable bounds
        result_df.loc[has_location, 'lat'] = [random.uniform(25, 50) for _ in range(sum(has_location))]
        result_df.loc[has_location, 'lon'] = [random.uniform(-125, -70) for _ in range(sum(has_location))]
    
    return result_df

def aggregate_by_time(df, freq='1H'):
    """
    Aggregate tweet data by time periods.
    
    Args:
        df (pandas.DataFrame): DataFrame containing tweet data
        freq (str): Frequency string for resampling (e.g., '1H' for hourly)
        
    Returns:
        pandas.DataFrame: Aggregated DataFrame
    """
    if df.empty or 'created_at' not in df.columns:
        return pd.DataFrame()
    
    # Set index to created_at for resampling
    temp_df = df.set_index('created_at')
    
    # Count tweets per time period
    counts = temp_df.resample(freq).size().reset_index(name='tweet_count')
    
    # Aggregate sentiment
    sentiment_counts = temp_df.groupby([pd.Grouper(freq=freq), 'sentiment']).size().unstack(fill_value=0)
    
    # Merge with counts
    result = pd.merge(counts, sentiment_counts.reset_index(), on='created_at', how='left')
    
    # Fill NaN with 0
    result = result.fillna(0)
    
    return result

def filter_by_keywords(df, keywords):
    """
    Filter dataframe by keywords in the text.
    
    Args:
        df (pandas.DataFrame): DataFrame containing tweet data
        keywords (list): List of keywords to filter by
        
    Returns:
        pandas.DataFrame: Filtered DataFrame
    """
    if df.empty or 'text' not in df.columns:
        return pd.DataFrame()
    
    if not keywords:
        return df
    
    # Create regex pattern from keywords
    pattern = '|'.join(keywords)
    
    # Filter tweets containing any of the keywords
    filtered_df = df[df['text'].str.contains(pattern, case=False, regex=True)]
    
    return filtered_df
