import pandas as pd
import json
import os
from datetime import datetime
import logging

# Initialize logger
logger = logging.getLogger(__name__)

# Define paths for caching
CACHE_DIR = "cache"
CACHE_FILE = os.path.join(CACHE_DIR, "tweets_cache.csv")

def filter_dataframe(df, query):
    """
    Filter DataFrame based on text query.
    
    Args:
        df (pandas.DataFrame): DataFrame to filter
        query (str): Text query to search for
        
    Returns:
        pandas.DataFrame: Filtered DataFrame
    """
    if df.empty or not query:
        return df
    
    # Convert query to lowercase for case-insensitive matching
    query = query.lower()
    
    # Filter based on text, username, or hashtags
    mask = (
        df['text'].str.lower().str.contains(query, na=False) |
        df['username'].str.lower().str.contains(query, na=False)
    )
    
    # Also check hashtags if available
    if 'hashtags' in df.columns:
        # If hashtags are stored as strings, convert to lowercase
        if df['hashtags'].dtype == 'object':
            hashtag_mask = df['hashtags'].astype(str).str.lower().str.contains(query, na=False)
        # If hashtags are stored as lists, check each list
        else:
            def check_hashtags(tags):
                if isinstance(tags, list):
                    return any(query in tag.lower() for tag in tags)
                return False
            
            hashtag_mask = df['hashtags'].apply(check_hashtags)
        
        mask = mask | hashtag_mask
    
    return df[mask]

def cache_data(df):
    """
    Cache DataFrame to disk for persistence.
    
    Args:
        df (pandas.DataFrame): DataFrame to cache
    """
    try:
        # Create cache directory if it doesn't exist
        if not os.path.exists(CACHE_DIR):
            os.makedirs(CACHE_DIR)
        
        # Save DataFrame to CSV
        df.to_csv(CACHE_FILE, index=False)
        logger.info(f"Data cached to {CACHE_FILE}")
        
    except Exception as e:
        logger.error(f"Error caching data: {e}")

def get_cached_data():
    """
    Load cached DataFrame from disk.
    
    Returns:
        pandas.DataFrame: Cached DataFrame or empty DataFrame if no cache exists
    """
    try:
        if os.path.exists(CACHE_FILE):
            df = pd.read_csv(CACHE_FILE)
            
            # Convert 'created_at' back to datetime
            if 'created_at' in df.columns:
                df['created_at'] = pd.to_datetime(df['created_at'])
            
            # Convert string representations of lists back to actual lists
            for col in ['hashtags', 'mentions']:
                if col in df.columns:
                    df[col] = df[col].apply(lambda x: 
                        eval(x) if isinstance(x, str) and x.startswith('[') and x.endswith(']') 
                        else x)
            
            logger.info(f"Loaded cached data from {CACHE_FILE}")
            return df
        else:
            logger.info("No cached data found")
            return pd.DataFrame()
        
    except Exception as e:
        logger.error(f"Error loading cached data: {e}")
        return pd.DataFrame()

def format_time_ago(timestamp):
    """
    Format timestamp as time ago (e.g., "2 hours ago").
    
    Args:
        timestamp (datetime): Timestamp to format
        
    Returns:
        str: Formatted time ago string
    """
    if not timestamp:
        return ""
    
    now = datetime.now()
    diff = now - timestamp
    
    seconds = diff.total_seconds()
    
    if seconds < 60:
        return f"{int(seconds)} seconds ago"
    elif seconds < 3600:
        minutes = int(seconds / 60)
        return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
    elif seconds < 86400:
        hours = int(seconds / 3600)
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    elif seconds < 604800:
        days = int(seconds / 86400)
        return f"{days} day{'s' if days != 1 else ''} ago"
    else:
        weeks = int(seconds / 604800)
        return f"{weeks} week{'s' if weeks != 1 else ''} ago"

def export_data(df, format="csv"):
    """
    Export DataFrame to a file.
    
    Args:
        df (pandas.DataFrame): DataFrame to export
        format (str): Export format ('csv' or 'json')
        
    Returns:
        str: Path to exported file
    """
    if df.empty:
        return None
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        if format.lower() == "json":
            filename = f"disaster_tweets_{timestamp}.json"
            
            # Convert datetime to string to make JSON serializable
            df_copy = df.copy()
            if 'created_at' in df_copy.columns:
                df_copy['created_at'] = df_copy['created_at'].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            df_copy.to_json(filename, orient="records", date_format="iso")
            
        else:  # default to CSV
            filename = f"disaster_tweets_{timestamp}.csv"
            df.to_csv(filename, index=False)
        
        logger.info(f"Data exported to {filename}")
        return filename
        
    except Exception as e:
        logger.error(f"Error exporting data: {e}")
        return None
