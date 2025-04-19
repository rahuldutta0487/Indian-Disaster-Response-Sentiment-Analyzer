import os
import logging
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Float, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import pandas as pd
from datetime import datetime

# Initialize logger
logger = logging.getLogger(__name__)

# Check if DATABASE_URL is defined (production/Replit)
if os.environ.get('DATABASE_URL'):
    SQLALCHEMY_DATABASE_URL = os.environ.get('DATABASE_URL')
else:
    # Use SQLite for local development
    SQLALCHEMY_DATABASE_URL = "sqlite:///./tweets.db"
# Create SQLAlchemy engine
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, 
    connect_args={} if SQLALCHEMY_DATABASE_URL.startswith('postgresql') else {"check_same_thread": False}
)

# Create declarative base
Base = declarative_base()

# Define Tweet model
class Tweet(Base):
    __tablename__ = 'tweets'
    
    id = Column(Integer, primary_key=True)
    tweet_id = Column(String(255), unique=True, nullable=False)
    text = Column(Text, nullable=False)
    clean_text = Column(Text)
    created_at = Column(DateTime, index=True)
    username = Column(String(255))
    display_name = Column(String(255))
    location = Column(String(255))
    retweet_count = Column(Integer, default=0)
    like_count = Column(Integer, default=0)
    reply_count = Column(Integer, default=0)
    hashtags = Column(JSON)
    mentions = Column(JSON)
    sentiment = Column(String(50))
    sentiment_score = Column(Float)
    disaster_impact = Column(String(50))
    disaster_type = Column(String(50))
    lat = Column(Float)
    lon = Column(Float)
    inserted_at = Column(DateTime, default=datetime.now)
    
# Create tables
def init_db():
    """Initialize database tables"""
    try:
        Base.metadata.create_all(engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error creating database tables: {e}")

# Create session factory
Session = sessionmaker(bind=engine)

def save_tweets(tweets_df):
    """
    Save tweets from DataFrame to database
    
    Args:
        tweets_df (pandas.DataFrame): DataFrame containing tweet data
        
    Returns:
        int: Number of tweets saved
    """
    if tweets_df.empty:
        return 0
    
    session = Session()
    count = 0
    
    try:
        for _, row in tweets_df.iterrows():
            # Check if tweet already exists
            existing = session.query(Tweet).filter_by(tweet_id=str(row['id'])).first()
            
            if existing:
                continue
                
            # Create new tweet
            tweet = Tweet(
                tweet_id=str(row['id']),
                text=row['text'],
                clean_text=row['clean_text'] if 'clean_text' in row else None,
                created_at=row['created_at'],
                username=row['username'],
                display_name=row['display_name'],
                location=row['location'] if 'location' in row else None,
                retweet_count=row['retweet_count'] if 'retweet_count' in row else 0,
                like_count=row['like_count'] if 'like_count' in row else 0,
                reply_count=row['reply_count'] if 'reply_count' in row else 0,
                hashtags=row['hashtags'] if 'hashtags' in row else [],
                mentions=row['mentions'] if 'mentions' in row else [],
                sentiment=row['sentiment'],
                sentiment_score=row['sentiment_score'],
                disaster_impact=row['disaster_impact'] if 'disaster_impact' in row else 'unknown',
                disaster_type=row.get('disaster_type', 'General'),
                lat=row['lat'] if 'lat' in row else None,
                lon=row['lon'] if 'lon' in row else None
            )
            
            session.add(tweet)
            count += 1
        
        session.commit()
        logger.info(f"Saved {count} new tweets to database")
        return count
        
    except Exception as e:
        session.rollback()
        logger.error(f"Error saving tweets to database: {e}")
        return 0
        
    finally:
        session.close()

def get_tweets(limit=1000, disaster_type=None, time_range=None):
    """
    Get tweets from database with optional filtering
    
    Args:
        limit (int): Maximum number of tweets to retrieve
        disaster_type (str, optional): Filter by disaster type
        time_range (tuple, optional): Filter by time range (start, end)
        
    Returns:
        pandas.DataFrame: DataFrame with tweets
    """
    session = Session()
    
    try:
        query = session.query(Tweet)
        
        # Apply filters
        if disaster_type and disaster_type != "All":
            query = query.filter(Tweet.disaster_type == disaster_type)
            
        if time_range:
            start_time, end_time = time_range
            if start_time:
                query = query.filter(Tweet.created_at >= start_time)
            if end_time:
                query = query.filter(Tweet.created_at <= end_time)
        
        # Order by created_at descending and limit
        query = query.order_by(Tweet.created_at.desc()).limit(limit)
        
        # Execute query
        tweets = query.all()
        
        if not tweets:
            return pd.DataFrame()
        
        # Convert to DataFrame
        data = []
        for tweet in tweets:
            data.append({
                'id': tweet.tweet_id,
                'text': tweet.text,
                'clean_text': tweet.clean_text,
                'created_at': tweet.created_at,
                'username': tweet.username,
                'display_name': tweet.display_name,
                'location': tweet.location,
                'retweet_count': tweet.retweet_count,
                'like_count': tweet.like_count,
                'reply_count': tweet.reply_count,
                'hashtags': tweet.hashtags,
                'mentions': tweet.mentions,
                'sentiment': tweet.sentiment,
                'sentiment_score': tweet.sentiment_score,
                'disaster_impact': tweet.disaster_impact,
                'disaster_type': tweet.disaster_type,
                'lat': tweet.lat,
                'lon': tweet.lon
            })
        
        df = pd.DataFrame(data)
        logger.info(f"Retrieved {len(df)} tweets from database")
        return df
        
    except Exception as e:
        logger.error(f"Error retrieving tweets from database: {e}")
        return pd.DataFrame()
        
    finally:
        session.close()

def get_tweet_count(disaster_type=None, time_range=None):
    """
    Get count of tweets in database with optional filtering
    
    Args:
        disaster_type (str, optional): Filter by disaster type
        time_range (tuple, optional): Filter by time range (start, end)
        
    Returns:
        int: Count of tweets
    """
    session = Session()
    
    try:
        query = session.query(Tweet)
        
        # Apply filters
        if disaster_type and disaster_type != "All":
            query = query.filter(Tweet.disaster_type == disaster_type)
            
        if time_range:
            start_time, end_time = time_range
            if start_time:
                query = query.filter(Tweet.created_at >= start_time)
            if end_time:
                query = query.filter(Tweet.created_at <= end_time)
        
        # Count tweets
        count = query.count()
        return count
        
    except Exception as e:
        logger.error(f"Error counting tweets in database: {e}")
        return 0
        
    finally:
        session.close()

def clear_old_tweets(days=30):
    """
    Delete tweets older than specified number of days
    
    Args:
        days (int): Number of days to keep
        
    Returns:
        int: Number of tweets deleted
    """
    session = Session()
    
    try:
        # Calculate cutoff date
        cutoff_date = datetime.now() - pd.Timedelta(days=days)
        
        # Delete tweets older than cutoff
        result = session.query(Tweet).filter(Tweet.created_at < cutoff_date).delete()
        session.commit()
        
        logger.info(f"Deleted {result} tweets older than {days} days")
        return result
        
    except Exception as e:
        session.rollback()
        logger.error(f"Error deleting old tweets: {e}")
        return 0
        
    finally:
        session.close()

# Initialize database on module import
init_db()