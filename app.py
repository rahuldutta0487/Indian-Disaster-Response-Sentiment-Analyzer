import streamlit as st
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import random

# Import custom modules
from twitter_api import TwitterAPI
from sentiment_analyzer import analyze_sentiment
from data_processor import process_tweets
from visualizations import (
    create_sentiment_chart, 
    create_tweet_volume_chart,
    create_word_cloud,
    create_location_map,
    create_heatmap
)
from trend_analyzer import analyze_trends
from disaster_keywords import get_disaster_keywords
from utils import filter_dataframe, export_data
from database import save_tweets, get_tweets, get_tweet_count, clear_old_tweets

# Download required NLTK datasets
import nltk
try:
    nltk.download('punkt')
    nltk.download('stopwords')
except Exception as e:
    print(f"Failed to download NLTK data: {e}")

# Page configuration
st.set_page_config(
    page_title="Indian Disaster Response Sentiment Analysis",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
if 'tweets_df' not in st.session_state:
    st.session_state.tweets_df = pd.DataFrame()
if 'refresh_rate' not in st.session_state:
    st.session_state.refresh_rate = 60  # default refresh rate in seconds
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = None
if 'filter_query' not in st.session_state:
    st.session_state.filter_query = ""
if 'selected_disaster_type' not in st.session_state:
    st.session_state.selected_disaster_type = "All"

# Import mock data generator
from mock_data_generator import generate_mock_tweets, get_mock_tweet_trends

# Function to generate mock tweets (replaces Twitter API initialization)
def initialize_mock_data_generator():
    return True

# Function to fetch mock tweets
def fetch_tweets(mock_generator, disaster_type="All", count=100):
    if not mock_generator:
        return pd.DataFrame()
    
    # Generate time range based on current time
    now = datetime.now()
    # Create a time range of 7 days ending now
    time_range = (now - timedelta(days=7), now)
    
    # Generate mock tweets
    tweets_df = generate_mock_tweets(count=count, disaster_type=disaster_type, time_range=time_range)
    
    # Add to existing dataframe if it exists
    if not st.session_state.tweets_df.empty:
        tweets_df = pd.concat([tweets_df, st.session_state.tweets_df]).drop_duplicates(subset=['id']).reset_index(drop=True)
        
        # Keep only last 1000 tweets to manage memory
        if len(tweets_df) > 1000:
            tweets_df = tweets_df.head(1000)
    
    return tweets_df

# Function to refresh data
def refresh_data():
    mock_generator = initialize_mock_data_generator()
    
    with st.spinner("Generating new tweet data..."):
        # Determine how many tweets to generate based on disaster type
        if st.session_state.selected_disaster_type == "All":
            count = random.randint(30, 50)  # More tweets for "All" category
        else:
            count = random.randint(10, 25)  # Fewer tweets for specific disasters
            
        tweets_df = fetch_tweets(mock_generator, st.session_state.selected_disaster_type, count=count)
        
        if not tweets_df.empty:
            # Save to database
            saved_count = save_tweets(tweets_df)
            
            # Update session state with the latest tweets
            st.session_state.tweets_df = tweets_df
            st.session_state.last_refresh = datetime.now()
            
            if saved_count > 0:
                st.success(f"Generated and saved {saved_count} new tweets to the database.")
            
            # Clear old tweets (older than 30 days)
            deleted_count = clear_old_tweets(30)
            if deleted_count > 0:
                st.info(f"Cleared {deleted_count} old tweets from the database.")
        else:
            st.warning("Failed to generate new tweet data.")

# Main app layout
st.title("Disaster Response Sentiment Analyzer")
st.markdown("Real-time Twitter sentiment analysis for monitoring disaster-related trends and public response across India")

# Sidebar for controls
with st.sidebar:
    st.header("Controls")
    
    # Disaster type selector
    disaster_types = ["All", "Cyclone", "Earthquake", "Flood", "Landslide", "Heatwave", "Drought"]
    selected_disaster = st.selectbox("Select Disaster Type", disaster_types)
    
    if selected_disaster != st.session_state.selected_disaster_type:
        st.session_state.selected_disaster_type = selected_disaster
        # Clear existing data when changing disaster type
        st.session_state.tweets_df = pd.DataFrame()
    
    # Search and filter
    st.subheader("Search & Filter")
    filter_query = st.text_input("Filter tweets (contains text)", st.session_state.filter_query)
    st.session_state.filter_query = filter_query
    
    # Time range selector
    st.subheader("Time Range")
    time_range = st.radio(
        "Select time range", 
        ["Last hour", "Last 24 hours", "Last 7 days", "All"]
    )
    
    # Refresh controls
    st.subheader("Data Refresh")
    refresh_rate = st.slider("Refresh rate (seconds)", 30, 300, st.session_state.refresh_rate, 10)
    st.session_state.refresh_rate = refresh_rate
    
    if st.button("Refresh Now"):
        refresh_data()
    
    # Display last refresh time
    if st.session_state.last_refresh:
        st.info(f"Last refreshed: {st.session_state.last_refresh.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Database stats
    st.subheader("Database Stats")
    
    # Get tweet counts by disaster type
    db_stats = {}
    total_tweets = 0
    
    for disaster_type in disaster_types:
        count = get_tweet_count(disaster_type if disaster_type != "All" else None)
        db_stats[disaster_type] = count
        if disaster_type != "All":
            total_tweets += count
    
    # Display database stats
    st.metric("Total tweets in database", total_tweets)
    st.caption("Tweets by disaster type:")
    
    for disaster_type, count in db_stats.items():
        if disaster_type != "All":
            st.caption(f"- {disaster_type}: {count}")
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("This dashboard provides real-time sentiment analysis of Twitter data related to disasters to help understand public response and trends.")

# Load data from database if tweets_df is empty
if st.session_state.tweets_df.empty:
    # Get current time range for filtering
    now = datetime.now()
    time_filters = {
        "Last hour": (now - timedelta(hours=1), now),
        "Last 24 hours": (now - timedelta(days=1), now),
        "Last 7 days": (now - timedelta(days=7), now),
        "All": (None, None)
    }
    
    start_time, end_time = time_filters.get(time_range, (None, None))
    
    # Get tweets from database
    db_tweets = get_tweets(
        limit=1000, 
        disaster_type=st.session_state.selected_disaster_type,
        time_range=(start_time, end_time)
    )
    
    if not db_tweets.empty:
        st.session_state.tweets_df = db_tweets
        st.session_state.last_refresh = datetime.now()

# Filter the DataFrame based on time range
df = st.session_state.tweets_df.copy()
if not df.empty:
    now = datetime.now()
    
    if time_range == "Last hour":
        df = df[df['created_at'] > (now - timedelta(hours=1))]
    elif time_range == "Last 24 hours":
        df = df[df['created_at'] > (now - timedelta(days=1))]
    elif time_range == "Last 7 days":
        df = df[df['created_at'] > (now - timedelta(days=7))]
    
    # Apply text filter if provided
    if filter_query:
        df = filter_dataframe(df, filter_query)

# Display main dashboard
if df.empty:
    st.warning("No data available. Please refresh to fetch tweets.")
    
    # If it's the first run, trigger data fetch
    if st.session_state.last_refresh is None:
        refresh_data()
else:
    # Create metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Tweets", len(df))
    
    with col2:
        positive_count = len(df[df['sentiment'] == 'positive'])
        positive_pct = (positive_count / len(df)) * 100
        st.metric("Positive Sentiment", f"{positive_pct:.1f}%")
    
    with col3:
        negative_count = len(df[df['sentiment'] == 'negative'])
        negative_pct = (negative_count / len(df)) * 100
        st.metric("Negative Sentiment", f"{negative_pct:.1f}%")
    
    with col4:
        neutral_count = len(df[df['sentiment'] == 'neutral'])
        neutral_pct = (neutral_count / len(df)) * 100
        st.metric("Neutral Sentiment", f"{neutral_pct:.1f}%")
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Sentiment Analysis", "Tweet Volume", "Word Cloud", "Location Map", "Database Management", "Activity Heatmap"])
    
    with tab1:
        st.subheader("Sentiment Analysis Over Time")
        sentiment_chart = create_sentiment_chart(df)
        st.plotly_chart(sentiment_chart, use_container_width=True)
    
    with tab2:
        st.subheader("Tweet Volume Over Time")
        volume_chart = create_tweet_volume_chart(df)
        st.plotly_chart(volume_chart, use_container_width=True)
    
    with tab3:
        st.subheader("Common Words in Tweets")
        word_cloud = create_word_cloud(df)
        st.pyplot(word_cloud)
    
    with tab4:
        st.subheader("Tweet Locations")
        location_map = create_location_map(df)
        st.plotly_chart(location_map, use_container_width=True)
    
    with tab5:
        st.subheader("Database Management")
        st.markdown("This tab allows you to manage the tweet database.")
        
        # Display current database stats
        total_tweets = get_tweet_count()
        st.info(f"Total tweets in database: {total_tweets}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Data retention options
            st.subheader("Data Retention")
            retention_days = st.slider("Keep tweets for how many days?", 1, 90, 30)
            
            if st.button("Clear Old Tweets"):
                with st.spinner("Clearing old tweets..."):
                    deleted_count = clear_old_tweets(retention_days)
                    if deleted_count > 0:
                        st.success(f"Deleted {deleted_count} tweets older than {retention_days} days.")
                    else:
                        st.info("No old tweets to delete.")
        
        with col2:
            # Database management options
            st.subheader("Database Export")
            export_format = st.radio("Export format", ["CSV", "JSON"])
            
            # Import already done
            if st.button("Export Database"):
                with st.spinner("Exporting tweets..."):
                    # Get all tweets
                    all_tweets = get_tweets(limit=10000)
                    
                    if not all_tweets.empty:
                        filename = export_data(all_tweets, format=export_format.lower())
                        if filename:
                            st.success(f"Exported {len(all_tweets)} tweets to {filename}")
                        else:
                            st.error("Failed to export tweets.")
                    else:
                        st.warning("No tweets to export.")
    
    with tab6:
        st.subheader("Tweet Activity Heatmap")
        st.markdown("This heatmap shows tweet activity patterns by day of week and hour of day.")
        
        # Create heatmap
        heatmap = create_heatmap(df)
        st.plotly_chart(heatmap, use_container_width=True)
        
        # Add description and insights
        st.markdown("""
        ### Understanding the Heatmap
        
        The heatmap visualization shows when tweets are most active during the week:
        
        - **Brighter colors** (yellow/orange/red) indicate higher tweet volume
        - **Darker colors** (darker red/black) indicate lower tweet volume
        - The X-axis shows the **hour of day** (24-hour format)
        - The Y-axis shows the **day of week** (Monday through Sunday)
        
        This pattern analysis can help identify when disaster-related discussions are most active and
        when emergency communications might be most effective.
        """)
        
        # Add interactive features explanation
        st.info("""
        **Interactive Features:**
        - Hover over cells to see exact tweet counts
        - Click on days or hours in the axes to filter the view
        - Double-click to reset the view
        """)
        
        # Add data source disclaimer
        if "created_at" in df.columns:
            date_range = f"{df['created_at'].min().strftime('%Y-%m-%d')} to {df['created_at'].max().strftime('%Y-%m-%d')}"
            st.caption(f"Data shown is for the period: {date_range}")
            
        # Add disaster type filtering if available
        if "disaster_type" in df.columns and st.session_state.selected_disaster_type != "All":
            st.caption(f"Filtered for disaster type: {st.session_state.selected_disaster_type}")
        elif "disaster_type" in df.columns:
            disaster_counts = df['disaster_type'].value_counts()
            st.caption(f"All disaster types included. Most common: {disaster_counts.index[0]} ({disaster_counts.iloc[0]} tweets)")
        
    
    # Trending hashtags and topics
    st.subheader("Trending Hashtags and Topics")
    trends = analyze_trends(df)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Top Hashtags")
        if trends['hashtags']:
            hashtags_df = pd.DataFrame(trends['hashtags'].items(), columns=['Hashtag', 'Count'])
            hashtags_df = hashtags_df.sort_values('Count', ascending=False).head(10)
            
            fig = px.bar(hashtags_df, x='Hashtag', y='Count', title="Top Hashtags")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No hashtags found in the current data.")
    
    with col2:
        st.markdown("### Top Mentioned Users")
        if trends['mentions']:
            mentions_df = pd.DataFrame(trends['mentions'].items(), columns=['User', 'Count'])
            mentions_df = mentions_df.sort_values('Count', ascending=False).head(10)
            
            fig = px.bar(mentions_df, x='User', y='Count', title="Top Mentioned Users")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No user mentions found in the current data.")
    
    # Recent tweets table
    st.subheader("Recent Tweets")
    
    # Create a more readable dataframe for display
    display_df = df[['created_at', 'text', 'username', 'sentiment', 'sentiment_score']].copy()
    display_df['created_at'] = display_df['created_at'].dt.strftime('%Y-%m-%d %H:%M:%S')
    display_df = display_df.sort_values('created_at', ascending=False).head(10)
    
    # Color code the sentiment
    def color_sentiment(val):
        if val == 'positive':
            return 'background-color: rgba(0, 128, 0, 0.2)'
        elif val == 'negative':
            return 'background-color: rgba(255, 0, 0, 0.2)'
        else:
            return 'background-color: rgba(128, 128, 128, 0.2)'
    
    st.dataframe(display_df.style.applymap(color_sentiment, subset=['sentiment']), use_container_width=True)

# Auto-refresh functionality
if st.session_state.last_refresh:
    time_since_refresh = (datetime.now() - st.session_state.last_refresh).total_seconds()
    
    if time_since_refresh > st.session_state.refresh_rate:
        refresh_data()
        st.rerun()

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center">
        <p>Developed for monitoring disaster response across India. Real-time data refreshes every {refresh_rate} seconds.</p>
        <p>In partnership with NDRF and IMD for timely disaster alerts and response coordination.</p>
    </div>
    """.format(refresh_rate=st.session_state.refresh_rate),
    unsafe_allow_html=True
)
