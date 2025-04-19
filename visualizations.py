import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
import logging

# Initialize logger
logger = logging.getLogger(__name__)

# Try to download stopwords if needed
try:
    nltk.download('stopwords', quiet=True)
    STOPWORDS = set(stopwords.words('english'))
except Exception as e:
    logger.warning(f"Failed to download NLTK stopwords: {e}")
    STOPWORDS = set()

# Add common Twitter terms and disaster-related terms to stopwords
TWITTER_STOPWORDS = {
    'rt', 'amp', 'http', 'https', 'co', 't.co', 'twitter', 'tweet',
    'retweet', 'disaster', 'emergency', 'breaking', 'news', 'update',
    'updates', 'reported', 'reports', 'just', 'says', 'via', 'today',
    'watch', 'watching', 'video', 'photo', 'photos', 'pictures', 'pic',
    'pics', 'live', 'happening', 'now', 'breaking'
}

# Combine stopwords
STOPWORDS = STOPWORDS.union(TWITTER_STOPWORDS)

def create_sentiment_chart(df):
    """
    Create an interactive time-based sentiment analysis chart with zoom and hover details.
    
    Args:
        df (pandas.DataFrame): DataFrame containing tweet data with sentiment
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    if df.empty or 'created_at' not in df.columns or 'sentiment' not in df.columns:
        # Return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for sentiment analysis",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14)
        )
        return fig
    
    # Group by hour and sentiment - creating a more detailed time series for better zoom functionality
    df_copy = df.copy()
    df_copy['hour'] = df_copy['created_at'].dt.floor('h')  # Using 'h' instead of 'H' to avoid FutureWarning
    
    # Also group by 15-minute intervals for a more detailed view when zoomed in
    df_copy['minute_15'] = df_copy['created_at'].dt.floor('15min')
    
    # Create two DataFrames: one for hourly display (when zoomed out) and one for detailed view (when zoomed in)
    hourly_counts = df_copy.groupby(['hour', 'sentiment']).size().reset_index(name='count')
    detailed_counts = df_copy.groupby(['minute_15', 'sentiment']).size().reset_index(name='count')
    
    # Process tweet text information for hover details
    # Using a simpler approach to avoid the ambiguous truth value error
    df_copy['short_text'] = df_copy['text'].apply(
        lambda x: (x[:70] + '...') if len(x) > 70 else x
    )
    
    # Just prepare count data by hour and sentiment - we'll use customdata for hover info
    tweet_info = df_copy.groupby(['hour', 'sentiment']).size().reset_index(name='tweet_count')
    
    # Add sentiment percentages for hover information
    total_by_hour = df_copy.groupby('hour').size().reset_index(name='total')
    sentiment_pcts = df_copy.groupby(['hour', 'sentiment']).size().reset_index(name='count')
    sentiment_pcts = sentiment_pcts.merge(total_by_hour, on='hour')
    sentiment_pcts['percentage'] = (sentiment_pcts['count'] / sentiment_pcts['total'] * 100).round(1)
    
    # Pivot the data for plotting
    pivot_df = hourly_counts.pivot(index='hour', columns='sentiment', values='count').fillna(0)
    pivot_df = pivot_df.reset_index()
    
    # Ensure all sentiment categories exist
    for sentiment in ['positive', 'negative', 'neutral']:
        if sentiment not in pivot_df.columns:
            pivot_df[sentiment] = 0
    
    # Add the percentage data to the pivot DataFrame for hover text
    positive_pcts = sentiment_pcts[sentiment_pcts['sentiment'] == 'positive']
    neutral_pcts = sentiment_pcts[sentiment_pcts['sentiment'] == 'neutral']
    negative_pcts = sentiment_pcts[sentiment_pcts['sentiment'] == 'negative']
    
    # Create the stacked area chart with enhanced interactivity
    fig = go.Figure()
    
    # Add positive sentiment with hover text
    positive_hover = []
    for i, row in positive_pcts.iterrows():
        time_str = row['hour'].strftime('%Y-%m-%d %H:%M')
        hover_text = f"Time: {time_str}<br>Positive tweets: {int(row['count'])}<br>Percentage: {row['percentage']}%"
        positive_hover.append(hover_text)
    
    fig.add_trace(go.Scatter(
        x=pivot_df['hour'],
        y=pivot_df['positive'],
        mode='lines',
        stackgroup='one',
        name='Positive',
        line=dict(width=1, color='rgba(0, 128, 0, 0.8)'),
        fillcolor='rgba(0, 128, 0, 0.4)',
        customdata=positive_pcts['percentage'] if not positive_pcts.empty else [],
        hovertemplate='<b>Positive</b><br>Count: %{y}<br>Percentage: %{customdata:.1f}%<extra></extra>'
    ))
    
    # Add neutral sentiment with hover text
    neutral_hover = []
    for i, row in neutral_pcts.iterrows():
        time_str = row['hour'].strftime('%Y-%m-%d %H:%M')
        hover_text = f"Time: {time_str}<br>Neutral tweets: {int(row['count'])}<br>Percentage: {row['percentage']}%"
        neutral_hover.append(hover_text)
        
    fig.add_trace(go.Scatter(
        x=pivot_df['hour'],
        y=pivot_df['neutral'],
        mode='lines',
        stackgroup='one',
        name='Neutral',
        line=dict(width=1, color='rgba(128, 128, 128, 0.8)'),
        fillcolor='rgba(128, 128, 128, 0.4)',
        customdata=neutral_pcts['percentage'] if not neutral_pcts.empty else [],
        hovertemplate='<b>Neutral</b><br>Count: %{y}<br>Percentage: %{customdata:.1f}%<extra></extra>'
    ))
    
    # Add negative sentiment with hover text
    negative_hover = []
    for i, row in negative_pcts.iterrows():
        time_str = row['hour'].strftime('%Y-%m-%d %H:%M')
        hover_text = f"Time: {time_str}<br>Negative tweets: {int(row['count'])}<br>Percentage: {row['percentage']}%"
        negative_hover.append(hover_text)
        
    fig.add_trace(go.Scatter(
        x=pivot_df['hour'],
        y=pivot_df['negative'],
        mode='lines',
        stackgroup='one',
        name='Negative',
        line=dict(width=1, color='rgba(255, 0, 0, 0.8)'),
        fillcolor='rgba(255, 0, 0, 0.4)',
        customdata=negative_pcts['percentage'] if not negative_pcts.empty else [],
        hovertemplate='<b>Negative</b><br>Count: %{y}<br>Percentage: %{customdata:.1f}%<extra></extra>'
    ))
    
    # Update layout with interactive elements
    fig.update_layout(
        title='Interactive Sentiment Analysis Timeline',
        xaxis_title='Time',
        yaxis_title='Tweet Count',
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        # Add range slider and selector for time navigation
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1h", step="hour", stepmode="backward"),
                    dict(count=6, label="6h", step="hour", stepmode="backward"),
                    dict(count=12, label="12h", step="hour", stepmode="backward"),
                    dict(count=1, label="1d", step="day", stepmode="backward"),
                    dict(count=7, label="1w", step="day", stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(visible=True),
            type="date"
        )
    )
    
    return fig

def create_tweet_volume_chart(df):
    """
    Create an interactive chart showing tweet volume over time with zoom and hover details.
    
    Args:
        df (pandas.DataFrame): DataFrame containing tweet data
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    if df.empty or 'created_at' not in df.columns:
        # Return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for tweet volume analysis",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14)
        )
        return fig
    
    # Create multiple time aggregations for different zoom levels
    df_copy = df.copy()
    df_copy['hour'] = df_copy['created_at'].dt.floor('h')  # Using 'h' instead of 'H' to avoid FutureWarning
    df_copy['minute_15'] = df_copy['created_at'].dt.floor('15min')
    df_copy['minute_5'] = df_copy['created_at'].dt.floor('5min')
    
    # Create aggregations at different time scales
    hourly_volume = df_copy.groupby('hour').size().reset_index(name='count')
    detailed_volume_15min = df_copy.groupby('minute_15').size().reset_index(name='count')
    detailed_volume_5min = df_copy.groupby('minute_5').size().reset_index(name='count')
    
    # Add disaster type information if available
    if 'disaster_type' in df_copy.columns:
        # Calculate top disaster types for each hour
        disaster_info = df_copy.groupby(['hour', 'disaster_type']).size().reset_index(name='type_count')
        top_disasters = disaster_info.sort_values(['hour', 'type_count'], ascending=[True, False])
        top_disasters = top_disasters.groupby('hour').first().reset_index()
        
        # Merge with hourly volume
        hourly_volume = hourly_volume.merge(top_disasters[['hour', 'disaster_type', 'type_count']], on='hour', how='left')
        hourly_volume['disaster_pct'] = (hourly_volume['type_count'] / hourly_volume['count'] * 100).round(1)
    
    # Create the volume chart with enhanced interactivity
    fig = go.Figure()
    
    # Add main volume line with hover information
    fig.add_trace(go.Scatter(
        x=hourly_volume['hour'],
        y=hourly_volume['count'],
        mode='lines+markers',
        name='Tweet Volume',
        line=dict(width=2, color='rgba(0, 128, 255, 0.8)'),
        marker=dict(size=8, color='rgba(0, 128, 255, 0.8)'),
        hovertemplate='<b>Time</b>: %{x}<br><b>Tweets</b>: %{y}' + 
                      ('<br><b>Top Disaster</b>: %{customdata[0]} (%{customdata[1]}%)' 
                       if 'disaster_type' in hourly_volume.columns else '') +
                      '<extra></extra>',
        customdata=hourly_volume[['disaster_type', 'disaster_pct']].values if 'disaster_type' in hourly_volume.columns else None
    ))
    
    # Add annotation for peak volume
    if not hourly_volume.empty:
        peak_hour = hourly_volume.loc[hourly_volume['count'].idxmax()]
        fig.add_annotation(
            x=peak_hour['hour'],
            y=peak_hour['count'],
            text=f"Peak: {peak_hour['count']} tweets",
            showarrow=True,
            arrowhead=1,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="#636363",
            ax=20,
            ay=-30,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="#c7c7c7"
        )
    
    # Update layout with interactive elements
    fig.update_layout(
        title='Interactive Tweet Volume Timeline',
        xaxis_title='Time',
        yaxis_title='Tweet Count',
        hovermode='x unified',
        # Add range slider and selector for time navigation
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1h", step="hour", stepmode="backward"),
                    dict(count=6, label="6h", step="hour", stepmode="backward"),
                    dict(count=12, label="12h", step="hour", stepmode="backward"),
                    dict(count=1, label="1d", step="day", stepmode="backward"),
                    dict(count=7, label="1w", step="day", stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(visible=True),
            type="date"
        ),
        # Add a shaded background for visual appeal
        plot_bgcolor='rgba(240, 240, 240, 0.8)'
    )
    
    return fig

def create_word_cloud(df, column='text', max_words=100):
    """
    Create a word cloud visualization from tweet text.
    
    Args:
        df (pandas.DataFrame): DataFrame containing tweet data
        column (str): Column containing text to analyze
        max_words (int): Maximum number of words to include
        
    Returns:
        matplotlib.figure.Figure: Matplotlib figure with word cloud
    """
    if df.empty or column not in df.columns:
        # Return empty figure with message
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No data available for word cloud generation',
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=14)
        ax.axis('off')
        return fig
    
    # Combine all text
    text = ' '.join(df[column].dropna().astype(str).tolist())
    
    if not text.strip():
        # Return empty figure if no text
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No text content available for word cloud',
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=14)
        ax.axis('off')
        return fig
    
    # Define color map for word cloud (blue gradient)
    colors = [(0.12, 0.47, 0.71), (0.2, 0.6, 0.86), (0.4, 0.7, 0.96)]
    cmap = LinearSegmentedColormap.from_list('TwitterBlue', colors)
    
    # Generate word cloud
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        max_words=max_words,
        stopwords=STOPWORDS,
        colormap=cmap,
        contour_width=1,
        contour_color='steelblue'
    ).generate(text)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    plt.tight_layout()
    
    return fig

def create_location_map(df):
    """
    Create a map visualization of tweet locations.
    
    Args:
        df (pandas.DataFrame): DataFrame containing tweet data with location info
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure with map
    """
    # Check if we have location data
    if df.empty or 'location' not in df.columns:
        # Return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No location data available for mapping",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14)
        )
        return fig
    
    # For this example, we'll create a simplified map with random points
    # In a real application, you would geocode the locations
    
    # Filter to tweets with location
    location_df = df[df['location'].notna() & (df['location'] != '')].copy()
    
    if location_df.empty:
        # Return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No location data available for mapping",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14)
        )
        return fig
    
    # Use geocoded coordinates if available, otherwise use coordinates within India
    import random
    
    # Check if lat and lon columns exist and have values
    if 'lat' not in location_df.columns or 'lon' not in location_df.columns or \
       location_df['lat'].isna().all() or location_df['lon'].isna().all():
        # Use coordinates within India if not available
        location_df['lat'] = [random.uniform(8, 35) for _ in range(len(location_df))]  # India latitude range
        location_df['lon'] = [random.uniform(68, 97) for _ in range(len(location_df))]  # India longitude range
    
    # Set marker colors based on sentiment
    colors = {
        'positive': 'green',
        'neutral': 'gray',
        'negative': 'red'
    }
    
    location_df['color'] = location_df['sentiment'].map(colors)
    
    # Create map centered on India
    fig = px.scatter_mapbox(
        location_df,
        lat='lat',
        lon='lon',
        hover_name='username',
        hover_data=['text', 'sentiment', 'created_at'],
        color='sentiment',
        color_discrete_map=colors,
        zoom=4,  # Adjusted zoom level for India
        center={"lat": 20.5937, "lon": 78.9629},  # Center coordinates of India
        height=600,
        mapbox_style="carto-positron"
    )
    
    # Update layout
    fig.update_layout(
        title='Tweet Locations Across India',
        margin={"r":0,"t":50,"l":0,"b":0},
        legend_title_text='Sentiment'
    )
    
    return fig

def create_heatmap(df):
    """
    Create a heatmap visualization showing tweet activity by day and hour.
    
    Args:
        df (pandas.DataFrame): DataFrame containing tweet data with timestamps
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure with heatmap
    """
    if df.empty or 'created_at' not in df.columns:
        # Return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for heatmap visualization",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14)
        )
        return fig
    
    # Extract day of week and hour from timestamp
    df_copy = df.copy()
    df_copy['day_of_week'] = df_copy['created_at'].dt.day_name()
    df_copy['hour_of_day'] = df_copy['created_at'].dt.hour
    
    # Define day order (Monday first)
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    # Count tweets by day and hour
    heatmap_data = df_copy.groupby(['day_of_week', 'hour_of_day']).size().reset_index(name='count')
    
    # Create pivot table
    pivot_table = heatmap_data.pivot(index='day_of_week', columns='hour_of_day', values='count').fillna(0)
    
    # Reorder days to start with Monday
    pivot_table = pivot_table.reindex(day_order)
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=pivot_table.values,
        x=pivot_table.columns,
        y=pivot_table.index,
        colorscale='YlOrRd',  # Yellow-Orange-Red color scale
        hoverongaps=False,
        hovertemplate='Day: %{y}<br>Hour: %{x}:00<br>Tweets: %{z}<extra></extra>'
    ))
    
    # Add sentiment breakdown in hover if available
    if 'sentiment' in df_copy.columns:
        # Prepare sentiment breakdown data
        sentiment_data = {}
        for day in day_order:
            sentiment_data[day] = {}
            for hour in range(24):
                day_hour_data = df_copy[(df_copy['day_of_week'] == day) & (df_copy['hour_of_day'] == hour)]
                if not day_hour_data.empty:
                    sentiment_counts = day_hour_data['sentiment'].value_counts()
                    total = len(day_hour_data)
                    positive_pct = round((sentiment_counts.get('positive', 0) / total) * 100, 1)
                    negative_pct = round((sentiment_counts.get('negative', 0) / total) * 100, 1)
                    neutral_pct = round((sentiment_counts.get('neutral', 0) / total) * 100, 1)
                    
                    sentiment_data[day][hour] = {
                        'positive': positive_pct,
                        'negative': negative_pct,
                        'neutral': neutral_pct
                    }
    
    # Update layout
    fig.update_layout(
        title='Tweet Activity Heatmap by Day and Hour',
        xaxis_title='Hour of Day',
        yaxis_title='Day of Week',
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(24)),
            ticktext=[f"{h}:00" for h in range(24)]
        ),
        height=500,
        margin=dict(l=50, r=20, t=50, b=50)
    )
    
    return fig

def create_impact_chart(df):
    """
    Create a chart showing disaster impact levels from tweets.
    
    Args:
        df (pandas.DataFrame): DataFrame containing tweet data with impact analysis
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    if df.empty or 'disaster_impact' not in df.columns:
        # Return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No impact data available for analysis",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14)
        )
        return fig
    
    # Count impact levels
    impact_counts = df['disaster_impact'].value_counts().reset_index()
    impact_counts.columns = ['impact', 'count']
    
    # Define order and colors
    impact_order = ['severe', 'moderate', 'minor', 'unknown']
    impact_colors = {
        'severe': 'rgba(255, 0, 0, 0.7)',
        'moderate': 'rgba(255, 165, 0, 0.7)',
        'minor': 'rgba(255, 255, 0, 0.7)',
        'unknown': 'rgba(128, 128, 128, 0.7)'
    }
    
    # Filter and sort by impact level
    impact_counts = impact_counts[impact_counts['impact'].isin(impact_order)]
    impact_counts['impact'] = pd.Categorical(impact_counts['impact'], categories=impact_order, ordered=True)
    impact_counts = impact_counts.sort_values('impact')
    
    # Create the bar chart
    fig = px.bar(
        impact_counts,
        x='impact',
        y='count',
        color='impact',
        color_discrete_map=impact_colors,
        title='Disaster Impact Levels from Tweets',
        labels={'impact': 'Impact Level', 'count': 'Tweet Count'}
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title='Impact Level',
        yaxis_title='Tweet Count',
        showlegend=False
    )
    
    return fig
