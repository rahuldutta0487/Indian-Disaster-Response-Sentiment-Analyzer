import tweepy
import time
import logging
from datetime import datetime

class TwitterAPI:
    def __init__(self, api_key, api_secret, access_token, access_token_secret, bearer_token):
        """
        Initialize the Twitter API client with credentials.
        
        Args:
            api_key (str): Twitter API key
            api_secret (str): Twitter API secret
            access_token (str): Twitter access token
            access_token_secret (str): Twitter access token secret
            bearer_token (str): Twitter bearer token for v2 API
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.access_token = access_token
        self.access_token_secret = access_token_secret
        self.bearer_token = bearer_token
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Initialize API clients
        self._init_api()
    
    def _init_api(self):
        """Initialize both v1.1 and v2 Twitter API clients."""
        try:
            # V1.1 API (for compatibility with some functions)
            auth = tweepy.OAuth1UserHandler(
                self.api_key, 
                self.api_secret, 
                self.access_token, 
                self.access_token_secret
            )
            self.api_v1 = tweepy.API(auth)
            
            # V2 API (for newer features)
            self.client = tweepy.Client(
                bearer_token=self.bearer_token,
                consumer_key=self.api_key,
                consumer_secret=self.api_secret,
                access_token=self.access_token,
                access_token_secret=self.access_token_secret
            )
            
            self.logger.info("Twitter API initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing Twitter API: {e}")
            self.api_v1 = None
            self.client = None
    
    def search_tweets(self, keywords, count=100, lang="en"):
        """
        Search for tweets containing specific keywords.
        
        Args:
            keywords (list): List of keywords or phrases to search for
            count (int): Maximum number of tweets to return
            lang (str): Language filter for tweets (default: English)
            
        Returns:
            list: List of tweet objects
        """
        if not self.client:
            self.logger.error("Twitter API client not initialized")
            return []
        
        try:
            tweets = []
            
            # Split keywords into smaller batches to avoid query length issues
            if isinstance(keywords, list):
                # If we have many keywords, split them into smaller groups
                if len(keywords) > 10:
                    # Create batches of 5 keywords each
                    keyword_batches = [keywords[i:i+5] for i in range(0, min(len(keywords), 30), 5)]
                else:
                    keyword_batches = [keywords]
            else:
                # If it's a string, just use it as is
                keyword_batches = [[keywords]]
            
            # Search for each batch to avoid rate limits and query length issues
            for batch in keyword_batches:
                if len(tweets) >= count:
                    break
                    
                # Create query from keywords in this batch
                query = " OR ".join(batch[:5])  # Limit to 5 keywords per query
                
                # Add language filter
                query += f" lang:{lang}"
                
                # Add filter to exclude retweets for better quality
                query += " -is:retweet"
                
                try:
                    # We use the v2 API with expanded tweet fields
                    response = self.client.search_recent_tweets(
                        query=query,
                        max_results=min(count - len(tweets), 25),  # API limit is 100 per request, but use smaller batches
                        tweet_fields=["created_at", "public_metrics", "geo", "entities"],
                        user_fields=["name", "username", "location"],
                        expansions=["author_id", "geo.place_id"]
                    )
                    
                    if not response or not response.data:
                        self.logger.warning(f"No tweets found for query batch: {query}")
                        continue
                    
                    # Extract users to a dictionary for easy lookup
                    users = {}
                    if response.includes and "users" in response.includes:
                        for user in response.includes["users"]:
                            users[user.id] = user
                    
                    # Process tweets
                    for tweet in response.data:
                        tweet_dict = tweet.data
                        
                        # Add user information
                        if hasattr(tweet, "author_id") and tweet.author_id in users:
                            user = users[tweet.author_id]
                            tweet_dict["user"] = {
                                "id": user.id,
                                "name": user.name,
                                "username": user.username,
                                "location": user.location if hasattr(user, "location") else None
                            }
                        
                        tweets.append(tweet_dict)
                    
                    self.logger.info(f"Retrieved {len(response.data)} tweets for query batch: {query}")
                    
                    # Sleep a bit to avoid rate limits
                    time.sleep(2)
                    
                except Exception as e:
                    self.logger.error(f"Error searching tweets for batch: {e}")
                    # Continue with next batch instead of failing completely
                    time.sleep(5)  # Sleep longer on error
                    continue
            
            self.logger.info(f"Total tweets retrieved: {len(tweets)}")
            return tweets
            
        except Exception as e:
            self.logger.error(f"Error in search_tweets: {e}")
            return []
    
    def get_trends(self, woeid=1):
        """
        Get trending topics from Twitter.
        
        Args:
            woeid (int): The Yahoo! Where On Earth ID of the location to get trends for.
                         Default is 1 which is worldwide.
        
        Returns:
            list: List of trending topics
        """
        if not self.api_v1:
            self.logger.error("Twitter API v1 client not initialized")
            return []
        
        try:
            trends = self.api_v1.get_place_trends(woeid)
            
            if trends and len(trends) > 0:
                return trends[0]["trends"]
            else:
                return []
                
        except Exception as e:
            self.logger.error(f"Error getting trends: {e}")
            return []
    
    def stream_tweets(self, keywords, callback, time_limit=60):
        """
        Stream tweets in real-time based on keywords.
        This is a simplified version that runs for a specified time limit.
        
        Args:
            keywords (list): List of keywords to track
            callback (function): Function to call for each tweet
            time_limit (int): Time limit in seconds
        """
        class TweetListener(tweepy.StreamingClient):
            def __init__(self, bearer_token, callback_func, **kwargs):
                super().__init__(bearer_token, **kwargs)
                self.callback_func = callback_func
                self.start_time = time.time()
                self.time_limit = time_limit
            
            def on_tweet(self, tweet):
                if time.time() - self.start_time > self.time_limit:
                    self.disconnect()
                    return
                
                self.callback_func(tweet)
            
            def on_error(self, status):
                if status == 420:  # Rate limit
                    return False
        
        try:
            stream = TweetListener(self.bearer_token, callback)
            
            # Delete existing rules
            existing_rules = stream.get_rules()
            if existing_rules.data:
                rule_ids = [rule.id for rule in existing_rules.data]
                stream.delete_rules(rule_ids)
            
            # Add new rules based on keywords
            if isinstance(keywords, list):
                for keyword in keywords:
                    stream.add_rules(tweepy.StreamRule(keyword))
            else:
                stream.add_rules(tweepy.StreamRule(keywords))
            
            # Start streaming
            stream.filter(
                tweet_fields=["created_at", "public_metrics", "geo", "entities"],
                user_fields=["name", "username", "location", "profile_image_url"],
                expansions=["author_id", "geo.place_id"]
            )
            
        except Exception as e:
            self.logger.error(f"Error streaming tweets: {e}")
