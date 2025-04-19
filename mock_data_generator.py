"""
This module provides mock data generation for disaster-related tweets.
It's used when Twitter API access is limited or unavailable.
"""

import pandas as pd
import random
import uuid
from datetime import datetime, timedelta
import numpy as np
from sentiment_analyzer import analyze_sentiment, analyze_disaster_impact

# Sample usernames
USERNAMES = [
    "DisasterAlert", "WeatherWatcher", "StormChaser", "EmergencyInfo", "SafetyFirst",
    "CrisisResponse", "WeatherChannel", "DisasterRelief", "EmergencyUpdate", "NewsFeed",
    "WeatherUpdates", "StormTracker", "DisasterMonitor", "EmergencyServices", "FirstResponder",
    "ReliefWorker", "WeatherForecast", "DisasterRecovery", "EmergencyNotice", "SafetyTips"
]

# Sample locations in India
LOCATIONS = [
    "Mumbai, Maharashtra", "Delhi, NCR", "Bangalore, Karnataka", "Chennai, Tamil Nadu", 
    "Kolkata, West Bengal", "Hyderabad, Telangana", "Pune, Maharashtra", "Ahmedabad, Gujarat", 
    "Jaipur, Rajasthan", "Surat, Gujarat", "Lucknow, Uttar Pradesh", "Kanpur, Uttar Pradesh", 
    "Nagpur, Maharashtra", "Patna, Bihar", "Indore, Madhya Pradesh", "Thane, Maharashtra", 
    "Bhopal, Madhya Pradesh", "Visakhapatnam, Andhra Pradesh", "Vadodara, Gujarat", 
    "Ghaziabad, Uttar Pradesh", "Ludhiana, Punjab", "Agra, Uttar Pradesh", "Kochi, Kerala",
    "Bhubaneswar, Odisha", "Coimbatore, Tamil Nadu", "Guwahati, Assam"
]

# Sample coordinates (lat, lon) for major Indian cities
COORDINATES = {
    "Mumbai, Maharashtra": (19.0760, 72.8777),
    "Delhi, NCR": (28.6139, 77.2090),
    "Bangalore, Karnataka": (12.9716, 77.5946),
    "Chennai, Tamil Nadu": (13.0827, 80.2707),
    "Kolkata, West Bengal": (22.5726, 88.3639),
    "Hyderabad, Telangana": (17.3850, 78.4867),
    "Pune, Maharashtra": (18.5204, 73.8567),
    "Ahmedabad, Gujarat": (23.0225, 72.5714),
    "Jaipur, Rajasthan": (26.9124, 75.7873),
    "Surat, Gujarat": (21.1702, 72.8311),
    "Lucknow, Uttar Pradesh": (26.8467, 80.9462),
    "Kanpur, Uttar Pradesh": (26.4499, 80.3319),
    "Nagpur, Maharashtra": (21.1458, 79.0882),
    "Patna, Bihar": (25.5941, 85.1376),
    "Indore, Madhya Pradesh": (22.7196, 75.8577),
    "Thane, Maharashtra": (19.2183, 72.9781),
    "Bhopal, Madhya Pradesh": (23.2599, 77.4126),
    "Visakhapatnam, Andhra Pradesh": (17.6868, 83.2185),
    "Vadodara, Gujarat": (22.3072, 73.1812),
    "Ghaziabad, Uttar Pradesh": (28.6692, 77.4538),
    "Ludhiana, Punjab": (30.9010, 75.8573),
    "Agra, Uttar Pradesh": (27.1767, 78.0081),
    "Kochi, Kerala": (9.9312, 76.2673),
    "Bhubaneswar, Odisha": (20.2961, 85.8245),
    "Coimbatore, Tamil Nadu": (11.0168, 76.9558),
    "Guwahati, Assam": (26.1445, 91.7362)
}

# Dictionary for tweet templates by disaster type
TWEET_TEMPLATES = {
    "Cyclone": [
        "Cyclone {name} is approaching the coast of {location}. Stay safe everyone! #Cyclone #StaySafe",
        "Wind speeds of {wind_speed} kmph reported as Cyclone {name} makes landfall in {location}. #WeatherAlert",
        "Evacuations underway in {location} as Cyclone {name} strengthens to Category {category}. #Cyclone",
        "Storm surge expected to reach {surge_height} meters in {location} due to Cyclone {name}. #StormSurge",
        "Cyclone {name} has been downgraded to a deep depression near {location}. Stay cautious. #CycloneRecovery",
        "Flooding reported in {location} after Cyclone {name} passed through. #FloodWarning",
        "Power outages affecting {outage_count} homes in {location} due to Cyclone {name}. #PowerOutage"
    ],
    
    "Earthquake": [
        "Magnitude {magnitude} earthquake reported in {location}. #Earthquake #Breaking",
        "Aftershocks continue in {location} following yesterday's {magnitude} earthquake. #Aftershock",
        "Building damage reported in {location} after the {magnitude} earthquake. #EarthquakeDamage",
        "Rescue teams searching for survivors in {location} after the devastating earthquake. #RescueEfforts",
        "Tsunami warning issued for coastal {location} following offshore earthquake. #TsunamiWarning",
        "Earthquake of {magnitude} magnitude felt across {location}. No major damage reported. #Earthquake",
        "Seismic activity continues in {location} region. Experts monitoring closely. #SeismicActivity"
    ],
    
    "Flood": [
        "Flash flood warning for {location}. Seek higher ground immediately! #FlashFlood",
        "River levels rising rapidly in {location}. Flood stage expected by {time}. #FloodWarning",
        "Evacuation orders issued for low-lying areas in {location} due to flooding. #Evacuation",
        "Roads closed in {location} due to severe flooding. Avoid travel if possible. #RoadClosure",
        "Flood waters receding in {location}, but damage assessment still ongoing. #FloodRecovery",
        "Emergency shelters open in {location} for those displaced by flooding. #EmergencyShelter",
        "Monsoon flooding worsens in {location} as rainfall continues. #MonsoonAlert"
    ],
    
    "Landslide": [
        "Major landslide reported in {location}. Roads blocked and homes damaged. #Landslide",
        "Heavy rains trigger landslides in hilly areas around {location}. Exercise caution. #LandslideWarning",
        "Rescue operations underway after landslide in {location} traps residents. #RescueOps",
        "Several villages cut off after landslide blocks main road to {location}. #Isolated",
        "Landslide risk high in {location} due to continuous rainfall. Avoid hill travel. #HighAlert",
        "Geological team assessing landslide damage in {location}. Further slides possible. #GeologicalHazard",
        "Residents evacuated from hillside areas in {location} due to landslide risk. #Evacuation"
    ],
    
    "Heatwave": [
        "Severe heatwave continues in {location} with temperatures reaching {temperature}°C. #Heatwave",
        "Health alert issued for {location} as extreme temperatures forecast to continue. #HeatAlert",
        "Schools closed in {location} due to dangerous heat conditions. #SchoolClosure",
        "Heat stroke cases increasing in hospitals across {location}. Stay hydrated. #HeatAdvisory",
        "Temperature records broken in {location} as heatwave intensifies. #RecordHeat",
        "Cooling centers opened across {location} to provide relief from extreme heat. #CoolingCenters",
        "Power grid under stress in {location} due to increased air conditioning use. #PowerStrain"
    ],
    
    "Drought": [
        "Water rationing implemented in {location} as drought conditions worsen. #WaterCrisis",
        "Farmers in {location} reporting crop failures due to prolonged drought. #CropLoss",
        "Emergency water supplies being distributed in {location} communities. #WaterShortage",
        "Reservoir levels in {location} drop to {level}% of capacity amid continuing drought. #Drought",
        "Government declares drought emergency for {location} region. Relief measures announced. #DroughtEmergency",
        "Groundwater depletion reaching critical levels in {location}. Conservation essential. #WaterConservation",
        "Rainfall deficit in {location} now at {deficit}% below normal for the season. #RainfallDeficit"
    ]
}

# For mixing in the content, to have a variety of tweets
GENERAL_TWEETS = [
    "Emergency response teams deployed to {location} for {disaster} relief. #EmergencyResponse",
    "Latest update on the {disaster} in {location}: {status}. Stay tuned for more information. #DisasterUpdate",
    "Resources available for those affected by the {disaster} in {location}. Visit {website} for details. #DisasterRelief",
    "Our thoughts are with everyone affected by the {disaster} in {location}. #StaySafe",
    "Volunteers needed for {disaster} recovery efforts in {location}. #HelpNeeded",
    "Weather conditions improving in {location} following the {disaster}. #WeatherUpdate",
    "Road closures in effect around {location} due to {disaster}. Check local traffic updates. #TrafficAlert",
    "Schools closed in {location} tomorrow due to {disaster}. #SchoolClosure",
    "Remember to check on elderly neighbors during this {disaster} in {location}. #CommunitySupport",
    "Donation center for {disaster} victims open at {location} community center. #Donations"
]

# Indian Ocean Cyclone names
CYCLONE_NAMES = ["Amphan", "Nisarga", "Gati", "Nivar", "Burevi", "Tauktae", "Yaas", "Gulab", "Shaheen", "Jawad", 
                "Asani", "Karim", "Mandous", "Mocha", "Biparjoy", "Tej", "Hamoon", "Dana", "Fengal", "Remal", 
                "Asna", "Gagea", "Manahil", "Shakhti", "Montha", "Thianyot", "Sadang", "Matsa", "Guchol", "Prapiroon"]

# Hashtags by disaster type
HASHTAGS = {
    "Cyclone": ["Cyclone", "StormAlert", "WeatherWarning", "Evacuation", "StormSurge", "CycloneSeason", "StormPrep", "FloodWatch", "NDRF", "IMD"],
    "Earthquake": ["Earthquake", "Quake", "Seismic", "TsunamiWarning", "Aftershock", "EarthquakeSafety", "QuakeDamage", "SeismicActivity"],
    "Flood": ["Flood", "FloodWarning", "HighWater", "FlashFlood", "RisingWater", "FloodSafety", "RiverWatch", "EvacuationOrder", "MonsoonFloods"],
    "Landslide": ["Landslide", "MudSlide", "LandslipAlert", "HillCollapse", "GeologicalHazard", "EvacuationOrder", "HillsideDanger", "RainfallWarning"],
    "Heatwave": ["Heatwave", "TemperatureAlert", "HeatAdvisory", "IMDAlert", "ExtremeHeat", "HeatStroke", "HydrationAlert", "HealthEmergency"],
    "Drought": ["Drought", "WaterCrisis", "RainfallDeficit", "CropFailure", "WaterScarcity", "DroughtRelief", "WaterConservation", "FarmCrisis"]
}

# Status updates for general tweets
STATUS_UPDATES = [
    "emergency response ongoing", "situation stabilizing", "damage assessment in progress", 
    "evacuations continuing", "relief efforts underway", "conditions worsening", 
    "recovery beginning", "emergency teams on scene", "shelters at capacity", 
    "volunteers needed", "conditions improving"
]

def generate_mock_tweet(disaster_type, time_range=None):
    """Generate a mock tweet based on disaster type and time range."""
    # Set time range (default to last 7 days)
    if time_range is None:
        end_time = datetime.now()
        start_time = end_time - timedelta(days=7)
    else:
        start_time, end_time = time_range
        
    # Create mock tweet data
    tweet_time = random.uniform(start_time.timestamp(), end_time.timestamp())
    tweet_datetime = datetime.fromtimestamp(tweet_time)
    
    # Select a template
    if random.random() < 0.7:  # 70% chance of disaster-specific tweet
        if disaster_type == "All":
            selected_type = random.choice(list(TWEET_TEMPLATES.keys()))
            templates = TWEET_TEMPLATES[selected_type]
            hashtag_list = HASHTAGS[selected_type]
        else:
            templates = TWEET_TEMPLATES[disaster_type]
            hashtag_list = HASHTAGS[disaster_type]
    else:  # 30% chance of general tweet
        templates = GENERAL_TWEETS
        if disaster_type == "All":
            selected_type = random.choice(list(TWEET_TEMPLATES.keys()))
            hashtag_list = HASHTAGS[selected_type]
        else:
            hashtag_list = HASHTAGS[disaster_type]
    
    template = random.choice(templates)
    
    # Select a location and get coordinates
    location = random.choice(LOCATIONS)
    lat, lon = COORDINATES.get(location, (0, 0))
    
    # Add some randomness to coordinates (within ~5 miles)
    lat += random.uniform(-0.07, 0.07)
    lon += random.uniform(-0.07, 0.07)
    
    # Format the template with relevant info
    try:
        tweet_text = template.format(
            location=location,
            disaster=disaster_type if disaster_type != "All" else random.choice(list(TWEET_TEMPLATES.keys())),
            name=random.choice(CYCLONE_NAMES),
            wind_speed=random.randint(75, 180),
            category=random.randint(1, 5),
            surge_height=random.randint(3, 20),
            magnitude=round(random.uniform(4.0, 8.5), 1),
            time=tweet_datetime.strftime("%H:%M"),
            acres=random.randint(500, 50000),
            containment=random.randint(0, 100),
            wave_height=random.randint(1, 10),
            outage_count=f"{random.randint(1, 100)},000",
            status=random.choice(STATUS_UPDATES),
            website="www.ndrf.gov.in",
            temperature=random.randint(38, 49),
            level=random.randint(10, 75),
            deficit=random.randint(30, 80)
        )
    except Exception as e:
        # Fallback in case of formatting error
        print(f"Error formatting tweet: {e}")
        tweet_text = f"Disaster alert for {location}: {disaster_type} situation developing. Stay tuned for updates."
    
    # Add hashtags (1-3 random ones)
    hashtag_count = random.randint(1, 3)
    selected_hashtags = random.sample(hashtag_list, min(hashtag_count, len(hashtag_list)))
    hashtag_text = " " + " ".join([f"#{tag}" for tag in selected_hashtags])
    tweet_text += hashtag_text
    
    # Clean text for analysis
    clean_text = tweet_text.replace("#", " ")
    
    # Generate sentiment
    sentiment, sentiment_score = analyze_sentiment(clean_text)
    
    # Generate impact level
    impact_level = analyze_disaster_impact(clean_text)
    
    # Generate user info
    username = random.choice(USERNAMES)
    display_name = username
    if random.random() < 0.5:  # 50% chance of having a real name
        display_name = f"{random.choice(['Rahul', 'Aishwarya', 'Vikram', 'Priya', 'Amit', 'Meera', 'Arjun', 'Anjali', 'Rajesh', 'Sunita', 'Vijay', 'Divya', 'Sanjay', 'Neha', 'Anand', 'Pooja'])} {random.choice(['Sharma', 'Patel', 'Singh', 'Verma', 'Gupta', 'Kumar', 'Reddy', 'Rao', 'Shah', 'Agarwal', 'Joshi', 'Mehta', 'Iyer', 'Nair', 'Das', 'Chatterjee'])}"
    
    # Generate tweet metrics
    retweet_count = int(np.random.exponential(10))
    like_count = int(np.random.exponential(25))
    reply_count = int(np.random.exponential(5))
    
    # Create hashtags and mentions arrays
    hashtags = selected_hashtags
    mentions = []
    if random.random() < 0.3:  # 30% chance of mentioning someone
        mention_count = random.randint(1, 2)
        mentions = random.sample(USERNAMES, mention_count)
    
    # Create mock tweet object
    tweet = {
        "id": str(uuid.uuid4()),
        "text": tweet_text,
        "clean_text": clean_text,
        "created_at": tweet_datetime,
        "username": username,
        "display_name": display_name,
        "location": location,
        "retweet_count": retweet_count,
        "like_count": like_count,
        "reply_count": reply_count,
        "hashtags": hashtags,
        "mentions": mentions,
        "sentiment": sentiment,
        "sentiment_score": sentiment_score,
        "disaster_impact": impact_level,
        "disaster_type": disaster_type if disaster_type != "All" else random.choice(list(TWEET_TEMPLATES.keys())),
        "lat": lat,
        "lon": lon
    }
    
    return tweet

def generate_mock_tweets(count=100, disaster_type="All", time_range=None):
    """Generate a list of mock tweets for testing."""
    tweets = []
    for _ in range(count):
        tweet = generate_mock_tweet(disaster_type, time_range)
        tweets.append(tweet)
    
    # Convert to DataFrame
    df = pd.DataFrame(tweets)
    return df

def get_mock_tweet_trends(df):
    """Generate mock trends from the DataFrame of tweets."""
    tweets_count = len(df)
    trends = {
        "hashtags": {},
        "mentions": {},
        "terms": {},
        "phrases": [],
        "emerging_topics": []
    }
    
    # Process hashtags
    if "hashtags" in df.columns:
        all_hashtags = []
        for hashtags_list in df["hashtags"]:
            if isinstance(hashtags_list, list):
                all_hashtags.extend(hashtags_list)
        
        # Count occurrences
        for hashtag in all_hashtags:
            if hashtag in trends["hashtags"]:
                trends["hashtags"][hashtag] += 1
            else:
                trends["hashtags"][hashtag] = 1
    
    # Process mentions
    if "mentions" in df.columns:
        all_mentions = []
        for mentions_list in df["mentions"]:
            if isinstance(mentions_list, list):
                all_mentions.extend(mentions_list)
        
        # Count occurrences
        for mention in all_mentions:
            if mention in trends["mentions"]:
                trends["mentions"][mention] += 1
            else:
                trends["mentions"][mention] = 1
    
    # Generate mock terms
    common_terms = ["emergency", "disaster", "relief", "help", "evacuation", "damage", 
                    "warning", "alert", "safety", "shelter", "recovery", "response", 
                    "crisis", "impact", "flood", "fire", "storm", "earthquake", "tornado"]
    
    # Assign random counts to terms
    for term in common_terms:
        trends["terms"][term] = random.randint(1, max(1, tweets_count // 3))
    
    # Generate mock phrases
    phrases = ["stay safe", "emergency response", "evacuation order", "rescue teams", 
               "disaster relief", "immediate evacuation", "take shelter", "flash flood", 
               "weather update", "road closed"]
    
    # Add random phrase counts
    for phrase in phrases:
        trends["phrases"].append({"phrase": phrase, "count": random.randint(1, max(1, tweets_count // 5))})
    
    # Generate mock emerging topics
    emerging_topics = ["power outage", "shelter locations", "road closures", "volunteer coordination", 
                       "donation centers", "emergency contacts", "medical assistance", "pet rescue"]
    
    # Add random emerging topic scores
    for topic in emerging_topics:
        trends["emerging_topics"].append({"topic": topic, "score": random.random() * 10})
    
    return trends