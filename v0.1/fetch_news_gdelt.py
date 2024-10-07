import json
import time
import requests
import pandas as pd
from datetime import datetime, timedelta

# Function to fetch GDELT data for a specific date range with retries and handling rate limits (HTTP 429)
def fetch_gdelt_data_with_retry(start_date, end_date, retries=3, delay=5):
    base_url = 'https://api.gdeltproject.org/api/v2/doc/doc'
    
    # Format the query
    query_params = {
        'query': 'AAPL',
        'mode': 'artlist',
        'format': 'json',
        'startdatetime': start_date.strftime('%Y%m%d%H%M%S'),
        'enddatetime': end_date.strftime('%Y%m%d%H%M%S'),
    }
    
    for attempt in range(retries):
        try:
            # Send the request to GDELT
            response = requests.get(base_url, params=query_params)
            
            # If the server returns a 429 status, wait longer and retry
            if response.status_code == 429:
                print(f"Rate limit hit (429). Retrying after {delay} seconds...")
                time.sleep(delay)  # Wait before retrying
                delay *= 2  # Exponential backoff (increase the wait time with each attempt)
                continue  # Retry the request
            
            # Check if the request was successful
            if response.status_code == 200:
                try:
                    # Try to parse the response as JSON
                    return response.json().get('articles', [])
                except json.JSONDecodeError:
                    print(f"Error decoding JSON on attempt {attempt + 1} for {start_date} to {end_date}")
                    print("Response content (partial):", response.text[:500])  # Log first 500 chars
                    return []
            else:
                print(f"Error fetching data: {response.status_code}")
                return []
        
        except Exception as e:
            print(f"Error on attempt {attempt + 1}: {e}")
        
        # Wait before retrying
        time.sleep(delay)
    
    return []

# Function to iterate over the date range and fetch data in smaller chunks
def fetch_gdelt_data_over_time(start_date, end_date, days_per_chunk=7):
    all_articles = []
    error_dates = []
    
    # Loop through the date range in chunks
    current_start = start_date
    while current_start < end_date:
        current_end = min(current_start + timedelta(days=days_per_chunk), end_date)
        
        # Fetch the data for the current chunk using retry logic
        articles = fetch_gdelt_data_with_retry(current_start, current_end, retries=3, delay=10)  # Start with a 10-second delay
        
        # Check if any articles were returned
        if articles:
            all_articles.extend(articles)
            print(f"Fetched {len(articles)} articles from {current_start} to {current_end}")
        else:
            print(f"No articles fetched from {current_start} to {current_end}")
            error_dates.append((current_start, current_end))
        
        # Move to the next chunk
        current_start = current_end
    
    # Log date ranges where errors occurred
    if error_dates:
        print("Errors occurred for the following date ranges:")
        for start, end in error_dates:
            print(f"From {start} to {end}")
    
    return all_articles

# Define the overall date range
start_date = datetime(2017, 1, 1)
end_date = datetime(2024, 8, 31)

# Fetch data over the date range (chunking by 7 days)
all_articles = fetch_gdelt_data_over_time(start_date, end_date)

# Save the articles to a JSON file
filename = 'gdelt_aapl_news_articles.json'
with open(filename, 'w') as json_file:
    json.dump(all_articles, json_file, indent=4)

# Convert JSON to CSV
# Load the JSON data from the file
with open('gdelt_aapl_news_articles.json', 'r') as json_file:
    all_articles = json.load(json_file)

# Convert the articles into a DataFrame
df_sentiment = pd.DataFrame(all_articles)

# Format the date and select relevant columns
df_sentiment['published_date'] = pd.to_datetime(df_sentiment['seendate'])
df_sentiment = df_sentiment[['title', 'url', 'published_date', 'domain']]

# Save the DataFrame to a CSV file
df_sentiment.to_csv('gdelt_aapl_news_articles.csv', index=False)