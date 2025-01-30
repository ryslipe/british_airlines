from datetime import datetime
import requests
from bs4 import BeautifulSoup
import pandas as pd
import sys
from pathlib import Path
import sys
import os

from src.paths import RAW_DATA_DIR

# function to scrape data
# Function to scrape the data
def scraper(base_url: str, start_page: int, end_page: int) -> pd.DataFrame:
    '''Scrape the website for all available reviews.'''
    reviews = []

    for page_num in range(start_page, end_page + 1):
        url = f"{base_url}/page/{page_num}/?sortby=post_date%3ADesc&pagesize=100"
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')

        review_elements = soup.find_all('article', itemprop='review')

        for element in review_elements:
            review_data = {}

            # Define the verification status based on the presence of specific text
            if "✅ Trip Verified" in element.get_text():
                review_data['verification_status'] = "Trip Verified"
            elif "✅ Verified Review" in element.get_text():
                review_data['verification_status'] = "Review Verified"
            else:
                review_data['verification_status'] = "Not Verified"

            # Clean review body text
            review_body = element.find('div', class_='text_content').get_text(strip=True)
            review_body = review_body.replace("✅Trip Verified|", "").replace("✅Verified Review|", "").replace("Not Verified|", "").strip()
            review_data['review_body'] = review_body

            # Extract published date
            published_date = element.find('time', itemprop='datePublished')['datetime']
            review_data['published_date'] = published_date

            # Extract rating
            rating_element = element.find('div', itemprop='reviewRating')
            if rating_element:
                rating_value = rating_element.find('span', itemprop='ratingValue').get_text(strip=True)
                best_rating = rating_element.find('span', itemprop='bestRating').get_text(strip=True)
                review_data['rating'] = f"{rating_value}/{best_rating}"

            # Extract additional data
            rows = element.find_all('tr')
            for row in rows:
                header = row.find('td', class_='review-rating-header')
                value = row.find('td', class_='review-value')
                if header and value:
                    review_data[header.get_text(strip=True)] = value.get_text(strip=True)

            reviews.append(review_data)

    reviews_df = pd.DataFrame(reviews)
    return reviews_df

# function to check for new reviews
def check_and_update_reviews(base_url: str, start_page: int, end_page: int, existing_reviews_df: pd.DataFrame, current_date: str) -> pd.DataFrame:
    '''Scrape the webpage for new reviews and update the DataFrame with unique reviews.'''
    # # use our scraper function but only scrape the first page
    new_reviews_df = scraper(base_url, start_page, end_page)
    
    # # should have a full page of reviews
    if new_reviews_df.empty:
        print("No reviews scraped or DataFrame is empty.")
        return existing_reviews_df

    # Combine new reviews with existing reviews
    combined_reviews_df = pd.concat([existing_reviews_df, new_reviews_df], ignore_index=True)
    
    # Sort by published_date in ascending order to keep the oldest reviews when we drop duplicates
    combined_reviews_df['published_date'] = pd.to_datetime(combined_reviews_df['published_date'])
    combined_reviews_df = combined_reviews_df.sort_values(by='published_date', ascending=True)

    # Identify and remove duplicates, keeping the first occurrence (oldest date)
    unique_reviews_df = combined_reviews_df.drop_duplicates(subset=['review_body'], keep='first')

    # Identify new unique reviews - this will be empty if there are no new reviews
    new_unique_reviews_df = unique_reviews_df[~unique_reviews_df['review_body'].isin(existing_reviews_df['review_body'])]

    if new_unique_reviews_df.empty:
        print("No new unique reviews found.")
        return existing_reviews_df
    else:
        print("New unique reviews found.")
        
        # Save new unique reviews to a new DataFrame with the current date in the filename
        new_reviews_filename = RAW_DATA_DIR / f'new_unique_reviews_{current_date}.parquet'
        new_unique_reviews_df.to_parquet(new_reviews_filename, index=False)
        print(f"New unique reviews saved to '{new_reviews_filename}'.")

        # Update existing reviews DataFrame by concatenating with new unique reviews and dropping duplicates
        updated_reviews_df = pd.concat([existing_reviews_df, new_unique_reviews_df], ignore_index=True).drop_duplicates(subset=['review_body'], keep='first')
        updated_reviews_filename = RAW_DATA_DIR / 'existing_reviews.parquet'
        updated_reviews_df.to_parquet(updated_reviews_filename, index=False)
        print("Existing reviews DataFrame updated.")
        
        return updated_reviews_df

# function to rename columns
def rename_cols(reviews_df: pd.DataFrame) -> pd.DataFrame:
    '''Change column names to snake case.'''
    return reviews_df.rename(columns={
        'Aircraft': 'aircraft',
        'Type Of Traveller': 'type_of_traveler', 
        'Seat Type': 'seat_type', 
        'Route': 'route', 
        'Date Flown': 'date_flown', 
        'Recommended': 'recommended'},
        inplace=True
)



