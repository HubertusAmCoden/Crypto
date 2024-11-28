# 0 Packages
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import pandas as pd

from textblob import TextBlob
import re
import nltk
from nltk.corpus import stopwords

#nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# 1 Class
class CryptoNewsLoader:
    # Initialisierung
    def __init__(self, cryptocurrencies = ['Bitcoin', 'Ether', 'Solana'], number_pages = 50):
        self.cryptocurrencies = cryptocurrencies
        self.number_pages = number_pages

    # Hilfsfunktionen
    def fetch_news(self, crypto_name):
        news = []
        base_url1 = "https://crypto.news/page/"
        base_url2 = f"/?s={crypto_name}"

        for page in range(1, self.number_pages + 1):
            if page%10 == 1:
                print(page)
            url = base_url1 + str(page) + base_url2
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
            }

            response = requests.get(url, headers=headers)
            if response.status_code != 200:
                print(f"Failed to retrieve news for {crypto_name} on page {page}")
                break

            soup = BeautifulSoup(response.content, 'html.parser')
            articles = soup.find_all('div', class_='search-result-loop__content')

            if not articles:
                # No more articles found, stop the loop
                break

            for article in articles:
                title_tag = article.find('a', class_='search-result-loop__link')
                title = title_tag.text.strip() if title_tag else 'No title'
                link = title_tag['href'] if title_tag else 'No link'
                abstract_tag = article.find('p')
                abstract = abstract_tag.text.strip() if abstract_tag else 'No abstract'
                published_time_tag = article.find('div', class_='search-result-loop__date')
                published_time = published_time_tag.text.strip() if published_time_tag else 'No date'

                # Convert time to readable format
                try:
                    published_datetime = datetime.strptime(published_time, "%B %d, %Y at %I:%M %p")
                    # add 2 hours due to different news page timezone
                    published_datetime = published_datetime + timedelta(hours=2)
                    published_day = published_datetime.strftime("%Y-%m-%d")
                    published_hour = published_datetime.hour
                except ValueError:
                    published_day = '-1'
                    published_hour = -1

                news.append({
                    'coin' : crypto_name,
                    'published_day': published_day,
                    'published_hour': published_hour,
                    'title': title,
                    'abstract': abstract
                })

        return news
    
    def print_news(self, news):
        for article in news:
            print(f"Title: {article['title']}")
            print(f"Link: {article['link']}")
            print(f"Abstract: {article['abstract']}")
            print(f"Published Day: {article['published_day']}")
            print(f"Published Hour: {article['published_hour']}")
            print("\n")

    def clean_text(self, text):
        text = text.lower()  # Convert to lowercase
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stop words
        return text
    
    def calculate_sentiment(self, df):
        df['cleaned_title'] = df['title'].apply(self.clean_text)
        df['cleaned_abstract'] = df['abstract'].apply(self.clean_text)
        df['title_sentiment'] = df['cleaned_title'].apply(lambda x: TextBlob(x).sentiment.polarity)
        df['abstract_sentiment'] = df['cleaned_abstract'].apply(lambda x: TextBlob(x).sentiment.polarity)
        return df
    
    def enlarge_dataframe(self, df):
        # Get the range of days and hours
        min_date = df['published_day'].min()#.date()
        max_date = df['published_day'].max()#.date()
        date_range = pd.date_range(start=min_date, end=max_date, freq='D')
        hours = range(24)
        
        # Create a complete grid of all combinations of cryptocurrencies, dates, and hours
        grid = pd.MultiIndex.from_product(
            [df['coin'].unique(), date_range, hours], 
            names=['coin', 'published_day', 'published_hour']
        ).to_frame(index=False)
        grid['published_day'] = grid['published_day'].apply(lambda x: str(x.date()))
        
        # Merge the grid with the original DataFrame
        enlarged_df = pd.merge(grid, df, how='left', on=['coin', 'published_day', 'published_hour'])
        
        return enlarged_df
    
    def rolling_count_sum(self, df, window_range = 12):
        df = df.sort_values(by=['coin', 'published_day', 'published_hour'])

        # Create a rolling count for each cryptocurrency
        name_count = 'news_count_' + str(window_range)
        name_sent = 'news_sentiment_' + str(window_range)
        df[name_count] = df.groupby('coin')['news_count'].rolling(window=window_range, min_periods=1).sum().reset_index(level=0, drop=True)
        df[name_sent] = df.groupby('coin')['news_sentiment'].rolling(window=window_range, min_periods=1).sum().reset_index(level=0, drop=True)

        return df
    
    # Hauptfunktion
    def prepare_crypto_news(self):
        # fetch all news
        all_news = []
        for crypto in self.cryptocurrencies:
            print(f"Fetching news for {crypto}...")
            news = self.fetch_news(crypto)
            all_news.extend(news)

        # create DataFrame
        df_all_news =  pd.DataFrame(all_news)
        df_all_news = df_all_news.loc[df_all_news.published_hour > -1]
        df_all_news.drop_duplicates(inplace=True)
        
        # calculate sentiment
        df_all_news_sent = self.calculate_sentiment(df_all_news)
        # aggregate to ensure that no duplicates per hour
        df_all_news_sent = df_all_news_sent.groupby(['coin', 'published_day', 'published_hour'], as_index = False)['abstract_sentiment'].agg(['count', 'sum'])
        df_all_news_sent.rename(columns = {'count' : 'news_count', 'sum' : 'news_sentiment'}, inplace = True)
        df_all_news_sent['news_count'] = df_all_news_sent['news_count'].fillna(0)
        df_all_news_sent['news_sentiment'] = df_all_news_sent['news_sentiment'].fillna(0)
        # get rolling metrics for last 12/24 hours 
        df_grid_news = self.enlarge_dataframe(df_all_news_sent)
        df_grid_news  = self.rolling_count_sum(df = df_grid_news, window_range=3)
        df_grid_news  = self.rolling_count_sum(df = df_grid_news, window_range=6)
        df_grid_news  = self.rolling_count_sum(df = df_grid_news, window_range=12)
        df_grid_news  = self.rolling_count_sum(df = df_grid_news, window_range=24)

        return df_grid_news
        