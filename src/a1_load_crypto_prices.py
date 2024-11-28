import pandas as pd
import numpy as np
import ccxt
import pytz
from datetime import datetime, timezone, timedelta
import time

class CryptoPriceLoader:
    """
    A class for loading and preparing cryptocurrency price data.

    Attributes:
        symbols (list): A list of cryptocurrency symbols.
        timeframe (str): The timeframe for fetching the price data (default is '1h').
        days_ago (int): The number of days ago from which to start fetching the price data (default is 180).

    Methods:
        fetch_all_ohlcv(symbol, limit): Fetches the OHLCV (Open, High, Low, Close, Volume) data for a specific symbol.
        prep_ohlcv_df(fetched_ohlcv, coin_name): Prepares the fetched OHLCV data as a DataFrame.
        fetch_prepare_crypto_prices(): Fetches and prepares the cryptocurrency price data for all symbols.
    """

    def __init__(self, symbols, timeframe='1h', days_ago=180):
        """
        Initializes a CryptoPriceLoader object.

        Args:
            symbols (list): A list of cryptocurrency symbols.
            timeframe (str): The timeframe for fetching the price data (default is '1h').
            days_ago (int): The number of days ago from which to start fetching the price data (default is 180).
        """
        self.exchange = ccxt.coinbase()
        self.symbols = symbols
        self.timeframe = timeframe
        self.now = self.exchange.milliseconds()
        self.first_retrieval_date = self.now - days_ago * 24 * 60 * 60 * 1000  # days_ago in milliseconds
        self.berlin_tz = pytz.timezone('Europe/Berlin')
        self.now_dt = datetime.fromtimestamp(self.now / 1000, tz=timezone.utc).astimezone(self.berlin_tz)
        self.first_retrieval_date_dt = datetime.fromtimestamp(self.first_retrieval_date / 1000, tz=timezone.utc).astimezone(self.berlin_tz)

    def fetch_all_ohlcv(self, symbol, limit=60):
        """
        Fetches the OHLCV (Open, High, Low, Close, Volume) data for a specific symbol.

        Args:
            symbol (str): The cryptocurrency symbol.
            limit (int): The maximum number of data points to fetch (default is 300).

        Returns:
            list: A list of OHLCV data points.
        """
        all_ohlcv = []
        since = self.first_retrieval_date
        while since < self.now:
            ohlcv = self.exchange.fetch_ohlcv(symbol, self.timeframe, since=since, limit=limit)
            if not ohlcv:
                since = since + limit * 60 * 60 * 1000  # Move to the next timestamp
            else:
                all_ohlcv.extend(ohlcv)
                since = ohlcv[-1][0] + 1  # Move to the next timestamp
                time.sleep(1)  # Avoid hitting rate limits
        return all_ohlcv


    def basic_prep_ohlcv_df(self, fetched_ohlcv, coin_name, agg_level = ['coin', 'weekday', 'day']):
        df = pd.DataFrame(fetched_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['coin'] = coin_name
        df['weekday'] = df['timestamp'].dt.weekday
        df['day'] = df['timestamp'].dt.date
        df['hour'] = df['timestamp'].dt.hour

        for cc in ['open', 'high', 'low', 'close']:
            df[cc] = df[cc] * df['volume']

        df = df.groupby(agg_level, as_index = False)[['open', 'high', 'low', 'close', 'volume']].sum()

        for cc in ['open', 'high', 'low', 'close']:
            df[cc] = df[cc] / df['volume']

        # Rename 'open' to 'price'
        df['price'] = df[['open', 'close']].apply(lambda x: x.mean(), axis = 1)

        df['volume_usd'] = df['volume'] * df['price']
        
        return df

    def prep_ohlcv_df(self, fetched_ohlcv, coin_name, agg_level):
        """
        Prepares the fetched OHLCV data as a DataFrame.

        Args:
            fetched_ohlcv (list): A list of OHLCV data points.
            coin_name (str): The name of the cryptocurrency.

        Returns:
            pandas.DataFrame: The prepared DataFrame containing the OHLCV data.
        """
        df = self.basic_prep_ohlcv_df(fetched_ohlcv, coin_name, agg_level=agg_level)
        
        # Calculate price developments
        for hours in [1, 3, 6, 12, 18, 24, 48, 72, 168]:
            df[f'change_{hours}h'] = df['price'].pct_change(periods=hours) * 100
        # Calculate future price developments
        for hours in [3, 6, 12, 24, 48, 72, 168]:
            df[f'future_change_{hours}h'] = df['price'].shift(-hours).pct_change(periods=hours, fill_method=None) * 100

        # Calculate average volumes and volume volatility
        for hours in [6, 12, 24, 48, 72, 168, 336]:
            df[f'avg_volume_{hours}h'] = df['volume'].rolling(hours).mean()
            df[f'volatility_volume_{hours}h'] = df['volume'].rolling(hours).std()
            df[f'volatility_volume_{hours}h'] = df[f'volatility_volume_{hours}h'] / df[f'avg_volume_{hours}h']

        df['avg_volume_level_24_336'] = df['avg_volume_24h'] / df['avg_volume_336h']
        df['avg_volume_level_12_336'] = df['avg_volume_12h'] / df['avg_volume_336h']
        df['avg_volume_level_6_336'] = df['avg_volume_6h'] / df['avg_volume_336h']


        # Select and order relevant columns
        df_prep = df[['coin', 'timestamp', 'price', 'volume', 'volume_usd', 'weekday', 'day', 'hour'] +
                     [f'avg_volume_level_{hours}_336' for hours in [6, 12, 24]] +
                    [f'avg_volume_{hours}h' for hours in [6, 12, 24, 48, 72, 168, 336]] +
                    [f'volatility_volume_{hours}h' for hours in [6, 12, 24, 48, 72, 168, 336]] +
                     [f'change_{hours}h' for hours in [1, 3, 6, 12, 18, 24, 48, 72, 168]] +
                     [f'future_change_{hours}h' for hours in [3, 6, 12, 24, 48, 72, 168]]]
        return df_prep

    def fetch_prepare_crypto_prices(self, agg_level = ['coin', 'weekday', 'day'], prepare=True):
        """
        Fetches and prepares the cryptocurrency price data for all symbols.

        Returns:
            pandas.DataFrame: The prepared DataFrame containing the cryptocurrency price data.
        """
        df_prep = pd.DataFrame()
        for sym in self.symbols:
            try:
                print(sym)
                ohlcv_sym = self.fetch_all_ohlcv(sym)
                if prepare:
                    df_sym = self.prep_ohlcv_df(fetched_ohlcv=ohlcv_sym, coin_name=sym, agg_level=agg_level)
                else:
                    df_sym = self.basic_prep_ohlcv_df(fetched_ohlcv=ohlcv_sym, coin_name=sym, agg_level=agg_level)
                df_prep = pd.concat([df_prep, df_sym])
            except Exception as e:
                print(f'Error fetching {sym}: {e}')
        
        df_prep['day'] = pd.to_datetime(df_prep['day'])
        return df_prep

# Beispiel für die Verwendung der Klasse
if __name__ == "__main__":
    symbols = ['BTC/EUR', 'ETH/EUR', 'SOL/EUR']
    loader = CryptoPriceLoader(symbols=symbols)
    #df_prep = loader.fetch_prepare_crypto_prices()
    #print(df_prep.head())
