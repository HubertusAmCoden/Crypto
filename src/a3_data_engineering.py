import pandas as pd
import warnings
# Suppress FutureWarning messages
warnings.simplefilter(action='ignore', category=FutureWarning)

def calculate_relative_price_to_btc(df):
    df_price_btc = df.loc[df.coin == 'BTC/USD',['day', 'price']]
    df_price_btc.rename(columns={'price': 'price_btc'}, inplace=True)
    df = df.merge(df_price_btc, on='day', how='left')
    df['price_relative_to_btc'] = df['price'] / df['price_btc']
    return df

def weighted_average(df, col_price, colname_new, window_length = 5,col_group = 'coin', col_sort = 'timestamp'):
    # weighted by volume
    df['weighted_price'] = df[col_price] * df['volume']
    # colname output
    colname_new = colname_new + str(window_length) 
    # calculation
    df.sort_values(by = col_sort, inplace = True)
    df['nominator'] = df.groupby(col_group)['weighted_price'].transform(lambda d: d.rolling(window=window_length, min_periods=1, center=False).sum())
    df['denominator'] = df.groupby(col_group)['volume'].transform(lambda d: d.rolling(window=window_length, min_periods=1, center=False).sum())
    df[colname_new] = df['nominator'] / df['denominator']
    # drop the weighted price
    df.drop(columns = ['weighted_price','nominator','denominator'], inplace = True)
    return df

def define_local_min_max(df, col_price, col_group = 'coin', period_window_min_max = 30, suffix = '', col_sort = 'timestamp'):
    # Defining "local maxima/minima"
    col_max = 'local_max' + suffix
    col_min = 'local_min' + suffix
    df.sort_values(by = col_sort, inplace = True)
    df[col_max] = df.groupby(col_group)[col_price].transform(lambda d: d.rolling(window=period_window_min_max, min_periods=period_window_min_max, center=True).max())
    df[col_max] = df[[col_price, col_max]].apply(lambda x: 1* (x[0] == x[1]), axis=1)
    df[col_min] = df.groupby(col_group)['avg_price_5'].transform(lambda d: d.rolling(window=period_window_min_max, min_periods=period_window_min_max, center=True).min())
    df[col_min] = df[[col_price, col_min]].apply(lambda x: 1* (x[0] == x[1]), axis=1)
    
    return df

def define_local_min_max2(df, col_price_min = 'low', col_price_max = 'high', col_group = 'coin', period_window_min_max = 30, suffix = '', col_sort = 'timestamp'):
    # Defining "local maxima/minima"
    col_max = 'local_max' + suffix
    col_min = 'local_min' + suffix
    df.sort_values(by = col_sort, inplace = True)
    df[col_max] = df.groupby(col_group)[col_price_max].transform(lambda d: d.rolling(window=period_window_min_max, min_periods=period_window_min_max, center=True).max())
    df[col_max] = df[[col_price_max, col_max]].apply(lambda x: 1* (x[0] == x[1]), axis=1)
    df[col_min] = df.groupby(col_group)[col_price_min].transform(lambda d: d.rolling(window=period_window_min_max, min_periods=period_window_min_max, center=True).min())
    df[col_min] = df[[col_price_min, col_min]].apply(lambda x: 1* (x[0] == x[1]), axis=1)
    
    return df

def aggregate_support_levels(df, col_min, col_max, col_price, col_group, thres = 0.025, suffix = ''):
    df_cluster = pd.DataFrame()
    # define column names for each cluster (level: 1,2,3,...), the price of the min/max, and the number of hits to that min/max
    col_cluster = 'level' + suffix
    col_cluster_price = 'price_level' + suffix
    col_cluster_counter = 'counter_level' + suffix
    # filter the local min/max
    df_locals = df.loc[(df[col_min] == 1) | (df[col_max] == 1) ,['day', col_group, col_price, col_min, col_max]]
    # loop over each coin
    for g in df_locals[col_group].unique():
        # we iterate through local min/max by price and then define clusters of similar prices to define support/resistance levels
        df_g = df_locals.loc[df_locals[col_group] == g].sort_values(col_price).reset_index(drop = True)
        # initialize each row as an individual cluster
        df_g[col_cluster] = range(1, df_g.shape[0]+1)
        # iterate through the rows to define the clusters
        for i in range(df_g.shape[0]-1):
            v_i = df_g.iloc[i][col_price]
            v_i2 = df_g.iloc[i+1][col_price]
            if abs(v_i-v_i2) / v_i <= thres:
                df_g.loc[i+1, col_cluster] = df_g.loc[i, col_cluster]
        df_g[col_cluster_price] = df_g.groupby([col_cluster])[col_price].transform(lambda x: x.median())
        df_g[col_cluster_counter] = df_g.groupby([col_cluster])[col_price].transform(lambda x: x.count())
        df_cluster = pd.concat([df_cluster, df_g])
        df_cluster = df_cluster[['day', col_group, col_cluster, col_cluster_price, col_cluster_counter]]
    df_result = df.merge(df_cluster, on = ['day', col_group], how = 'left')
    return df_result

def calculate_momentum(df, col_new, col_price1, col_price2, col_price3, col_price4):
    
    df['momentum_1_2'] = df[[col_price1, col_price2]].apply(lambda x: (x[0] - x[1]) / x[1], axis=1)
    df['momentum_2_3'] = df[[col_price2, col_price3]].apply(lambda x: (x[0] - x[1]) / x[1], axis=1)
    df['momentum_3_4'] = df[[col_price3, col_price4]].apply(lambda x: (x[0] - x[1]) / x[1], axis=1)
    
    cols_momentum = ['momentum_1_2', 'momentum_2_3', 'momentum_3_4']
    df[col_new] = df[cols_momentum].sum(axis=1)
    df.drop(cols_momentum, axis=1, inplace=True)
    
    return df

def find_level(row, df_levels, direction = 'below', n = 1, col_find = 'price_level'):
        try:
            if direction == 'below':
                return df_levels.loc[(df_levels['price_level'] <= row['price']) & (df_levels['coin'] == row['coin'])].tail(n)[col_find].values[0]
            elif direction == 'above':
                return df_levels.loc[(df_levels['price_level'] >= row['price']) & (df_levels['coin'] == row['coin'])].head(n)[col_find].values[0]
        except:
            return row[col_find]