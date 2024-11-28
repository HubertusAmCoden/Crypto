import seaborn as sns
import matplotlib.pyplot as plt

def plot_scatter(df, col_pred = 'predicted', coins=None):
    """
    Plots a scatter plot of actual vs predicted values for each coin in the given DataFrame.

    Parameters:
    - df (pandas.DataFrame): The DataFrame containing the data.
    - coins (list or None): The list of coins to plot. If None, all unique coins in the DataFrame will be plotted.

    Returns:
    None
    """
    if coins is None:
        coins = df['coin'].unique()
    for coin in coins:
        df_coin = df[df['coin'] == coin]
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df_coin, x=col_pred, y='actual', hue='coin', palette='viridis')
        plt.axline((0, 0), slope=1, color='gray', linestyle='--', linewidth=1, label='Reference Line (slope=1)')
        plt.axhline(0, color='gray', linestyle='--', linewidth=1)
        plt.axvline(0, color='gray', linestyle='--', linewidth=1)
        plt.title(f'Scatter Plot of Actual vs Predicted for {coin}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.legend(title='Coin')
        plt.show()
  