import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import asyncio
import aiohttp

# Your Alpha Vantage API Key
ALPHA_VANTAGE_API_KEY = 'IKZ6G28VB3K76Q9R'

async def fetch_stock_data(symbol='AAPL'):
    url = 'https://www.alphavantage.co/query'
    params = {
        'function': 'TIME_SERIES_INTRADAY',
        'symbol': symbol,
        'interval': '1min',
        'apikey': ALPHA_VANTAGE_API_KEY
    }
    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params) as response:
            data = await response.json()
    return data

async def fetch_crypto_data(symbol='bitcoin'):
    url = f'https://api.coingecko.com/api/v3/coins/{symbol}/market_chart'
    params = {
        'vs_currency': 'usd',
        'days': '1',
        'interval': 'minute'
    }
    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params) as response:
            data = await response.json()
    return data

def process_stock_data(data):
    time_series = data.get('Time Series (1min)', {})
    if not time_series:
        return pd.DataFrame(), pd.DataFrame()

    dates = []
    prices = []

    for date, info in sorted(time_series.items()):
        dates.append(pd.to_datetime(date))
        prices.append(float(info['1. open']))

    df = pd.DataFrame({'date': dates, 'price': prices})
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)

    df['day_num'] = (df.index - df.index[0]).days
    X = df[['day_num']]
    y = df['price']
    model = LinearRegression()
    model.fit(X, y)
    
    future_days = np.array([(datetime.now() - df.index[0]).days + i for i in range(1, 8)])
    future_prices = model.predict(future_days.reshape(-1, 1))
    
    future_dates = [datetime.now() + timedelta(days=i) for i in range(1, 8)]
    future_df = pd.DataFrame({'date': future_dates, 'predicted_price': future_prices})

    return df, future_df

def process_crypto_data(data):
    prices = data.get('prices', [])
    if not prices:
        return pd.DataFrame(), pd.DataFrame()

    dates = [pd.to_datetime(price[0], unit='ms') for price in prices]
    prices = [price[1] for price in prices]

    df = pd.DataFrame({'date': dates, 'price': prices})
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)

    df['day_num'] = (df.index - df.index[0]).days
    X = df[['day_num']]
    y = df['price']
    model = LinearRegression()
    model.fit(X, y)
    
    future_days = np.array([(datetime.now() - df.index[0]).days + i for i in range(1, 8)])
    future_prices = model.predict(future_days.reshape(-1, 1))
    
    future_dates = [datetime.now() + timedelta(days=i) for i in range(1, 8)]
    future_df = pd.DataFrame({'date': future_dates, 'predicted_price': future_prices})

    return df, future_df

def update_graph():
    global current_symbol, current_data_type, ax, canvas, root

    try:
        if current_data_type == 'Stock':
            data = asyncio.run(fetch_stock_data(current_symbol))
            df, future_df = process_stock_data(data)
        else:
            data = asyncio.run(fetch_crypto_data(current_symbol))
            df, future_df = process_crypto_data(data)

        ax.clear()
        ax.plot(df.index, df['price'], label='Historical Prices', color='blue', linestyle='-')
        ax.plot(future_df['date'], future_df['predicted_price'], label='Predicted Prices', linestyle='--', color='red')

        ax.set_title(f'{current_symbol} Price History and Prediction', fontsize=14, fontweight='bold', color='green')
        ax.set_xlabel('Date', fontsize=12, fontweight='bold', color='black')
        ax.set_ylabel('Price (USD)', fontsize=12, fontweight='bold', color='black')
        ax.legend()
        ax.grid(True)

        # Ensure layout is updated
        ax.figure.tight_layout()
        canvas.draw()

    except Exception as e:
        print(f"Error: {e}")

    root.after(update_interval, update_graph)

def on_symbol_change(*args):
    global current_symbol, ax, canvas

    current_symbol = symbol_var.get()
    ax.clear()  # Clear previous plot
    update_graph()

def on_data_type_change(*args):
    global current_data_type
    current_data_type = data_type_var.get()
    on_symbol_change()  # Trigger symbol change to refresh data

def create_gui():
    global symbol_var, data_type_var, graph_frame, root, ax, canvas, update_interval, current_symbol, current_data_type

    root = tk.Tk()
    root.title("Real-Time Price Tracker")
    root.configure(bg='white')

    current_symbol = 'bitcoin'
    current_data_type = 'Crypto'
    
    symbol_var = tk.StringVar(value=current_symbol)
    data_type_var = tk.StringVar(value=current_data_type)
    
    tk.Label(root, text="Select Data Type:", bg='white', fg='green', font=('Arial', 14, 'bold')).pack(pady=10)

    data_type_options = ['Stock', 'Crypto']
    data_type_menu = tk.OptionMenu(root, data_type_var, *data_type_options, command=on_data_type_change)
    data_type_menu.config(bg='green', fg='white', font=('Arial', 12))
    data_type_menu.pack(pady=10)

    tk.Label(root, text="Select Symbol:", bg='white', fg='green', font=('Arial', 14, 'bold')).pack(pady=10)

    symbol_options = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'bitcoin', 'ethereum', 'dogecoin']
    symbol_menu = tk.OptionMenu(root, symbol_var, *symbol_options, command=on_symbol_change)
    symbol_menu.config(bg='green', fg='white', font=('Arial', 12))
    symbol_menu.pack(pady=10)

    update_interval = 60000  # Update interval in milliseconds (60000 ms = 1 minute)

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('white')
    graph_frame = tk.Frame(root, bg='white')
    graph_frame.pack(pady=20, fill=tk.BOTH, expand=True)

    canvas = FigureCanvasTkAgg(fig, master=graph_frame)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    update_graph()  # Initial fetch and plot

    root.mainloop()

if __name__ == "__main__":
    create_gui()
