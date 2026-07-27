# download_stock.py
# Run: python download_stock.py
# This script downloads real stock data and saves as CSV file

try:
    import yfinance as yf
except:
    import subprocess, sys
    print("Installing yfinance library...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance"])
    import yfinance as yf

import pandas as pd
from datetime import datetime, timedelta

print("=" * 50)
print("  Real Stock Data Downloader")
print("=" * 50)

stocks = {
    "1": ("RELIANCE.NS", "Reliance Industries"),
    "2": ("TCS.NS",      "TCS"),
    "3": ("INFY.NS",     "Infosys"),
    "4": ("SBIN.NS",     "SBI Bank"),
    "5": ("HDFCBANK.NS", "HDFC Bank"),
    "6": ("WIPRO.NS",    "Wipro"),
}

print("\nSelect a stock:")
for k, (sym, name) in stocks.items():
    print(f"  {k}. {name} ({sym})")

choice        = input("\nEnter number (1-6): ").strip()
symbol, company = stocks.get(choice, ("RELIANCE.NS", "Reliance Industries"))

print(f"\nDownloading {company} ({symbol}) data...")
print("Period: Jan 2020 to Today (4 years)")

end   = datetime.today()
start = end - timedelta(days=365 * 4)

ticker = yf.Ticker(symbol)
df     = ticker.history(
    start = start.strftime("%Y-%m-%d"),
    end   = end.strftime("%Y-%m-%d")
)

if df.empty:
    print("No data found! Please check your internet connection.")
else:
    # Keep only Date and Close price columns
    df = df[["Close"]].reset_index()
    df.columns = ["Date", "Close"]
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
    df = df.dropna()

    # Round close price to 2 decimal places
    df["Close"] = df["Close"].round(2)

    filename = f"{symbol.replace('.', '_')}_stock_data.csv"
    df.to_csv(filename, index=False)

    print(f"\nSuccess! File saved: {filename}")
    print(f"Total trading days: {len(df)}")
    print(f"Date range : {df['Date'].iloc[0]}  to  {df['Date'].iloc[-1]}")
    print(f"Price range: Rs.{df['Close'].min():.2f}  to  Rs.{df['Close'].max():.2f}")
    print(f"\nNext steps:")
    print(f"  1. Run:  python app.py")
    print(f"  2. Open: http://localhost:5000")
    print(f"  3. Click UPLOAD tab and upload: {filename}")