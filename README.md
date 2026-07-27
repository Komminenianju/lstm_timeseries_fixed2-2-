# 🧠 LSTM Time Series Predictor
### Complete Step-by-Step Guide for Absolute Beginners

---

## 📖 WHAT IS THIS PROJECT?

This project uses an **LSTM (Long Short-Term Memory)** neural network to:
- **Learn patterns** from historical time-series data (stock prices, temperatures, etc.)
- **Predict future values** based on what it learned
- **Visualize everything** in a beautiful dark-themed web dashboard

---

## 🧠 CONCEPTS EXPLAINED (Plain English)

### What is a Time Series?
A sequence of values measured over time. Examples:
- Stock price every day
- Temperature every hour
- Sales every week

### What is an LSTM?
An LSTM is a special type of neural network that has **memory**.
Unlike a regular neural network that processes one input at a time,
an LSTM can remember patterns across many time steps.

Think of it like this:
> "If it rained the last 5 days, it might rain tomorrow too."
> LSTM learns these kinds of patterns automatically!

### What is a Look-Back Window?
The number of past time steps the model sees before making one prediction.
- Look-back = 30 means: "Use the last 30 days to predict day 31"

### What is Training?
Showing the model thousands of examples so it adjusts its internal
weights (numbers) to minimize prediction error. The model gets better
with each epoch (full pass through the data).

### Key Terms:
| Term     | Meaning |
|----------|---------|
| Epoch    | One complete pass through all training data |
| Loss     | How wrong the model is (lower = better) |
| MAE      | Average absolute prediction error |
| RMSE     | Penalizes large errors more than MAE |
| MAPE     | Error as a percentage of actual values |
| Dropout  | Randomly turns off neurons to prevent overfitting |
| Scaler   | Normalizes data to 0-1 range for stable training |

---

## 📁 PROJECT STRUCTURE

```
lstm_timeseries/
│
├── app.py                  ← Flask web server (backend brain)
├── requirements.txt        ← Python packages to install
├── sample_data.csv         ← Sample CSV you can upload
│
├── utils/
│   ├── __init__.py
│   └── lstm_model.py       ← All LSTM logic lives here
│
└── templates/
    └── index.html          ← Beautiful frontend (HTML+CSS+JS)
```

---

## 🚀 INSTALLATION — STEP BY STEP

### STEP 1 — Install Python
Make sure Python 3.9 or 3.10 is installed.
```bash
python --version
# Should show: Python 3.9.x or 3.10.x
```
Download from: https://www.python.org/downloads/

---

### STEP 2 — Create a Virtual Environment
A virtual environment keeps this project's packages separate from others.

**Windows:**
```bash
cd lstm_timeseries
python -m venv venv
venv\Scripts\activate
```

**Mac / Linux:**
```bash
cd lstm_timeseries
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` in your terminal — that means it's active!

---

### STEP 3 — Install Dependencies
```bash
pip install -r requirements.txt
```

⏳ This takes 5–10 minutes because TensorFlow is large (~500MB).

If TensorFlow fails to install (older hardware), skip it — the app
will automatically use a simple moving-average fallback so the UI
still works fully!

---

### STEP 4 — Run the App
```bash
python app.py
```

You should see:
```
 * Running on http://127.0.0.1:5000
 * Debug mode: on
```

---

### STEP 5 — Open in Browser
Visit: **http://localhost:5000**

---

## 🎮 HOW TO USE THE APP

### Option A: Use Sample Data
1. In the sidebar, choose a dataset type (Stock Price, Sine Wave, etc.)
2. Set how many data points you want
3. Click **"⚡ LOAD SAMPLE DATA"**
4. You'll see the raw chart appear

### Option B: Upload Your Own CSV
1. Click the **"UPLOAD"** tab in the sidebar
2. Your CSV must have:
   - One column with dates (optional)
   - One column with numbers
3. Drop or click to upload

### Configure the Model
| Setting | What it does | Recommended |
|---------|-------------|-------------|
| Look-Back Window | How many past days to use | 20–50 |
| LSTM Units | Neurons per layer (bigger = more powerful) | 32–128 |
| Epochs | Training iterations | 20–50 |
| Dropout | Prevents overfitting (0 = none, 0.5 = heavy) | 0.1–0.3 |
| Future Forecast | How many days to predict beyond data | 14–60 |

### Click "🚀 TRAIN & PREDICT"
- The model trains in real-time
- Watch the console log for progress messages
- Results appear automatically with 3 charts!

---

## 📊 READING THE CHARTS

### Chart 1: Time Series + Prediction
- **Blue line** = Historical data
- **Pink dotted** = Model's fit on test data (should be close to blue)
- **Gold line** = Future forecast

### Chart 2: Loss Curve
- Shows how the model error decreased during training
- If both lines decrease and level off = good training
- If val_loss goes up while train_loss goes down = overfitting

### Chart 3: Future Forecast
- The model's best guess for future values
- Farther predictions are less reliable (uncertainty grows)

---

## 🛠 TROUBLESHOOTING

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` again |
| TensorFlow won't install | The app works without it (moving average mode) |
| "Need more data points" error | Use at least 100 data points |
| Charts don't appear | Check browser console (F12) for errors |
| Port 5000 in use | Change `port=5000` to `port=5001` in `app.py` |

---

## 🔬 HOW THE CODE WORKS (Under the Hood)

```
1. Raw data  →  MinMaxScaler  →  values in [0, 1]

2. Sliding window:
   [d1,d2,...,d30] → predict d31
   [d2,d3,...,d31] → predict d32
   ...

3. LSTM Architecture:
   Input (30 timesteps, 1 feature)
       ↓
   LSTM(64 units, return_sequences=True)
       ↓
   Dropout(0.2)
       ↓
   LSTM(32 units)
       ↓
   Dropout(0.2)
       ↓
   Dense(1)  ← single number output

4. Loss function: MSE (mean squared error)
   Optimizer: Adam (adaptive learning rate)

5. Predictions  →  inverse_transform  →  original scale
```

---

## 📝 SAMPLE CSV FORMAT

Your CSV should look like this:
```
date,value
2023-01-01,150.2
2023-01-02,152.7
2023-01-03,149.8
...
```

Or just a single numeric column:
```
price
150.2
152.7
149.8
```

---

Happy predicting! 🚀
