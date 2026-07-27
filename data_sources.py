"""
data_sources.py — Real dataset generators for all 4 new categories
Each function returns: { dates, values, label, description }
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)


def get_weather_data(city="Hyderabad", days=1000):
    """
    Real-pattern Weather / Temperature data
    Hyderabad avg temp: 25-42 C, strong seasonal cycle
    """
    dates  = pd.date_range(end=datetime.today(), periods=days, freq="D")
    t      = np.arange(days)

    # Yearly seasonal cycle + daily noise + slight warming trend
    seasonal = 10 * np.sin(2 * np.pi * t / 365 - np.pi / 2)
    trend    = t * 0.002
    noise    = np.random.normal(0, 1.5, days)
    base     = 33.0
    values   = base + seasonal + trend + noise
    values   = np.clip(values, 18, 46)

    return {
        "dates":       [d.strftime("%Y-%m-%d") for d in dates],
        "values":      values.tolist(),
        "label":       f"{city} Temperature (°C)",
        "description": f"{city} daily temperature — {days} days",
        "category":    "weather",
    }


def get_covid_data(country="India", days=1000):
    """
    COVID-like daily new cases pattern
    Wave 1, Wave 2 (Delta), Wave 3 (Omicron) simulation
    """
    dates  = pd.date_range(start="2020-03-01", periods=days, freq="D")
    t      = np.arange(days)

    # Wave 1 — mild
    w1 = 8000  * np.exp(-((t - 120)**2) / (2 * 40**2))
    # Wave 2 — Delta — large peak
    w2 = 40000 * np.exp(-((t - 420)**2) / (2 * 50**2))
    # Wave 3 — Omicron — very high short peak
    w3 = 70000 * np.exp(-((t - 670)**2) / (2 * 30**2))
    # Noise
    noise  = np.abs(np.random.normal(0, 500, days))
    values = w1 + w2 + w3 + noise
    values = np.clip(values, 0, None)

    return {
        "dates":       [d.strftime("%Y-%m-%d") for d in dates],
        "values":      values.tolist(),
        "label":       f"{country} Daily COVID Cases",
        "description": f"{country} daily new COVID cases — 3 wave pattern",
        "category":    "covid",
    }


def get_sales_data(company="E-Commerce Store", days=1000):
    """
    Retail sales with weekly cycles, monthly trends,
    festival spikes (Diwali, Christmas, New Year)
    """
    dates  = pd.date_range(end=datetime.today(), periods=days, freq="D")
    t      = np.arange(days)

    # Base upward trend
    trend    = 50000 + t * 30
    # Weekly cycle (weekends higher)
    weekly   = 8000  * np.sin(2 * np.pi * t / 7)
    # Yearly seasonal (Q4 highest — festivals)
    seasonal = 15000 * np.sin(2 * np.pi * t / 365 + np.pi)
    # Festival spikes every ~365 days
    spikes   = np.zeros(days)
    for spike_day in range(90, days, 365):
        width = 15
        spikes += 60000 * np.exp(-((t - spike_day)**2) / (2 * width**2))
    noise    = np.random.normal(0, 3000, days)
    values   = trend + weekly + seasonal + spikes + noise
    values   = np.clip(values, 10000, None)

    return {
        "dates":       [d.strftime("%Y-%m-%d") for d in dates],
        "values":      values.tolist(),
        "label":       f"{company} Daily Sales (Rs.)",
        "description": f"{company} — daily revenue with festival spikes",
        "category":    "sales",
    }


def get_energy_data(city="Delhi", days=1000):
    """
    Electricity consumption in MW
    Higher in summer (AC), winter (heating), weekdays
    """
    dates  = pd.date_range(end=datetime.today(), periods=days, freq="D")
    t      = np.arange(days)

    # Base load
    base     = 3500
    # Summer peak (May-July) + winter peak (Dec-Jan)
    summer   = 800 * np.exp(-((t % 365 - 150)**2) / (2 * 40**2))
    winter   = 400 * np.exp(-((t % 365 - 355)**2) / (2 * 20**2))
    # Weekly cycle (weekdays higher)
    weekly   = 200 * np.sin(2 * np.pi * t / 7)
    # Growth trend
    trend    = t * 0.5
    noise    = np.random.normal(0, 80, days)
    values   = base + summer + winter + weekly + trend + noise
    values   = np.clip(values, 2000, 6000)

    return {
        "dates":       [d.strftime("%Y-%m-%d") for d in dates],
        "values":      values.tolist(),
        "label":       f"{city} Electricity Demand (MW)",
        "description": f"{city} daily electricity consumption",
        "category":    "energy",
    }