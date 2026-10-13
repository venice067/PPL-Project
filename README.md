# 🌾 Agricultural Pest Outbreak Analysis and Forecasting Using Environmental Factors

A Python capstone project that cleans a rice pest and weather dataset, then uses Matplotlib and Seaborn to explore which pests break out, which weather conditions favour them, and whether outbreaks can be forecast from the previous week's data.

## Overview

Pest outbreaks can cause serious yield losses in rice. This project analyses weekly pest observations from several research stations in India, together with weather readings for the same weeks, to answer three questions:

1. **Outbreak analysis:** Which pests are most common, and when do they appear?
2. **Environmental factors:** How do humidity, temperature and rainfall relate to pest presence?
3. **Forecasting:** Does last week's pest level help predict this week's?

## Dataset

- **Source:** Rice pest and weather dataset from Kaggle (`RICE.csv`)
- **Contents:** Weekly pest observations (counts or percent damage) by station, with max/min temperature, humidity, rainfall, wind speed, sunshine hours and evaporation
- **After cleaning:** 19,387 rows, 30 columns, 11 pests

## Project Structure

```
.
├── pyproj.py                 # Pipeline: cleaning, summaries, charts
├── RICE.csv                  # Original dataset (input)
├── cleaned_rice_final.csv    # Cleaned dataset (output)
├── summary_by_pest.csv       # Per-pest summary (output)
├── summary_by_season.csv     # Pest presence by season (output)
├── chart1_top_pests.png ... chart9_forecast.png
└── README.md
```

## Methodology

### Data cleaning and manipulation
- Corrected evaporation unit errors at one station and removed years with unreliable evaporation data
- Set physically impossible weather values to missing, using limits for each variable
- Merged duplicate observations by averaging them
- Filled missing weather values using each station's typical value for that week of the year
- Flagged extreme pest values as outliers (kept in the data)
- Added new columns: season (Kharif or Rabi/Summer), month, mean temperature, mean humidity, pest-present flag, log-scaled pest value, and last-week lag features

### Data visualization

| # | Chart | Type | Question answered |
|---|-------|------|-------------------|
| 1 | Top 10 pests by share of weeks present | Horizontal bar | Which pests are most common? |
| 2 | Pest presence by season | Heatmap | How do Kharif and Rabi/Summer differ? |
| 3 | Pest presence across the year | Line | Which months are worst? |
| 4 | Pests with the most outbreak records | Horizontal bar | Which pests spike most often? |
| 5 | Pest presence by humidity band | Bar | Does humidity matter? |
| 6 | Pest presence by temperature band | Bar | Is there a favourable temperature range? |
| 7 | Pest presence by rainfall band | Bar | Does rainfall matter? |
| 8 | Weather vs pest value correlations | Heatmap | Which variables move together? |
| 9 | Last week's vs this week's pest level | Scatter with trend line | Can pest levels be forecast? |

## Key Findings

- **Humidity:** Pest presence rises from about 32% of observations below 60% humidity to about 72% above 90%.
- **Temperature:** Presence peaks at 25-30 °C (about 67%) and falls at both extremes.
- **Rainfall:** The effect is weak, ranging from about 49% to 59% across bands.
- **Seasonality:** Leaf Blast and Neck Blast occur almost entirely in the Kharif season, while Zig Zag Leafhopper is present year-round.
- **Forecasting:** Chart 9 compares this week's pest level with last week's, which supports using lag features in a forecasting model.

## How to Run

### Requirements
- Python 3.8 or later
- pandas, numpy, matplotlib, seaborn

```bash
pip install -r requirements.txt
```

### Steps
1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd <repo-folder>
   ```
2. Make sure `RICE.csv` is in the same folder as `pyproj.py`.
3. Run the script:
   ```bash
   python pyproj.py
   ```
   To also display charts interactively in pop-up windows, run:
   ```bash
   python pyproj.py --show
   ```

The script prints the cleaning steps, saves the cleaned CSV, both summary CSVs, and exports all 9 chart images into the project folder.

## Acknowledgements

Dataset from Kaggle. Built as a capstone project for the Python programming lab.
