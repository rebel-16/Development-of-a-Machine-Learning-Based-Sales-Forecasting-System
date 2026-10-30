# Sales Forecasting System

A beginner-friendly college project scaffold for a machine learning-based sales forecasting system. The current version serves a responsive Flask frontend, summarizes sales data with Pandas, and forecasts future sales with Scikit-learn Linear Regression.

## Current setup

- `app.py` provides routes for the dashboard, upload, analysis, forecasting, and evaluation pages. It reads and cleans sales CSV files with Pandas.
- `templates/base.html` contains the shared page layout and navigation. The other templates extend it to avoid repeating the sidebar.
- Bootstrap 5 and Bootstrap Icons are loaded from a CDN. `static/css/style.css` adds the project-specific styling.
- `static/js/charts.js` updates the forecast horizon options and draws historical and predicted values with Chart.js.
- `sales_data.csv` contains two years of synthetic demonstration data, not actual company sales. Run `py generate_sales_data.py` to regenerate the same dataset using a fixed random seed.
- Upload accepts CSV files up to 5 MB with `Date` and `Sales` columns. Validated uploads are saved under `uploaded_data/` with generated filenames; the browser session remembers which dataset to use.
- The dashboard and monthly analysis use the selected uploaded CSV, or the synthetic sample when no upload is selected.
- `ml_model.py` aggregates daily or monthly sales, evaluates a chronological Linear Regression model, then trains on all history and predicts future periods. The model requires at least 12 observations at the selected frequency.
- Forecast options are 7, 14, or 30 days and 3, 6, or 12 months. Missing dates/months are filled with zero sales. Negative predictions are clipped to zero and marked in the results.
- `evaluation.html` displays the latest forecast's validation metrics for the current browser session. MAE and RMSE also appear with each forecast.
- `outputs/` is reserved for generated files.

## Run locally on Windows

Open the project folder in VS Code, then use its PowerShell terminal:

```powershell
cd "sales-forecasting-system"
py -m venv .venv
.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
py app.py
```

Open the local address printed in the terminal (usually `http://127.0.0.1:5000/`). To stop the server, press `Ctrl+C`.