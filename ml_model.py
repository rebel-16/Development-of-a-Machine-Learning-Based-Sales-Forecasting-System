from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


MINIMUM_OBSERVATIONS = 12
ALLOWED_HORIZONS = {
    "daily": {7, 14, 30},
    "monthly": {3, 6, 12},
}


class ForecastError(ValueError):
    """A validation or model-training problem safe to show to the user."""


def load_sales_series(csv_path, frequency):
    """Read and aggregate sales into a complete daily or monthly series."""
    if frequency not in ALLOWED_HORIZONS:
        raise ForecastError("Choose either daily or monthly forecasting.")

    try:
        sales_data = pd.read_csv(Path(csv_path))
    except FileNotFoundError as error:
        raise ForecastError("The selected sales dataset could not be found.") from error
    except pd.errors.EmptyDataError as error:
        raise ForecastError("The selected sales dataset is empty.") from error
    except (pd.errors.ParserError, UnicodeDecodeError, OSError, ValueError) as error:
        raise ForecastError("The selected sales dataset could not be read as a CSV.") from error

    missing_columns = {"Date", "Sales"}.difference(sales_data.columns)
    if missing_columns:
        names = ", ".join(sorted(missing_columns))
        raise ForecastError(f"The selected dataset is missing required column(s): {names}.")

    sales_data["Date"] = pd.to_datetime(
        sales_data["Date"], format="mixed", errors="coerce"
    )
    sales_data["Sales"] = pd.to_numeric(sales_data["Sales"], errors="coerce")
    sales_data = sales_data.dropna(subset=["Date", "Sales"])
    sales_data = sales_data[np.isfinite(sales_data["Sales"])]

    if sales_data.empty:
        raise ForecastError("The selected dataset has no valid Date and Sales rows.")

    if frequency == "daily":
        sales_data["Period"] = sales_data["Date"].dt.normalize()
        period_frequency = "D"
    else:
        sales_data["Period"] = sales_data["Date"].dt.to_period("M").dt.to_timestamp()
        period_frequency = "MS"

    aggregated_sales = sales_data.groupby("Period")["Sales"].sum().sort_index()
    complete_dates = pd.date_range(
        start=aggregated_sales.index.min(),
        end=aggregated_sales.index.max(),
        freq=period_frequency,
    )
    # A missing date/month means no recorded sales, so it is filled with zero.
    return aggregated_sales.reindex(complete_dates, fill_value=0).astype(float)


def train_sales_forecast(csv_path, frequency, horizon_value):
    """Evaluate a time-trend model and forecast the requested future periods."""
    try:
        horizon = int(horizon_value)
    except (TypeError, ValueError) as error:
        raise ForecastError("Choose a valid forecast horizon.") from error

    if horizon not in ALLOWED_HORIZONS.get(frequency, set()):
        if frequency in ALLOWED_HORIZONS:
            allowed = ", ".join(str(value) for value in sorted(ALLOWED_HORIZONS[frequency]))
            raise ForecastError(f"For {frequency} forecasts, choose a horizon of {allowed}.")
        raise ForecastError("Choose either daily or monthly forecasting.")

    sales_series = load_sales_series(csv_path, frequency)
    observation_count = len(sales_series)
    if observation_count < MINIMUM_OBSERVATIONS:
        raise ForecastError(
            f"At least {MINIMUM_OBSERVATIONS} {frequency} observations are needed "
            "to train and validate a meaningful forecast."
        )

    # X is the time-step number; y is the sales amount for that time step.
    time_steps = np.arange(observation_count).reshape(-1, 1)
    sales_values = sales_series.to_numpy(dtype=float)
    validation_count = max(3, int(np.ceil(observation_count * 0.2)))
    training_end = observation_count - validation_count
    if training_end < 3:
        raise ForecastError("There are not enough earlier observations for model training.")

    try:
        validation_model = LinearRegression()
        validation_model.fit(time_steps[:training_end], sales_values[:training_end])
        validation_predictions = validation_model.predict(time_steps[training_end:])
        validation_mae = float(
            mean_absolute_error(sales_values[training_end:], validation_predictions)
        )
        validation_rmse = float(
            np.sqrt(mean_squared_error(sales_values[training_end:], validation_predictions))
        )
        validation_r2 = None
        validation_actuals = sales_values[training_end:]
        if len(validation_actuals) >= 3 and np.ptp(validation_actuals) > 0:
            validation_r2 = float(r2_score(validation_actuals, validation_predictions))

        # Refit on all known history before creating future predictions.
        final_model = LinearRegression()
        final_model.fit(time_steps, sales_values)
        future_steps = np.arange(observation_count, observation_count + horizon).reshape(-1, 1)
        predicted_sales = final_model.predict(future_steps)
    except (ValueError, FloatingPointError) as error:
        raise ForecastError("The model could not train on this dataset. Check the sales values.") from error

    negative_prediction_count = int(np.count_nonzero(predicted_sales < 0))
    predicted_sales = np.maximum(predicted_sales, 0)
    last_period = sales_series.index[-1]
    if frequency == "daily":
        future_dates = pd.date_range(last_period + pd.Timedelta(days=1), periods=horizon, freq="D")
        date_format = "%Y-%m-%d"
        frequency_label = "Daily"
    else:
        future_dates = pd.date_range(
            last_period + pd.offsets.MonthBegin(1), periods=horizon, freq="MS"
        )
        date_format = "%B %Y"
        frequency_label = "Monthly"

    history_dates = [value.strftime(date_format) for value in sales_series.index]
    forecast_dates = [value.strftime(date_format) for value in future_dates]
    forecast_values = predicted_sales.astype(float).tolist()
    return {
        "model_name": "Linear Regression",
        "frequency_label": frequency_label,
        "horizon": horizon,
        "history_dates": history_dates,
        "history_sales": sales_values.tolist(),
        "forecast_dates": forecast_dates,
        "forecast_sales": forecast_values,
        "total_predicted_sales": float(np.sum(predicted_sales)),
        "average_predicted_sales": float(np.mean(predicted_sales)),
        "validation_mae": validation_mae,
        "validation_rmse": validation_rmse,
        "validation_r2": validation_r2,
        "negative_predictions_clipped": negative_prediction_count,
    }