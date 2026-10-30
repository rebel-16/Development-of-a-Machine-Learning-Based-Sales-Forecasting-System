import os
import re
import secrets
from pathlib import Path
from uuid import uuid4

import pandas as pd
from flask import Flask, flash, has_request_context, redirect, render_template, request, send_file, session, url_for
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

from ml_model import ForecastError, train_sales_forecast

app = Flask(__name__)
PROJECT_FOLDER = Path(__file__).resolve().parent
SALES_DATA_PATH = PROJECT_FOLDER / "sales_data.csv"
UPLOAD_FOLDER = PROJECT_FOLDER / "uploaded_data"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024
app.secret_key = os.environ.get("FLASK_SECRET_KEY") or secrets.token_hex(32)


def get_active_dataset_path():
    """Resolve the session's generated ID, or use the sample CSV."""
    dataset_id = session.get("selected_dataset_id") if has_request_context() else None
    if isinstance(dataset_id, str) and re.fullmatch(r"[a-f0-9]{32}", dataset_id):
        upload_path = Path(app.config["UPLOAD_FOLDER"]) / f"{dataset_id}.csv"
        if upload_path.is_file():
            return upload_path
    return SALES_DATA_PATH


def get_active_dataset_name():
    """Return a display name without using it to build a filesystem path."""
    if get_active_dataset_path() == SALES_DATA_PATH:
        return SALES_DATA_PATH.name
    return secure_filename(session.get("selected_dataset_name", "uploaded.csv")) or "uploaded.csv"


def load_sales_data(data_path=None):
    """Load, clean, and sort the selected sales records."""
    sales_data = pd.read_csv(data_path or get_active_dataset_path())
    sales_data["Date"] = pd.to_datetime(sales_data["Date"], errors="coerce")
    sales_data["Sales"] = pd.to_numeric(sales_data["Sales"], errors="coerce")

    # Rows without a usable date or sales amount cannot be included in summaries.
    sales_data = sales_data.dropna(subset=["Date", "Sales"])
    return sales_data.sort_values("Date").reset_index(drop=True)


def render_upload_page():
    sales_data = load_sales_data()
    preview_records = sales_data.head(10).to_dict(orient="records")
    for record in preview_records:
        record["Date"] = record["Date"].strftime("%Y-%m-%d")
        for column, value in record.items():
            if pd.isna(value):
                record[column] = "-"

    return render_template(
        "upload.html",
        active_page="upload",
        page_title="Upload Sales Data",
        dataset_filename=get_active_dataset_name(),
        preview_columns=sales_data.columns.tolist(),
        preview_records=preview_records,
        record_count=len(sales_data),
        total_sales=sales_data["Sales"].sum(),
        average_sales=sales_data["Sales"].mean(),
    )


@app.route("/")
def home():
    sales_data = load_sales_data()
    statistics = {
        "total_sales": sales_data["Sales"].sum(),
        "record_count": len(sales_data),
        "average_sales": sales_data["Sales"].mean(),
        "highest_sales": sales_data["Sales"].max(),
        "lowest_sales": sales_data["Sales"].min(),
    }
    return render_template(
        "index.html",
        active_page="dashboard",
        page_title="Dashboard",
        dataset_filename=get_active_dataset_name(),
        **statistics,
    )


@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        uploaded_file = request.files.get("sales_file")
        if uploaded_file is None or not uploaded_file.filename:
            flash("Choose a CSV file before uploading.", "danger")
            return redirect(url_for("upload"))

        original_filename = secure_filename(uploaded_file.filename)
        if not original_filename.lower().endswith(".csv"):
            flash("Only files with a .csv extension are accepted.", "danger")
            return redirect(url_for("upload"))

        try:
            uploaded_data = pd.read_csv(uploaded_file.stream)
        except pd.errors.EmptyDataError:
            flash("The CSV is empty or does not contain a header row.", "danger")
            return redirect(url_for("upload"))
        except (pd.errors.ParserError, UnicodeDecodeError, ValueError):
            flash("The file could not be read as a valid CSV. Check its format and try again.", "danger")
            return redirect(url_for("upload"))

        missing_columns = {"Date", "Sales"}.difference(uploaded_data.columns)
        if missing_columns:
            missing_names = ", ".join(sorted(missing_columns))
            flash(f"The CSV must include these required columns: {missing_names}.", "danger")
            return redirect(url_for("upload"))

        uploaded_data["Date"] = pd.to_datetime(
            uploaded_data["Date"], format="mixed", errors="coerce"
        )
        uploaded_data["Sales"] = pd.to_numeric(uploaded_data["Sales"], errors="coerce")
        valid_data = uploaded_data.dropna(subset=["Date", "Sales"])
        valid_data = valid_data.sort_values("Date").reset_index(drop=True)

        if valid_data.empty:
            flash("The CSV has no valid rows. Include at least one valid Date and Sales value.", "danger")
            return redirect(url_for("upload"))

        dataset_id = uuid4().hex
        upload_folder = Path(app.config["UPLOAD_FOLDER"])
        upload_folder.mkdir(parents=True, exist_ok=True)
        saved_path = upload_folder / f"{dataset_id}.csv"
        try:
            valid_data.to_csv(saved_path, index=False)
        except OSError:
            flash("The uploaded data could not be saved. Please try again.", "danger")
            return redirect(url_for("upload"))

        session["selected_dataset_id"] = dataset_id
        session["selected_dataset_name"] = original_filename or "uploaded.csv"
        skipped_rows = len(uploaded_data) - len(valid_data)
        if skipped_rows:
            flash(
                f"Uploaded {original_filename}: {len(valid_data)} valid records saved; "
                f"{skipped_rows} invalid row(s) skipped.",
                "warning",
            )
        else:
            flash(f"Uploaded {original_filename}: {len(valid_data)} valid records saved.", "success")
        return redirect(url_for("upload"))

    return render_upload_page()


@app.route("/download-sample")
def download_sample_data():
    return send_file(SALES_DATA_PATH, as_attachment=True, download_name="sales_data.csv", mimetype="text/csv")


@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(_error):
    flash("The file is larger than the 5 MB upload limit.", "danger")
    return render_upload_page(), 413


@app.route("/analysis")
def analysis():
    sales_data = load_sales_data()
    monthly_sales = (
        sales_data.assign(Month=sales_data["Date"].dt.to_period("M"))
        .groupby("Month", as_index=False)["Sales"]
        .sum()
    )
    monthly_sales["Month"] = monthly_sales["Month"].dt.strftime("%B %Y")
    monthly_records = monthly_sales.to_dict(orient="records")
    return render_template(
        "analysis.html",
        active_page="analysis",
        page_title="Sales Analysis",
        dataset_filename=get_active_dataset_name(),
        monthly_records=monthly_records,
    )


@app.route("/forecast", methods=["GET", "POST"])
def forecast():
    frequency = request.values.get("frequency", "monthly")
    horizon = request.values.get("horizon", "6")
    forecast_result = None

    if request.method == "POST":
        try:
            forecast_result = train_sales_forecast(
                get_active_dataset_path(), frequency, horizon
            )
            session["last_evaluation"] = {
                "dataset_filename": get_active_dataset_name(),
                "frequency_label": forecast_result["frequency_label"],
                "horizon": forecast_result["horizon"],
                "model_name": forecast_result["model_name"],
                "validation_mae": forecast_result["validation_mae"],
                "validation_rmse": forecast_result["validation_rmse"],
                "validation_r2": forecast_result["validation_r2"],
            }
        except ForecastError as error:
            flash(str(error), "warning")

    chart_data = None
    if forecast_result:
        history_length = len(forecast_result["history_sales"])
        chart_data = {
            "labels": forecast_result["history_dates"] + forecast_result["forecast_dates"],
            "historical": forecast_result["history_sales"] + [None] * forecast_result["horizon"],
            "forecast": [None] * history_length + forecast_result["forecast_sales"],
        }

    return render_template(
        "forecast.html",
        active_page="forecast",
        page_title="Sales Forecasting",
        dataset_filename=get_active_dataset_name(),
        frequency=frequency,
        horizon=horizon,
        forecast_result=forecast_result,
        chart_data=chart_data,
    )


@app.route("/evaluation")
def evaluation():
    return render_template(
        "evaluation.html",
        active_page="evaluation",
        page_title="Model Evaluation",
        evaluation_result=session.get("last_evaluation"),
    )


if __name__ == "__main__":
    app.run(debug=True)