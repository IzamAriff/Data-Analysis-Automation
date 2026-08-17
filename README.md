# 📊 DataPilot — Automated Data Analysis Studio

> A self-serve, interactive web application that **profiles, visualises, diagnoses and models any tabular dataset** — no hard-coded column names, no code required from the user.

Upload a CSV/Excel/Parquet/JSON file (or paste a URL), and DataPilot automatically:

1. **Detects everything it can about the data** — encoding, delimiter, date columns, numeric columns stored as formatted strings (e.g. `$1,234.50`), categorical vs. free-text vs. ID columns.
2. **Profiles data quality** — row/column counts, duplicate rows, missing-value percentages, per-column statistics and a data dictionary.
3. **Builds an interactive dashboard** — KPI cards, time trends, distributions, comparisons, relationships, compositions, correlation heatmaps — all driven by dynamic sidebar filters.
4. **Runs diagnostics** — correlation analysis with p-values, ANOVA/Kruskal–Wallis group comparisons, chi-square + Cramér's V association tests, IQR outlier detection.
5. **Fits predictive models on demand** — regression, classification, k-means clustering, and time-series forecasting, each with cross-validated metrics, baselines and stated assumptions.
6. **Exports everything** — filtered CSV downloads, chart downloads (PNG/HTML), the auto-generated data dictionary, and a reproducible Markdown report.

---

## 🚀 Quick start (local)

```bash
git clone https://github.com/IzamAriff/Data-Analysis-Automation.git
cd Data-Analysis-Automation

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

streamlit run app.py
```

Then open the printed URL (usually **http://localhost:8501**). No database, no API keys, no configuration needed.

> **Try it instantly:** click **“🎁 Sample dataset → 🚀 Load sample data”** on the landing screen to explore the bundled *Sample Superstore* retail dataset (9,994 orders) or the *Video Game Sales* dataset (16,595 titles).

---

## 📂 Data requirements

| Format | Extensions | Notes |
|---|---|---|
| Delimited text | `.csv`, `.tsv`, `.txt`, `.dat` | Delimiter & encoding detected automatically (`utf-8` → `latin-1` → `cp1252`) |
| Excel | `.xlsx`, `.xlsm`, `.xls` | Multi-sheet workbooks: pick the sheet to analyse after loading |
| Parquet | `.parquet`, `.pq` | Read via PyArrow |
| JSON | `.json` | Records or JSON-lines |

- **Maximum file size: 250 MB** (uploads and URLs).
- Tabular data works best; each column should be a single variable, each row an observation.
- Missing values are fine — charts skip them per column, models impute them (see below).
- **Privacy:** data never leaves your machine; everything runs in your browser session, and nothing is uploaded to any server (unless you deploy the app yourself).

---

## 🧭 How to use the app

### 1. Load data
Choose **Upload a file**, **From a URL**, or a bundled **Sample dataset**. All parsing steps are logged and shown to you.

### 2. Review the profile
DataPilot shows row/column counts, duplicates, missing cells, memory use and the **inferred role of every column** (`date`, `year`, `numeric`, `binary`, `boolean`, `category`, `text`, `id`). You can override any role before continuing — roles drive the entire app.

### 3. Explore the dashboard
- **📊 Overview** — filter-aware KPI cards, data-quality chart, column roles, descriptive statistics and the full preprocessing log.
- **📈 Charts** — a chart builder with *Auto* mode: trend, distribution, comparison, relationship, composition and correlation charts, each with PNG/HTML download.
- **🔎 Diagnostics** — correlation matrix + strongest pairs, ANOVA/Kruskal–Wallis group comparisons with box plots, chi-square association tests, outlier detection.
- **🤖 Predictive** — pick a target and press *Run*:
  - *Regression* — linear + random forest, 5-fold cross-validated R²/RMSE/MAE vs. a mean baseline, feature importances, actual-vs-predicted chart.
  - *Classification* — random forest, stratified 5-fold CV accuracy & macro-F1 vs. a majority-class baseline, confusion matrix, feature importances (max 20 classes).
  - *Clustering* — k-means over standardised features with elbow & silhouette plots, PCA projection and cluster profiles.
  - *Forecasting* — Holt–Winters exponential smoothing (automatic seasonality) with a holdout MAPE vs. a naive baseline and a confidence band.
- **📚 Data Dictionary** — auto-generated per-column dictionary (role, type, description, missing %, examples) — downloadable.
- **🗂️ Raw Data** — browse the filtered rows, choose columns, download CSV.
- **📄 Report** — one-click reproducible Markdown summary of the whole session.

### Sidebar filters (dynamic per dataset)
Date-range sliders, year sliders, category multi-selects, numeric range sliders and free-text search — all generated from the inferred column roles. Empty selections mean “no filter”. A one-click reset restores the full data. If filters match no rows you get a friendly notice instead of a crash.

---

## 🧹 Preprocessing (documented & reproducible)

Every transformation is recorded and shown in the **preprocessing log** on the Overview tab (and in the report):

1. **Column-name cleanup** — whitespace trimmed, duplicate names suffixed.
2. **Fully-empty rows removed**; **exact duplicate rows** dropped by default (optional).
3. **Datetime parsing** — string columns where ≥90 % of sampled values parse as dates are converted (pure 4-digit years are kept as years).
4. **Numeric-string parsing** — currency/percentage strings (`$1,234.50`, `12,5 %`) are converted to numbers.
5. **Missing values for models** — your choice, applied to modelling only:
   - `median` (default) / `mean` — impute numeric features; categoricals get a `(missing)` level;
   - `drop` — discard incomplete rows.
6. **Model encoding** — categorical features one-hot encoded (top 20 levels per column, the rest binned to `(other)`); dates converted to days since the earliest date; ID/text columns excluded (logged).

Charts and descriptive statistics always skip missing values per column; no rows are silently invented.

---

## 🧠 Analysis methods & honest reporting

- **Cross-validation** — regression/classification metrics come from 5-fold CV: every row is scored by a model that never saw it. A naive baseline (mean / majority class) is always shown so “good” is never a matter of scale.
- **Hypothesis tests** — ANOVA with Levene-driven fallback to Kruskal–Wallis (effect size η² reported); chi-square with Cramér's V; correlation p-values derived from the t-distribution.
- **Assumptions & limitations** are stated in an expander under every model (linearity, independence, class imbalance, causality vs. association, forecast uncertainty).
- Results describe **association, not causation**, and the app tells users when a model fails to beat its baseline.

### Findings on the bundled demo data (Sample Superstore)

| Analysis | Result |
|---|---|
| Monthly Sales trend | 48 months, strong growth + year-end seasonality |
| Sales ↔ Profit correlation | r = 0.48, p < 0.001 (moderate) |
| Sales by Region (ANOVA) | **No significant difference** (p = 0.49) — regions are balanced |
| Segment ↔ Ship Mode | Statistically associated (p < 0.001) but weak (Cramér's V = 0.04) |
| Profit outliers | ~19 % of orders outside the IQR bounds (heavy-tailed target) |
| Forecast monthly Sales | Holt–Winters holdout MAPE **22.6 % vs. naive 97.3 %** |
| Regression: Profit from Sales/Quantity/Discount/Category | RF R² ≈ 0.64 (CV) vs. mean baseline — profit is noisy but learnable |
| Classification: Category | 74.6 % accuracy vs. 60.3 % baseline |
| Classification: Segment | Model does **not** beat the majority baseline — the app reports this honestly |

---

## 📦 Project structure

```
Data-Analysis-Automation/
├── app.py                    # Streamlit entry point (UI orchestration)
├── requirements.txt          # Pinned dependencies (Python 3.11 validated)
├── README.md
├── Dockerfile                # Optional containerised deployment
├── .streamlit/config.toml    # Theme + server settings
├── data/                     # Bundled public demo datasets
│   ├── sample_superstore.csv          # Tableau "Sample Superstore" (9,994 rows)
│   └── sample_video_game_sales.csv    # "Video Game Sales" / vgsales (16,595 rows)
├── src/
│   ├── loader.py             # Ingestion: encoding/delimiter/date detection, URL safety
│   ├── profile.py            # Column-role inference, profiling, data dictionary
│   ├── analysis.py           # Descriptive stats, correlations, hypothesis tests
│   ├── modeling.py           # Regression, classification, clustering, forecasting
│   ├── plots.py              # Colorblind-safe Plotly chart builders
│   └── ui.py                 # Dynamic filters, KPI cards, download buttons
└── tests/
    ├── test_modules.py       # Unit tests for every analytics module
    └── test_app.py           # End-to-end tests via Streamlit's AppTest harness
```

---

## 🧪 Testing

```bash
pytest tests/ -v
```

- **Unit tests** cover parsing edge cases (encodings, delimiters, JSON), role inference on both samples, statistics, every model family, chart builders and the filter logic.
- **AppTest end-to-end tests** drive the real `app.py`: load both samples, confirm the profile step, apply date/category/numeric/search filters, reset them, and run all four predictive model flows — asserting no exceptions and sane metrics.

---

## ☁️ Deployment

### Streamlit Community Cloud (recommended, free)
1. Push this repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app** → pick the repo, branch and `app.py`.
3. Deploy. Requirements are installed automatically.

### Hugging Face Spaces
1. Create a new **Space** with the *Streamlit* SDK.
2. Push this repository to the Space (it reads `requirements.txt` and `app.py` automatically).

### Internal server
```bash
docker build -t datapilot .
docker run -p 8501:8501 datapilot
```
or run under any WSGI-capable host with `streamlit run app.py --server.port 8501`.

**Note on chart PNG export:** server-side PNG export uses Kaleido, which needs Chrome. On headless servers run `kaleido_get_chrome` once; otherwise the app automatically offers an interactive **HTML download** whose Plotly toolbar can still save the chart as PNG.

---

## 🔒 Security & robustness

- URL loading is restricted to `http(s)`, with a 30-second timeout and a 250 MB cap; credentials are stripped from any displayed URL.
- Uploads are parsed only by format-specific readers (never pickled/executed); Excel readers do not run macros.
- Free-text search uses literal matching (`regex=False`) — no regex-injection surface.
- No user input is ever `eval`/`exec`-ed; column names are sanitised before use.
- Files use **relative paths** only; runtime logs go to `logs/app.log`.
- Per-section error handling keeps the app alive and shows friendly messages instead of stack traces.

## 🎨 Accessibility

- Colorblind-safe palettes throughout: Okabe–Ito for categorical series, Cividis for sequential scales, BrBG for diverging scales.
- Hover tooltips, labelled axes, readable font sizes and clear legends on all charts.

## 📚 Dataset provenance

- `sample_superstore.csv` — the public Tableau **“Sample – Superstore”** dataset (© Tableau, widely redistributed for learning purposes).
- `sample_video_game_sales.csv` — the public **“Video Game Sales” (vgsales)** dataset compiled from VGChartz data.
Both are bundled only as convenient, clearly-labelled demo data — your own files never leave your session.
