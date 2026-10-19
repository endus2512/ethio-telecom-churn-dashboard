# Ethio Telecom Customer Churn Analytics Dashboard

A production-ready **Streamlit** portfolio project for analyzing customer churn across Ethio Telecom segments. It combines synthetic customer data (CSV + SQLite), interactive Plotly charts, KPI metrics, CSV export, and an individual **churn risk predictor**.

> **Note:** All data is synthetically generated for demonstration purposes. This project is not affiliated with Ethio Telecom.

---

## Features

- **Synthetic dataset generator** — 1,000 realistic Ethio Telecom customers with region, contract, internet tier, telebirr usage, billing, tenure, service calls, and churn labels
- **SQLite + CSV persistence** — query-friendly local database plus a portable CSV export
- **Sidebar filters** — Region, Contract Type, Internet Service, Churn status
- **KPI strip** — total customers, churned count, churn rate (%), average monthly bill (ETB)
- **Interactive Plotly visuals**
  - Churn rate by region (bar chart)
  - Internet service mix among churned customers (donut/pie)
  - Monthly bill vs churn status (box plot)
- **Data table & CSV download** of the filtered view
- **Risk Predictor** — Gradient Boosting model blended with transparent rule-based scoring
- **Ethio Telecom-inspired UI** — green, yellow, white, and dark-blue accents

---

## Project Structure

```text
ethio/
├── app.py                 # Main Streamlit dashboard
├── generate_data.py       # Synthetic data + SQLite/CSV writer
├── requirements.txt       # Python dependencies
├── README.md              # This file
└── data/                  # Created on first data generation
    ├── ethio_telecom_customers.csv
    └── ethio_telecom.db
```

---

## Tech Stack

| Layer        | Tools                                      |
|-------------|---------------------------------------------|
| Language    | Python 3.9+                                 |
| UI          | Streamlit                                   |
| Data        | Pandas, NumPy, SQLite                       |
| Charts      | Plotly Express / Graph Objects              |
| ML          | scikit-learn (GradientBoostingClassifier)   |

---

## Setup

### 1. Clone / open the project

```bash
cd ethio
```

### 2. Create and activate a virtual environment (recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Generate the synthetic dataset

```bash
python generate_data.py
```

This creates `data/ethio_telecom_customers.csv` and `data/ethio_telecom.db`.

> If you skip this step, the Streamlit app will auto-generate the dataset on first launch.

### 5. Run the dashboard

```bash
streamlit run app.py
```

Open the local URL shown in the terminal (typically `http://localhost:8501`).

---

## Dataset Fields

| Field                    | Description                                      |
|--------------------------|--------------------------------------------------|
| `Customer_ID`            | Unique ID (e.g. `ET-1001`)                       |
| `Region`                 | Addis Ababa, Oromia, Amhara, SNNPR, …            |
| `Contract_Type`          | Prepaid / Postpaid                               |
| `Monthly_Bill_ETB`       | Monthly spend in Ethiopian Birr                  |
| `Internet_Service`       | 3G / 4G / 5G / Fiber                             |
| `telebirr_User`          | Yes / No                                         |
| `Customer_Service_Calls` | Support contact count                            |
| `Tenure_Months`          | Months as a customer                             |
| `Churn`                  | Yes / No                                         |

Churn labels are simulated from tenure, bill size, service calls, contract type, telebirr adoption, and network technology — so the risk model has a learnable signal.

---

## Risk Predictor

The **Risk Predictor** tab estimates churn probability from:

- Tenure, monthly bill, and service-call volume
- Contract type, internet service, telebirr usage, and region

It reports:

1. **ML probability** — `GradientBoostingClassifier` trained on the synthetic population  
2. **Rule-based score** — transparent heuristic for demo explainability  
3. **Blended risk** — 70% ML + 30% rules, with Low / Medium / High guidance  

---

## Portfolio Tips

- Capture screenshots of the KPI strip, regional churn chart, and risk gauge
- Mention SQLite + Pandas ETL and Plotly interactivity in your write-up
- Highlight the blended ML / rules approach as an interview talking point
- Optionally deploy free on [Streamlit Community Cloud](https://streamlit.io/cloud)

---

## License

MIT — feel free to adapt for your portfolio. Ethio Telecom names and colors are used only for thematic demonstration.
