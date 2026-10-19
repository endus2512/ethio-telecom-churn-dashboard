"""
generate_data.py
----------------
Generate a realistic synthetic dataset of Ethio Telecom customers
and persist it to both CSV and SQLite for the churn analytics dashboard.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEED = 42
N_CUSTOMERS = 1000
DATA_DIR = Path(__file__).resolve().parent / "data"
CSV_PATH = DATA_DIR / "ethio_telecom_customers.csv"
DB_PATH = DATA_DIR / "ethio_telecom.db"
TABLE_NAME = "customers"

REGIONS = [
    "Addis Ababa",
    "Oromia",
    "Amhara",
    "SNNPR",
    "Sidama",
    "Somali",
    "Tigray",
    "Afar",
    "Dire Dawa",
    "Harari",
]

# Weighted toward denser / more urban markets
REGION_WEIGHTS = {
    "Addis Ababa": 0.28,
    "Oromia": 0.18,
    "Amhara": 0.14,
    "SNNPR": 0.10,
    "Sidama": 0.08,
    "Somali": 0.06,
    "Tigray": 0.07,
    "Afar": 0.04,
    "Dire Dawa": 0.03,
    "Harari": 0.02,
}

CONTRACT_TYPES = ["Prepaid", "Postpaid"]
INTERNET_SERVICES = ["3G", "4G", "5G", "Fiber"]
INTERNET_WEIGHTS = [0.15, 0.50, 0.25, 0.10]


def _simulate_churn(
    tenure: np.ndarray,
    monthly_bill: np.ndarray,
    service_calls: np.ndarray,
    contract: np.ndarray,
    telebirr: np.ndarray,
    internet: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Probabilistic churn labels driven by realistic telecom risk signals.
    Higher service calls, short tenure, prepaid, high bills, and older
    network tech increase churn likelihood.
    """
    # Baseline logit contribution
    logit = np.full(len(tenure), -1.8, dtype=float)

    logit += np.clip(service_calls - 1, 0, None) * 0.55
    logit += np.where(tenure < 6, 0.90, 0.0)
    logit += np.where(tenure < 12, 0.35, 0.0)
    logit += np.where(tenure > 36, -0.50, 0.0)
    logit += np.where(contract == "Prepaid", 0.45, -0.25)
    logit += np.where(telebirr == "No", 0.30, -0.20)
    logit += np.where(internet == "3G", 0.40, 0.0)
    logit += np.where(internet == "Fiber", -0.35, 0.0)
    logit += np.clip((monthly_bill - 450) / 300, -0.4, 1.2)

    # Logistic transform → probability, then Bernoulli draw
    prob = 1.0 / (1.0 + np.exp(-logit))
    churned = rng.random(len(tenure)) < prob
    return np.where(churned, "Yes", "No")


def generate_customers(n: int = N_CUSTOMERS, seed: int = SEED) -> pd.DataFrame:
    """Create a synthetic Ethio Telecom customer DataFrame."""
    rng = np.random.default_rng(seed)

    customer_ids = [f"ET-{1001 + i}" for i in range(n)]
    regions = rng.choice(
        REGIONS,
        size=n,
        p=[REGION_WEIGHTS[region] for region in REGIONS],
    )
    contract_types = rng.choice(CONTRACT_TYPES, size=n, p=[0.72, 0.28])
    internet_services = rng.choice(INTERNET_SERVICES, size=n, p=INTERNET_WEIGHTS)
    telebirr_users = rng.choice(["Yes", "No"], size=n, p=[0.62, 0.38])

    # Tenure: mostly short–medium, long-tail of loyal customers
    tenure = rng.integers(1, 73, size=n)

    # Monthly bill (ETB): log-normal-ish, prepaid lower than postpaid
    base_bill = rng.lognormal(mean=5.8, sigma=0.45, size=n)  # ~ETB 330 median-ish
    postpaid_bump = np.where(contract_types == "Postpaid", rng.uniform(80, 250, size=n), 0)
    fiber_bump = np.where(internet_services == "Fiber", rng.uniform(100, 400, size=n), 0)
    g5_bump = np.where(internet_services == "5G", rng.uniform(40, 150, size=n), 0)
    monthly_bill = np.round(base_bill + postpaid_bump + fiber_bump + g5_bump, 2)
    monthly_bill = np.clip(monthly_bill, 49.0, 2500.0)

    # Customer service calls: zero-inflated count
    service_calls = rng.poisson(lam=1.4, size=n)
    service_calls = np.clip(service_calls, 0, 12)

    churn = _simulate_churn(
        tenure=tenure,
        monthly_bill=monthly_bill,
        service_calls=service_calls,
        contract=contract_types,
        telebirr=telebirr_users,
        internet=internet_services,
        rng=rng,
    )

    df = pd.DataFrame(
        {
            "Customer_ID": customer_ids,
            "Region": regions,
            "Contract_Type": contract_types,
            "Monthly_Bill_ETB": monthly_bill,
            "Internet_Service": internet_services,
            "telebirr_User": telebirr_users,
            "Customer_Service_Calls": service_calls.astype(int),
            "Tenure_Months": tenure.astype(int),
            "Churn": churn,
        }
    )

    # Inject a few missing values to exercise graceful handling in the app
    missing_idx = rng.choice(df.index, size=max(1, n // 200), replace=False)
    df.loc[missing_idx[: len(missing_idx) // 2], "Internet_Service"] = np.nan
    df.loc[missing_idx[len(missing_idx) // 2 :], "Monthly_Bill_ETB"] = np.nan

    return df


def save_to_csv(df: pd.DataFrame, path: Path = CSV_PATH) -> Path:
    """Write the dataset to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def save_to_sqlite(df: pd.DataFrame, path: Path = DB_PATH, table: str = TABLE_NAME) -> Path:
    """Write the dataset to a SQLite database."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        df.to_sql(table, conn, if_exists="replace", index=False)
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{table}_region ON {table}(Region)"
        )
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{table}_churn ON {table}(Churn)"
        )
    return path


def load_from_sqlite(path: Path = DB_PATH, table: str = TABLE_NAME) -> pd.DataFrame:
    """Load customer records from SQLite (used by the Streamlit app)."""
    if not path.exists():
        raise FileNotFoundError(
            f"Database not found at {path}. Run: python generate_data.py"
        )
    with sqlite3.connect(path) as conn:
        return pd.read_sql_query(f"SELECT * FROM {table}", conn)


def main() -> None:
    print(f"Generating {N_CUSTOMERS} synthetic Ethio Telecom customers...")
    df = generate_customers()

    csv_path = save_to_csv(df)
    db_path = save_to_sqlite(df)

    churn_rate = (df["Churn"] == "Yes").mean() * 100
    print(f"✓ CSV saved  → {csv_path}")
    print(f"✓ SQLite saved → {db_path} (table: {TABLE_NAME})")
    print(f"  Rows: {len(df):,} | Churn rate: {churn_rate:.1f}%")
    print(f"  Missing values:\n{df.isna().sum()[df.isna().sum() > 0].to_string()}")


if __name__ == "__main__":
    main()
