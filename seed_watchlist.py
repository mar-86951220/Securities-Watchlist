"""
Seeder: uploads Watchlist.xlsx to the Neon database.
Reads DB credentials from .streamlit/secrets.toml (gitignored).
Run via run_seeder.bat or: python seed_watchlist.py
"""

import sys
import os
import tomllib
import pandas as pd
import psycopg2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SECRETS_PATH = os.path.join(SCRIPT_DIR, ".streamlit", "secrets.toml")
EXCEL_PATH = os.path.join(SCRIPT_DIR, "Watchlist.xlsx")


def load_db_url() -> str:
    if not os.path.exists(SECRETS_PATH):
        sys.exit(f"ERROR: secrets file not found at {SECRETS_PATH}")
    with open(SECRETS_PATH, "rb") as f:
        secrets = tomllib.load(f)
    url = secrets.get("DATABASE_URL", "")
    if not url:
        sys.exit("ERROR: DATABASE_URL not found in secrets.toml")
    # Strip channel_binding — not supported by psycopg2
    if "channel_binding" in url:
        base, qs = url.split("?", 1)
        params = [p for p in qs.split("&") if not p.startswith("channel_binding")]
        url = base + "?" + "&".join(params)
    return url


def load_excel() -> pd.DataFrame:
    if not os.path.exists(EXCEL_PATH):
        sys.exit(f"ERROR: Excel file not found at {EXCEL_PATH}")
    try:
        df = pd.read_excel(EXCEL_PATH, sheet_name="Sheet1", usecols=[0, 1, 2])
    except PermissionError:
        sys.exit("ERROR: Watchlist.xlsx is open in Excel. Please close it and try again.")
    df.columns = ["ticker", "name", "security_type"]
    df = df.dropna(subset=["ticker"]).reset_index(drop=True)
    df["ticker"] = df["ticker"].str.strip().str.upper()
    return df


def seed(df: pd.DataFrame, db_url: str):
    conn = psycopg2.connect(db_url)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS watchlist (
            id            SERIAL PRIMARY KEY,
            ticker        VARCHAR(20)  NOT NULL UNIQUE,
            name          VARCHAR(200),
            security_type VARCHAR(100)
        )
    """)

    # Upsert all rows
    upserted = 0
    for _, row in df.iterrows():
        cur.execute("""
            INSERT INTO watchlist (ticker, name, security_type)
            VALUES (%s, %s, %s)
            ON CONFLICT (ticker) DO UPDATE
                SET name          = EXCLUDED.name,
                    security_type = EXCLUDED.security_type
        """, (row["ticker"], row["name"], row["security_type"]))
        upserted += 1

    # Remove tickers no longer in the Excel file
    tickers = df["ticker"].tolist()
    cur.execute("DELETE FROM watchlist WHERE ticker != ALL(%s)", (tickers,))
    deleted = cur.rowcount

    conn.commit()

    # Summary
    cur.execute("SELECT security_type, COUNT(*) FROM watchlist GROUP BY security_type ORDER BY security_type")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return upserted, deleted, rows


def main():
    print("=" * 50)
    print("  Watchlist Seeder")
    print("=" * 50)

    print(f"\n[1/3] Reading secrets from {SECRETS_PATH}...")
    db_url = load_db_url()
    print("      OK")

    print(f"\n[2/3] Reading Excel: {EXCEL_PATH}...")
    df = load_excel()
    print(f"      Loaded {len(df)} rows")

    print("\n[3/3] Uploading to Neon DB...")
    upserted, deleted, summary = seed(df, db_url)
    print(f"      Upserted : {upserted} rows")
    if deleted:
        print(f"      Deleted  : {deleted} stale rows")

    print("\n  Database summary:")
    for security_type, count in summary:
        print(f"    {security_type:<25} {count} rows")

    print("\n  Done! Watchlist is up to date in Neon.")
    print("=" * 50)


if __name__ == "__main__":
    main()
