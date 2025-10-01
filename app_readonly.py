# app_readonly.py
# Vendors Directory (READ-ONLY)
# - Wide layout with horizontal scrolling
# - Multi-term AND search across common fields
# - Optional Category / Service filters
# - Website shown as a clickable Link column, placed right after Address
# - No non-serializable callables in column_config (fixes TypeError)

from __future__ import annotations

import os
import re
from urllib.parse import urlparse

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text as sql_text

# -------------- Page/Layout -----------------
st.set_page_config(page_title="Vendors Directory", layout="wide")

# Expand the central container beyond Streamlit default
st.markdown(
    """
<style>
.block-container {
  max-width: 1900px;   /* increase to 2100–2400px if you want wider */
  padding-left: 1rem;
  padding-right: 1rem;
}
</style>
""",
    unsafe_allow_html=True,
)

# -------------- Config -----------------
DEFAULT_SQLITE_PATH = "/mount/src/vendors-readonly-app/vendors.db"

# Expected vendors columns (text schema)
EXPECTED_COLUMNS = [
    "id",
    "category",
    "service",
    "business_name",
    "contact_name",
    "phone",
    "address",
    "website",
    "notes",
    "keywords",
]

# Display column order & labels
DISPLAY_ORDER = [
    ("business_name", "Business Name"),
    ("contact_name", "Contact Name"),
    ("phone", "Phone"),
    ("address", "Address"),
    ("website", "Website"),  # immediately after Address
    ("category", "Category"),
    ("service", "Service"),
    ("notes", "Notes"),
    ("keywords", "Keywords"),
]

# Minimum pixel widths per display column (tune to taste)
COLUMN_MIN_WIDTHS = {
    "Business Name": 300,
    "Contact Name": 220,
    "Phone": 150,
    "Address": 360,
    "Website": 260,   # link column
    "Category": 220,
    "Service": 260,
    "Notes": 360,
    "Keywords": 160,
}

# -------------- Helpers -----------------
def get_engine():
    """Create a SQLAlchemy engine for SQLite (read-only app uses local file)."""
    db_path = os.getenv("SQLITE_PATH", DEFAULT_SQLITE_PATH)
    if not db_path.startswith("sqlite:///"):
        # support both '/path/to.db' and 'sqlite:////path/to.db'
        if db_path.startswith("/"):
            engine_url = f"sqlite:///{db_path}"
        else:
            engine_url = f"sqlite:///{os.path.abspath(db_path)}"
    else:
        engine_url = db_path
    return create_engine(engine_url, future=True)

def load_vendors_df(engine) -> pd.DataFrame:
    """Load vendors table and ensure missing expected columns exist as empty strings."""
    with engine.begin() as conn:
        df = pd.read_sql(sql_text("SELECT * FROM vendors"), conn)

    for col in EXPECTED_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    for col in EXPECTED_COLUMNS:
        if col in df.columns and col != "id":
            df[col] = df[col].fillna("").astype(str)

    return df

def normalize_url(url: str) -> str:
    """Prepend https:// if scheme is missing."""
    url = (url or "").strip()
    if not url:
        return ""
    parsed = urlparse(url)
    if not parsed.scheme:
        return "https://" + url
    return url

def build_display_df(raw: pd.DataFrame) -> pd.DataFrame:
    """Reorder/rename columns and normalize Website URL for clickable links."""
    df = raw.copy()
    df["website"] = df["website"].apply(normalize_url)

    cols, rename_map = [], {}
    for raw_col, label in DISPLAY_ORDER:
        if raw_col in df.columns:
            cols.append(raw_col)
            rename_map[raw_col] = label

    df = df[cols].rename(columns=rename_map)
    return df

def apply_search(df_disp: pd.DataFrame, query: str) -> pd.DataFrame:
    """
    Multi-term AND search across a subset of columns.
    Splits on commas or whitespace. Empty query returns df unchanged.
    """
    q = (query or "").strip()
    if not q:
        return df_disp

    tokens = [t.strip().lower() for t in re.split(r"[,\s]+", q) if t.strip()]
    if not tokens:
        return df_disp

    SEARCH_COLUMNS = [
        "Business Name",
        "Contact Name",
        "Phone",
        "Address",
        "Category",
        "Service",
        "Notes",
        "Keywords",
        "Website",
    ]
    cols_present = [c for c in SEARCH_COLUMNS if c in df_disp.columns]

    combined = (
        df_disp[cols_present]
        .astype(str)
        .fillna("")
        .agg(" ".join, axis=1)
        .str.lower()
    )

    mask = combined.apply(lambda s: all(t in s for t in tokens))
    return df_disp[mask].reset_index(drop=True)

# -------------- UI -----------------
st.title("Vendors Directory (Read-Only)")

# Load data
engine = get_engine()
try:
    raw_df = load_vendors_df(engine)
except Exception as e:
    st.error(f"Failed to load vendors: {e}")
    st.stop()

# Build display frame
df_disp = build_display_df(raw_df)

# Sidebar controls
with st.sidebar:
    st.subheader("Filters")

    # Category filter (optional)
    cat_vals = sorted(v for v in raw_df["category"].dropna().astype(str).unique() if v.strip())
    selected_cat = st.selectbox("Category", options=["(All)"] + cat_vals, index=0)

    # Service filter (optional)
    svc_vals = sorted(v for v in raw_df["service"].dropna().astype(str).unique() if v.strip())
    selected_svc = st.selectbox("Service", options=["(All)"] + svc_vals, index=0)

    st.markdown("---")
    st.subheader("Search")
    search_q = st.text_input(
        "Multi-term AND search (comma or space-separated)",
        value="",
        placeholder="plumber, water heater",
    )

    st.markdown("---")
    st.subheader("Columns")
    show_notes = st.checkbox("Show Notes", value=True)
    show_keywords = st.checkbox("Show Keywords", value=True)

# Apply filters/search
work = df_disp.copy()

if selected_cat != "(All)" and "Category" in work.columns:
    work = work[work["Category"].astype(str) == selected_cat]

if selected_svc != "(All)" and "Service" in work.columns:
    work = work[work["Service"].astype(str) == selected_svc]

work = apply_search(work, search_q)

# Optionally hide columns
cols_to_hide = []
if not show_notes and "Notes" in work.columns:
    cols_to_hide.append("Notes")
if not show_keywords and "Keywords" in work.columns:
    cols_to_hide.append("Keywords")
if cols_to_hide:
    work = work.drop(columns=cols_to_hide)

# Column configuration (NO callables passed to LinkColumn)
col_config = {}
for label, width in COLUMN_MIN_WIDTHS.items():
    if label not in work.columns:
        continue
    if label == "Website":
        col_config[label] = st.column_config.LinkColumn(
            label="Website",
            help="Open vendor website",
            width=width,
            # Do NOT pass display_text as a function; it breaks JSON serialization.
            # If you want pretty labels later, we can precompute a text column.
        )
    else:
        col_config[label] = st.column_config.Column(
            label=label,
            width=width,
        )

# Info bar + CSV export
left, right = st.columns([1, 1])
with left:
    st.caption(f"Showing {len(work):,} of {len(df_disp):,} vendors after filters/search.")
with right:
    csv_bytes = work.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download CSV (filtered)",
        data=csv_bytes,
        file_name="vendors_filtered.csv",
        mime="text/csv",
    )

# Main table
st.dataframe(
    work,
    use_container_width=True,
    hide_index=True,
    height=640,
    column_config=col_config,
)

st.caption(
    "Tip: widen your browser window or zoom out to view more columns at once. "
    "Use the horizontal scrollbar to reach the last column."
)
