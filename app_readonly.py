# app_readonly.py — Vendors Directory (Read-only, Live DB via Turso; local fallback for dev)
# - Non-FTS AND search across all fields
# - Clickable Website links
# - Configurable character-width clipping per column (edit CHAR_WIDTHS)
# - Case-insensitive sort by Business Name (falls back if missing)
# - Safe if some columns are missing (only renders what exists)

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine
from sqlalchemy.sql import text as sql_text

APP_TITLE = "Vendors Directory (Read-only)"
st.set_page_config(page_title=APP_TITLE, layout="wide")

# ===== Display configuration you can change =====
# Only columns listed here (if present in DB) will be shown, in this order.
DISPLAY_COLUMNS: List[str] = [
    "category", "service", "business_name", "contact_name",
    "phone", "address", "website", "notes",
]

# Character clip widths by column **name** (applied before display). "website" is not clipped.
CHAR_WIDTHS: Dict[str, int] = {
    "business_name": 36,
    "category": 32,
    "service": 32,
    "contact_name": 28,
    "phone": 18,
    "address": 40,
    "notes": 60,
    # "website": (not clipped; shown as a link labeled "Open")
}

# Friendly display labels (optional)
LABELS: Dict[str, str] = {
    "category": "Category",
    "service": "Service",
    "business_name": "Business Name",
    "contact_name": "Contact Name",
    "phone": "Phone",
    "address": "Address",
    "website": "Website",
    "notes": "Notes",
}

# ===== Database wiring (Turso first, then local SQLite fallback) =====
@st.cache_resource(show_spinner=False)
def get_engine():
    # Prefer Turso (live remote DB)
    turso_url = os.environ.get("TURSO_DATABASE_URL") or st.secrets.get("TURSO_DATABASE_URL", "")
    turso_tok = os.environ.get("TURSO_AUTH_TOKEN")   or st.secrets.get("TURSO_AUTH_TOKEN", "")
    if turso_url and turso_tok:
        # Turso gives libsql://<host>. SQLAlchemy expects sqlite+libsql://<host>
        driver_url = turso_url.replace("libsql://", "sqlite+libsql://")
        return create_engine(
            f"{driver_url}?secure=true",
            connect_args={"auth_token": turso_tok},
            future=True,
        )

    # Fallback: local SQLite file next to this script (dev only)
    default_db = Path(__file__).resolve().parent / "vendors.db"
    db_url = os.environ.get("DB_URL") or f"sqlite:///{default_db}"
    return create_engine(db_url, future=True)

# ===== Helpers =====
def _cols(engine) -> List[str]:
    with engine.begin() as con:
        info = con.execute(sql_text("PRAGMA table_info(vendors)")).fetchall()
    return [str(r[1]) for r in info]


def load_df(engine) -> pd.DataFrame:
    cols = _cols(engine)
    desired = [c for c in DISPLAY_COLUMNS if c in cols]
    # Always fetch id internally for stable ops, even if we don't display it
    sel = (["id"] + desired) if "id" in cols else desired
    if not sel:
        return pd.DataFrame()
    order_col = "business_name" if "business_name" in cols else sel[0]
    with engine.begin() as con:
        df = pd.read_sql_query(
            f"SELECT {', '.join(sel)} FROM vendors ORDER BY {order_col} COLLATE NOCASE",
            con,
        )
    return df


def _s(x) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x.strip()
    return str(x).strip()


def clip_value(val: object, width: int) -> str:
    if width <= 0:
        return _s(val)
    t = _s(val)
    if len(t) <= width:
        return t
    return t[: max(width - 1, 0)] + "…"


def filter_df(df: pd.DataFrame, query: str) -> pd.DataFrame:
    q = _s(query)
    if not q:
        return df
    tokens = [t.lower() for t in re.split(r"\s+", q) if t]
    if not tokens:
        return df
    search_cols = [c for c in df.columns if c != "id"]
    lowered = df[search_cols].astype(str).apply(lambda s: s.str.lower())

    mask = pd.Series(True, index=df.index)
    for tok in tokens:
        mask &= lowered.apply(lambda s: s.str.contains(tok, na=False)).any(axis=1)
    return df[mask]


# ===== UI =====
def main():
    st.title(APP_TITLE)

    engine = get_engine()
    df = load_df(engine)

    if df.empty:
        st.info("No vendor data found.")
        return

    # Search (AND across all fields)
    q = st.text_input(
        "Search (AND across all fields)",
        value=st.session_state.get("q", ""),
        placeholder="e.g., plumber alamo heights",
        key="q",
        help="Type one or more words. We use AND logic across all fields (case-insensitive).",
    )

    df = filter_df(df, q)

    # Keep only the desired display columns that exist
    present = [c for c in DISPLAY_COLUMNS if c in df.columns]
    if not present:
        st.info("No displayable columns found.")
        return
    df = df[present]

    # Apply character clipping (except website)
    disp = df.copy()
    for col, width in CHAR_WIDTHS.items():
        if col in disp.columns and col != "website":
            disp[col] = disp[col].map(lambda v: clip_value(v, width))

    # Rename columns for friendly labels
    disp = disp.rename(columns={c: LABELS.get(c, c) for c in disp.columns})

    # Configure Website as a clickable link labeled "Open"
    col_config = {}
    if "Website" in disp.columns:
        try:
            col_config["Website"] = st.column_config.LinkColumn("Website", display_text="Open")
        except Exception:
            # Older Streamlit versions may not have LinkColumn; leave as plain text
            pass

    st.dataframe(disp, use_container_width=True, hide_index=True, column_config=col_config)

    with st.expander("About this directory", expanded=False):
        st.markdown(
            """
            **Live data**: This page reads directly from the configured database (Turso if set, otherwise local SQLite).
            Use the search box to filter; multiple words are combined with AND.
            To adjust column clipping widths, edit the `CHAR_WIDTHS` dict near the top of the file.
            """
        )


if __name__ == "__main__":
    main()
