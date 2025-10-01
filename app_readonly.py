# app_readonly.py — Vendors Directory (Read-only, Live DB via Turso; local fallback)
# - Uses the SAME live DB as Admin: supports LIBSQL_URL/LIBSQL_AUTH_TOKEN and TURSO_DATABASE_URL/TURSO_AUTH_TOKEN
# - Non-FTS AND search across all fields
# - Clickable Website links (scheme normalized)
# - Configurable character-width clipping per column (CHAR_WIDTHS)
# - Page width knob (PAGE_MAX_WIDTH_PX) so you can see entire last column with horizontal scroll
# - Case-insensitive sort by Business Name (falls back if missing)
# - Safe if some columns are missing

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List
from urllib.parse import urlparse

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine
from sqlalchemy.sql import text as sql_text

APP_TITLE = "Vendors Directory (Read-only)"
st.set_page_config(page_title=APP_TITLE, layout="wide")

# ===== Display configuration (edit as needed) =====
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
    # "website": (not clipped)
}

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

# Page width knob: increase if you still can't fully see the last column.
PAGE_MAX_WIDTH_PX = int(os.getenv("PAGE_MAX_WIDTH_PX", "2300"))  # try 2400–2600 on small monitors
WEBSITE_WIDTH_PX = int(os.getenv("WEBSITE_COL_WIDTH_PX", "300"))

# Rough “characters to pixels” conversion for non-website columns in st.dataframe
# This isn’t perfect but keeps columns readable. Tweak if needed.
CHARS_TO_PX = int(os.getenv("CHARS_TO_PX", "10"))  # ~10 px per character as a starting point
EXTRA_COL_PADDING = int(os.getenv("EXTRA_COL_PADDING_PX", "24"))

# ===== Page width CSS =====
st.markdown(f"""
<style>
.block-container {{
  max-width: {PAGE_MAX_WIDTH_PX}px;
  padding-left: 1rem;
  padding-right: 1rem;
}}
</style>
""", unsafe_allow_html=True)

# ===== Database wiring (Turso first; SQLite fallback) =====
def _get_secret(*keys: str) -> str:
    """Return first non-empty among env vars or st.secrets for given keys."""
    for k in keys:
        v = os.environ.get(k, "") or st.secrets.get(k, "")
        if v:
            return str(v)
    return ""

@st.cache_resource(show_spinner=False)
def get_engine():
    # Accept BOTH naming styles used across your apps:
    # Old/Admin style:
    libsql_url = _get_secret("LIBSQL_URL", "LIBSQL_DATABASE_URL")
    libsql_tok = _get_secret("LIBSQL_AUTH_TOKEN")
    # Turso naming:
    turso_url = _get_secret("TURSO_DATABASE_URL")
    turso_tok = _get_secret("TURSO_AUTH_TOKEN")

    # Prefer Turso/libsql remote if either pair is present (Admin & Read-only share these)
    db_url_raw = libsql_url or turso_url
    auth_tok = libsql_tok or turso_tok

    if db_url_raw and auth_tok:
        # SQLAlchemy needs sqlite+libsql://... (not libsql://...)
        driver_url = db_url_raw.replace("libsql://", "sqlite+libsql://")
        # Add secure flag for remote connection
        driver_url = f"{driver_url}?secure=true"
        return create_engine(driver_url, connect_args={"auth_token": auth_tok}, future=True)

    # Fallback: local SQLite file (dev only). Supports explicit path via env/secrets.
    sqlite_path = _get_secret("SQLITE_PATH")
    if not sqlite_path:
        sqlite_path = str(Path(__file__).resolve().parent / "vendors.db")
    if not sqlite_path.startswith("sqlite:///"):
        if sqlite_path.startswith("/"):
            sqlite_url = f"sqlite:///{sqlite_path}"
        else:
            sqlite_url = f"sqlite:///{os.path.abspath(sqllite_path)}"  # noqa: F821 (typo caught below)
    else:
        sqlite_url = sqlite_path
    # Fix potential variable typo
    try:
        return create_engine(sqlite_url, future=True)
    except NameError:
        sqlite_url = f"sqlite:///{sqlite_path}" if not sqlite_path.startswith("sqlite:///") else sqlite_path
        return create_engine(sqlite_url, future=True)

# ===== Helpers =====
def _cols(engine) -> List[str]:
    with engine.begin() as con:
        info = con.execute(sql_text("PRAGMA table_info(vendors)")).fetchall()
    return [str(r[1]) for r in info]

def _s(x) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x.strip()
    return str(x).strip()

def normalize_url(u: str) -> str:
    s = _s(u)
    if not s:
        return ""
    parsed = urlparse(s)
    return s if parsed.scheme else f"https://{s}"

def load_df(engine) -> pd.DataFrame:
    cols = _cols(engine)
    desired = [c for c in DISPLAY_COLUMNS if c in cols]
    sel = (["id"] + desired) if "id" in cols else desired
    if not sel:
        return pd.DataFrame()
    order_col = "business_name" if "business_name" in cols else sel[0]
    with engine.begin() as con:
        df = pd.read_sql_query(
            f"SELECT {', '.join(sel)} FROM vendors ORDER BY {order_col} COLLATE NOCASE",
            con,
        )
    # Ensure strings and normalize website links for clickability
    for c in desired:
        if c in df.columns and c != "id":
            df[c] = df[c].fillna("").astype(str)
    if "website" in df.columns:
        df["website"] = df["website"].map(normalize_url)
    return df

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
    # split on spaces or commas; AND semantics
    tokens = [t.lower() for t in re.split(r"[,\s]+", q) if t]
    if not tokens:
        return df
    search_cols = [c for c in df.columns if c != "id"]
    lowered = df[search_cols].astype(str).apply(lambda s: s.str.lower())
    mask = pd.Series(True, index=df.index)
    for tok in tokens:
        mask &= lowered.apply(lambda s: s.str.contains(tok, na=False)).any(axis=1)
    return df[mask]

def build_col_config(displayed_cols: List[str]) -> Dict[str, st.column_config.BaseColumn]:
    """
    Build Streamlit column_config with explicit widths.
    Approximate char widths -> px so the last column stays visible when you scroll.
    """
    col_config: Dict[str, st.column_config.BaseColumn] = {}
    # In dataframe, we’ll rename columns to user-friendly labels;
    # create a reverse map from labels so we can assign configs by label.
    label_map = {c: LABELS.get(c, c) for c in displayed_cols}

    for raw_col in displayed_cols:
        label = label_map[raw_col]
        if raw_col == "website":
            try:
                col_config[label] = st.column_config.LinkColumn(
                    label=label,
                    width=WEBSITE_WIDTH_PX,
                )
            except Exception:
                # older Streamlit versions may not have LinkColumn
                pass
            continue

        # For text columns, compute a width based on desired character clipping width if present.
        w_chars = CHAR_WIDTHS.get(raw_col, 24)  # default a reasonable char width
        width_px = w_chars * CHARS_TO_PX + EXTRA_COL_PADDING
        col_config[label] = st.column_config.Column(
            label=label,
            width=width_px,
        )
    return col_config

# ===== UI =====
def main():
    st.title(APP_TITLE)

    engine = get_engine()
    df = load_df(engine)

    if df.empty:
        st.info("No vendor data found.")
        with st.expander("Debug"):
            st.json({"engine_url": str(engine.url)})
        return

    # Search (AND across all fields)
    q = st.text_input(
        "Search (AND across all fields)",
        value=st.session_state.get("q", ""),
        placeholder="e.g., plumber alamo heights",
        key="q",
        help="Type words separated by space or comma. AND logic (case-insensitive).",
    )
    df = filter_df(df, q)

    # Keep only desired display columns that exist
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

    # Rename to friendly labels for display
    disp = disp.rename(columns={c: LABELS.get(c, c) for c in disp.columns})

    # Build explicit column widths so the last column stays fully visible when scrolling
    col_config = build_col_config(displayed_cols=[c for c in df.columns])

    st.dataframe(
        disp,
        use_container_width=True,
        hide_index=True,
        column_config=col_config,
        height=680,  # tweak if you want more vertical space
    )

    with st.expander("Debug", expanded=False):
        st.json({
            "engine_url": str(engine.url),
            "using_remote": str(engine.url).startswith("sqlite+libsql://"),
            "page_max_width_px": PAGE_MAX_WIDTH_PX,
            "chars_to_px": CHARS_TO_PX,
            "extra_col_padding_px": EXTRA_COL_PADDING,
            "displayed_columns": list(disp.columns),
            "row_count": int(len(disp)),
        })

if __name__ == "__main__":
    main()
