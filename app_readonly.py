# app_readonly.py — Vendors Directory (Read-only, Live DB via Turso; local fallback)
# - Uses the SAME live DB as Admin (LIBSQL_* or TURSO_* secrets)
# - NO content clipping (shows full strings from DB)
# - Admin-set default column widths via secrets/env/file; end users can still drag-resize
# - Explicit page/table width so rightmost column is fully reachable with horizontal scroll
# - AND search across ALL DB columns (case-insensitive, substring match)
# - Search label text: "Search (all partial or full word or words):" at ~16pt, not bold
# - Display columns: Category, Business Name, Contact Name, Phone, Address, Website, Notes, Keywords
# - Wider horizontal & vertical scrollbars on the table (WebKit + Firefox)

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Dict, List
from urllib.parse import urlparse

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine
from sqlalchemy.sql import text as sql_text

APP_TITLE = "Vendors Directory (Read-only)"
st.set_page_config(page_title=APP_TITLE, layout="wide")

# ===== Display configuration =====
# Only these are SHOWN (in this order). Search will use ALL DB columns.
DISPLAY_COLUMNS: List[str] = [
    "category",
    "business_name",
    "contact_name",
    "phone",
    "address",
    "website",
    "notes",
    "keywords",
]

LABELS: Dict[str, str] = {
    "category": "Category",
    "business_name": "Business Name",
    "contact_name": "Contact Name",
    "phone": "Phone",
    "address": "Address",
    "website": "Website",
    "notes": "Notes",
    "keywords": "Keywords",
}

# ===== Int config helper (reads secrets first, then env, then default) =====
def _int_config(key: str, default: int) -> int:
    val = st.secrets.get(key, None)
    if val is not None:
        try:
            return int(val)
        except Exception:
            pass
    env = os.getenv(key, "")
    if env:
        try:
            return int(env)
        except Exception:
            pass
    return int(default)

# Page width & sizing knobs (secrets override env)
PAGE_MAX_WIDTH_PX = _int_config("PAGE_MAX_WIDTH_PX", 2300)
WEBSITE_WIDTH_PX  = _int_config("WEBSITE_COL_WIDTH_PX", 300)
CHARS_TO_PX       = _int_config("CHARS_TO_PX", 10)   # heuristic only for fallback widths
EXTRA_COL_PADDING = _int_config("EXTRA_COL_PADDING_PX", 24)

# ===== Page, Search, and SCROLLBAR CSS =====
# - Scrollbars widened *only* inside the table’s grid to avoid affecting the whole page.
# - WebKit (::webkit-scrollbar*) and Firefox (scrollbar-width/scrollbar-color) supported.
st.markdown(f"""
<style>
/* Page width */
.block-container {{
  max-width: {PAGE_MAX_WIDTH_PX}px;
  padding-left: 1rem;
  padding-right: 1rem;
}}

/* Search label styling: ~16pt, not bold */
.search-label {{ font-size: 16pt; font-weight: normal; margin: 0 0 .25rem 0; }}
/* Make the search input text also ~16pt */
div[data-baseweb="input"] input {{ font-size: 16pt; }}

/* === Wider scrollbars for the dataframe grid === */
[data-testid="stDataFrame"] div[role="grid"] {{
  scrollbar-width: thick;                 /* Firefox width */
  scrollbar-color: #a3a3a3 #e7e7e7;       /* thumb color, track color */
}}

/* WebKit (Chrome, Edge, Safari) */
[data-testid="stDataFrame"] div[role="grid"]::-webkit-scrollbar {{
  height: 16px;   /* horizontal bar thickness */
  width: 16px;    /* vertical bar thickness */
}}
[data-testid="stDataFrame"] div[role="grid"]::-webkit-scrollbar-track {{
  background: #e7e7e7;
  border-radius: 8px;
}}
[data-testid="stDataFrame"] div[role="grid"]::-webkit-scrollbar-thumb {{
  background: #a3a3a3;
  border-radius: 8px;
  border: 3px solid #e7e7e7; /* increases grab area visually */
}}
[data-testid="stDataFrame"] div[role="grid"]::-webkit-scrollbar-thumb:hover {{
  background: #8a8a8a;
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
    libsql_url = _get_secret("LIBSQL_URL", "LIBSQL_DATABASE_URL")
    libsql_tok = _get_secret("LIBSQL_AUTH_TOKEN")
    turso_url  = _get_secret("TURSO_DATABASE_URL")
    turso_tok  = _get_secret("TURSO_AUTH_TOKEN")

    db_url_raw = libsql_url or turso_url
    auth_tok   = libsql_tok or turso_tok

    if db_url_raw and auth_tok:
        # SQLAlchemy needs sqlite+libsql://... (not libsql://...)
        driver_url = db_url_raw.replace("libsql://", "sqlite+libsql://")
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
            sqlite_url = f"sqlite:///{os.path.abspath(sqlite_path)}"
    else:
        sqlite_url = sqlite_path
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
    """
    Load **all columns** from vendors (plus id if present) so search can see everything.
    Display slicing happens later using DISPLAY_COLUMNS.
    """
    cols = _cols(engine)
    if not cols:
        return pd.DataFrame()

    fetch_cols = list(cols)  # all table columns
    sel = (["id"] + [c for c in fetch_cols if c != "id"]) if "id" in fetch_cols else fetch_cols

    # Choose a stable order-by if we have business_name, else first column
    order_col = "business_name" if "business_name" in fetch_cols else fetch_cols[0]

    with engine.begin() as con:
        df = pd.read_sql_query(
            f"SELECT {', '.join(sel)} FROM vendors ORDER BY {order_col} COLLATE NOCASE",
            con,
        )

    # Normalize strings and website
    for c in df.columns:
        if c != "id":
            df[c] = df[c].fillna("").astype(str)
    if "website" in df.columns:
        df["website"] = df["website"].map(normalize_url)
    return df

def filter_df(df: pd.DataFrame, query: str) -> pd.DataFrame:
    """
    AND search across **all non-id columns** (case-insensitive, substring).
    """
    q = _s(query)
    if not q:
        return df
    tokens = [t.lower() for t in re.split(r"[,\s]+", q) if t]
    if not tokens:
        return df

    search_cols = [c for c in df.columns if c != "id"]
    if not search_cols:
        return df

    lowered = df[search_cols].astype(str).apply(lambda s: s.str.lower())
    mask = pd.Series(True, index=df.index)
    for tok in tokens:
        # row matches if *any* column contains tok; AND across tokens
        mask &= lowered.apply(lambda s: s.str.contains(tok, na=False)).any(axis=1)
    return df[mask]

# ===== Admin width loading (no user UI) =====
def _load_admin_widths_px(displayed_cols: List[str]) -> Dict[str, int]:
    """
    Load initial widths (px) from:
      1) st.secrets["COLUMN_WIDTHS_PX"]  (table/dict: {raw_col: px})
      2) env var COLUMN_WIDTHS_JSON      (JSON string)
      3) ./column_widths.json            (JSON file in repo)
    Fallback: simple heuristic (chars→px) + padding if no explicit widths provided.
    """
    # 1) secrets table/mapping
    try:
        cfg = st.secrets.get("COLUMN_WIDTHS_PX", None)
        if isinstance(cfg, Mapping):
            return {k: int(cfg[k]) for k in cfg if k in displayed_cols}
    except Exception:
        pass

    # 2) env var JSON
    try:
        raw = os.getenv("COLUMN_WIDTHS_JSON", "")
        if raw:
            data = json.loads(raw)
            if isinstance(data, dict):
                return {k: int(data[k]) for k in data if k in displayed_cols}
    except Exception:
        pass

    # 3) local JSON file
    try:
        p = Path(__file__).resolve().parent / "column_widths.json"
        if p.exists():
            with open(p, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                return {k: int(data[k]) for k in data if k in displayed_cols}
    except Exception:
        pass

    # Fallback: provide a reasonable starting width per column (no clipping)
    approx_chars = {
        "business_name": 36,
        "category": 32,
        "contact_name": 28,
        "phone": 18,
        "address": 40,
        "website": 30,
        "notes": 60,
        "keywords": 40,
    }
    return {
        col: approx_chars.get(col, 24) * CHARS_TO_PX + EXTRA_COL_PADDING
        for col in displayed_cols
    }

def build_col_config(displayed_cols: List[str], default_widths_px: Dict[str, int]) -> Dict[str, st.column_config.BaseColumn]:
    """
    Build column_config with **admin-set** initial widths (no user controls here).
    Users can still drag-resize in the UI; we do not expose any inputs in this app.
    """
    col_config: Dict[str, st.column_config.BaseColumn] = {}
    label_map = {c: LABELS.get(c, c) for c in displayed_cols}

    for raw_col in displayed_cols:
        label = label_map[raw_col]
        target_px = int(default_widths_px.get(raw_col, 240))  # guardrail

        if raw_col == "website":
            try:
                col_config[label] = st.column_config.LinkColumn(
                    label=label,
                    width=target_px,
                )
            except Exception:
                # Older Streamlit may not support LinkColumn
                col_config[label] = st.column_config.Column(label=label, width=target_px)
        else:
            col_config[label] = st.column_config.Column(label=label, width=target_px)

    return col_config

# ===== UI =====
def main():
    engine = get_engine()
    df = load_df(engine)  # includes ALL columns

    if df.empty:
        st.info("No vendor data found.")
        with st.expander("Debug"):
            st.json({"engine_url": str(engine.url)})
        return

    # Search — custom label at ~16pt, not bold
    st.markdown('<div class="search-label">Search (all partial or full word or words):</div>', unsafe_allow_html=True)
    q = st.text_input(
        label="Search (all partial or full word or words):",
        value=st.session_state.get("q", ""),
        placeholder="e.g., plumb returns any record with plumb, plumber, plumbers, plumbing, etc.",
        key="q",
        help="Type one or more words (partial or full). Each word is matched as a substring across all columns; all words must match (AND, case-insensitive).",
        label_visibility="collapsed",
    )

    # AND search across ALL columns
    df = filter_df(df, q)

    # Slice to ONLY the display columns (in order) that actually exist
    present = [c for c in DISPLAY_COLUMNS if c in df.columns]
    if not present:
        st.info("No displayable columns found.")
        return
    df = df[present]

    # Rename to friendly labels for display
    disp = df.rename(columns={c: LABELS.get(c, c) for c in df.columns})

    # Admin-defined initial widths (no UI). Users can still drag-resize in the table.
    default_widths_px = _load_admin_widths_px(displayed_cols=[c for c in df.columns])
    col_config = build_col_config(displayed_cols=[c for c in df.columns],
                                  default_widths_px=default_widths_px)

    # Keep column order explicit to avoid surprises
    column_order = [LABELS.get(c, c) for c in df.columns]

    # IMPORTANT: don't stretch; use explicit width so defaults "take"
    st.dataframe(
        disp,
        use_container_width=False,
        width=PAGE_MAX_WIDTH_PX,
        hide_index=True,
        column_config=col_config,
        column_order=column_order,
        height=680,
    )

    with st.expander("Debug", expanded=False):
        st.json({
            "engine_url": str(engine.url),
            "using_remote": str(engine.url).startswith("sqlite+libsql://"),
            "page_max_width_px": PAGE_MAX_WIDTH_PX,
            "displayed_columns": list(disp.columns),
            "default_widths_px": default_widths_px,
            "row_count": int(len(disp)),
        })

if __name__ == "__main__":
    main()
