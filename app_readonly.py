# app_readonly.py
# Vendors — Read-only (AG Grid)
# - st_aggrid table with ONE global Quick Search (AND across tokens), NO per-column filters/menus
# - Turso/libSQL preferred (sqlite+libsql with auth token), clean fallback to local SQLite vendors.db
# - Column labels & pixel widths from secrets (with sane defaults; partial overrides OK)
# - Website column rendered as a clickable "Open" link (new tab)
# - Wide page layout & sidebar state from secrets
# - CSV download (respects current quick search) + a thorough debug/status expander
#
# Secrets you can set (all optional except DB if using Turso):
#   page_title, page_max_width_px, sidebar_state,
#   READONLY_HELP_TITLE, READONLY_HELP_MD,
#   READONLY_COLUMN_LABELS, COLUMN_WIDTHS_PX_READONLY,
#   TURSO_DATABASE_URL, TURSO_AUTH_TOKEN
#
# Requirements (requirements.txt):
#   streamlit==1.39.0
#   pandas==2.2.2
#   sqlalchemy==2.0.34
#   streamlit-aggrid==0.3.4.post3
#   libsql-experimental==0.1.7
#
# Optional runtime pin (runtime.txt):
#   python-3.11.9

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text as sql_text
from sqlalchemy.engine import Engine

from st_aggrid import (
    AgGrid,
    GridOptionsBuilder,
    GridUpdateMode,
    JsCode,
    ColumnsAutoSizeMode,
)

# =========================
# Early secret/env access
# =========================
def _read_secret_early(name: str, default=None):
    """Read from st.secrets if available, else environment, else default."""
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.environ.get(name, default)

# =========================
# Page config (must be early)
# =========================
PAGE_TITLE = _read_secret_early("page_title", "Vendors (Read-only)")
SIDEBAR_STATE = (_read_secret_early("sidebar_state", "auto") or "auto").lower()
if SIDEBAR_STATE not in {"auto", "expanded", "collapsed"}:
    SIDEBAR_STATE = "auto"
st.set_page_config(page_title=PAGE_TITLE, layout="wide", initial_sidebar_state=SIDEBAR_STATE)

def _apply_page_width_css():
    max_width_px = _read_secret_early("page_max_width_px", None)
    try:
        max_width_px = int(max_width_px) if max_width_px is not None else None
    except Exception:
        max_width_px = None

    css = ["<style>"]
    if max_width_px:
        css.append(
            f""".block-container {{
                    max-width: {max_width_px}px;
                }}"""
        )
    # Ensure cells wrap inside AG Grid
    css.append(
        """
        .ag-theme-streamlit .ag-cell {
            white-space: normal !important;
            line-height: 1.25rem !important;
        }
        """
    )
    css.append("</style>")
    st.markdown("\n".join(css), unsafe_allow_html=True)

_apply_page_width_css()

# =========================
# Help content & column config
# =========================
HELP_TITLE = _read_secret_early("READONLY_HELP_TITLE", "Provider Help / Tips")
HELP_MD = _read_secret_early(
    "READONLY_HELP_MD",
    "Use the global Quick Search (below) for partial word matches; tokens are ANDed. Click headers to sort."
)

# Default DB→Display label mapping
DEFAULT_LABELS: Dict[str, str] = {
    "id": "ID",
    "category": "Category",
    "service": "Service",
    "business_name": "Business Name",
    "contact_name": "Contact Name",
    "phone": "Phone",
    "address": "Address",
    "website": "Website",
    "notes": "Notes",
    "keywords": "Keywords",
}
LABEL_OVERRIDES = _read_secret_early("READONLY_COLUMN_LABELS", {})
DISPLAY_LABELS = {**DEFAULT_LABELS, **LABEL_OVERRIDES} if isinstance(LABEL_OVERRIDES, dict) else DEFAULT_LABELS

# Pixel widths per DB column key
DEFAULT_WIDTHS_PX: Dict[str, int] = {
    "id": 60,
    "business_name": 300,
    "category": 220,
    "service": 220,
    "contact_name": 160,
    "phone": 140,
    "address": 300,
    "website": 240,
    "notes": 320,
    "keywords": 140,
}
WIDTHS_RAW = _read_secret_early("COLUMN_WIDTHS_PX_READONLY", None)
COLUMN_WIDTHS_PX: Dict[str, int] = {**DEFAULT_WIDTHS_PX, **WIDTHS_RAW} if isinstance(WIDTHS_RAW, dict) else DEFAULT_WIDTHS_PX

# Preferred display order (URL right after Address; ID last)
DISPLAY_ORDER = [
    "category", "business_name", "contact_name", "phone",
    "address", "website", "notes", "keywords", "id"
]
DISPLAY_ORDER = [c for c in DISPLAY_ORDER if c in DEFAULT_LABELS]

# =========================
# DB Engine (Turso → SQLite)
# =========================
def build_engine() -> Tuple[Engine, dict]:
    """Create SQLAlchemy engine preferring Turso/libSQL; fallback to local SQLite."""
    libsql_url = _read_secret_early("TURSO_DATABASE_URL", None)
    if not libsql_url:
        raw = _read_secret_early("LIBSQL_URL", None)
        if raw and raw.startswith("libsql://"):
            libsql_url = f"sqlite+libsql://{raw[len('libsql://'):]}"
    auth_token = _read_secret_early("TURSO_AUTH_TOKEN", _read_secret_early("LIBSQL_AUTH_TOKEN", None))

    if libsql_url and libsql_url.startswith("libsql://"):
        libsql_url = "sqlite+libsql://" + libsql_url[len("libsql://"):]

    # Try remote first
    if libsql_url and auth_token:
        try:
            eng = create_engine(libsql_url, connect_args={"auth_token": auth_token})
            with eng.connect() as conn:
                conn.execute(sql_text("SELECT 1"))
            return eng, {"using_remote": True, "engine_url": libsql_url, "driver": "libsql"}
        except Exception as e:
            st.warning(f"Turso connection failed ({e!r}). Falling back to local SQLite vendors.db.")

    # Local SQLite fallback (portable path detection)
    candidates = [
        os.path.join(os.getcwd(), "vendors.db"),
        "/mount/src/vendors-readonly-app/vendors.db",
        "/app/vendors-readonly-app/vendors.db",
    ]
    db_path = next((p for p in candidates if os.path.exists(p)), candidates[0])
    eng = create_engine(f"sqlite:///{db_path}")
    return eng, {"using_remote": False, "engine_url": f"sqlite:///{db_path}", "driver": "sqlite"}

engine, engine_info = build_engine()

# =========================
# Data Load
# =========================
@st.cache_data(show_spinner=False)
def load_vendors(_engine: Engine) -> pd.DataFrame:
    cols = [
        "id", "category", "service", "business_name", "contact_name",
        "phone", "address", "website", "notes", "keywords"
    ]
    with _engine.begin() as conn:
        info = pd.read_sql(sql_text("PRAGMA table_info(vendors)"), conn)
        available = set(info["name"].tolist())
        select_cols = [c for c in cols if c in available]
        if not select_cols:
            raise RuntimeError("vendors table exists but none of the expected columns were found.")
        df = pd.read_sql(sql_text(f"SELECT {', '.join(select_cols)} FROM vendors"), conn)

    # Normalize types
    for c in df.columns:
        if c != "id":
            df[c] = df[c].astype("string").fillna("")
    return df

try:
    df_raw = load_vendors(engine)
except Exception as e:
    st.error(f"Error loading vendors: {e}")
    st.stop()

# =========================
# UI — Header & Help
# =========================
st.title(PAGE_TITLE)

# Show Help/Tips via popover if available, else expander
try:
    with st.popover(HELP_TITLE):  # Streamlit >= 1.30
        st.markdown(HELP_MD or "_No help content configured._")
except AttributeError:
    with st.expander(HELP_TITLE, expanded=False):
        st.markdown(HELP_MD or "_No help content configured._")

# =========================
# Global Quick Search (under Help)
# =========================
def _tokenize(q: str) -> List[str]:
    return [t.strip() for t in q.split() if t.strip()]

search_query = st.text_input(
    "Quick Search (partial words; AND across words). e.g., `plumb repair`",
    key="quick_search",
    help="Applies to all visible columns. Case-insensitive partial matches; tokens are ANDed."
).strip()

# We'll use AG Grid's built-in quickFilter for display AND also filter in Python
# so that the CSV download matches the search results exactly.
def py_filter(df: pd.DataFrame, query: str) -> pd.DataFrame:
    if not query:
        return df
    tokens = _tokenize(query.lower())
    if not tokens:
        return df
    text_cols = [c for c in df.columns if c != "id"]
    df2 = df.copy()
    for c in text_cols:
        df2[c] = df2[c].astype("string").fillna("")
    joined = df2[text_cols].agg(" ".join, axis=1).str.lower()
    mask = pd.Series(True, index=df2.index)
    for t in tokens:
        mask &= joined.str.contains(t, na=False)
    return df2[mask]

df_filtered = py_filter(df_raw, search_query)

# =========================
# Prepare display frame (labels + order)
# =========================
def apply_labels_and_order(df: pd.DataFrame) -> pd.DataFrame:
    for c in DEFAULT_LABELS:
        if c not in df.columns:
            df[c] = "" if c != "id" else pd.NA
    cols = [c for c in DISPLAY_ORDER if c in df.columns]
    dfx = df[cols].copy()
    rename_map = {c: DISPLAY_LABELS.get(c, c) for c in cols}
    dfx.rename(columns=rename_map, inplace=True)
    return dfx

df_display = apply_labels_and_order(df_filtered)
st.caption(f"{len(df_display):,} vendor(s) shown")

# Reverse map labels -> DB keys (used for widths)
label_to_key = {DISPLAY_LABELS.get(k, k): k for k in DISPLAY_LABELS}

# =========================
# AG Grid setup (NO per-column filters/menus)
# =========================
# Renderer for Website links
link_renderer = JsCode(
    """
    class LinkRenderer {
      init(params) {
        const url = (params.value || '').trim();
        const a = document.createElement('a');
        a.href = url || '#';
        a.target = '_blank';
        a.rel = 'noopener noreferrer';
        a.innerText = url ? 'Open' : '';
        this.eGui = a;
      }
      getGui() { return this.eGui; }
    }
    """
)

g = GridOptionsBuilder.from_dataframe(df_display)

# Global defaults: no per-column filters, no menus, sorting & resizing on, wrap text
g.configure_default_column(
    filter=False,               # disable per-column filter UI
    sortable=True,
    resizable=True,
    wrapText=True,
    autoHeight=True,
    suppressMenu=True,          # hide the column menu
)

# Column widths + special Website renderer
for disp_col in df_display.columns:
    db_key = label_to_key.get(disp_col, disp_col)
    width_px = COLUMN_WIDTHS_PX.get(db_key, None)

    if db_key == "website":
        g.configure_column(
            disp_col,
            width=width_px,
            cellRenderer=link_renderer,
            suppressSizeToFit=True,
        )
    else:
        g.configure_column(
            disp_col,
            width=width_px,
            suppressSizeToFit=True,
        )

grid_options = g.build()
# Wire the grid's built-in quick filter to our search box
grid_options["quickFilterText"] = search_query

AgGrid(
    df_display,
    gridOptions=grid_options,
    update_mode=GridUpdateMode.NO_UPDATE,
    allow_unsafe_jscode=True,
    theme="streamlit",
    height=600,
    fit_columns_on_grid_load=False,
    columns_auto_size_mode=ColumnsAutoSizeMode.NO_AUTOSIZE,
)

# =========================
# Download (CSV uses DB keys; respects Quick Search)
# =========================
csv_bytes = df_filtered.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download providers CSV",
    data=csv_bytes,
    file_name="vendors_readonly_export.csv",
    mime="text/csv"
)

# =========================
# Debug / Status
# =========================
with st.expander("Status & Secrets (debug)", expanded=False):
    st.subheader("DB")
    st.code({
        "backend": engine_info.get("driver"),
        "dsn": engine_info.get("engine_url"),
        "auth": "token_set" if _read_secret_early("TURSO_AUTH_TOKEN", _read_secret_early("LIBSQL_AUTH_TOKEN")) else "none"
    }, language="json")

    keys_present = []
    try:
        keys_present = list(st.secrets.keys())
    except Exception:
        pass

    st.subheader("Secrets keys (present)")
    st.write(keys_present if keys_present else "No st.secrets or cannot enumerate in this environment.")

    st.subheader("Help Content")
    st.write({"title": HELP_TITLE, "md_len": len(HELP_MD or "")})

    st.subheader("Raw column widths (from secrets, merged with defaults)")
    st.code(COLUMN_WIDTHS_PX, language="json")

    st.subheader("Display labels (merged)")
    st.code(DISPLAY_LABELS, language="json")

    st.subheader("Database Status & Schema (snapshot)")
    try:
        with engine.begin() as conn:
            try:
                pragma = pd.read_sql(sql_text("PRAGMA table_info(vendors)"), conn)
                cols = pragma[["name", "type"]].values.tolist()
            except Exception:
                cols = []
            try:
                ct = pd.read_sql(sql_text("SELECT COUNT(*) AS c FROM vendors"), conn)["c"].iloc[0]
            except Exception:
                ct = None
    except Exception as e:
        cols, ct = [], None
        st.write(f"Error checking schema/counts: {e!r}")

    st.code({
        "engine_url": engine_info.get("engine_url"),
        "using_remote": engine_info.get("using_remote"),
        "page_max_width_px": _read_secret_early("page_max_width_px", None),
        "displayed_columns": list(df_display.columns),
        "counts": {"vendors": int(ct) if ct is not None else "unknown"},
        "vendors_columns": [c for c, _t in cols] if cols else "unknown"
    }, language="json")
