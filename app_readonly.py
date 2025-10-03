# app_readonly.py — HCR Vendors (Read-Only) v3.4 with st_aggrid auto-height
# - Labels, widths, Help all come from secrets (same as Admin)
# - Real auto-expanding rows via st_aggrid (wrapText + autoHeight); falls back to st.dataframe if not installed
# - Website column shows "Website" (clickable) only if URL is valid; adjacent "Website URL" shows the full link
# - CSV download; Status & Secrets (debug) at end
# - No edits/audit here — strictly read-only

from __future__ import annotations

import os
from urllib.parse import urlparse
from typing import Dict, List

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine
from sqlalchemy.sql import text as sql_text

# Try to load st_aggrid (auto-expand rows)
try:
    from st_aggrid import AgGrid, GridOptionsBuilder, JsCode, GridUpdateMode, ColumnsAutoSizeMode
    _AGGRID_AVAILABLE = True
except Exception:
    _AGGRID_AVAILABLE = False


# -----------------------------
# Secrets / env helpers
# -----------------------------
def _read_secret(name: str, default=None):
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.environ.get(name, default)

def _get_bool(val, default=False):
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() in ("1", "true", "yes", "on")
    return bool(default)


# -----------------------------
# Page config / layout
# -----------------------------
def _apply_layout():
    st.set_page_config(
        page_title=_read_secret("page_title", "HCR Vendors (Read-Only)"),
        layout="wide",
        initial_sidebar_state=_read_secret("sidebar_state", "expanded"),
    )
    maxw = _read_secret("page_max_width_px", 2300)
    try:
        maxw = int(maxw)
    except Exception:
        maxw = 2300
    st.markdown(
        f"""
        <style>
        .block-container {{ max-width: {maxw}px; }}
        </style>
        """,
        unsafe_allow_html=True,
    )

_apply_layout()


# -----------------------------
# Config from secrets
# -----------------------------
LABEL_OVERRIDES: Dict[str, str] = _read_secret("READONLY_COLUMN_LABELS", {}) or {}
COLUMN_WIDTHS: Dict[str, int]   = _read_secret("COLUMN_WIDTHS_PX_READONLY", {}) or {}
STICKY_FIRST: bool              = _get_bool(_read_secret("READONLY_STICKY_FIRST_COL", False), False)

HELP_TITLE: str = (
    _read_secret("READONLY_HELP_TITLE")
    or LABEL_OVERRIDES.get("readonly_help_title")
    or "Providers Help / Tips"
)
HELP_MD: str = (
    _read_secret("READONLY_HELP_MD")
    or LABEL_OVERRIDES.get("readonly_help_md")
    or ""
)
HELP_DEBUG: bool = _get_bool(
    _read_secret("READONLY_HELP_DEBUG"),
    _get_bool(LABEL_OVERRIDES.get("readonly_help_debug"), False),
)

RAW_COLS = ["id","business_name","category","service","contact_name","phone","address","website","notes","keywords"]


# -----------------------------
# Engine (Turso/libSQL with fallback)
# -----------------------------
def _engine():
    url = _read_secret("TURSO_DATABASE_URL", "sqlite+libsql://vendors-prod-mrbookend.aws-us-west-2.turso.io?secure=true")
    token = _read_secret("TURSO_AUTH_TOKEN", None)
    if isinstance(url, str) and url.startswith("sqlite+libsql://"):
        try:
            eng = create_engine(url, connect_args={"auth_token": token} if token else {}, pool_pre_ping=True)
            with eng.connect() as c:
                c.execute(sql_text("SELECT 1"))
            return eng
        except Exception as e:
            st.warning(f"Turso connection failed ({e}). Falling back to local SQLite vendors.db.")
    local = "/mount/src/vendors-readonly-app/vendors.db"
    if not os.path.exists(local):
        local = "vendors.db"
    return create_engine(f"sqlite:///{local}")

eng = _engine()


# -----------------------------
# CSS (applies to fallback table only; aggrid handles its own)
# -----------------------------
def _apply_css(field_order: List[str]):
    rules = []
    for idx, col in enumerate(field_order, start=1):
        px = COLUMN_WIDTHS.get(col)
        if not px:
            continue
        rules.append(
            f"""
            div[data-testid='stDataFrame'] table thead tr th:nth-child({idx}),
            div[data-testid='stDataFrame'] table tbody tr td:nth-child({idx}) {{
                min-width:{px}px !important; max-width:{px}px !important; width:{px}px !important;
                white-space: normal !important; overflow-wrap:anywhere !important; word-break:break-word !important;
            }}
            """
        )
    sticky = ""
    if STICKY_FIRST and field_order:
        sticky = """
        div[data-testid='stDataFrame'] table thead tr th:nth-child(1),
        div[data-testid='stDataFrame'] table tbody tr td:nth-child(1){
            position:sticky;left:0;z-index:2;background:var(--background-color,white);box-shadow:1px 0 0 rgba(0,0,0,0.06);
        }
        div[data-testid='stDataFrame'] table thead tr th:nth-child(1){z-index:3;}
        """
    st.markdown(
        f"""
        <style>
        div[data-testid='stDataFrame'] table {{ table-layout: fixed !important; }}
        div[data-testid='stDataFrame'] table td, div[data-testid='stDataFrame'] table th {{
            white-space: normal !important; overflow-wrap:anywhere !important; word-break:break-word !important;
        }}
        {''.join(rules)}
        {sticky}
        </style>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------
# URL helper
# -----------------------------
def _normalize_url(u: str) -> str:
    s = (u or "").strip()
    if not s:
        return ""
    if not s.lower().startswith(("http://","https://")):
        s = "http://" + s
    try:
        p = urlparse(s)
        if p.scheme not in ("http","https"):
            return ""
        if not p.netloc or "." not in p.netloc:
            return ""
        return s
    except Exception:
        return ""


# -----------------------------
# Help
# -----------------------------
def render_help():
    with st.expander(HELP_TITLE, expanded=False):
        if HELP_MD:
            st.markdown(HELP_MD)
            if HELP_DEBUG:
                st.caption("Raw help (debug preview)")
                st.code(HELP_MD, language=None)
        else:
            st.info("No help content has been configured yet.")


# -----------------------------
# Data access
# -----------------------------
def _fetch_df() -> pd.DataFrame:
    with eng.connect() as c:
        df = pd.read_sql_query(sql_text("SELECT * FROM vendors"), c)
    for col in RAW_COLS:
        if col not in df.columns:
            df[col] = ""
    return df[RAW_COLS]

def _apply_labels(df: pd.DataFrame) -> pd.DataFrame:
    if not LABEL_OVERRIDES:
        return df
    mapping = {k:v for k,v in LABEL_OVERRIDES.items() if k in df.columns and isinstance(v,str) and v}
    return df.rename(columns=mapping)


# -----------------------------
# AG Grid renderer (autoHeight)
# -----------------------------
def _aggrid_view(df_show: pd.DataFrame, website_label: str = "website"):
    """AgGrid with wrapped cells and autoHeight ONLY on long-text columns, using a normal scrollable grid."""
    if df_show.empty:
        st.info("No rows to display.")
        return

    # Detect displayed website column (handles label overrides)
    website_key = website_label if website_label in df_show.columns else next((c for c in df_show.columns if c.lower()=="website"), None)

    # Build display DF + normalized URL column to the right of website
    _df = df_show.copy()
    url_col = None
    if website_key:
        norm = _df.get("website", _df[website_key]).map(_normalize_url)
        url_col = f"{website_key} URL" if website_key.lower() != "website" else "Website URL"
        widx = _df.columns.get_loc(website_key)
        _df.insert(widx + 1, url_col, norm)

    gob = GridOptionsBuilder.from_dataframe(_df)

    # Base column config
    gob.configure_default_column(
        resizable=True,
        sortable=True,
        filter=True,
        wrapText=False,      # default: no wrap (keeps rows compact)
        autoHeight=False,    # default: fixed row height
    )

    # Columns that should wrap and auto-height
    long_cols = {"notes", "address"}
    if url_col:
        long_cols.add(url_col)

    # Map raw width config to displayed names
    display_to_raw = {disp: raw for raw, disp in LABEL_OVERRIDES.items() if isinstance(disp, str) and disp}
    for col in _df.columns:
        raw_key = display_to_raw.get(col, col)
        px = COLUMN_WIDTHS.get(raw_key)
        if px:
            gob.configure_column(col, width=px)

        # Enable wrapping/autoHeight only on long text columns
        if col.lower() in long_cols:
            gob.configure_column(col, wrapText=True, autoHeight=True, cellStyle={"white-space": "normal"})

    # Clickable "Website" label (only if we created URL column)
    if website_key and url_col:
        link_renderer = JsCode(f"""
            function(params){{
                const url = params.data["{url_col}"] || "";
                if (!url) return "";
                return `<a href="${{url}}" target="_blank" rel="noopener noreferrer">Website</a>`;
            }}
        """)
        gob.configure_column(website_key, cellRenderer=link_renderer)

    grid_options = gob.build()

    # IMPORTANT: use normal layout (scroll inside grid) so heights are stable
    # We also set a reasonable grid height; adjust if you want more/less visible rows
    AgGrid(
        _df,
        gridOptions=grid_options,
        update_mode=GridUpdateMode.NO_UPDATE,
        fit_columns_on_grid_load=False,
        allow_unsafe_jscode=True,
        enable_enterprise_modules=False,
        columns_auto_size_mode=ColumnsAutoSizeMode.FIT_CONTENTS,
        height=600,     # scrollable grid; no domLayout:'autoHeight'
        theme="streamlit",
    )


# -----------------------------
# View body
# -----------------------------
def render_view():
    df = _fetch_df()
    df_show = _apply_labels(df)

    # CSV download (first; grid can be tall)
    st.download_button(
        label="Download Providers (CSV)",
        data=df_show.to_csv(index=False).encode("utf-8"),
        file_name="providers_readonly.csv",
        mime="text/csv",
    )

    if _AGGRID_AVAILABLE:
        _aggrid_view(df_show, website_label=LABEL_OVERRIDES.get("website", "website"))
    else:
        # Fallback table (no true auto-height)
        st.warning("`streamlit-aggrid` not installed — showing basic table without auto-expanding rows.")
        website_key = LABEL_OVERRIDES.get("website", "website")
        _df = df_show.copy()
        if website_key in _df.columns:
            try:
                base_col = "website" if "website" in df.columns else website_key
                norm = df[base_col].apply(_normalize_url) if base_col in df.columns else _df[website_key].apply(_normalize_url)
                url_col = f"{website_key} URL" if website_key.lower() != "website" else "Website URL"
                w_idx = _df.columns.get_loc(website_key)
                _df.insert(w_idx + 1, url_col, norm)
                _apply_css(df.columns.tolist())
                st.dataframe(
                    _df.assign(**{website_key: norm}),
                    use_container_width=True,
                    hide_index=True,
                    column_config={website_key: st.column_config.LinkColumn(display_text="Website")},
                )
            except Exception:
                _apply_css(df.columns.tolist())
                st.dataframe(_df, use_container_width=True, hide_index=True)
        else:
            _apply_css(df.columns.tolist())
            st.dataframe(_df, use_container_width=True, hide_index=True)


# -----------------------------
# Status & Secrets (debug) — END
# -----------------------------
def render_status_debug():
    with st.expander("Status & Secrets (debug)", expanded=False):
        backend = "libsql" if str(_read_secret("TURSO_DATABASE_URL", "")).startswith("sqlite+libsql://") else "sqlite"
        st.write("DB")
        st.code(
            {
                "backend": backend,
                "dsn": _read_secret("TURSO_DATABASE_URL",""),
                "auth": "token_set" if bool(_read_secret("TURSO_AUTH_TOKEN")) else "none",
            }
        )
        try:
            keys = list(st.secrets.keys())
        except Exception:
            keys = []
        st.write("Secrets keys (present)")
        st.code(keys)
        st.write("Help MD:", "present" if bool(HELP_MD) else "(missing or empty)")
        st.write("Sticky first col enabled:", STICKY_FIRST)
        st.write("Raw COLUMN_WIDTHS_PX_READONLY (type)", type(COLUMN_WIDTHS).__name__)
        st.code(COLUMN_WIDTHS)
        st.write("Column label overrides (if any)")
        st.code(LABEL_OVERRIDES)


# -----------------------------
# Main
# -----------------------------
def main():
    render_help()
    st.header("View", anchor=False)
    render_view()

    # Debug LAST
    render_status_debug()

if __name__ == "__main__":
    main()
