# app_readonly.py — Read‑Only Vendors (stability‑first v3)
# 
# Goals
# - Preserve the original look/feel and column order (zero surprise)
# - Use secrets for labels, widths, sticky first column, help text
# - Fix prior Help crash (no nested expanders; Markdown rendering)
# - Turso (libSQL) primary DB, local SQLite fallback
# - Simple, robust quick filter (client‑side contains across all string cols)
# - Minimal CSS just for wrapping and column widths; no other visual drift
# - Keep code compact and readable without cleverness
# - Place "Download Providers (CSV)" and then "Status & Secrets (debug)" at the END of the page
#
# Expected schema: vendors(id, category, service, business_name, contact_name, phone, address, website, notes, keywords)
#
# Version: 3.0 (2025‑10‑03)

from __future__ import annotations

import os
from typing import Dict, List

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text as sql_text
from sqlalchemy.engine import Engine


# -----------------------------
# Early secret/env helpers
# -----------------------------
def _read_secret_early(name: str, default=None):
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.environ.get(name, default)


def _get_secret_bool(val, default=False) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() in ("1", "true", "yes", "on")
    return bool(default)


# -----------------------------
# Page layout (must run first)
# -----------------------------
def _apply_page_layout():
    title = _read_secret_early("page_title", "HCR Vendors (Read‑Only)")
    sidebar_state = _read_secret_early("sidebar_state", "expanded")
    max_width = _read_secret_early("page_max_width_px", 2300)

    st.set_page_config(page_title=title, layout="wide", initial_sidebar_state=sidebar_state)

    try:
        max_width = int(max_width)
    except Exception:
        max_width = 2300

    st.markdown(
        f"""
        <style>
        .block-container {{ max-width: {max_width}px; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


_apply_page_layout()


# -----------------------------
# Secrets-driven config
# -----------------------------
COLUMN_WIDTHS_PX: Dict[str, int] = _read_secret_early("COLUMN_WIDTHS_PX_READONLY", {}) or {}
LABEL_OVERRIDES: Dict[str, str] = _read_secret_early("READONLY_COLUMN_LABELS", {}) or {}
STICKY_FIRST = _get_secret_bool(_read_secret_early("READONLY_STICKY_FIRST_COL", False), False)

HELP_TITLE = (
    _read_secret_early("READONLY_HELP_TITLE")
    or LABEL_OVERRIDES.get("readonly_help_title")
    or "Providers Help / Tips"
)
HELP_MD = (
    _read_secret_early("READONLY_HELP_MD")
    or LABEL_OVERRIDES.get("readonly_help_md")
    or ""
)
HELP_DEBUG = _get_secret_bool(
    _read_secret_early("READONLY_HELP_DEBUG"),
    _get_secret_bool(LABEL_OVERRIDES.get("readonly_help_debug"), False),
)

# Allow optional column order override via secrets; default to canonical order
DEFAULT_ORDER = [
    "id",
    "business_name",
    "category",
    "service",
    "contact_name",
    "phone",
    "address",
    "website",
    "notes",
    "keywords",
]
RAW_OVERRIDE_ORDER = _read_secret_early("READONLY_COLUMN_ORDER", None)
COLUMN_ORDER: List[str] = (
    [c.strip() for c in RAW_OVERRIDE_ORDER] if isinstance(RAW_OVERRIDE_ORDER, (list, tuple)) else DEFAULT_ORDER
)


# -----------------------------
# Database engine
# -----------------------------
def _make_engine() -> Engine:
    url = _read_secret_early(
        "TURSO_DATABASE_URL",
        "sqlite+libsql://vendors-prod-mrbookend.aws-us-west-2.turso.io?secure=true",
    )
    token = _read_secret_early("TURSO_AUTH_TOKEN", None)

    if isinstance(url, str) and url.startswith("sqlite+libsql://"):
        connect_args = {"auth_token": token} if token else {}
        try:
            eng = create_engine(url, connect_args=connect_args, pool_pre_ping=True)
            with eng.connect() as conn:
                conn.execute(sql_text("SELECT 1"))
            return eng
        except Exception as e:
            st.warning(f"Turso connection failed ({e}). Falling back to local SQLite vendors.db.")

    # Fallback local path (Streamlit Cloud mount or repo root)
    local_path = "/mount/src/vendors-readonly-app/vendors.db"
    if not os.path.exists(local_path):
        local_path = "vendors.db"
    return create_engine(f"sqlite:///{local_path}")


engine = _make_engine()


# -----------------------------
# Data access
# -----------------------------
EXPECTED_COLS = set(DEFAULT_ORDER)


def _effective_order(df_cols: List[str]) -> List[str]:
    # Keep requested order but only include columns that exist; append any extra columns at the end
    order = [c for c in COLUMN_ORDER if c in df_cols]
    extras = [c for c in df_cols if c not in order]
    return order + extras


def load_vendors() -> pd.DataFrame:
    # Select only columns that likely exist; if not, select * and reorder later
    try:
        query = f"SELECT {', '.join(DEFAULT_ORDER)} FROM vendors"
        with engine.connect() as conn:
            df = pd.read_sql_query(sql_text(query), conn)
    except Exception:
        # Fallback to SELECT * then trim/reorder
        with engine.connect() as conn:
            df = pd.read_sql_query(sql_text("SELECT * FROM vendors"), conn)

    # Ensure all expected columns exist (fill in empty if missing), but do not coerce user data
    for col in DEFAULT_ORDER:
        if col not in df.columns:
            df[col] = ""

    # Reorder columns with optional override, without dropping unknown columns
    df = df[_effective_order(df.columns.tolist())]

    return df


# -----------------------------
# CSS: wrapping, widths, optional sticky first column
# -----------------------------
def _apply_table_css(field_order: List[str], widths_px: Dict[str, int], sticky_first: bool):
    width_rules = []
    # Build nth-child selectors aligned to the *visible* order
    for idx, field in enumerate(field_order, start=1):
        px = widths_px.get(field)
        if not px:
            continue
        width_rules.append(
            f"""
            /* {idx}: {field} width */
            div[data-testid='stDataFrame'] table thead tr th:nth-child({idx}),
            div[data-testid='stDataFrame'] table tbody tr td:nth-child({idx}) {{
                min-width: {px}px !important;
                max-width: {px}px !important;
                width: {px}px !important;
                white-space: normal !important;
                overflow-wrap: anywhere !important;
                word-break: break-word !important;
            }}
            """
        )

    sticky_rule = ""
    if sticky_first and field_order:
        sticky_rule = f"""
        /* Sticky first column */
        div[data-testid='stDataFrame'] table thead tr th:nth-child(1),
        div[data-testid='stDataFrame'] table tbody tr td:nth-child(1) {{
            position: sticky; left: 0; z-index: 2; background: var(--background-color, white);
            box-shadow: 1px 0 0 0 rgba(0,0,0,0.06);
        }}
        div[data-testid='stDataFrame'] table thead tr th:nth-child(1) {{ z-index: 3; }}
        """

    st.markdown(
        f"""
        <style>
        div[data-testid='stDataFrame'] table {{ table-layout: fixed !important; }}
        div[data-testid='stDataFrame'] table td, div[data-testid='stDataFrame'] table th {{
            white-space: normal !important; overflow-wrap: anywhere !important; word-break: break-word !important;
        }}
        {''.join(width_rules)}
        {sticky_rule}
        </style>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------
# Column label overrides (display only)
# -----------------------------
def _apply_label_overrides(df: pd.DataFrame, overrides: Dict[str, str]) -> pd.DataFrame:
    if not overrides:
        return df
    mapping = {k: v for k, v in overrides.items() if k in df.columns and isinstance(v, str) and v}
    return df.rename(columns=mapping)


# -----------------------------
# Quick filter
# -----------------------------
def _quick_filter(df: pd.DataFrame) -> pd.DataFrame:
    st.subheader("Browse Providers")
    q = st.text_input("Quick filter (matches any column)", value="", placeholder="e.g., plumb, roof, 78245…").strip()
    if not q:
        return df

    ql = q.lower()
    mask = pd.Series(False, index=df.index)
    for c in df.columns:
        if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c]):
            mask = mask | df[c].fillna("").astype(str).str.lower().str.contains(ql, na=False)
    return df[mask]


# -----------------------------
# Help (single expander, Markdown; optional raw debug)
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
# Debug panel (kept concise)
# -----------------------------
DEBUG_KEYS = (
    "COLUMN_WIDTHS_PX_READONLY",
    "READONLY_COLUMN_LABELS",
    "READONLY_STICKY_FIRST_COL",
    "READONLY_HELP_TITLE",
    "READONLY_HELP_MD",
    "READONLY_HELP_DEBUG",
    "READONLY_COLUMN_ORDER",
    "page_title",
    "page_max_width_px",
    "sidebar_state",
    "TURSO_DATABASE_URL",
)


def render_status_debug():
    with st.expander("Status & Secrets (debug)", expanded=False):
        try:
            keys = list(st.secrets.keys())
        except Exception:
            keys = []

        backend = (
            "libsql" if str(_read_secret_early("TURSO_DATABASE_URL", "")).startswith("sqlite+libsql://") else "sqlite"
        )
        dsn = _read_secret_early("TURSO_DATABASE_URL", "")
        auth = "token_set" if bool(_read_secret_early("TURSO_AUTH_TOKEN", None)) else "none"

        st.write("DB")
        st.code({"backend": backend, "dsn": dsn, "auth": auth})

        st.write("Secrets keys (present)")
        st.code(keys)

        st.write("Help MD:", "present" if bool(HELP_MD) else "(missing or empty)")
        st.write("Sticky first col enabled:", STICKY_FIRST)

        st.write("Raw COLUMN_WIDTHS_PX_READONLY (type)", type(COLUMN_WIDTHS_PX).__name__)
        st.code(COLUMN_WIDTHS_PX)

        st.write("Column label overrides (if any)")
        st.code(LABEL_OVERRIDES)

        st.write("Effective column order")
        st.code(COLUMN_ORDER)


# -----------------------------
# CSV export helper
# -----------------------------
def _to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


# -----------------------------
# Main app
# -----------------------------

def main():
    # Load data (robust to minor schema drift)
    try:
        df = load_vendors()
    except Exception as e:
        st.error(f"Failed to load vendors: {e}")
        return

    # Compute the field order we will *display* (after label overrides)
    visible_order = _effective_order(df.columns.tolist())

    # CSS for wrapping / widths / optional sticky
    _apply_table_css(visible_order, COLUMN_WIDTHS_PX, STICKY_FIRST)

    # Help (safe, single expander)
    render_help()

    # Apply label overrides for display only (no mutation of underlying data)
    df_display = _apply_label_overrides(df, LABEL_OVERRIDES)

    # Quick filter (client‑side)
    df_filtered = _quick_filter(df_display)

    # Show table
    st.dataframe(df_filtered, use_container_width=True, hide_index=True)

    # --- Download CSV button just before debug section ---
    st.download_button(
        label="Download Providers (CSV)",
        data=_to_csv_bytes(df_filtered),
        file_name="providers.csv",
        mime="text/csv",
        help="Exports exactly what you see (after filter and label overrides).",
    )

    # --- Debug panel LAST on the page ---
    render_status_debug()


if __name__ == "__main__":
    main()
