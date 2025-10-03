# app_readonly.py — Providers Read-Only (v3.6)
# Tabs: View | Categories | Services | Changelog
# - No top-level Streamlit renders; init happens inside main()
# - Read-only: NEVER mutates DB (no schema changes, no writes)
# - Hardened debug (gated via ADMIN_DEBUG), safe prints (no None)
# - Cross-platform timestamp formatting, CSS widths/sticky, link column
# - Cached lookups for categories/services

from __future__ import annotations

import os
import re
from typing import Dict, List

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine
from sqlalchemy.sql import text as sql_text
from sqlalchemy.engine import Engine
from urllib.parse import urlparse
from datetime import datetime, timezone

# Global engine placeholder (initialized in main())
eng: Engine | None = None

# -----------------------------
# Secrets / env helpers
# -----------------------------
def _read_secret(name: str, default=None):
    """
    Read from Streamlit secrets first (if available), then environment.
    Guarded to avoid errors when st.secrets is unavailable.
    """
    try:
        s = getattr(st, "secrets", None)
        if s is not None:
            try:
                if name in s:
                    return s[name]
            except Exception:
                try:
                    return s[name]
                except Exception:
                    pass
    except Exception:
        pass
    return os.environ.get(name, default)

def _get_bool(val, default=False):
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() in ("1", "true", "yes", "on")
    return bool(default)

def _now_iso() -> str:
    # UTC ISO 8601 (seconds precision)
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

def _nz(x, default=""):
    return default if x is None else x

# -----------------------------
# Page config
# -----------------------------
def _apply_layout():
    st.set_page_config(
        page_title=_read_secret("page_title", "HCR Providers"),
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

# -----------------------------
# Global config from secrets
# -----------------------------
LABEL_OVERRIDES: Dict[str, str] = _read_secret("READONLY_COLUMN_LABELS", {}) or {}
COLUMN_WIDTHS: Dict[str, int] = _read_secret("COLUMN_WIDTHS_PX_READONLY", {}) or {}
STICKY_FIRST: bool = _get_bool(_read_secret("READONLY_STICKY_FIRST_COL", False), False)

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

# Gate the debug panel (A)
SHOW_DEBUG = bool(_read_secret("ADMIN_DEBUG", False))

# Columns expected from vendors table
RAW_COLS = [
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
AUDIT_COLS = ["created_at", "updated_at", "updated_by"]  # may or may not be present

# -----------------------------
# Engine (read-only app still needs a connection)
# -----------------------------
def _engine() -> Engine:
    url = _read_secret(
        "TURSO_DATABASE_URL",
        "sqlite+libsql://vendors-prod-mrbookend.aws-us-west-2.turso.io?secure=true",
    )
    token = _read_secret("TURSO_AUTH_TOKEN", None)
    if isinstance(url, str) and url.startswith("sqlite+libsql://"):
        try:
            eng_local = create_engine(
                url,
                connect_args={"auth_token": token} if token else {},
                pool_pre_ping=True,
            )
            with eng_local.connect() as c:
                c.execute(sql_text("SELECT 1"))
            return eng_local
        except Exception as e:
            st.warning(f"Turso connection failed ({e}). Falling back to local SQLite vendors.db.")
    local = "/mount/src/vendors-readonly-app/vendors.db"
    if not os.path.exists(local):
        local = "vendors.db"
    return create_engine(f"sqlite:///{local}")

# -----------------------------
# CSS widths / sticky (same as Admin)
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
# URL + phone helpers (validators/formatters)
# -----------------------------
def _normalize_url(u: str) -> str:
    """Return normalized URL or empty string if invalid."""
    s = (u or "").strip()
    if not s:
        return ""
    if not s.lower().startswith(("http://", "https://")):
        s = "http://" + s
    try:
        p = urlparse(s)
        if p.scheme not in ("http", "https"):
            return ""
        if not p.netloc or "." not in p.netloc:
            return ""
        return s
    except Exception:
        return ""

def _digits_only(s: str) -> str:
    return re.sub(r"[^0-9]", "", s or "")

def _format_phone_10(digits: str) -> str:
    d = _digits_only(digits)
    if len(d) != 10:
        return ""
    return f"({d[0:3]}) {d[3:6]}-{d[6:10]}"

# -----------------------------
# Help (single expander; Markdown)
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
# Data access (strictly read-only)
# -----------------------------
def _fetch_df() -> pd.DataFrame:
    assert eng is not None, "Engine not initialized"
    with eng.connect() as c:
        df = pd.read_sql_query(sql_text("SELECT * FROM vendors"), c)
    # Ensure columns exist for display
    for col in RAW_COLS + AUDIT_COLS:
        if col not in df.columns:
            df[col] = ""
    return df[[c for c in df.columns if c in (RAW_COLS + AUDIT_COLS)]]

@st.cache_data(ttl=30)
def _cats_cached():
    assert eng is not None, "Engine not initialized"
    with eng.connect() as c:
        try:
            return pd.read_sql_query(sql_text("SELECT id, name FROM categories ORDER BY name COLLATE NOCASE"), c)
        except Exception:
            return pd.DataFrame(columns=["id", "name"]).astype({"id": int, "name": str})

@st.cache_data(ttl=30)
def _svcs_cached():
    assert eng is not None, "Engine not initialized"
    with eng.connect() as c:
        try:
            return pd.read_sql_query(sql_text("SELECT id, name FROM services ORDER BY name COLLATE NOCASE"), c)
        except Exception:
            return pd.DataFrame(columns=["id", "name"]).astype({"id": int, "name": str})

def _cats() -> pd.DataFrame:
    return _cats_cached()

def _svcs() -> pd.DataFrame:
    return _svcs_cached()

# -----------------------------
# Label mapping (display only)
# -----------------------------
def _apply_labels(df: pd.DataFrame) -> pd.DataFrame:
    if not LABEL_OVERRIDES:
        return df
    mapping = {k: v for k, v in LABEL_OVERRIDES.items() if k in df.columns and isinstance(v, str) and v}
    return df.rename(columns=mapping)

# -----------------------------
# VIEW tab
# -----------------------------
def _apply_table_css_for_view(df_raw_cols: List[str]):
    _apply_css(df_raw_cols)

def tab_view():
    try:
        df = _fetch_df()
    except Exception as e:
        st.error(f"Failed to load vendors table: {e}")
        return

    df_show = _apply_labels(df)
    _apply_table_css_for_view([c for c in df.columns if c in COLUMN_WIDTHS or c in RAW_COLS])

    # Website link with adjacent full URL column
    website_key = LABEL_OVERRIDES.get("website", "website")
    _df = df_show.copy()
    if website_key in _df.columns:
        try:
            base_col = "website" if "website" in df.columns else website_key
            norm = df[base_col].apply(_normalize_url) if base_col in df.columns else _df[website_key].apply(_normalize_url)
            url_col = f"{website_key} URL" if website_key.lower() != "website" else "Website URL"
            w_idx = _df.columns.get_loc(website_key)
            _df.insert(w_idx + 1, url_col, norm)
            st.dataframe(
                _df.assign(**{website_key: norm}),
                use_container_width=True,
                hide_index=True,
                column_config={website_key: st.column_config.LinkColumn(display_text="Website")},
            )
        except Exception:
            st.dataframe(_df, use_container_width=True, hide_index=True)
    else:
        st.dataframe(_df, use_container_width=True, hide_index=True)

    st.download_button(
        label="Download Providers (CSV)",
        data=df_show.to_csv(index=False).encode("utf-8"),
        file_name="providers_readonly_view.csv",
        mime="text/csv",
    )

# -----------------------------
# CATEGORIES tab (read-only)
# -----------------------------
def tab_categories():
    st.subheader("Categories")
    cats = _cats()
    st.dataframe(cats, use_container_width=True, hide_index=True)

# -----------------------------
# SERVICES tab (read-only)
# -----------------------------
def tab_services():
    st.subheader("Services")
    svcs = _svcs()
    st.dataframe(svcs, use_container_width=True, hide_index=True)

# -----------------------------
# CHANGELOG tab (read-only)
# -----------------------------
def _fmt_when(s: str) -> str:
    """Human-friendly local time; works on Windows/macOS/Linux."""
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone()
        out = dt.strftime("%b %d, %Y, %I:%M %p")
        return out.replace(" 0", " ")
    except Exception:
        return s

def tab_changelog():
    st.subheader("Changelog")
    fcol1, fcol2 = st.columns(2)
    with fcol1:
        vendor_filter = st.text_input("Filter by vendor name contains", "")
    with fcol2:
        user_filter = st.text_input("Filter by edited-by contains", "")

    assert eng is not None, "Engine not initialized"
    with eng.connect() as c:
        try:
            q = """
                SELECT vc.id, vc.vendor_id, vc.changed_at, vc.changed_by, vc.action, vc.field, vc.old_value, vc.new_value,
                       v.business_name
                FROM vendor_changes vc
                LEFT JOIN vendors v ON v.id = vc.vendor_id
                ORDER BY vc.changed_at DESC, vc.id DESC
            """
            df = pd.read_sql_query(sql_text(q), c)
        except Exception:
            st.info("No changelog available (vendor_changes table missing or inaccessible).")
            return

    if vendor_filter.strip():
        df = df[df["business_name"].fillna("").str.contains(vendor_filter.strip(), case=False, na=False)]
    if user_filter.strip():
        df = df[df["changed_by"].fillna("").str.contains(user_filter.strip(), case=False, na=False)]

    lines = []
    for _, r in df.iterrows():
        when = _fmt_when(str(r["changed_at"]))
        by = r.get("changed_by") or "unknown"
        name = r.get("business_name") or f"Vendor #{r.get('vendor_id')}"
        action = r.get("action")
        if action == "insert":
            lines.append(f"{when} — Added **{name}** (by {by})")
        elif action == "delete":
            lines.append(f"{when} — Deleted **{name}** (by {by})")
        elif action == "update":
            field = r.get("field") or "field"
            oldv = r.get("old_value") or ""
            newv = r.get("new_value") or ""
            arrow = f"`{oldv}` → `{newv}`" if oldv or newv else ""
            lines.append(f"{when} — Updated **{field}** for **{name}** (by {by}) {arrow}")
        else:
            lines.append(f"{when} — Change on **{name}** (by {by})")

    if not lines:
        st.info("No changes recorded yet.")
    else:
        st.markdown("\n\n".join(f"- {ln}" for ln in lines))

# -----------------------------
# Status & Secrets (debug) — gated
# -----------------------------
def render_status_debug():
    with st.expander("Status & Secrets (debug)", expanded=False):
        backend = "libsql" if str(_read_secret("TURSO_DATABASE_URL", "")).startswith("sqlite+libsql://") else "sqlite"
        st.write("DB")
        st.code({
            "backend": backend,
            "dsn": _read_secret("TURSO_DATABASE_URL", ""),
            "auth": "token_set" if bool(_read_secret("TURSO_AUTH_TOKEN")) else "none",
        })
        try:
            keys = list(st.secrets.keys())
        except Exception:
            keys = []
        st.write("Secrets keys (present)")
        st.code(keys or [])
        st.write("Help MD:", "present" if bool(HELP_MD) else "(missing or empty)")
        st.write("Sticky first col enabled:", STICKY_FIRST)
        st.write("Raw COLUMN_WIDTHS_PX_READONLY (type)", type(COLUMN_WIDTHS).__name__)
        st.code(COLUMN_WIDTHS or {})
        st.write("Column label overrides (if any)")
        st.code(LABEL_OVERRIDES or {})

# -----------------------------
# Main
# -----------------------------
def main():
    _apply_layout()
    global eng
    eng = _engine()

    # Health check (no writes)
    try:
        with eng.connect() as c:
            c.execute(sql_text("SELECT 1"))
    except Exception as e:
        st.error(f"DB health check failed: {e}")
        st.stop()

    render_help()

    tabs = st.tabs([
        "View",
        "Categories",
        "Services",
        "Changelog",
    ])

    with tabs[0]:
        tab_view()
    with tabs[1]:
        tab_categories()
    with tabs[2]:
        tab_services()
    with tabs[3]:
        tab_changelog()

    # Debug LAST (gated)
    if SHOW_DEBUG:
        render_status_debug()

if __name__ == "__main__":
    main()
PY

python3 -m py_compile app_readonly.py && echo "Syntax OK"
