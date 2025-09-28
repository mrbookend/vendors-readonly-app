# app.py  (or app_readonly.py)
# Vendors Read-only — wide layout, collapsed sidebar, fixed character widths,
# Website hyperlink immediately after Address, and NO Service/Keywords columns in the UI.

from __future__ import annotations

import os
import re
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.sql import text as sql_text

# ---------- PAGE LAYOUT ----------
st.set_page_config(page_title="Vendors", layout="wide", initial_sidebar_state="collapsed")

# ---------- DB CONFIG ----------
LIBSQL_URL = (
    st.secrets.get("LIBSQL_URL")
    or os.getenv("LIBSQL_URL")
    or os.getenv("TURSO_DATABASE_URL")
)
LIBSQL_AUTH_TOKEN = (
    st.secrets.get("LIBSQL_AUTH_TOKEN")
    or os.getenv("LIBSQL_AUTH_TOKEN")
    or os.getenv("TURSO_AUTH_TOKEN")
)
VENDORS_DB_PATH = st.secrets.get("VENDORS_DB_PATH") or os.getenv("VENDORS_DB_PATH") or "vendors.db"

USE_TURSO = bool(LIBSQL_URL and LIBSQL_AUTH_TOKEN)

def _as_sqlalchemy_url(u: str) -> str:
    """Convert libsql://... to sqlite+libsql://... (SQLAlchemy dialect)."""
    if not u:
        return u
    if u.startswith("sqlite+libsql://"):
        return u
    if u.startswith("libsql://"):
        tail = u[len("libsql://"):]
        return f"sqlite+libsql://{tail}" + ("" if "?" in tail else "?secure=true")
    return u

def make_engine() -> Engine:
    if USE_TURSO:
        sa_url = _as_sqlalchemy_url(LIBSQL_URL)
        return create_engine(
            sa_url,
            connect_args={"auth_token": LIBSQL_AUTH_TOKEN},
            pool_pre_ping=True,
            pool_recycle=300,
        )
    return create_engine(f"sqlite:///{VENDORS_DB_PATH}")

engine: Engine = make_engine()
DB_SOURCE = f"(Turso) {_as_sqlalchemy_url(LIBSQL_URL) or ''}" if USE_TURSO else f"(SQLite) {VENDORS_DB_PATH}"

# ---------- HELPERS ----------
def run_df(sql: str, params: Dict | None = None) -> pd.DataFrame:
    with engine.begin() as conn:
        return pd.read_sql_query(sql_text(sql), conn, params=dict(params or {}))

def query_scalar(sql: str, params: Dict | None = None):
    with engine.begin() as conn:
        row = conn.execute(sql_text(sql), dict(params or {})).fetchone()
        return row[0] if row else None

def table_exists(name: str) -> bool:
    return query_scalar(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=:n LIMIT 1;", {"n": name}
    ) is not None

def table_columns(table: str) -> List[str]:
    with engine.begin() as conn:
        rows = conn.execute(sql_text(f"PRAGMA table_info({table});")).fetchall()
        return [r[1] for r in rows]

# ---------- SCHEMA ----------
def detect_schema():
    if not table_exists("vendors"):
        st.error("Table 'vendors' not found. Check DB connection/path.")
        st.stop()
    vcols = table_columns("vendors")
    has_categories = table_exists("categories")
    has_services = table_exists("services")
    has_fts = table_exists("vendors_fts")
    return {
        "vendors_columns": vcols,
        "has_categories_table": has_categories,
        "has_services_table": has_services,
        "has_fts": has_fts,
        "uses_cat_id": "category_id" in vcols and has_categories,
        "uses_svc_id": "service_id" in vcols and has_services,
        "uses_cat_text": "category" in vcols,
        "uses_svc_text": "service" in vcols,
        "has_keywords_col": "keywords" in vcols,
    }

SCHEMA = detect_schema()

# ---------- CATALOG LOADERS ----------
def get_categories() -> List[str]:
    if SCHEMA["has_categories_table"]:
        df = run_df("SELECT name FROM categories ORDER BY name;")
        if not df.empty:
            return df["name"].tolist()
    if SCHEMA["uses_cat_text"]:
        df = run_df(
            "SELECT DISTINCT TRIM(category) AS name FROM vendors "
            "WHERE TRIM(category) <> '' ORDER BY 1;"
        )
        return df["name"].tolist()
    return []

# ---------- SEARCH ----------
def _fts_query_string(q: str) -> str:
    toks = re.findall(r"[A-Za-z0-9]+", q or "")
    if not toks:
        return ""
    return " AND ".join(f"{t}*" for t in toks)

def load_vendors_df(q: str = "", use_fts: bool = False,
                    category_filter: str = "") -> pd.DataFrame:
    # Category expr/join
    if SCHEMA["uses_cat_id"]:
        cat_expr = "c.name AS Category"
        cat_join = "LEFT JOIN categories c ON c.id = v.category_id"
        cat_filter_expr = "c.name = :cat"
    elif SCHEMA["uses_cat_text"]:
        cat_expr, cat_join = "v.category AS Category", ""
        cat_filter_expr = "v.category = :cat"
    else:
        cat_expr, cat_join, cat_filter_expr = "NULL AS Category", "", None

    base_select = f"""
        SELECT
          v.id AS id,
          {cat_expr},
          v.business_name AS "Business Name",
          v.contact_name  AS "Contact Name",
          v.phone         AS Phone,
          v.address       AS Address,
          v.notes         AS Notes,
          v.website       AS Website
        FROM vendors v
        {cat_join}
    """

    params: Dict[str, object] = {}
    wheres: List[str] = []

    if category_filter and cat_filter_expr:
        wheres.append(cat_filter_expr); params["cat"] = category_filter

    if q:
        if use_fts and SCHEMA["has_fts"]:
            fts_q = _fts_query_string(q)
            if fts_q:
                base_select += "\nJOIN vendors_fts ON vendors_fts.rowid = v.id\n"
                wheres.append("vendors_fts MATCH :fts")
                params["fts"] = fts_q
        else:
            like = f"%{q}%"
            lconds = [
                "v.business_name LIKE :like",
                "v.notes LIKE :like",
                "v.address LIKE :like",
                "v.phone LIKE :like",
            ]
            if SCHEMA["uses_cat_text"]:
                lconds.append("v.category LIKE :like")
            wheres.append("(" + " OR ".join(lconds) + ")")
            params["like"] = like

    where_sql = ("WHERE " + " AND ".join(wheres)) if wheres else ""
    order_sql = 'ORDER BY "Business Name" COLLATE NOCASE ASC'
    sql = f"{base_select}\n{where_sql}\n{order_sql};"
    df = run_df(sql, params)

    # Normalize Website URLs so LinkColumn works
    if "Website" in df.columns:
        def _normalize_url(u):
            if not u:
                return None
            s = str(u).strip()
            if not s:
                return None
            if s.startswith(("http://", "https://")):
                return s
            if s.startswith("www."):
                return "https://" + s
            if "." in s and " " not in s:
                return "https://" + s
            return None
        df["Website"] = df["Website"].map(_normalize_url)

    return df

# ---------- CONFIGURABLE DISPLAY WIDTHS (characters) ----------
# Edit these numbers; we clip text to this length and add an ellipsis when needed.
CHAR_WIDTHS: Dict[str, int] = {
    "Business Name": 36,
    "Category": 36,
    "Phone": 30,
    "Address": 40,
    "Notes": 36,
    # NOTE: Service and Keywords are intentionally omitted (not shown).
    # Website is a hyperlink; we don't clip its display text here.
}

def _clip(s: Optional[str], n: int) -> str:
    if s is None:
        return ""
    t = str(s)
    if len(t) <= n:
        return t
    if n <= 1:
        return "…"
    return t[: n - 1] + "…"

# ---------- UI ----------
st.title("Vendors")

# Top row: search + Category filter + FTS toggle (no Service filter)
col_q, col_cat, col_fts = st.columns([5, 3, 2])
q = col_q.text_input("Search", value="", placeholder="Search name, notes, address…").strip()
cats = get_categories()
category_filter = col_cat.selectbox("Category", options=["(All)"] + cats, index=0)
category_filter = "" if category_filter == "(All)" else category_filter
use_fts = col_fts.checkbox("Use FTS", value=False)

# Data
df = load_vendors_df(q=q, use_fts=use_fts, category_filter=category_filter)

if df.empty:
    st.info("No vendors found.")
else:
    # Apply character clipping to configured columns (others pass through)
    clipped = df.copy()
    for col, n in CHAR_WIDTHS.items():
        if col in clipped.columns:
            clipped[col] = clipped[col].map(lambda x: _clip(x, n))

    # Column order: Website immediately AFTER Address
    desired_order = [
        "Business Name", "Category", "Phone",
        "Address", "Website",  # Website directly after Address
        "Notes",
        # Service/Keywords intentionally left out
    ]
    cols = [c for c in desired_order if c in clipped.columns]
    others = [c for c in clipped.columns if c not in cols and c != "id"]
    show = clipped[cols + others].drop(columns=["id"], errors="ignore")

    # Render with Website as a clickable link column
    col_cfg: Dict[str, st.column_config.Column] = {
        "Business Name": st.column_config.TextColumn("Business Name", width="medium"),
        "Category":      st.column_config.TextColumn("Category",      width="medium"),
        "Phone":         st.column_config.TextColumn("Phone",         width="medium"),
        "Address":       st.column_config.TextColumn("Address",       width="large"),
        "Notes":         st.column_config.TextColumn("Notes",         width="medium"),
    }
    if "Website" in show.columns:
        col_cfg["Website"] = st.column_config.LinkColumn("Website", help="Open in new tab", width="medium")

    st.dataframe(
        show,
        use_container_width=True,
        hide_index=True,
        column_config=col_cfg,
    )

# Debug footer
with st.expander("Debug / DB status"):
    counts = {
        "vendors": query_scalar("SELECT COUNT(*) FROM vendors;") or 0,
        "categories": query_scalar("SELECT COUNT(*) FROM categories;") if table_exists("categories") else None,
        "services": query_scalar("SELECT COUNT(*) FROM services;") if table_exists("services") else None,
    }
    st.write({"db_source": DB_SOURCE, "schema": SCHEMA, "counts": counts})
# EOF
