# app.py  (or app_readonly.py)
# Vendors Read-only — wide layout, collapsed sidebar, fixed character widths per column,
# and Website hyperlink column immediately after Address.

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

def get_services() -> List[str]:
    if SCHEMA["has_services_table"]:
        df = run_df("SELECT name FROM services ORDER BY name;")
        if not df.empty:
            return df["name"].tolist()
    if SCHEMA["uses_svc_text"]:
        df = run_df(
            "SELECT DISTINCT TRIM(service) AS name FROM vendors "
            "WHERE TRIM(service) <> '' ORDER BY 1;"
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
                    category_filter: str = "", service_filter: str = "") -> pd.DataFrame:
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

    # Service expr/join
    if SCHEMA["uses_svc_id"]:
        svc_expr = "s.name AS Service"
        svc_join = "LEFT JOIN services s ON s.id = v.service_id"
        svc_filter_expr = "s.name = :svc"
    elif SCHEMA["uses_svc_text"]:
        svc_expr, svc_join = "v.service AS Service", ""
        svc_filter_expr = "v.service = :svc"
    else:
        svc_expr, svc_join, svc_filter_expr = "NULL AS Service", "", None

    base_select = f"""
        SELECT
          v.id AS id,
          {cat_expr},
          {svc_expr},
          v.business_name AS "Business Name",
          v.contact_name  AS "Contact Name",
          v.phone         AS Phone,
          v.address       AS Address,
          v.notes         AS Notes,
          v.website       AS Website,
          {'v.keywords AS Keywords' if SCHEMA['has_keywords_col'] else 'NULL AS Keywords'}
        FROM vendors v
        {cat_join}
        {svc_join}
    """

    params: Dict[str, object] = {}
    wheres: List[str] = []

    if category_filter and cat_filter_expr:
        wheres.append(cat_filter_expr); params["cat"] = category_filter
    if service_filter and svc_filter_expr:
        wheres.append(svc_filter_expr); params["svc"] = service_filter

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
            if SCHEMA["has_keywords_col"]:
                lconds.append("v.keywords LIKE :like")
            if SCHEMA["uses_cat_text"]:
                lconds.append("v.category LIKE :like")
            if SCHEMA["uses_svc_text"]:
                lconds.append("v.service LIKE :like")
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
# Edit these numbers to control how many characters are shown per column.
# We clip text to this character count and add an ellipsis when needed.
CHAR_WIDTHS: Dict[str, int] = {
    "Business Name": 36,
    "Category": 30,
    "Service": 30,
    "Phone": 24,
    "Address": 40,
    "Notes": 36,
    "Keywords": 10,
    # Note: "Website" is a hyperlink column; character clipping is not applied to it.
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

# Top row: search + filters + FTS toggle
col_q, col_cat, col_svc, col_fts = st.columns([4, 3, 3, 2])
q = col_q.text_input("Search", value="", placeholder="Search name, notes, address, keywords…").strip()
cats = get_categories()
svcs = get_services()
category_filter = col_cat.selectbox("Category", options=["(All)"] + cats, index=0)
category_filter = "" if category_filter == "(All)" else category_filter
service_filter = col_svc.selectbox("Service", options=["(All)"] + svcs, index=0)
service_filter = "" if service_filter == "(All)" else service_filter
use_fts = col_fts.checkbox("Use FTS", value=False)

# Data
df = load_vendors_df(q=q, use_fts=use_fts,
                     category_filter=category_filter, service_filter=service_filter)

if df.empty:
    st.info("No vendors found.")
else:
    # Apply character clipping to text columns that exist (not applied to hyperlink column)
    clipped = df.copy()
    for col, n in CHAR_WIDTHS.items():
        if col in clipped.columns:
            clipped[col] = clipped[col].map(lambda x: _clip(x, n))

    # Column order: put Website immediately AFTER Address
    # Only include columns that exist; always drop internal id.
    desired_order = [
        "Business Name", "Category", "Service", "Phone",
        "Address", "Website",  # Website directly after Address
        "Notes", "Keywords",
    ]
    cols = [c for c in desired_order if c in clipped.columns]
    others = [c for c in clipped.columns if c not in cols and c != "id"]
    show = clipped[cols + others].drop(columns=["id"], errors="ignore")

    # Render with Website as a clickable link column
    col_cfg: Dict[str, st.column_config.Column] = {
        "Business Name": st.column_config.TextColumn("Business Name", width="medium"),
        "Category":      st.column_config.TextColumn("Category",      width="small"),
        "Service":       st.column_config.TextColumn("Service",       width="small"),
        "Phone":         st.column_config.TextColumn("Phone",         width="small"),
        "Address":       st.column_config.TextColumn("Address",       width="large"),
        "Notes":         st.column_config.TextColumn("Notes",         width="large"),
        "Keywords":      st.column_config.TextColumn("Keywords",      width="large"),
    }
    if "Website" in show.columns:
        # Note: LinkColumn's display_text is global, not per-row; we leave the URL visible or use a fixed label.
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
