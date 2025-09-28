# app_admin.py — Vendors Admin (auto-maps real DB columns to friendly headers)
# - Works even if your DB uses snake_case/lowercase column names.
# - Uses SQLite rowid as stable key for Update/Delete.
# - Turso/libSQL or local SQLite supported.

import os
import contextlib
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import streamlit as st

# -------------------------------
# Connect (Turso/libsql or SQLite)
# -------------------------------

def _connect():
    db_url = st.secrets.get("TURSO_DATABASE_URL", os.environ.get("TURSO_DATABASE_URL", "")).strip()
    db_tok = st.secrets.get("TURSO_AUTH_TOKEN", os.environ.get("TURSO_AUTH_TOKEN", "")).strip()
    if db_url:
        try:
            from libsql_client import create_client
            client = create_client(url=db_url, auth_token=(db_tok or None))
            return client, "libsql"
        except Exception as e:
            st.warning(f"libsql unavailable ({e}); falling back to SQLite.")
    import sqlite3
    db_path = os.environ.get("SQLITE_PATH", "vendors.db")
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn, "sqlite"

CONN, DRIVER = _connect()

# -------------------------------
# Thin DB helpers
# -------------------------------

def exec_query(sql: str, params: Sequence[Any] = ()) -> List[Tuple]:
    if DRIVER == "libsql":
        res = CONN.execute(sql, list(params))
        return list(res.rows)
    else:
        cur = CONN.execute(sql, params)
        rows = cur.fetchall()
        return [tuple(r) for r in rows]

def exec_write(sql: str, params: Sequence[Any] = ()) -> int:
    if DRIVER == "libsql":
        res = CONN.execute(sql, list(params))
        return getattr(res, "rowcount", 1)
    else:
        cur = CONN.execute(sql, params)
        CONN.commit()
        return cur.rowcount

def table_exists(name: str) -> bool:
    rows = exec_query("SELECT 1 FROM sqlite_schema WHERE type='table' AND name=? LIMIT 1", (name,))
    return bool(rows)

def list_tables() -> List[str]:
    rows = exec_query("SELECT name FROM sqlite_schema WHERE type='table' ORDER BY 1")
    return [r[0] for r in rows]

def get_table_columns(table: str) -> List[str]:
    # Cannot parametrize table name in PRAGMA; embed trusted table string
    rows = exec_query(f"PRAGMA table_info('{table}')")
    # row = (cid, name, type, notnull, dflt_value, pk)
    return [r[1] for r in rows]

# -------------------------------
# Schema mapping (auto detect)
# -------------------------------

VENDORS_TABLE = "vendors"

# Friendly headers we want to show in the UI
EXPECTED = [
    "Category",
    "Service",
    "Business Name",
    "Contact Name",
    "Phone",
    "Address",
    "URL",
    "Notes",
    "Keywords",
    "Website",
]

# For each friendly header, probe these candidate DB column names (in order)
CANDIDATES: Dict[str, List[str]] = {
    "Category":      ["Category", "category"],
    "Service":       ["Service", "service"],
    "Business Name": ["Business Name", "business_name", "name", "vendor", "vendor_name", "business"],
    "Contact Name":  ["Contact Name", "contact_name", "contact"],
    "Phone":         ["Phone", "phone", "phone_number"],
    "Address":       ["Address", "address", "street_address"],
    "URL":           ["URL", "url"],
    "Notes":         ["Notes", "notes", "note"],
    "Keywords":      ["Keywords", "keywor]()
