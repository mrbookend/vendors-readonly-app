# app_admin.py — Vendors Admin (single-table schema with spaces in column names)
# Production schema (columns on table "vendors"):
#   "Category", "Service", "Business Name", "Contact Name", "Phone",
#   "Address", "URL", "Notes", "Keywords", "Website"
#
# Key approach:
#   - Use SQLite's implicit ROWID as a stable primary key surrogate.
#   - All UPDATE/DELETE are done via WHERE rowid = ? to target a single row.
#
# Safe & portable:
#   - Works with Turso/libsql or local SQLite.
#   - Always fetches rows (no iteration over raw cursors).
#   - UI shows a grid without the key; selection uses a label that encodes [#rowid].

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
    """Run SELECT and return list of tuples."""
    if DRIVER == "libsql":
        res = CONN.execute(sql, list(params))
        return list(res.rows)
    else:
        cur = CONN.execute(sql, params)
        rows = cur.fetchall()
        return [tuple(r) for r in rows]

def exec_write(sql: str, params: Sequence[Any] = ()) -> int:
    """Run INSERT/UPDATE/DELETE and return affected count (best-effort)."""
    if DRIVER == "libsql":
        res = CONN.execute(sql, list(params))
        # libsql may not report rowcount; assume success if no exception
        return getattr(res, "rowcount", 1)
    else:
        cur = CONN.execute(sql, params)
        CONN.commit()
        return cur.rowcount

def tab
