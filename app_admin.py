# app_admin.py — Vendors Admin (Turso or SQLite)
# - Robust to missing columns (id/vendor_id/is_active/category)
# - Safe cursor usage (fetchall())
# - Keep key in df; hide only in display copy
# - Add / Update / Delete flows
# - Debug gated behind a checkbox; no secrets dumped by default

import os
import contextlib
import sqlite3
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import streamlit as st

# -------------------------------
# Connection handling (Turso/libsql or SQLite)
# -------------------------------

def _connect() -> Tuple[Any, str]:
    """Return (conn, driver) where driver in {'libsql','sqlite'}."""
    db_url  = st.secrets.get("TURSO_DATABASE_URL", os.environ.get("TURSO_DATABASE_URL", "")).strip()
    db_tok  = st.secrets.get("TURSO_AUTH_TOKEN", os.environ.get("TURSO_AUTH_TOKEN", "")).strip()
    if db_url:
        # Try libsql client
        try:
            from libsql_client import create_client
            client = create_client(url=db_url, auth_token=(db_tok or None))
            return client, "libsql"
        except Exception as e:
            st.warning(f"Falling back to SQLite (libsql unavailable): {e}")
    # SQLite fallback
    db_path = os.environ.get("SQLITE_PATH", "vendors.db")
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn, "sqlite"


conn, DRIVER = _connect()


# -------------------------------
# SQL helpers (uniform API)
# -------------------------------

def exec_query(sql: str, params: Sequence[Any] = ()) -> List[Tuple]:
    """SELECT; returns list of tuples."""
    if DRIVER == "libsql":
        # libsql_client returns rows with .rows (list of tuples)
        res = conn.execute(sql, list(params))
        return list(res.rows)
    else:
        cur = conn.execute(sql, params)
        rows = cur.fetchall()
        return [tuple(r) for r in rows]

def exec_write(sql: str, params: Sequence[Any] = ()) -> int:
    """INSERT/UPDATE/DELETE; returns affected rows (best-effort)."""
    if DRIVER == "libsql":
        res = conn.execute(sql, list(params))
        # libsql doesn't always report rowcount; return 1 if no exception
        return getattr(res, "rowcount", 1)
    else:
        cur = conn.execute(sql, params)
        conn.commit()
        return cur.rowcount

def table_has_column(table: str, column: str) -> bool:
    """Check if a column exists on a table (SQLite/Turso)."""
    # Cannot bind table name into PRAGMA; use literal (we control the string)
    sql = f"SELECT 1 FROM pragma_table_info('{table}') WHERE lower(name)=lower(?) LIMIT 1"
    row = None
    with contextlib.suppress(Exception):
        rows = exec_query(sql, (column,))
        row = rows[0] if rows else None
    return bool(row)

def list_tables() -> List[str]:
    rows = exec_query("SELECT name FROM sqlite_schema WHERE type='table' ORDER BY 1")
    return [r[0] for r in rows]


# -------------------------------
# Schema detection & constants
# -------------------------------

HAS_VENDOR_ID = table_has_column("vendors", "id") or table_has_column("vendors", "vendor_id")
KEY_COL = "id" if table_has_column("vendors", "id") else ("vendor_id" if table_has_column("vendors", "vendor_id") else None)

HAS_IS_ACTIVE = table_has_column("vendors", "is_active")
HAS_CATEGORY  = table_has_column("categories", "name")

REQUIRED_BASE = ["business_name", "phone", "address", "website"]  # can be empty if schema differs

# -------------------------------
# UI — Page header & diagnostics
# -------------------------------

st.set_page_config(page_title="Vendors - Admin", layout="wide")
st.title("Vendors - Admin")

with st.expander("Diagnostics (toggle)", expanded=False):
    st.write("Driver:", DRIVER)
    with contextlib.suppress(Exception):
        st.write("Tables:", list_tables())
    st.write("Key column:", KEY_COL)
    st.write("Has is_active:", HAS_IS_ACTIVE)
    st.write("Has category name:", HAS_CATEGORY)

# -------------------------------
# Search UI
# -------------------------------

st.subheader("Find / pick a vendor to edit")
query_text = st.text_input("Search name/address/phone", placeholder="Type to filter...").strip()

# -------------------------------
# Query vendors (robust to missing cols)
# -------------------------------

# SELECT list (schema-aware)
select_cols = []
if KEY_COL:
    select_cols.append(f"v.{KEY_COL}")
else:
    # Provide synthetic key in result to keep code paths working (value will be None)
    select_cols.append("NULL AS missing_key")

select_cols += [
    "v.business_name"                      if table_has_column("vendors", "business_name") else "'' AS business_name",
    "COALESCE(c.name,'') AS category"      if HAS_CATEGORY else "'' AS category",
    "v.phone"                              if table_has_column("vendors", "phone") else "'' AS phone",
    "v.address"                            if table_has_column("vendors", "address") else "'' AS address",
    "v.website"                            if table_has_column("vendors", "website") else "'' AS website",
    "v.is_active"                          if HAS_IS_ACTIVE else "1 AS is_active",
]
SELECT_SQL = ", ".join(select_cols)

base_sql = f"""
SELECT {SELECT_SQL}
FROM vendors v
LEFT JOIN categories c ON c.id = v.category_id
WHERE 1=1
"""

params: List[Any] = []

# Optional is_active filter
if HAS_IS_ACTIVE:
    base_sql += " AND v.is_active = 1"

# Search filter
if query_text:
    # Use COALESCE to avoid NULL match issues
    base_sql += """
      AND (
            COALESCE(v.business_name,'') LIKE ?
         OR COALESCE(v.address,'')       LIKE ?
         OR COALESCE(v.phone,'')         LIKE ?
      )
    """
    like = f"%{query_text}%"
    params.extend([like, like, like])

base_sql += "\nORDER BY v.business_name COLLATE NOCASE ASC\nLIMIT 500"

# Execute
rows = []
err = None
try:
    rows = exec_query(base_sql, params)
except Exception as e:
    err = e

if err:
    with st.expander("SQL failed", expanded=True):
        st.code(base_sql.strip())
        st.write("params:", params)
        st.error(f"{type(err).__name__}: {err}")
    st.stop()

# -------------------------------
# Build DataFrame and display copy
# -------------------------------

cols = ["key", "business_name", "category", "phone", "address", "website", "is_active"]
df = pd.DataFrame(rows, columns=cols)

if df.empty:
    st.info("No vendors match your filter. Clear the search to see all.")
    st.stop()

# Keep df intact (key preserved); hide key/is_active in display
hide_cols = [c for c in ("key", "is_active") if c in df.columns]
display_df = df.drop(columns=hide_cols, errors="ignore")

st.dataframe(display_df, use_container_width=True, hide_index=True)

# -------------------------------
# Selection control
# -------------------------------

def label_row(r: pd.Series) -> str:
    parts = [str(r.get("business_name") or "").strip()]
    ph = str(r.get("phone") or "").strip()
    addr = str(r.get("address") or "").strip()
    if ph:
        parts.append(ph)
    if addr:
        parts.append(addr)
    label = " — ".join([p for p in parts if p])
    return label or "(unnamed vendor)"

options = ["New vendor..."] + [label_row(r) for _, r in df.iterrows()]
idx = st.selectbox("Pick vendor", options=options, index=0 if not query_text else 1)

selected = None
if idx != "New vendor...":
    # Map back to key via displayed label index
    # We subtract 1 because option[0] is "New vendor..."
    selected = df.iloc[options.index(idx) - 1]

# -------------------------------
# Category helpers
# -------------------------------

def get_categories() -> List[Tuple[int, str]]:
    if HAS_CATEGORY:
        try:
            rows = exec_query("SELECT id, name FROM categories ORDER BY name COLLATE NOCASE ASC")
            return [(int(r[0]), str(r[1])) for r in rows]
        except Exception:
            return []
    return []

CATS = get_categories()

# -------------------------------
# Forms: Add / Update / Delete
# -------------------------------

def upsert_vendor(values: Dict[str, Any], key: Optional[Any] = None) -> str:
    """Insert if key is None, else update by key."""
    # Build column set based on actual schema
    cols_db = []
    params_db = []
    for c in ("business_name", "phone", "address", "website"):
        if table_has_column("vendors", c):
            cols_db.append(c)
            params_db.append(values.get(c) or "")

    # Category mapping (optional)
    if table_has_column("vendors", "category_id") and values.get("category_id") is not None:
        cols_db.append("category_id")
        params_db.append(int(values["category_id"]))

    if key is None:
        # INSERT
        placeholders = ",".join(["?"] * len(params_db))
        collist = ",".join(cols_db)
        sql = f"INSERT INTO vendors ({collist}) VALUES ({placeholders})"
        n = exec_write(sql, params_db)
        return f"Inserted ({n})"
    else:
        if KEY_COL is None:
            return "Cannot update: no key column in schema."
        # UPDATE
        sets = ",".join([f"{c}=?" for c in cols_db])
        sql = f"UPDATE vendors SET {sets} WHERE {KEY_COL}=?"
        n = exec_write(sql, params_db + [key])
        return f"Updated ({n})"

def delete_vendor(key: Any) -> str:
    if KEY_COL is None:
        return "Cannot delete: no key column."
    if HAS_IS_ACTIVE and table_has_column("vendors", "is_active"):
        # Soft delete
        n = exec_write(f"UPDATE vendors SET is_active=0 WHERE {KEY_COL}=?", (key,))
        return f"Soft-deleted ({n})"
    else:
        # Hard delete
        n = exec_write(f"DELETE FROM vendors WHERE {KEY_COL}=?", (key,))
        return f"Deleted ({n})"

# -------------------------------
# Render forms
# -------------------------------

st.markdown("---")

if selected is None:
    st.subheader("Add a new vendor")
    with st.form("add_vendor"):
        name = st.text_input("Business Name")
        phone = st.text_input("Phone")
        addr  = st.text_input("Address")
        web   = st.text_input("Website")

        cat_id = None
        if CATS and table_has_column("vendors", "category_id"):
            cat_labels = [c[1] for c in CATS]
            cat_choice = st.selectbox("Category", ["(none)"] + cat_labels, index=0)
            if cat_choice != "(none)":
                cat_id = CATS[cat_labels.index(cat_choice)][0]

        submitted = st.form_submit_button("Add Vendor")
        if submitted:
            res = upsert_vendor(
                {"business_name": name, "phone": phone, "address": addr, "website": web, "category_id": cat_id},
                key=None
            )
            st.success(res)
else:
    st.subheader("Edit vendor")
    initial = {
        "business_name": selected.get("business_name", ""),
        "phone":         selected.get("phone", ""),
        "address":       selected.get("address", ""),
        "website":       selected.get("website", ""),
        "category_id":   None,
    }

    with st.form("edit_vendor"):
        name = st.text_input("Business Name", value=initial["business_name"])
        phone = st.text_input("Phone", value=initial["phone"])
        addr  = st.text_input("Address", value=initial["address"])
        web   = st.text_input("Website", value=initial["website"])

        cat_id = None
        if CATS and table_has_column("vendors", "category_id"):
            cat_labels = [c[1] for c in CATS]
            # We cannot infer current category reliably without an extra join for id; leave unselected.
            cat_choice = st.selectbox("Category", ["(leave unchanged)"] + cat_labels, index=0)
            if cat_choice != "(leave unchanged)":
                cat_id = CATS[cat_labels.index(cat_choice)][0]

        c1, c2, c3 = st.columns([1,1,1])
        do_update = c1.form_submit_button("Update")
        do_delete = c2.form_submit_button("Delete")
        # Optional: a no-op write test
        # do_ping   = c3.form_submit_button("Write test")

        key_val = selected.get("key")

        if do_update:
            res = upsert_vendor(
                {"business_name": name, "phone": phone, "address": addr, "website": web, "category_id": cat_id},
                key=key_val
            )
            st.success(res)

        if do_delete:
            res = delete_vendor(key_val)
            st.warning(res)

# -------------------------------
# End
# -------------------------------
