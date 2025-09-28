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

def table_exists(name: str) -> bool:
    rows = exec_query("SELECT 1 FROM sqlite_schema WHERE type='table' AND name=? LIMIT 1", (name,))
    return bool(rows)

def list_tables() -> List[str]:
    rows = exec_query("SELECT name FROM sqlite_schema WHERE type='table' ORDER BY 1")
    return [r[0] for r in rows]

def table_has_column(table: str, column: str) -> bool:
    # Note: cannot bind table name in PRAGMA; embed trusted table string
    sql = f"SELECT 1 FROM pragma_table_info('{table}') WHERE name=? LIMIT 1"
    rows = exec_query(sql, (column,))
    return bool(rows)

# -------------------------------
# Constants / schema
# -------------------------------

VENDORS_TABLE = "vendors"
COLS = [
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

# Verify table & columns exist
HAS_VENDORS = table_exists(VENDORS_TABLE)
MISSING = [c for c in COLS if not table_has_column(VENDORS_TABLE, c)] if HAS_VENDORS else COLS

# -------------------------------
# Streamlit UI
# -------------------------------

st.set_page_config(page_title="Vendors - Admin", layout="wide")
st.title("Vendors - Admin")

with st.expander("Diagnostics (toggle)", expanded=False):
    st.write("Driver:", DRIVER)
    with contextlib.suppress(Exception):
        st.write("Tables:", list_tables())
    st.write("Using table:", VENDORS_TABLE)
    st.write("Missing expected columns:", MISSING)

if not HAS_VENDORS:
    st.error(f'Database missing required table: "{VENDORS_TABLE}".')
    st.stop()

if MISSING:
    st.error("Your production table is missing these required columns: " + ", ".join(MISSING))
    st.info("Fix the DB schema or adjust COLS in app_admin.py to match reality.")
    st.stop()

st.subheader("Find / pick a vendor to edit")
qtext = st.text_input("Search name/address/phone/keywords", placeholder="Type to filter...").strip()

# -------------------------------
# Build SELECT / WHERE (uses quoted identifiers)
# -------------------------------

# SELECT rowid as key + the exact production columns (quoted)
SEL = ', '.join([f'"{c}"' for c in COLS])
sql = f'SELECT rowid AS key, {SEL} FROM "{VENDORS_TABLE}" WHERE 1=1'
params: List[Any] = []

if qtext:
    like = f"%{qtext}%"
    # Search across common fields
    sql += """
      AND (
            COALESCE("Business Name",'') LIKE ?
         OR COALESCE("Contact Name",'')  LIKE ?
         OR COALESCE("Service",'')       LIKE ?
         OR COALESCE("Category",'')      LIKE ?
         OR COALESCE("Phone",'')         LIKE ?
         OR COALESCE("Address",'')       LIKE ?
         OR COALESCE("URL",'')           LIKE ?
         OR COALESCE("Website",'')       LIKE ?
         OR COALESCE("Notes",'')         LIKE ?
         OR COALESCE("Keywords",'')      LIKE ?
      )
    """
    params.extend([like]*10)

sql += '\nORDER BY "Business Name" COLLATE NOCASE ASC\nLIMIT 500'

# Execute
rows: List[Tuple] = []
err = None
try:
    rows = exec_query(sql, params)
except Exception as e:
    err = e

if err:
    with st.expander("SQL failed", expanded=True):
        st.code(sql.strip())
        st.write("params:", params)
        st.error(f"{type(err).__name__}: {err}")
    st.stop()

# -------------------------------
# DataFrame for display
# -------------------------------

cols_out = ["key"] + COLS
df = pd.DataFrame(rows, columns=cols_out)

if df.empty:
    st.info("No vendors match your filter. Clear the search to see all.")
    st.stop()

display_df = df.drop(columns=["key"], errors="ignore")
st.dataframe(display_df, use_container_width=True, hide_index=True)

# -------------------------------
# Selection widget
# -------------------------------

def make_label(row: pd.Series) -> str:
    name = (row.get("Business Name") or "").strip()
    svc  = (row.get("Service") or "").strip()
    phone = (row.get("Phone") or "").strip()
    addr  = (row.get("Address") or "").strip()
    bits = [b for b in [name, svc, phone, addr] if b]
    core = " — ".join(bits) if bits else "(unnamed vendor)"
    return f"{core}  [#{row['key']}]"

labels = ["New vendor..."] + [make_label(r) for _, r in df.iterrows()]
default_index = 0 if not qtext else 1
choice = st.selectbox("Pick vendor", options=labels, index=min(default_index, len(labels)-1))

selected_row: Optional[pd.Series] = None
if choice != "New vendor...":
    try:
        key_str = choice.rsplit("[#", 1)[1].rstrip("]")
        key_val = int(key_str) if key_str.isdigit() else None
    except Exception:
        key_val = None
    if key_val is not None:
        m = df[df["key"] == key_val]
        if not m.empty:
            selected_row = m.iloc[0]

# -------------------------------
# CRUD helpers (quoted identifiers)
# -------------------------------

def insert_vendor(values: Dict[str, Any]) -> str:
    cols = ', '.join([f'"{c}"' for c in COLS])
    placeholders = ', '.join(['?'] * len(COLS))
    sql = f'INSERT INTO "{VENDORS_TABLE}" ({cols}) VALUES ({placeholders})'
    params = [values.get(c) or "" for c in COLS]
    n = exec_write(sql, params)
    return f"Inserted ({n})"

def update_vendor(key: int, values: Dict[str, Any]) -> str:
    sets = ', '.join([f'"{c}"=?' for c in COLS])
    sql = f'UPDATE "{VENDORS_TABLE}" SET {sets} WHERE rowid=?'
    params = [values.get(c) or "" for c in COLS] + [key]
    n = exec_write(sql, params)
    return f"Updated ({n})"

def delete_vendor(key: int) -> str:
    sql = f'DELETE FROM "{VENDORS_TABLE}" WHERE rowid=?'
    n = exec_write(sql, (key,))
    return f"Deleted ({n})"

# -------------------------------
# Forms
# -------------------------------

st.markdown("---")

def render_inputs(prefix: str, init: Dict[str, Any]) -> Dict[str, Any]:
    c1, c2 = st.columns(2)
    with c1:
        category     = st.text_input("Category",     value=init.get("Category",""))
        service      = st.text_input("Service",      value=init.get("Service",""))
        business     = st.text_input("Business Name",value=init.get("Business Name",""))
        contact_name = st.text_input("Contact Name", value=init.get("Contact Name",""))
        phone        = st.text_input("Phone",        value=init.get("Phone",""))
    with c2:
        address  = st.text_area ("Address",  value=init.get("Address",""), height=80)
        url      = st.text_input("URL",      value=init.get("URL",""))
        website  = st.text_input("Website",  value=init.get("Website",""))
        keywords = st.text_input("Keywords", value=init.get("Keywords",""))
        notes    = st.text_area ("Notes",    value=init.get("Notes",""), height=80)

    out = {
        "Category": category,
        "Service": service,
        "Business Name": business,
        "Contact Name": contact_name,
        "Phone": phone,
        "Address": address,
        "URL": url,
        "Notes": notes,
        "Keywords": keywords,
        "Website": website,
    }
    return out

if selected_row is None:
    st.subheader("Add a new vendor")
    with st.form("add_vendor"):
        vals = render_inputs("add", init={})
        submitted = st.form_submit_button("Add Vendor")
        if submitted:
            if not (vals.get("Business Name") or "").strip():
                st.error("Business Name is required.")
            else:
                msg = insert_vendor(vals)
                st.success(msg)
                st.rerun()
else:
    st.subheader("Edit vendor")
    key_val = int(selected_row["key"])
    init_vals = {c: selected_row[c] for c in COLS}
    with st.form("edit_vendor"):
        vals = render_inputs("edit", init=init_vals)
        c1, c2 = st.columns([1,1])
        do_update = c1.form_submit_button("Update")
        do_delete = c2.form_submit_button("Delete")
        if do_update:
            if not (vals.get("Business Name") or "").strip():
                st.error("Business Name is required.")
            else:
                msg = update_vendor(key_val, vals)
                st.success(msg)
                st.rerun()
        if do_delete:
            with st.expander("Confirm delete", expanded=True):
                st.warning(f"About to delete: {selected_row.get('Business Name','(unnamed)')}  [#{key_val}]")
                really = st.checkbox("Yes, permanently delete this vendor.")
                if really and st.button("Confirm delete"):
                    msg = delete_vendor(key_val)
                    st.warning(msg)
                    st.rerun()

# -------------------------------
# End
# -------------------------------
