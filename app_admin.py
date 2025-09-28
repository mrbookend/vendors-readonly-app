# app_admin.py — Vendors Admin (works with Turso/libsql or local SQLite)
# - Schema-aware: handles missing tables/columns (categories, is_active, vendor_id vs id)
# - Safe DB access: always fetches rows (no iterating raw cursors)
# - UI-safe: keeps key for updates/deletes; hides key in display copy
# - Simple CRUD: add, update, delete (soft-delete if is_active exists)

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
        return getattr(res, "rowcount", 1)
    else:
        cur = CONN.execute(sql, params)
        CONN.commit()
        return cur.rowcount

def table_exists(name: str) -> bool:
    rows = exec_query("SELECT 1 FROM sqlite_schema WHERE type='table' AND name=? LIMIT 1", (name,))
    return bool(rows)

def table_has_column(table: str, column: str) -> bool:
    # pragma_table_info cannot parameterize table name; embed the (trusted) table
    sql = f"SELECT 1 FROM pragma_table_info('{table}') WHERE lower(name)=lower(?) LIMIT 1"
    rows = exec_query(sql, (column,))
    return bool(rows)

def list_tables() -> List[str]:
    rows = exec_query("SELECT name FROM sqlite_schema WHERE type='table' ORDER BY 1")
    return [r[0] for r in rows]

# -------------------------------
# Schema introspection
# -------------------------------

HAS_VENDORS      = table_exists("vendors")
HAS_CATS_TABLE   = table_exists("categories")
HAS_VENDOR_ID    = table_has_column("vendors", "id")
HAS_VENDORID_ALT = table_has_column("vendors", "vendor_id")
KEY_COL          = "id" if HAS_VENDOR_ID else ("vendor_id" if HAS_VENDORID_ALT else None)

HAS_BUSINESS     = table_has_column("vendors", "business_name")
HAS_PHONE        = table_has_column("vendors", "phone")
HAS_ADDRESS      = table_has_column("vendors", "address")
HAS_WEBSITE      = table_has_column("vendors", "website")
HAS_IS_ACTIVE    = table_has_column("vendors", "is_active")
HAS_CAT_NAME     = table_has_column("categories", "name")
HAS_CAT_FK       = table_has_column("vendors", "category_id")

# -------------------------------
# Streamlit UI
# -------------------------------

st.set_page_config(page_title="Vendors - Admin", layout="wide")
st.title("Vendors - Admin")

with st.expander("Diagnostics (toggle)", expanded=False):
    st.write("Driver:", DRIVER)
    with contextlib.suppress(Exception):
        st.write("Tables:", list_tables())
    st.write("Key column:", KEY_COL)
    st.write("Has vendors:", HAS_VENDORS)
    st.write("Has categories table:", HAS_CATS_TABLE)
    st.write("Has category FK:", HAS_CAT_FK)
    st.write("Has is_active:", HAS_IS_ACTIVE)

if not HAS_VENDORS:
    st.error("Database missing required table: vendors")
    st.stop()

st.subheader("Find / pick a vendor to edit")
qtext = st.text_input("Search name/address/phone", placeholder="Type to filter...").strip()

# -------------------------------
# Build SELECT & FROM (schema-aware)
# -------------------------------

select_cols: List[str] = []
if KEY_COL:
    select_cols.append(f"v.{KEY_COL} AS key")
else:
    select_cols.append("NULL AS key")

select_cols.append("v.business_name" if HAS_BUSINESS else "'' AS business_name")
if HAS_CATS_TABLE and HAS_CAT_NAME and HAS_CAT_FK:
    select_cols.append("COALESCE(c.name,'') AS category")
else:
    select_cols.append("'' AS category")
select_cols.append("v.phone" if HAS_PHONE else "'' AS phone")
select_cols.append("v.address" if HAS_ADDRESS else "'' AS address")
select_cols.append("v.website" if HAS_WEBSITE else "'' AS website")
select_cols.append("v.is_active" if HAS_IS_ACTIVE else "1 AS is_active")

SELECT_SQL = ", ".join(select_cols)

base_sql = f"SELECT {SELECT_SQL}\nFROM vendors v"
if HAS_CATS_TABLE and HAS_CAT_FK:
    base_sql += "\nLEFT JOIN categories c ON c.id = v.category_id"
base_sql += "\nWHERE 1=1"

params: List[Any] = []
if HAS_IS_ACTIVE:
    base_sql += "\n  AND v.is_active = 1"

if qtext:
    like = f"%{qtext}%"
    base_sql += """
      AND (
            COALESCE(v.business_name,'') LIKE ?
         OR COALESCE(v.address,'')       LIKE ?
         OR COALESCE(v.phone,'')         LIKE ?
      )
    """
    params.extend([like, like, like])

base_sql += "\nORDER BY v.business_name COLLATE NOCASE ASC\nLIMIT 500"

# -------------------------------
# Execute and display
# -------------------------------

rows: List[Tuple] = []
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

cols = ["key", "business_name", "category", "phone", "address", "website", "is_active"]
df = pd.DataFrame(rows, columns=cols)

if df.empty:
    st.info("No vendors match your filter. Clear the search to see all.")
    st.stop()

# Keep df intact; hide key/is_active in the rendered grid only
display_df = df.drop(columns=[c for c in ("key", "is_active") if c in df.columns], errors="ignore")
st.dataframe(display_df, use_container_width=True, hide_index=True)

# -------------------------------
# Selection widget
# -------------------------------

def make_label(row: pd.Series) -> str:
    name = (row.get("business_name") or "").strip()
    phone = (row.get("phone") or "").strip()
    addr = (row.get("address") or "").strip()
    key  = row.get("key")
    bits = [b for b in [name, phone, addr] if b]
    core = " — ".join(bits) if bits else "(unnamed vendor)"
    return f"{core}  [#{key}]"  # ensure uniqueness even for dup names

labels = ["New vendor..."] + [make_label(r) for _, r in df.iterrows()]
choice = st.selectbox("Pick vendor", options=labels, index=0 if not qtext else 1)

selected_row: Optional[pd.Series] = None
if choice != "New vendor...":
    # Extract key from trailing [#key]
    try:
        key_str = choice.rsplit("[#", 1)[1].rstrip("]")
        key_val = int(key_str) if key_str.isdigit() else key_str
    except Exception:
        key_val = None
    if key_val is not None and "key" in df.columns:
        matches = df[df["key"] == key_val]
        if not matches.empty:
            selected_row = matches.iloc[0]

# -------------------------------
# Categories helper
# -------------------------------

def get_categories() -> List[Tuple[int, str]]:
    if not (HAS_CATS_TABLE and HAS_CAT_NAME):
        return []
    try:
        rs = exec_query("SELECT id, name FROM categories ORDER BY name COLLATE NOCASE ASC")
        return [(int(r[0]), str(r[1])) for r in rs]
    except Exception:
        return []

CATS = get_categories()

# -------------------------------
# CRUD helpers
# -------------------------------

def upsert_vendor(values: Dict[str, Any], key: Optional[Any] = None) -> str:
    cols_db: List[str] = []
    vals_db: List[Any]  = []

    # only include columns that actually exist
    if HAS_BUSINESS:
        cols_db.append("business_name"); vals_db.append(values.get("business_name") or "")
    if HAS_PHONE:
        cols_db.append("phone");         vals_db.append(values.get("phone") or "")
    if HAS_ADDRESS:
        cols_db.append("address");       vals_db.append(values.get("address") or "")
    if HAS_WEBSITE:
        cols_db.append("website");       vals_db.append(values.get("website") or "")
    if HAS_CAT_FK and values.get("category_id") is not None:
        cols_db.append("category_id");   vals_db.append(int(values["category_id"]))

    if key is None:
        placeholders = ",".join(["?"] * len(vals_db))
        collist = ",".join(cols_db) if cols_db else ""
        if not collist:
            return "Nothing to insert: no writable columns in schema."
        sql = f"INSERT INTO vendors ({collist}) VALUES ({placeholders})"
        n = exec_write(sql, vals_db)
        return f"Inserted ({n})"
    else:
        if not KEY_COL:
            return "Cannot update: no key column in schema."
        if not cols_db:
            return "Nothing to update."
        sets = ",".join([f"{c}=?" for c in cols_db])
        sql = f"UPDATE vendors SET {sets} WHERE {KEY_COL}=?"
        n = exec_write(sql, vals_db + [key])
        return f"Updated ({n})"

def delete_vendor(key: Any) -> str:
    if not KEY_COL:
        return "Cannot delete: no key column."
    if HAS_IS_ACTIVE:
        n = exec_write(f"UPDATE vendors SET is_active=0 WHERE {KEY_COL}=?", (key,))
        return f"Soft-deleted ({n})"
    else:
        n = exec_write(f"DELETE FROM vendors WHERE {KEY_COL}=?", (key,))
        return f"Deleted ({n})"

# -------------------------------
# Forms
# -------------------------------

st.markdown("---")

if selected_row is None:
    st.subheader("Add a new vendor")
    with st.form("add_vendor"):
        name = st.text_input("Business Name" + ("" if HAS_BUSINESS else " (not in schema)"))
        phone = st.text_input("Phone" + ("" if HAS_PHONE else " (not in schema)"))
        addr  = st.text_input("Address" + ("" if HAS_ADDRESS else " (not in schema)"))
        web   = st.text_input("Website" + ("" if HAS_WEBSITE else " (not in schema)"))

        cat_id = None
        if CATS and HAS_CAT_FK:
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
    init = {
        "business_name": selected_row.get("business_name", "") if HAS_BUSINESS else "",
        "phone":         selected_row.get("phone", "") if HAS_PHONE else "",
        "address":       selected_row.get("address", "") if HAS_ADDRESS else "",
        "website":       selected_row.get("website", "") if HAS_WEBSITE else "",
    }
    key_val = selected_row.get("key")

    with st.form("edit_vendor"):
        name = st.text_input("Business Name" + ("" if HAS_BUSINESS else " (not in schema)"), value=init["business_name"])
        phone = st.text_input("Phone" + ("" if HAS_PHONE else " (not in schema)"), value=init["phone"])
        addr  = st.text_input("Address" + ("" if HAS_ADDRESS else " (not in schema)"), value=init["address"])
        web   = st.text_input("Website" + ("" if HAS_WEBSITE else " (not in schema)"), value=init["website"])

        cat_id = None
        if CATS and HAS_CAT_FK:
            cat_labels = [c[1] for c in CATS]
            cat_choice = st.selectbox("Category", ["(leave unchanged)"] + cat_labels, index=0)
            if cat_choice != "(leave unchanged)":
                cat_id = CATS[cat_labels.index(cat_choice)][0]

        c1, c2 = st.columns([1,1])
        do_update = c1.form_submit_button("Update")
        do_delete = c2.form_submit_button("Delete")

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
