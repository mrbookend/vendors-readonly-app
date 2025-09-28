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
    "Keywords":      ["Keywords", "keywords", "tags"],
    "Website":       ["Website", "website", "web", "site"],
}

def resolve_mapping(table: str) -> Tuple[Dict[str, str], List[str]]:
    """Return (mapping, missing) where mapping maps EXPECTED->actual DB column name."""
    if not table_exists(table):
        return {}, EXPECTED[:]  # everything missing
    actual_cols = get_table_columns(table)
    by_lower = {c.lower(): c for c in actual_cols}
    mapping: Dict[str, str] = {}
    missing: List[str] = []
    for friendly, options in CANDIDATES.items():
        found = None
        for opt in options:
            if opt.lower() in by_lower:
                found = by_lower[opt.lower()]
                break
        if found:
            mapping[friendly] = found
        else:
            missing.append(friendly)
    return mapping, missing

MAPPING, MISSING = resolve_mapping(VENDORS_TABLE)

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
    st.write("Detected mapping (friendly → actual):", MAPPING)
    if MISSING:
        st.write("Missing (will show blank fields & be skipped on write):", MISSING)

if not table_exists(VENDORS_TABLE):
    st.error(f'Database missing required table: "{VENDORS_TABLE}".')
    st.stop()

st.subheader("Find / pick a vendor to edit")
qtext = st.text_input("Search (name / service / phone / address / notes / keywords)", placeholder="Type to filter...").strip()

# -------------------------------
# Build SELECT / WHERE dynamically
# -------------------------------

# SELECT rowid and all friendly fields, aliasing actual cols; blanks for missing
select_parts = ['rowid AS key']
for friendly in EXPECTED:
    if friendly in MAPPING:
        actual = MAPPING[friendly]
        select_parts.append(f'"{actual}" AS "{friendly}"')
    else:
        select_parts.append(f"'' AS \"{friendly}\"")

select_sql = ", ".join(select_parts)
sql = f'SELECT {select_sql} FROM "{VENDORS_TABLE}" WHERE 1=1'
params: List[Any] = []

# Search across mapped fields only
if qtext:
    like = f"%{qtext}%"
    where_bits = []
    search_fields = ["Business Name", "Service", "Contact Name", "Category",
                     "Phone", "Address", "URL", "Website", "Notes", "Keywords"]
    for friendly in search_fields:
        if friendly in MAPPING:
            actual = MAPPING[friendly]
            where_bits.append(f'COALESCE("{actual}", \'\') LIKE ?')
            params.append(like)
    if where_bits:
        sql += "\n  AND (" + "\n       OR ".join(where_bits) + ")"

sql += '\nORDER BY ' + (
    f'"{MAPPING["Business Name"]}" COLLATE NOCASE' if "Business Name" in MAPPING
    else 'rowid'
) + '\nLIMIT 500'

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
# DataFrame + grid
# -------------------------------

cols_out = ["key"] + EXPECTED
df = pd.DataFrame(rows, columns=cols_out)

if df.empty:
    st.info("No vendors match your filter. Clear the search to see all.")
    st.stop()

st.dataframe(df.drop(columns=["key"], errors="ignore"),
             use_container_width=True, hide_index=True)

# -------------------------------
# Selection
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
choice = st.selectbox("Pick vendor", options=labels, index=0 if labels else 0)

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
# CRUD helpers (write only mapped cols)
# -------------------------------

def values_to_db_params(values: Dict[str, Any]) -> Tuple[List[str], List[Any]]:
    cols_db: List[str] = []
    vals_db: List[Any] = []
    for friendly in EXPECTED:
        if friendly in MAPPING:
            actual = MAPPING[friendly]
            cols_db.append(f'"{actual}"')
            vals_db.append(values.get(friendly, "") or "")
    return cols_db, vals_db

def insert_vendor(values: Dict[str, Any]) -> str:
    cols_db, vals_db = values_to_db_params(values)
    if not cols_db:
        return "Nothing to insert: no mapped columns."
    placeholders = ", ".join(["?"] * len(vals_db))
    sql = f'INSERT INTO "{VENDORS_TABLE}" ({", ".join(cols_db)}) VALUES ({placeholders})'
    n = exec_write(sql, vals_db)
    return f"Inserted ({n})"

def update_vendor(rowid_key: int, values: Dict[str, Any]) -> str:
    cols_db, vals_db = values_to_db_params(values)
    if not cols_db:
        return "Nothing to update: no mapped columns."
    sets = ", ".join([f"{c}=?" for c in cols_db])
    sql = f'UPDATE "{VENDORS_TABLE}" SET {sets} WHERE rowid=?'
    n = exec_write(sql, vals_db + [rowid_key])
    return f"Updated ({n})"

def delete_vendor(rowid_key: int) -> str:
    sql = f'DELETE FROM "{VENDORS_TABLE}" WHERE rowid=?'
    n = exec_write(sql, (rowid_key,))
    return f"Deleted ({n})"

# -------------------------------
# Forms
# -------------------------------

st.markdown("---")

def render_inputs(init: Dict[str, Any]) -> Dict[str, Any]:
    # Show all friendly fields; if a field isn't mapped, we still present it but it won't be written.
    c1, c2 = st.columns(2)
    with c1:
        v_category = st.text_input("Category",     value=init.get("Category", ""))
        v_service  = st.text_input("Service",      value=init.get("Service", ""))
        v_biz      = st.text_input("Business Name",value=init.get("Business Name", ""))
        v_contact  = st.text_input("Contact Name", value=init.get("Contact Name", ""))
        v_phone    = st.text_input("Phone",        value=init.get("Phone", ""))
    with c2:
        v_address  = st.text_area ("Address",  value=init.get("Address", ""),  height=80)
        v_url      = st.text_input("URL",      value=init.get("URL", ""))
        v_website  = st.text_input("Website",  value=init.get("Website", ""))
        v_keywords = st.text_input("Keywords", value=init.get("Keywords", ""))
        v_notes    = st.text_area ("Notes",    value=init.get("Notes", ""),    height=80)
    return {
        "Category": v_category,
        "Service": v_service,
        "Business Name": v_biz,
        "Contact Name": v_contact,
        "Phone": v_phone,
        "Address": v_address,
        "URL": v_url,
        "Website": v_website,
        "Keywords": v_keywords,
        "Notes": v_notes,
    }

if selected_row is None:
    st.subheader("Add a new vendor")
    with st.form("add_vendor"):
        vals = render_inputs({})
        submitted = st.form_submit_button("Add Vendor")
        if submitted:
            # Basic validation: require Business Name if it's mapped
            if "Business Name" in MAPPING and not (vals.get("Business Name") or "").strip():
                st.error('Business Name is required (because it maps to an existing DB column).')
            else:
                msg = insert_vendor(vals)
                st.success(msg)
                st.rerun()
else:
    st.subheader("Edit vendor")
    key_val = int(selected_row["key"])
    init_vals = {k: selected_row.get(k, "") for k in EXPECTED}
    with st.form("edit_vendor"):
        vals = render_inputs(init_vals)
        c1, c2 = st.columns([1,1])
        do_update = c1.form_submit_button("Update")
        do_delete = c2.form_submit_button("Delete")
        if do_update:
            if "Business Name" in MAPPING and not (vals.get("Business Name") or "").strip():
                st.error('Business Name is required (because it maps to an existing DB column).')
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
