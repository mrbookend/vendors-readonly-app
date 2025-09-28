# app_admin.py — Vendors Admin (auto-map columns; reliable Update/Delete; category/service dropdowns)
# - Works if your DB uses different column names (snake_case, etc.)
# - Uses SQLite rowid as the key (precise updates/deletes)
# - Delete uses checkbox + form_submit_button
# - Category & Service use dropdowns sourced from distinct table values
#   * Service list is filtered by currently selected Category (if both mapped)
# - Supports Turso/libSQL or local SQLite

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
    rows = exec_query(f"PRAGMA table_info('{table}')")
    return [r[1] for r in rows]  # (cid, name, type, notnull, dflt_value, pk)

def table_has_rowid(table: str) -> bool:
    rows = exec_query("SELECT sql FROM sqlite_schema WHERE type IN ('table','view') AND name=? LIMIT 1", (table,))
    if not rows:
        return False
    sql = (rows[0][0] or "").upper()
    if " VIEW " in sql:
        # Views are not updatable unless INSTEAD OF triggers exist; treat as not rowid-updateable
        return False
    return " WITHOUT ROWID" not in sql

# -------------------------------
# Schema mapping (auto detect)
# -------------------------------

VENDORS_TABLE = "vendors"

# Friendly headers shown in UI
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

# Candidate DB names we’ll probe for each friendly field
CANDIDATES: Dict[str, List[str]] = {
    "Category":      ["Category", "category", "cat"],
    "Service":       ["Service", "service", "svc"],
    "Business Name": ["Business Name", "business_name", "name", "vendor", "vendor_name", "business"],
    "Contact Name":  ["Contact Name", "contact_name", "contact"],
    "Phone":         ["Phone", "phone", "phone_number", "tel"],
    "Address":       ["Address", "address", "street_address"],
    "URL":           ["URL", "url"],
    "Notes":         ["Notes", "notes", "note"],
    "Keywords":      ["Keywords", "keywords", "tags"],
    "Website":       ["Website", "website", "web", "site"],
}

def resolve_mapping(table: str) -> Tuple[Dict[str, str], List[str]]:
    if not table_exists(table):
        return {}, EXPECTED[:]
    actual = get_table_columns(table)
    by_lower = {c.lower(): c for c in actual}
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
HAS_ROWID = table_has_rowid(VENDORS_TABLE)

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
    st.write("Has rowid (required for precise updates/deletes):", HAS_ROWID)
    st.write("Detected mapping (friendly → actual):", MAPPING)
    if MISSING:
        st.write("Missing (shown blank; skipped on write):", MISSING)

if not table_exists(VENDORS_TABLE):
    st.error(f'Database missing required table: "{VENDORS_TABLE}".')
    st.stop()

if not HAS_ROWID:
    st.error('This table is a VIEW or WITHOUT ROWID — cannot target exact rows for update/delete.\n'
             'Convert to a normal table or add an INTEGER PRIMARY KEY id.')
    st.stop()

st.subheader("Find / pick a vendor to edit")
qtext = st.text_input(
    "Search (name / service / phone / address / notes / keywords)",
    placeholder="Type to filter..."
).strip()

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
    for friendly in ["Business Name", "Service", "Contact Name", "Category",
                     "Phone", "Address", "URL", "Website", "Notes", "Keywords"]:
        if friendly in MAPPING:
            actual = MAPPING[friendly]
            where_bits.append(f'COALESCE("{actual}", \'\') LIKE ?')
            params.append(like)
    if where_bits:
        sql += "\n  AND (" + "\n       OR ".join(where_bits) + ")"

sql += '\nORDER BY ' + (
    f'"{MAPPING["Business Name"]}" COLLATE NOCASE' if "Business Name" in MAPPING else 'rowid'
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

st.dataframe(
    df.drop(columns=["key"], errors="ignore"),
    use_container_width=True,
    hide_index=True
)

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
# Dropdown helpers (Category/Service)
# -------------------------------

def distinct_options_for(
    friendly: str,
    limit: int = 500,
    filter_by: Optional[Dict[str, str]] = None
) -> List[str]:
    """Return sorted distinct non-empty values for a mapped column, optionally filtered by another mapped column."""
    if friendly not in MAPPING:
        return []
    actual = MAPPING[friendly]

    where = ['TRIM("{col}") IS NOT NULL AND TRIM("{col}") <> \'\' '.format(col=actual)]
    params: List[Any] = []

    # Optional dependency filter (e.g., Service filtered by Category)
    if filter_by:
        for fkey, fval in filter_by.items():
            if fkey in MAPPING and (fval or "").strip():
                fac = MAPPING[fkey]
                where.append(f'TRIM("{fac}") = ?')
                params.append((fval or "").strip())

    sql = f'''
        SELECT DISTINCT TRIM("{actual}") AS v
        FROM "{VENDORS_TABLE}"
        WHERE {' AND '.join(where)}
        ORDER BY v COLLATE NOCASE ASC
        LIMIT {int(limit)}
    '''
    rows = exec_query(sql, params)
    opts = [r[0] for r in rows if isinstance(r[0], str) and r[0].strip()]
    # Ensure uniqueness and deterministic ordering
    uniq = sorted({o.strip() for o in opts})
    return uniq

def select_or_text(
    label: str,
    friendly: str,
    current: str = "",
    filter_by: Optional[Dict[str, str]] = None
) -> str:
    """
    If the column is mapped, show a dropdown with distinct options (+ blank/custom).
    If '(custom…)' is chosen, show a text_input. Otherwise return the selected option.
    If not mapped, show a text_input.
    """
    current = (current or "").strip()
    if friendly not in MAPPING:
        return st.text_input(label, value=current)

    opts = distinct_options_for(friendly, filter_by=filter_by)
    base_items = ["(blank)"] + opts + ["(custom…)"]

    # Default selection logic
    if not current:
        default_item = "(blank)"
    elif current in opts:
        default_item = current
    else:
        default_item = "(custom…)"

    try:
        default_index = base_items.index(default_item)
    except ValueError:
        default_index = 0

    choice = st.selectbox(label, options=base_items, index=default_index, key=f"{label}_select")

    if choice == "(custom…)":
        return st.text_input(f"{label} (custom)", value=(current if default_item == "(custom…)" else ""))
    elif choice == "(blank)":
        return ""
    else:
        return choice

# -------------------------------
# CRUD helpers (write only mapped cols)
# -------------------------------

def values_to_db_params(values: Dict[str, Any]) -> Tuple[List[str], List[Any]]:
    cols_db: List[str] = []
    vals_db: List[Any] = []
    for friendly in EXPECTED:
        if friendly in MAPPING:  # only write to columns that actually exist
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
# Forms (dropdowns + dependent service)
# -------------------------------

st.markdown("---")

def render_inputs(init: Dict[str, Any]) -> Dict[str, Any]:
    # Category first (so Service can depend on it)
    c1, c2 = st.columns(2)
    with c1:
        v_category = select_or_text("Category", "Category", init.get("Category", ""))
        # Service options filtered by chosen Category (when both columns exist)
        svc_filter = {"Category": v_category} if ("Category" in MAPPING) else None
        v_service  = select_or_text("Service",  "Service",  init.get("Service", ""), filter_by=svc_filter)
        v_biz      = st.text_input("Business Name", value=init.get("Business Name", ""))
        v_contact  = st.text_input("Contact Name",  value=init.get("Contact Name", ""))
        v_phone    = st.text_input("Phone",         value=init.get("Phone", ""))
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
            # Require Business Name if it maps to a real column
            if "Business Name" in MAPPING and not (vals.get("Business Name") or "").strip():
                st.error('Business Name is required (it maps to a DB column).')
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
        delete_confirmed = st.checkbox("Yes, permanently delete this vendor.")
        c1, c2 = st.columns([1,1])
        do_update = c1.form_submit_button("Update")
        do_delete = c2.form_submit_button("Delete")
        if do_update:
            if "Business Name" in MAPPING and not (vals.get("Business Name") or "").strip():
                st.error('Business Name is required (it maps to a DB column).')
            else:
                msg = update_vendor(key_val, vals)
                st.success(msg)
                st.rerun()
        if do_delete:
            if not delete_confirmed:
                st.warning("Check the confirmation box above, then click Delete.")
            else:
                msg = delete_vendor(key_val)
                st.warning(msg)
                st.rerun()

# -------------------------------
# End
# -------------------------------
