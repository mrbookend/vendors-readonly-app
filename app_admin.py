# app_admin.py — Vendors Admin (normalized, strict category→service filtering)
# Requirements (SQLite or Turso/libSQL):
# - Table categories(id INTEGER PK, name TEXT UNIQUE NOT NULL)
# - Table services(id INTEGER PK, category_id INTEGER NOT NULL, name TEXT NOT NULL, UNIQUE(category_id,name))
# - Table vendors(
#       id INTEGER PRIMARY KEY,
#       category_id INTEGER, service_id INTEGER,
#       business_name TEXT, contact_name TEXT, phone TEXT, address TEXT,
#       url TEXT, website TEXT, notes TEXT, keywords TEXT
#   )
#   (foreign keys recommended but not required for this app to run)
#
# What this does:
# - Lists vendors with Category/Service names
# - Search across category, service, and vendor text fields
# - Add / Update / Delete vendors
# - Category dropdown drives Service dropdown (STRICT: only services for selected category)
# - "(add new…)" flow for both, created just-in-time on save
#
# If your vendors table uses different column names, add them to VENDOR_FIELDS below.

import os
import contextlib
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import streamlit as st

# -------------------------------
# Connection (libSQL -> fallback SQLite)
# -------------------------------

def _connect():
    db_url = st.secrets.get("TURSO_DATABASE_URL", os.environ.get("TURSO_DATABASE_URL", "")).strip()
    db_tok = st.secrets.get("TURSO_AUTH_TOKEN", os.environ.get("TURSO_AUTH_TOKEN", "")).strip()
    if db_url:
        try:
            from libsql_client import create_client  # type: ignore
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

def get_table_columns(table: str) -> List[str]:
    rows = exec_query(f"PRAGMA table_info('{table}')")
    return [r[1] for r in rows]  # (cid, name, type, notnull, dflt, pk)

# -------------------------------
# Schema & field setup
# -------------------------------

VENDORS_TABLE = "vendors"
CATS_TABLE    = "categories"
SRVS_TABLE    = "services"

REQUIRED_TABLES = [VENDORS_TABLE, CATS_TABLE, SRVS_TABLE]
MISSING = [t for t in REQUIRED_TABLES if not table_exists(t)]

VENDOR_FIELDS = [
    # (column_name_in_db, friendly_label_for_UI)
    ("business_name", "Business Name"),
    ("contact_name",  "Contact Name"),
    ("phone",         "Phone"),
    ("address",       "Address"),
    ("url",           "URL"),
    ("website",       "Website"),
    ("notes",         "Notes"),
    ("keywords",      "Keywords"),
]

# Only keep fields that actually exist
if table_exists(VENDORS_TABLE):
    existing = set(get_table_columns(VENDORS_TABLE))
    VENDOR_FIELDS = [f for f in VENDOR_FIELDS if f[0] in existing]

HAS_VENDOR_ID = ("id" in get_table_columns(VENDORS_TABLE)) if table_exists(VENDORS_TABLE) else False
HAS_CAT_ID    = ("category_id" in get_table_columns(VENDORS_TABLE)) if table_exists(VENDORS_TABLE) else False
HAS_SRV_ID    = ("service_id"  in get_table_columns(VENDORS_TABLE)) if table_exists(VENDORS_TABLE) else False

# -------------------------------
# UI header & guards
# -------------------------------

st.set_page_config(page_title="Vendors - Admin", layout="wide")
st.title("Vendors - Admin (normalized, strict service filtering)")

with st.expander("Diagnostics (toggle)", expanded=False):
    st.write("Driver:", DRIVER)
    st.write("Missing tables:", MISSING)
    st.write("vendors has id:", HAS_VENDOR_ID, "| has category_id:", HAS_CAT_ID, "| has service_id:", HAS_SRV_ID)
    st.write("Vendor text fields (present):", [lbl for _, lbl in VENDOR_FIELDS])

if MISSING:
    st.error(f"Missing required table(s): {', '.join(MISSING)}. Run the normalization migration first.")
    st.stop()

if not (HAS_VENDOR_ID and HAS_CAT_ID and HAS_SRV_ID):
    st.error("vendors must have id, category_id, service_id columns. Add/backfill them and retry.")
    st.stop()

# -------------------------------
# Catalog helpers
# -------------------------------

def get_categories() -> List[Tuple[int, str]]:
    rows = exec_query(f'SELECT id, name FROM "{CATS_TABLE}" ORDER BY name COLLATE NOCASE')
    return [(int(r[0]), str(r[1])) for r in rows]

def get_services(category_id: Optional[int] = None) -> List[Tuple[int, str]]:
    if category_id is None:
        return []
    rows = exec_query(f'SELECT id, name FROM "{SRVS_TABLE}" WHERE category_id=? ORDER BY name COLLATE NOCASE', (int(category_id),))
    return [(int(r[0]), str(r[1])) for r in rows]

def ensure_category(name: str) -> int:
    name = (name or "").strip()
    if not name:
        raise ValueError("Category name is empty")
    row = exec_query(f'SELECT id FROM "{CATS_TABLE}" WHERE name=?', (name,))
    if row:
        return int(row[0][0])
    exec_write(f'INSERT INTO "{CATS_TABLE}"(name) VALUES(?)', (name,))
    # reselect by name (portable across drivers)
    row = exec_query(f'SELECT id FROM "{CATS_TABLE}" WHERE name=?', (name,))
    return int(row[0][0])

def ensure_service(category_id: int, name: str) -> int:
    name = (name or "").strip()
    if not name:
        raise ValueError("Service name is empty")
    row = exec_query(f'SELECT id FROM "{SRVS_TABLE}" WHERE category_id=? AND name=?', (int(category_id), name))
    if row:
        return int(row[0][0])
    exec_write(f'INSERT INTO "{SRVS_TABLE}"(category_id, name) VALUES(?, ?)', (int(category_id), name))
    row = exec_query(f'SELECT id FROM "{SRVS_TABLE}" WHERE category_id=? AND name=?', (int(category_id), name))
    return int(row[0][0])

# -------------------------------
# Search / list vendors
# -------------------------------

st.subheader("Find / pick a vendor to edit")
qtext = st.text_input("Search (business / service / phone / address / notes / keywords)", placeholder="Type to filter...").strip()

select_parts = ['v."id" AS key', 'c.name AS Category', 's.name AS Service', 'v.category_id', 'v.service_id']
for col, lbl in VENDOR_FIELDS:
    select_parts.append(f'v."{col}" AS "{lbl}"')
select_sql = ", ".join(select_parts)

sql = f'''
SELECT {select_sql}
FROM "{VENDORS_TABLE}" v
LEFT JOIN "{CATS_TABLE}" c ON c.id = v.category_id
LEFT JOIN "{SRVS_TABLE}" s ON s.id = v.service_id
WHERE 1=1
'''
params: List[Any] = []

if qtext:
    like = f"%{qtext}%"
    where_bits = ['COALESCE(c.name, \'\') LIKE ?', 'COALESCE(s.name, \'\') LIKE ?']
    params.extend([like, like])
    for col, _lbl in VENDOR_FIELDS:
        where_bits.append(f'COALESCE(v."{col}", \'\') LIKE ?')
        params.append(like)
    sql += "\n  AND (" + "\n       OR ".join(where_bits) + ")"

order_by = 'v."business_name" COLLATE NOCASE' if any(c == "business_name" for c, _ in VENDOR_FIELDS) else "c.name, s.name, v.id"
sql += f"\nORDER BY {order_by}\nLIMIT 500"

rows = exec_query(sql, params)

cols = ["key", "Category", "Service", "category_id", "service_id"] + [lbl for _, lbl in VENDOR_FIELDS]
df = pd.DataFrame(rows, columns=cols)

if df.empty:
    st.info("No vendors match your filter. Clear the search to see all.")
    st.stop()

st.dataframe(
    df.drop(columns=["key", "category_id", "service_id"], errors="ignore"),
    use_container_width=True, hide_index=True
)

# -------------------------------
# Pick vendor
# -------------------------------

def label_row(r: pd.Series) -> str:
    name = str(r.get("Business Name") or "").strip()
    svc  = str(r.get("Service") or "").strip()
    ph   = str(r.get("Phone") or "").strip()
    adr  = str(r.get("Address") or "").strip()
    bits = [b for b in [name, svc, ph, adr] if b]
    core = " — ".join(bits) if bits else "(unnamed vendor)"
    return f"{core}  [#{int(r['key'])}]"

labels = ["New vendor..."] + [label_row(r) for _, r in df.iterrows()]
choice = st.selectbox("Pick vendor", options=labels, index=0 if labels else 0)

selected: Optional[pd.Series] = None
if choice != "New vendor...":
    key = int(choice.rsplit("[#", 1)[1].rstrip("]"))
    sel = df[df["key"] == key]
    if not sel.empty:
        selected = sel.iloc[0]

# -------------------------------
# Write helpers
# -------------------------------

def build_write_fields(values: Dict[str, Any]) -> Dict[str, Any]:
    write_fields: Dict[str, Any] = {}
    existing_cols = set(get_table_columns(VENDORS_TABLE))
    for col, lbl in VENDOR_FIELDS:
        if col in existing_cols:
            write_fields[col] = values.get(lbl, "") or ""
    return write_fields

def insert_vendor(cat_id: Optional[int], srv_id: Optional[int], write_fields: Dict[str, Any]) -> str:
    cols = []
    vals = []
    params: List[Any] = []
    # Always include category/service columns (even if NULL)
    cols.append("category_id"); vals.append("?"); params.append(cat_id)
    cols.append("service_id");  vals.append("?"); params.append(srv_id)
    for col, val in write_fields.items():
        cols.append(col); vals.append("?"); params.append(val)
    sql = f'INSERT INTO "{VENDORS_TABLE}" ({", ".join(f\'"{c}"\' for c in cols)}) VALUES ({", ".join(vals)})'
    n = exec_write(sql, params)
    return f"Inserted ({n})"

def update_vendor(key_id: int, cat_id: Optional[int], srv_id: Optional[int], write_fields: Dict[str, Any]) -> str:
    sets = ['"category_id" = ?', '"service_id" = ?']
    params: List[Any] = [cat_id, srv_id]
    for col, val in write_fields.items():
        sets.append(f'"{col}" = ?'); params.append(val)
    sql = f'UPDATE "{VENDORS_TABLE}" SET {", ".join(sets)} WHERE "id"=?'
    params.append(int(key_id))
    n = exec_write(sql, params)
    return f"Updated ({n})"

def delete_vendor(key_id: int) -> str:
    n = exec_write(f'DELETE FROM "{VENDORS_TABLE}" WHERE "id"=?', (int(key_id),))
    return f"Deleted ({n})"

# -------------------------------
# Forms (STRICT: service options depend on selected category)
# -------------------------------

st.markdown("---")

def render_inputs(init: Dict[str, Any]) -> Dict[str, Any]:
    cats = get_categories()
    cat_names = [n for _, n in cats]
    cat_id_by_name = {n: i for i, n in cats}
    cat_id_by_name_norm = {n.strip().lower(): i for i, n in cats}

    current_cat_name = (init.get("Category") or "").strip()
    current_srv_name = (init.get("Service") or "").strip()

    cat_options = ["(blank)"] + cat_names + ["(add new…)"]
    cat_default = current_cat_name if current_cat_name in cat_names else ("(blank)" if not current_cat_name else "(add new…)")
    try:
        cat_index = cat_options.index(cat_default)
    except ValueError:
        cat_index = 0

    c1, c2 = st.columns(2)
    with c1:
        cat_choice = st.selectbox("Category", options=cat_options, index=cat_index, key="category_select")
        new_cat = ""
        if cat_choice == "(add new…)":
            new_cat = st.text_input("New Category", value=(current_cat_name if cat_default == "(add new…)" else ""))

        # Resolve selected category_id
        if cat_choice == "(add new…)" and new_cat.strip():
            category_id = None         # will be created on save
            category_name = new_cat.strip()
        elif cat_choice == "(blank)":
            category_id = None
            category_name = ""
        else:
            category_name = cat_choice.strip()
            category_id = (
                cat_id_by_name.get(category_name)
                or cat_id_by_name_norm.get(category_name.lower())
            )

        # STRICT: if we do not have a concrete category_id yet, do NOT load all services
        if category_id is not None:
            srvs = get_services(category_id)
        else:
            srvs = []

        srv_names = [n for _, n in srvs]
        srv_id_by_name = {n: i for i, n in srvs}

        srv_options = ["(blank)"] + srv_names + ["(add new…)"]
        srv_default = (current_srv_name if current_srv_name in srv_names
                       else ("(blank)" if not current_srv_name else "(add new…)"))
        try:
            srv_index = srv_options.index(srv_default)
        except ValueError:
            srv_index = 0

        srv_choice = st.selectbox("Service", options=srv_options, index=srv_index, key="service_select")
        new_srv = ""
        if srv_choice == "(add new…)":
            new_srv = st.text_input("New Service", value=(current_srv_name if srv_default == "(add new…)" else ""))

    with c2:
        text_vals: Dict[str, Any] = {}
        for col, lbl in VENDOR_FIELDS:
            default = str(init.get(lbl) or "")
            if lbl in ("Address", "Notes"):
                text_vals[lbl] = st.text_area(lbl, value=default, height=80)
            else:
                text_vals[lbl] = st.text_input(lbl, value=default)

    return {
        "category_choice": cat_choice,
        "category_name": category_name,
        "new_category": new_cat.strip(),
        "service_choice": srv_choice,
        "service_name": current_srv_name if srv_choice != "(add new…)" else new_srv.strip(),
        "new_service": new_srv.strip(),
        **text_vals,
    }

def resolve_fk_ids(values: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
    # Category
    if values["category_choice"] == "(add new…)":
        if not values["new_category"]:
            raise ValueError("Enter a new Category name.")
        cat_id = ensure_category(values["new_category"])
    elif values["category_choice"] == "(blank)":
        cat_id = None
    else:
        # existing by name
        name = (values["category_name"] or "").strip()
        if name:
            row = exec_query(f'SELECT id FROM "{CATS_TABLE}" WHERE name=?', (name,))
            cat_id = int(row[0][0]) if row else None
        else:
            cat_id = None

    # Service (STRICT: requires a category if adding new)
    if values["service_choice"] == "(add new…)":
        if not cat_id:
            raise ValueError("Pick or create a Category before adding a Service.")
        if not values["new_service"]:
            raise ValueError("Enter a new Service name.")
        srv_id = ensure_service(cat_id, values["new_service"])
    elif values["service_choice"] == "(blank)":
        srv_id = None
    else:
        name = (values["service_name"] or "").strip()
        if name and cat_id:
            row = exec_query(f'SELECT id FROM "{SRVS_TABLE}" WHERE category_id=? AND name=?', (int(cat_id), name))
            srv_id = int(row[0][0]) if row else None
        else:
            # If no category chosen, do not resolve service; leave NULL
            srv_id = None

    return cat_id, srv_id

# -------------------------------
# Add / Edit / Delete flows
# -------------------------------

if selected is None:
    st.subheader("Add a new vendor")
    with st.form("add_vendor"):
        vals = render_inputs({})
        submitted = st.form_submit_button("Add Vendor")
        if submitted:
            try:
                # Basic validation: Business Name encouraged if present in schema
                if any(lbl == "Business Name" for _c, lbl in VENDOR_FIELDS):
                    if not (vals.get("Business Name") or "").strip():
                        st.error("Business Name is required.")
                        st.stop()
                cat_id, srv_id = resolve_fk_ids(vals)
                write_fields = build_write_fields(vals)
                msg = insert_vendor(cat_id, srv_id, write_fields)
                st.success(msg)
                st.rerun()
            except Exception as e:
                st.error(f"{type(e).__name__}: {e}")
else:
    st.subheader("Edit vendor")
    key_id = int(selected["key"])
    init_vals = {k: selected.get(k, "") for k in ["Category", "Service"] + [lbl for _c, lbl in VENDOR_FIELDS]}
    with st.form("edit_vendor"):
        vals = render_inputs(init_vals)
        delete_confirmed = st.checkbox("Yes, permanently delete this vendor.")
        c1, c2 = st.columns([1,1])
        do_update = c1.form_submit_button("Update")
        do_delete = c2.form_submit_button("Delete")

        if do_update:
            try:
                if any(lbl == "Business Name" for _c, lbl in VENDOR_FIELDS):
                    if not (vals.get("Business Name") or "").strip():
                        st.error("Business Name is required.")
                        st.stop()
                cat_id, srv_id = resolve_fk_ids(vals)
                write_fields = build_write_fields(vals)
                msg = update_vendor(key_id, cat_id, srv_id, write_fields)
                st.success(msg)
                st.rerun()
            except Exception as e:
                st.error(f"{type(e).__name__}: {e}")

        if do_delete:
            if not delete_confirmed:
                st.warning("Check the confirmation box above, then click Delete.")
            else:
                try:
                    msg = delete_vendor(key_id)
                    st.warning(msg)
                    st.rerun()
                except Exception as e:
                    st.error(f"{type(e).__name__}: {e}")

# -------------------------------
# Optional: quick catalog tools
# -------------------------------

with st.expander("Manage Categories & Services", expanded=False):
    st.caption("Seed or correct catalog values (uses UNIQUE constraints to avoid dupes).")
    new_cat = st.text_input("Add Category", "")
    if st.button("Create Category"):
        if not new_cat.strip():
            st.error("Enter a category name.")
        else:
            cid = ensure_category(new_cat.strip())
            st.success(f"Category created id={cid}")
            st.rerun()

    cats = get_categories()
    if cats:
        idx = st.selectbox("Select Category", options=range(len(cats)), format_func=lambda i: cats[i][1])
        new_srv = st.text_input("Add Service to selected Category", "")
        if st.button("Create Service"):
            if not new_srv.strip():
                st.error("Enter a service name.")
            else:
                sid = ensure_service(cats[int(idx)][0], new_srv.strip())
                st.success(f"Service created id={sid}")
                st.rerun()
    else:
        st.info("No categories yet.")
