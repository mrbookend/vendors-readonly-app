# app_admin.py — Vendors Admin (DB-derived Service options; Python normalization; safe SQL)
# - Single vendors table
# - Distinct (Category, Service) pairs loaded with simple SQL
# - Category→Service map built in Python with robust normalization
# - Writes to snake_case (category/service) when present
# - Safe SQL debug everywhere
# - One-click coalesce legacy TitleCase → snake_case

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple
from collections import defaultdict, Counter

import pandas as pd
import streamlit as st

# -------------------------------
# DB connect (libSQL → fallback SQLite)
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
    path = os.environ.get("SQLITE_PATH", "vendors.db")
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
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

def safe_query(sql: str, params: Sequence[Any] = ()) -> Tuple[List[Tuple], Optional[str]]:
    try:
        return exec_query(sql, params), None
    except Exception as e:
        st.error("SQL failed")
        st.code(sql)
        st.write("params:", list(params))
        st.write("driver:", DRIVER)
        st.exception(e)
        return [], f"{type(e).__name__}: {e}"

def safe_write(sql: str, params: Sequence[Any] = ()) -> Tuple[int, Optional[str]]:
    try:
        return exec_write(sql, params), None
    except Exception as e:
        st.error("SQL write failed")
        st.code(sql)
        st.write("params:", list(params))
        st.write("driver:", DRIVER)
        st.exception(e)
        return 0, f"{type(e).__name__}: {e}"

# -------------------------------
# Schema helpers
# -------------------------------

VENDORS_TABLE = "vendors"

def table_exists(name: str) -> bool:
    rows = exec_query("SELECT 1 FROM sqlite_schema WHERE type='table' AND name=? LIMIT 1", (name,))
    return bool(rows)

def get_table_columns(table: str) -> List[str]:
    rows = exec_query(f"PRAGMA table_info('{table}')")
    return [r[1] for r in rows]

def has_without_rowid(table: str) -> bool:
    rows = exec_query("SELECT sql FROM sqlite_schema WHERE type='table' AND name=? LIMIT 1", (table,))
    if not rows:
        return False
    return "WITHOUT ROWID" in (rows[0][0] or "").upper()

def detect_key_col(table: str) -> Optional[str]:
    cols = get_table_columns(table)
    if "id" in cols:
        return "id"
    info = exec_query(f"PRAGMA table_info('{table}')")
    pk_cols = [r[1] for r in info if int(r[5] or 0) > 0]
    if len(pk_cols) == 1:
        return pk_cols[0]
    return None

# -------------------------------
# Mapping candidates & resolver (non category/service fields)
# -------------------------------

EXPECTED = [
    "Category", "Service", "Business Name", "Contact Name", "Phone",
    "Address", "URL", "Notes", "Keywords", "Website",
]

CANDIDATES: Dict[str, List[str]] = {
    "Category":      ["category", "Category", "cat"],
    "Service":       ["service", "Service", "svc"],
    "Business Name": ["business_name", "Business Name", "name", "vendor_name", "vendor", "business"],
    "Contact Name":  ["contact_name", "Contact Name", "contact"],
    "Phone":         ["phone", "Phone", "phone_number", "tel"],
    "Address":       ["address", "Address", "street_address"],
    "URL":           ["url", "URL"],
    "Notes":         ["notes", "Notes", "note"],
    "Keywords":      ["keywords", "Keywords", "tags"],
    "Website":       ["website", "Website", "web", "site"],
}

def resolve_mapping(table: str) -> Dict[str, str]:
    if not table_exists(table):
        return {}
    actual = set(get_table_columns(table))
    mapping: Dict[str, str] = {}
    for friendly, options in CANDIDATES.items():
        present = [opt for opt in options if opt in actual]
        if not present:
            continue
        best_col = present[0]; best_cnt = -1
        for col in present:
            try:
                cnt = exec_query(
                    f'SELECT COUNT(*) FROM "{table}" WHERE "{col}" IS NOT NULL AND TRIM("{col}") <> \'\''
                )[0][0]
            except Exception:
                cnt = -1
            if cnt is not None and int(cnt) > best_cnt:
                best_cnt = int(cnt); best_col = col
        mapping[friendly] = best_col
    return mapping

MAPPING = resolve_mapping(VENDORS_TABLE)
KEY_COL = detect_key_col(VENDORS_TABLE)
WITHOUT_ROWID = has_without_rowid(VENDORS_TABLE)
COLS_SET = set(get_table_columns(VENDORS_TABLE)) if table_exists(VENDORS_TABLE) else set()

# Prefer snake_case for writes if present
for friendly, canonical in [("Category", "category"), ("Service", "service")]:
    if canonical in COLS_SET:
        MAPPING[friendly] = canonical

# -------------------------------
# Simple expressions for reading Category/Service
# -------------------------------

def cat_expr() -> str:
    if "category" in COLS_SET and "Category" in COLS_SET:
        return 'COALESCE(NULLIF(TRIM("category"), \'\'), NULLIF(TRIM("Category"), \'\'))'
    if "category" in COLS_SET:
        return 'NULLIF(TRIM("category"), \'\')'
    if "Category" in COLS_SET:
        return 'NULLIF(TRIM("Category"), \'\')'
    return "''"

def srv_expr() -> str:
    if "service" in COLS_SET and "Service" in COLS_SET:
        return 'COALESCE(NULLIF(TRIM("service"), \'\'), NULLIF(TRIM("Service"), \'\'))'
    if "service" in COLS_SET:
        return 'NULLIF(TRIM("service"), \'\')'
    if "Service" in COLS_SET:
        return 'NULLIF(TRIM("Service"), \'\')'
    return "''"

CAT_SQL = cat_expr()
SRV_SQL = srv_expr()

# -------------------------------
# Normalization in Python (robust)
# -------------------------------

def norm_strict_py(s: Any) -> str:
    if s is None:
        return ""
    s = str(s)
    s = s.replace("\u00A0", " ").replace("\t", " ").replace("\r", "").replace("\n", "")
    return s.strip().upper()

def norm_loose_py(s: Any) -> str:
    return norm_strict_py(s).replace(" ", "")

# -------------------------------
# Streamlit UI Header
# -------------------------------

st.set_page_config(page_title="Vendors - Admin", layout="wide")
st.title("Vendors - Admin (DB-derived services; Python-normalized)")

with st.expander("Diagnostics / Tools (toggle)", expanded=False):
    st.write("Driver:", DRIVER)
    st.write("Using table:", VENDORS_TABLE)
    st.write("Mapping (friendly → actual):", MAPPING)
    st.write("Detected key column:", KEY_COL or "(none; using rowid if possible)")
    st.write("Table WITHOUT ROWID:", WITHOUT_ROWID)

    # One-click coalesce legacy → snake_case when snake_case blank
    def coalesce_legacy_to_snake() -> Tuple[int, int, Optional[str]]:
        sql1 = f'''
            UPDATE "{VENDORS_TABLE}"
            SET category = TRIM(COALESCE(NULLIF(category,''), NULLIF("Category",'')))
            WHERE COALESCE(NULLIF("Category",''),'') <> ''
              AND (category IS NULL OR TRIM(category) = '');
        '''
        sql2 = f'''
            UPDATE "{VENDORS_TABLE}"
            SET service = TRIM(COALESCE(NULLIF(service,''), NULLIF("Service",'')))
            WHERE COALESCE(NULLIF("Service",''),'') <> ''
              AND (service IS NULL OR TRIM(service) = '');
        '''
        n1, e1 = safe_write(sql1)
        n2, e2 = safe_write(sql2)
        if not e1 and not e2:
            safe_write(f'UPDATE "{VENDORS_TABLE}" SET category = TRIM(category) WHERE category IS NOT NULL;')
            safe_write(f'UPDATE "{VENDORS_TABLE}" SET service  = TRIM(service)  WHERE service  IS NOT NULL;')
        return n1, n2, e1 or e2

    if st.button("Coalesce legacy Category/Service → snake_case"):
        c_cat, c_srv, err = coalesce_legacy_to_snake()
        if err:
            st.error("Coalesce encountered an error (details above).")
        else:
            st.success(f"Coalesced rows — category: {c_cat}, service: {c_srv}.")
            st.rerun()

if not table_exists(VENDORS_TABLE):
    st.error(f'Table "{VENDORS_TABLE}" not found.')
    st.stop()

if not KEY_COL and WITHOUT_ROWID:
    st.error("vendors is WITHOUT ROWID and has no single-column PRIMARY KEY. Add 'id INTEGER PRIMARY KEY' or a single PK.")
    st.stop()

# -------------------------------
# Load grid rows (LIMIT prevents giant tables; independent from options map)
# -------------------------------

select_parts: List[str] = []
if KEY_COL:
    select_parts.append(f'v."{KEY_COL}" AS key')
else:
    select_parts.append("rowid AS key")

select_parts.append(f"{CAT_SQL} AS \"Category\"")
select_parts.append(f"{SRV_SQL} AS \"Service\"")

for friendly in ["Business Name","Contact Name","Phone","Address","URL","Notes","Keywords","Website"]:
    if friendly in MAPPING:
        actual = MAPPING[friendly]
        select_parts.append(f'v."{actual}" AS "{friendly}"')
    else:
        select_parts.append(f"'' AS \"{friendly}\"")

select_sql = ", ".join(select_parts)

st.subheader("Find / pick a vendor to edit")
qtext = st.text_input(
    "Search (business / service / phone / address / notes / keywords)",
    placeholder="Type to filter..."
).strip()

sql = f' SELECT {select_sql} FROM "{VENDORS_TABLE}" v WHERE 1=1 '
params: List[Any] = []
if qtext:
    like = f"%{qtext}%"
    where_bits: List[str] = []
    where_bits.append(f"COALESCE({CAT_SQL}, '') LIKE ?"); params.append(like)
    where_bits.append(f"COALESCE({SRV_SQL}, '') LIKE ?"); params.append(like)
    for friendly in ["Business Name","Contact Name","Phone","Address","URL","Website","Notes","Keywords"]:
        if friendly in MAPPING:
            where_bits.append(f'COALESCE(v."{MAPPING[friendly]}", \'\') LIKE ?')
            params.append(like)
    sql += "\n  AND (" + "\n       OR ".join(where_bits) + ")"

order_by = f'v."{MAPPING["Business Name"]}" COLLATE NOCASE' if "Business Name" in MAPPING else "key"
sql += f"\nORDER BY {order_by} ASC\nLIMIT 500"

rows, err = safe_query(sql, params)
cols_out = ["key"] + EXPECTED
df = pd.DataFrame(rows, columns=cols_out) if rows else pd.DataFrame(columns=cols_out)

if df.empty and err:
    st.stop()
if df.empty:
    st.info("No vendors match your filter. Clear the search to see all.")
    st.stop()

st.dataframe(df.drop(columns=["key"], errors="ignore"),
             use_container_width=True, hide_index=True)

# -------------------------------
# Build Category→Service options DIRECTLY from DB (not from grid / not filtered)
# -------------------------------

def load_category_service_pairs() -> Tuple[Dict[str, List[str]], List[str]]:
    """
    Returns:
      - services_by_key: dict of normalized category key -> sorted service list
      - category_display_options: canonical display list for categories (sorted)
    """
    sql_pairs = f'''
        SELECT DISTINCT c, s FROM (
            SELECT {CAT_SQL} AS c, {SRV_SQL} AS s
            FROM "{VENDORS_TABLE}"
        )
        WHERE c IS NOT NULL AND TRIM(c) <> ''
          AND s IS NOT NULL AND TRIM(s) <> '';
    '''
    rows, _ = safe_query(sql_pairs)
    # Build normalization + display preference
    disp_counts: Dict[str, Counter] = defaultdict(Counter)  # key -> Counter(display->count)
    svc_map: Dict[str, set] = defaultdict(set)             # key -> set(services)
    for c, s in rows:
        cd = (str(c) if c is not None else "").strip()
        sd = (str(s) if s is not None else "").strip()
        if not cd or not sd:
            continue
        key = norm_loose_py(cd)
        disp_counts[key][cd] += 1
        svc_map[key].add(sd)

    # pick canonical category display per key: highest count, then shortest, then alpha
    def pick_display(cnt: Counter) -> str:
        if not cnt:
            return ""
        maxn = max(cnt.values())
        cands = [name for name, n in cnt.items() if n == maxn]
        cands.sort(key=lambda x: (len(x), x.upper()))
        return cands[0]

    services_by_key: Dict[str, List[str]] = {
        k: sorted(list(v), key=lambda s: s.upper())
        for k, v in svc_map.items()
        if k  # non-empty key
    }
    category_display = [pick_display(cnt) for k, cnt in disp_counts.items() if k]
    category_display = sorted([c for c in category_display if c], key=lambda s: s.upper())
    return services_by_key, category_display

SERVICES_BY_KEY, CATEGORY_OPTIONS = load_category_service_pairs()

with st.expander("Category/Service diagnostics", expanded=False):
    st.write(f"Categories found: {len(CATEGORY_OPTIONS)}")
    sample = ", ".join(CATEGORY_OPTIONS[:12]) + (" …" if len(CATEGORY_OPTIONS) > 12 else "")
    st.caption(f"Sample categories: {sample}")
    # count total pairs
    total_pairs = sum(len(v) for v in SERVICES_BY_KEY.values())
    st.write(f"Distinct (Category, Service) pairs: {total_pairs}")

# -------------------------------
# UI helpers (category/service from DB-derived map)
# -------------------------------

def select_or_text_category(current: str = "") -> str:
    base = ["(blank)"] + CATEGORY_OPTIONS + ["(custom…)"]
    cur = (current or "").strip()
    default = "(blank)" if not cur else (cur if cur in CATEGORY_OPTIONS else "(custom…)")
    try:
        idx = base.index(default)
    except ValueError:
        idx = 0
    choice = st.selectbox("Category", options=base, index=idx, key="Category_select")
    if choice == "(custom…)":
        return st.text_input("Category (custom)", value=(cur if default == "(custom…)" else ""))
    elif choice == "(blank)":
        return ""
    else:
        return choice

def services_for_category(category_value: str) -> List[str]:
    key = norm_loose_py(category_value)
    return SERVICES_BY_KEY.get(key, [])

def select_or_text_service(current: str, category_value: str) -> str:
    cur = (current or "").strip()
    opts = []
    if category_value and category_value not in ("(blank)", "(custom…)"):
        opts = services_for_category(category_value)
    base = ["(blank)"] + opts + ["(custom…)"]
    default = "(blank)" if not cur else (cur if cur in opts else "(custom…)")
    try:
        idx = base.index(default)
    except ValueError:
        idx = 0
    choice = st.selectbox("Service", options=base, index=idx, key="Service_select")
    if choice == "(custom…)":
        return st.text_input("Service (custom)", value=(cur if default == "(custom…)" else ""))
    elif choice == "(blank)":
        return ""
    else:
        return choice

# -------------------------------
# Selection label (grid)
# -------------------------------

def make_label(row: pd.Series) -> str:
    name = (row.get("Business Name") or "").strip()
    svc  = (row.get("Service") or "").strip()
    ph   = (row.get("Phone") or "").strip()
    adr  = (row.get("Address") or "").strip()
    bits = [b for b in [name, svc, ph, adr] if b]
    core = " — ".join(bits) if bits else "(unnamed vendor)"
    return f"{core}  [#{row['key']}]"

labels = ["New vendor..."] + [make_label(r) for _, r in df.iterrows()]
choice = st.selectbox("Pick vendor", options=labels, index=0 if labels else 0)

selected_row: Optional[pd.Series] = None
if choice != "New vendor...":
    try:
        key_val = int(choice.rsplit("[#", 1)[1].rstrip("]"))
    except Exception:
        key_val = None
    if key_val is not None:
        m = df[df["key"] == key_val]
        if not m.empty:
            selected_row = m.iloc[0]

# -------------------------------
# CRUD (writes to snake_case when available)
# -------------------------------

def values_to_db_params(values: Dict[str, Any]) -> Tuple[List[str], List[Any]]:
    cols_db: List[str] = []; vals_db: List[Any] = []
    cat_col = "category" if "category" in COLS_SET else MAPPING.get("Category")
    srv_col = "service"  if "service"  in COLS_SET else MAPPING.get("Service")
    if cat_col:
        cols_db.append(f'"{cat_col}"'); vals_db.append(values.get("Category", "") or "")
    if srv_col:
        cols_db.append(f'"{srv_col}"'); vals_db.append(values.get("Service", "") or "")
    for friendly in ["Business Name","Contact Name","Phone","Address","URL","Notes","Keywords","Website"]:
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
    n, e = safe_write(sql, vals_db)
    return f"Inserted ({n})" if not e else "Insert error (see above)"

def update_vendor(key: Any, values: Dict[str, Any]) -> str:
    cols_db, vals_db = values_to_db_params(values)
    if not cols_db:
        return "Nothing to update: no mapped columns."
    sets = ", ".join([f"{c}=?" for c in cols_db])
    if KEY_COL:
        sql = f'UPDATE "{VENDORS_TABLE}" SET {sets} WHERE "{KEY_COL}"=?'
    else:
        sql = f'UPDATE "{VENDORS_TABLE}" SET {sets} WHERE rowid=?'
    n, e = safe_write(sql, vals_db + [key])
    return f"Updated ({n})" if not e else "Update error (see above)"

def delete_vendor(key: Any) -> str:
    if KEY_COL:
        sql = f'DELETE FROM "{VENDORS_TABLE}" WHERE "{KEY_COL}"=?'
    else:
        sql = f'DELETE FROM "{VENDORS_TABLE}" WHERE rowid=?'
    n, e = safe_write(sql, (key,))
    return f"Deleted ({n})" if not e else "Delete error (see above)"

# -------------------------------
# Forms (Category→Service from DB-derived map)
# -------------------------------

st.markdown("---")

def render_inputs(init: Dict[str, Any]) -> Dict[str, Any]:
    c1, c2 = st.columns(2)
    with c1:
        v_category = select_or_text_category(init.get("Category", ""))
        v_service  = select_or_text_service(init.get("Service", ""), v_category)
        v_biz      = st.text_input("Business Name", value=init.get("Business Name", ""))
        v_contact  = st.text_input("Contact Name",  value=init.get("Contact Name", ""))
        v_phone    = st.text_input("Phone",         value=init.get("Phone", ""))
    with c2:
        v_address  = st.text_area ("Address",  value=init.get("Address", ""),  height=80)
        v_url      = st.text_input("URL",      value=init.get("URL", ""))
        v_website  = st.text_input("Website",  value=init.get("Website", ""))
        v_keywords = st.text_input("Keywords", value=init.get("Keywords", ""))
        v_notes    = st.text_area ("Notes",    value=init.get("Notes", ""),    height=80)

    # Inline diagnostics for your selection
    if v_category:
        found = services_for_category(v_category)
        st.caption(f"Services for '{v_category}': {len(found)} — {', '.join(found[:12])}{' …' if len(found) > 12 else ''}")

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
            if "Business Name" in MAPPING and not (vals.get("Business Name") or "").strip():
                st.error("Business Name is required.")
            else:
                msg = insert_vendor(vals)
                st.success(msg)
                st.rerun()
else:
    st.subheader("Edit vendor")
    key_val = selected_row["key"]
    init_vals = {k: selected_row.get(k, "") for k in EXPECTED}
    with st.form("edit_vendor"):
        vals = render_inputs(init_vals)
        delete_confirmed = st.checkbox("Yes, permanently delete this vendor.")
        c1, c2 = st.columns([1,1])
        do_update = c1.form_submit_button("Update")
        do_delete = c2.form_submit_button("Delete")
        if do_update:
            if "Business Name" in MAPPING and not (vals.get("Business Name") or "").strip():
                st.error("Business Name is required.")
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
