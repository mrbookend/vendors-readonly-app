import os
import contextlib
import pandas as pd
import streamlit as st
import libsql

st.set_page_config(page_title="Vendors - Admin", layout="wide")

# ---------- Password gate ----------
def _pw_ok() -> bool:
    expected = st.secrets.get("ADMIN_PASSWORD") or os.getenv("ADMIN_PASSWORD")
    entered = st.session_state.get("admin_pw", "")
    return bool(expected) and entered == expected

with st.sidebar:
    st.subheader("Admin Login")
    st.session_state["admin_pw"] = st.text_input("Password", type="password", placeholder="Enter admin password")
    if not _pw_ok():
        st.info("Enter the admin password to continue.")
        st.stop()

# ---------- DB connection (embedded replica; syncs to Turso) ----------
@st.cache_resource
def _conn():
    url = st.secrets["LIBSQL_URL"]
    tok = st.secrets["LIBSQL_AUTH_TOKEN"]
    c = libsql.connect(
        "vendors_local.db",
        sync_url=url,
        auth_token=tok,
        sync_interval=60,
    )
    c.execute("PRAGMA foreign_keys = ON;")
    return c

conn = _conn()

def _sync():
    with contextlib.suppress(Exception):
        conn.sync()

# ---------- Schema bootstrap (safe if tables already exist) ----------
def init_schema_if_needed():
    conn.execute("""
    CREATE TABLE IF NOT EXISTS categories (
      id INTEGER PRIMARY KEY,
      name TEXT NOT NULL UNIQUE COLLATE NOCASE
    );
    """)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS services (
      id INTEGER PRIMARY KEY,
      name TEXT NOT NULL UNIQUE COLLATE NOCASE
    );
    """)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS vendors (
      id INTEGER PRIMARY KEY,
      business_name TEXT NOT NULL,
      contact_name TEXT,
      phone TEXT,
      address TEXT,
      notes TEXT,
      website TEXT,
      category_id INTEGER,
      is_active INTEGER NOT NULL DEFAULT 1,
      created_at TEXT NOT NULL DEFAULT (datetime('now')),
      updated_at TEXT NOT NULL DEFAULT (datetime('now')),
      FOREIGN KEY (category_id) REFERENCES categories(id) ON DELETE SET NULL
    );
    """)
    conn.execute("""
    CREATE TRIGGER IF NOT EXISTS vendors_updated_at
    AFTER UPDATE ON vendors
    BEGIN
      UPDATE vendors SET updated_at = datetime('now') WHERE id = NEW.id;
    END;
    """)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS vendor_services (
      vendor_id INTEGER NOT NULL,
      service_id INTEGER NOT NULL,
      PRIMARY KEY (vendor_id, service_id),
      FOREIGN KEY (vendor_id) REFERENCES vendors(id) ON DELETE CASCADE,
      FOREIGN KEY (service_id) REFERENCES services(id) ON DELETE RESTRICT
    );
    """)
    conn.commit()

init_schema_if_needed()

# ---------- DEBUG (temporary; remove after green) ----------
with st.expander("DEBUG (remove after)", expanded=True):
    st.write("Secrets present:",
             bool(st.secrets.get("LIBSQL_URL")),
             bool(st.secrets.get("LIBSQL_AUTH_TOKEN")))
    try:
        conn.execute("SELECT 1")
        conn.execute("CREATE TABLE IF NOT EXISTS _write_test (x INT)")
        conn.execute("DROP TABLE _write_test")
        st.write("Write test: OK (token can write)")
    except Exception as e:
        st.write("Write test FAILED (likely read-only token):")
        st.exception(e)
    try:
        names = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY 1")]
        st.write("Tables:", names)
    except Exception as e:
        st.write("Listing tables failed:")
        st.exception(e)

# ---------- Query helpers (hardened) ----------
def q(sql: str, params=None):
    try:
        if params:
            p = tuple(params) if isinstance(params, (list, tuple)) else params
            cur = conn.execute(sql, p)
        else:
            cur = conn.execute(sql)
        return cur.fetchall()
    except Exception as e:
        st.error("SQL failed")
        st.code(sql)
        st.write("params:", params)
        st.exception(e)
        raise

def one(sql: str, params=None):
    try:
        if params:
            p = tuple(params) if isinstance(params, (list, tuple)) else params
            cur = conn.execute(sql, p)
        else:
            cur = conn.execute(sql)
        return cur.fetchone()
    except Exception as e:
        st.error("SQL failed (one)")
        st.code(sql)
        st.write("params:", params)
        st.exception(e)
        raise

# ---------- UI ----------
st.title("Vendors - Admin")

with st.expander("Find / pick a vendor to edit", expanded=True):
    colf, colc = st.columns([2, 1])
    with colf:
        search = st.text_input("Search name/address/phone", placeholder="Type to filter...")
    with colc:
        show_inactive = st.checkbox("Show inactive (soft-deleted)", value=False)

    clauses, params = [], []
    if search:
        like = f"%{search}%"
        clauses.append("(v.business_name LIKE ? OR v.contact_name LIKE ? OR v.phone LIKE ? OR v.address LIKE ?)")
        params += [like, like, like, like]
    if not show_inactive:
        clauses.append("v.is_active = 1")
    where_clause = (" WHERE " + " AND ".join(clauses)) if clauses else ""

    rows = q(
        f"""
        SELECT v.id, v.business_name, COALESCE(c.name,''), v.phone, v.address, v.website, v.is_active
        FROM vendors v
        LEFT JOIN categories c ON c.id = v.category_id
        {where_clause}
        ORDER BY v.business_name COLLATE NOCASE ASC
        LIMIT 500
        """,
        params,
    )

    df = pd.DataFrame(rows, columns=["id", "Business", "Category", "Phone", "Address", "Website", "Active"])
    # Stop early if no rows after filtering
if df is None or df.empty:
    st.info("No vendors match your filter. Clear the search to see all.")
    st.stop()

# Hide key columns only in the display copy; keep df intact for edits/deletes
_cols_to_hide = [c for c in ("id", "vendor_id") if c in df.columns]
display_df = df.drop(columns=_cols_to_hide, errors="ignore") if _cols_to_hide else df

st.dataframe(display_df, use_container_width=True, hide_index=True)

options = ["New vendor..."] + [f"{r[1]} - {r[2]}" for r in rows]
pick_ix = st.selectbox("Select to edit", range(len(options)), index=0, format_func=lambda i: options[i])
vid = None if pick_ix == 0 else rows[pick_ix - 1][0]

current = (
    one(
        """SELECT id, business_name, contact_name, phone, address, notes, website, category_id, is_active
           FROM vendors WHERE id = ?""",
        (vid,),
    )
    if vid
    else None
)
cur_services = set([r[0] for r in q("SELECT service_id FROM vendor_services WHERE vendor_id = ?", (vid,))]) if vid else set()

st.subheader("Edit / Create Vendor")
with st.form("vendor_form", clear_on_submit=False):
    business_name = st.text_input("Business Name *", value=(current[1] if current else ""))
    contact_name = st.text_input("Contact Name", value=(current[2] if current else ""))
    phone = st.text_input("Phone", value=(current[3] if current else ""))
    address = st.text_input("Address", value=(current[4] if current else ""))
    website = st.text_input("Website (URL)", value=(current[6] if current else ""))
    notes = st.text_area("Notes", value=(current[5] if current else ""), height=100)

    # Category selection (existing or new)
    cats = q("SELECT id, name FROM categories ORDER BY name")
    cat_ids = [None] + [c[0] for c in cats]
    cat_names = ["(none)"] + [c[1] for c in cats]
    default_cat_index = 0
    if current and current[7] in cat_ids:
        default_cat_index = cat_ids.index(current[7])
    cat_index = st.selectbox("Category", range(len(cat_ids)), index=default_cat_index, format_func=lambda i: cat_names[i])
    new_cat_name = st.text_input("New Category (optional)")

    # Services multiselect (existing + add new via CSV)
    svcs = q("SELECT id, name FROM services ORDER BY name")
    svc_ids = [s[0] for s in svcs]
    svc_names = [s[1] for s in svcs]
    default_svc_indices = [svc_ids.index(sid) for sid in cur_services if sid in svc_ids]
    svc_sel_ix = st.multiselect("Services", range(len(svc_ids)), default=default_svc_indices, format_func=lambda i: svc_names[i])
    new_svcs_csv = st.text_input("New Services (comma-separated, optional)")

    is_active = st.checkbox("Active", value=(bool(current[8]) if current else True))

    cL, cM, cR = st.columns([1, 1, 1])
    with cL:
        save = st.form_submit_button("Save / Update", type="primary")
    with cM:
        delete_soft = st.form_submit_button("Soft Delete")
    with cR:
        hard_confirm = st.text_input("Type DELETE to hard-delete", value="")
        delete_hard = st.form_submit_button("Hard Delete", disabled=(hard_confirm.strip() != "DELETE"))

    if save:
        if not business_name.strip():
            st.error("Business Name is required.")
            st.stop()

        category_id = cat_ids[cat_index]
        if new_cat_name.strip():
            conn.execute("INSERT OR IGNORE INTO categories(name) VALUES (?)", (new_cat_name.strip(),))
            r = one("SELECT id FROM categories WHERE name = ? COLLATE NOCASE", (new_cat_name.strip(),))
            if r:
                category_id = r[0]

        selected_ids = [svc_ids[i] for i in svc_sel_ix] if svc_ids else []
        if new_svcs_csv.strip():
            for raw in [n.strip() for n in new_svcs_csv.split(",") if n.strip()]:
                conn.execute("INSERT OR IGNORE INTO services(name) VALUES (?)", (raw,))
                r = one("SELECT id FROM services WHERE name = ? COLLATE NOCASE", (raw,))
                if r:
                    selected_ids.append(r[0])
        selected_ids = sorted(set(selected_ids))

        with conn:
            if vid:
                conn.execute(
                    """UPDATE vendors
                       SET business_name=?, contact_name=?, phone=?, address=?, notes=?, website=?, category_id=?, is_active=?
                       WHERE id=?""",
                    (business_name.strip(), contact_name.strip(), phone.strip(), address.strip(), notes.strip(),
                     website.strip(), category_id, 1 if is_active else 0, vid),
                )
                conn.execute("DELETE FROM vendor_services WHERE vendor_id = ?", (vid,))
                for sid in selected_ids:
                    conn.execute("INSERT OR IGNORE INTO vendor_services(vendor_id, service_id) VALUES (?,?)", (vid, sid))
                new_vid = vid
            else:
                cur = conn.execute(
                    """INSERT INTO vendors(business_name, contact_name, phone, address, notes, website, category_id, is_active)
                       VALUES (?,?,?,?,?,?,?,?)""",
                    (business_name.strip(), contact_name.strip(), phone.strip(), address.strip(), notes.strip(),
                     website.strip(), category_id, 1 if is_active else 0),
                )
                new_vid = cur.lastrowid
                for sid in selected_ids:
                    conn.execute("INSERT OR IGNORE INTO vendor_services(vendor_id, service_id) VALUES (?,?)", (new_vid, sid))
        _sync()
        st.success(f"Saved vendor ID {new_vid}.")
        st.rerun()

    if delete_soft and vid:
        conn.execute("UPDATE vendors SET is_active=0 WHERE id=?", (vid,))
        _sync()
        st.warning(f"Vendor {vid} soft-deleted (inactive).")
        st.rerun()

    if delete_hard and vid and hard_confirm.strip() == "DELETE":
        with conn:
            conn.execute("DELETE FROM vendors WHERE id=?", (vid,))
        _sync()
        st.error(f"Vendor {vid} permanently deleted.")
        st.rerun()

st.caption("Writes go to your Turso database; the public read-only app should reflect updates immediately.")
