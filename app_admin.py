git add app_admin.py requirements.txt
git commit -m "Replace admin stub with full CRUD"
git push
import os
import streamlit as st
import libsql

st.set_page_config(page_title="Vendors Admin", layout="wide")

# --- Simple password gate ---
def ok():
    pw = st.session_state.get("admin_pw","")
    expected = st.secrets.get("ADMIN_PASSWORD") or os.getenv("ADMIN_PASSWORD")
    return bool(expected) and pw == expected

st.sidebar.subheader("Admin Login")
st.session_state["admin_pw"] = st.sidebar.text_input("Password", type="password")
if not ok():
    st.stop()

# --- Connect to Turso/libSQL ---
conn = libsql.connect(
    "vendors_local.db",
    sync_url=st.secrets["LIBSQL_URL"],
    auth_token=st.secrets["LIBSQL_AUTH_TOKEN"],
    sync_interval=60,
)
conn.execute("PRAGMA foreign_keys = ON;")

st.title("Vendors — Admin (Smoke Test)")
# Basic health checks
c_vendors = conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='vendors'").fetchone()[0]
st.write("Table 'vendors' present:", bool(c_vendors))
if c_vendors:
    count = conn.execute("SELECT COUNT(*) FROM vendors").fetchone()[0]
    st.write("Vendors rows:", count)
st.success("Connected and authenticated.")
# app_admin.py — Vendors Admin (Full CRUD)
# Requires: streamlit>=1.36, libsql>=0.3, pandas>=2.2
import os
import contextlib
import pandas as pd
import streamlit as st
import libsql

st.set_page_config(page_title="Vendors Admin", layout="wide")

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
        "vendors_local.db",           # local replica file; safe to keep
        sync_url=url,
        auth_token=tok,
        sync_interval=60,             # seconds between background syncs
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

# ---------- Query helpers ----------
def q(sql: str, params=()):
    cur = conn.execute(sql, params)
    return cur.fetchall()

def one(sql: str, params=()):
    cur = conn.execute(sql, params)
    return cur.fetchone()

def upsert_category(name: str):
    name = (name or "").strip()
    if not name:
        return None
    with conn:
        conn.execute("INSERT OR IGNORE INTO categories(name) VALUES (?)", (name,))
    row = one("SELECT id FROM categories WHERE name = ? COLLATE NOCASE", (name,))
    return row[0] if row else None

def upsert_services(csv_names: str):
    if not csv_names:
        return []
    ids = []
    for raw in [n.strip() for n in csv_names.split(",") if n.strip()]:
        with conn:
            conn.execute("INSERT OR IGNORE INTO services(name) VALUES (?)", (raw,))
        r = one("SELECT id FROM services WHERE name = ? COLLATE NOCASE", (raw,))
        if r:
            ids.append(r[0])
    return ids

def vendor_services_ids(vendor_id: int):
    rows = q("SELECT service_id FROM vendor_services WHERE vendor_id = ?", (vendor_id,))
    return [r[0] for r in rows]

def save_vendor(vid, business_name, contact_name, phone, address, notes, website, category_id, service_ids, is_active=True):
    with conn:
        if vid:
            conn.execute(
                """UPDATE vendors
                   SET business_name=?, contact_name=?, phone=?, address=?, notes=?, website=?, category_id=?, is_active=?
                   WHERE id=?""",
                (business_name, contact_name, phone, address, notes, website, category_id, 1 if is_active else 0, vid),
            )
            conn.execute("DELETE FROM vendor_services WHERE vendor_id = ?", (vid,))
            for sid in service_ids:
                conn.execute("INSERT OR IGNORE INTO vendor_services(vendor_id, service_id) VALUES (?,?)", (vid, sid))
        else:
            cur = conn.execute(
                """INSERT INTO vendors(business_name, contact_name, phone, address, notes, website, category_id, is_active)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (business_name, contact_name, phone, address, notes, website, category_id, 1 if is_active else 0),
            )
            vid = cur.lastrowid
            for sid in service_ids:
                conn.execute("INSERT OR IGNORE INTO vendor_services(vendor_id, service_id) VALUES (?,?)", (vid, sid))
    return vid

def soft_delete_vendor(vid: int):
    with conn:
        conn.execute("UPDATE vendors SET is_active=0 WHERE id=?", (vid,))

def hard_delete_vendor(vid: int):
    with conn:
        conn.execute("DELETE FROM vendors WHERE id=?", (vid,))

# ---------- UI ----------
st.title("Vendors — A
