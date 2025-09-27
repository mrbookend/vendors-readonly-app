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
