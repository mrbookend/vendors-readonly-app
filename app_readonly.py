
import os, sqlite3
from pathlib import Path
import streamlit as st
import pandas as pd

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
st.set_page_config(page_title="Vendor Directory (Read-Only)", layout="wide")

REPO_DB = Path(__file__).with_name("vendors.db")
DB_PATH = Path(os.getenv("VENDORS_DB", REPO_DB))

# ------------------------------------------------------------
# DB helpers (READ-ONLY connection)
# ------------------------------------------------------------
def get_conn_ro(path: Path):
    uri = f"file:{path.as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def q(conn, sql, params=()):
    cur = conn.cursor()
    cur.execute(sql, params)
    return cur.fetchall()

def get_distinct(conn, col):
    rows = q(conn, f"SELECT DISTINCT {col} AS v FROM vendors WHERE {col} IS NOT NULL AND {col}<>'' ORDER BY {col} COLLATE NOCASE")
    return [""] + [r["v"] for r in rows]

def fts_available(conn):
    rows = q(conn, "SELECT name FROM sqlite_master WHERE type='table' AND name='vendors_fts'")
    return len(rows) > 0

def render_link(u):
    if not u:
        return ""
    s = str(u).strip()
    if s.startswith(("http://","https://")):
        return f"[{s}]({s})"
    if s.startswith("www."):
        return f"[http://{s}](http://{s})"
    return s

# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------
st.sidebar.header("Data Source")
st.sidebar.write(f"DB path: `{DB_PATH}`")

try:
    conn = get_conn_ro(DB_PATH)
    st.sidebar.success("Connected (read-only)")
except Exception as e:
    st.sidebar.error(f"DB error: {e}")
    st.stop()

use_fts = st.sidebar.checkbox("Use FTS (exact-word search)", value=fts_available(conn))

# ------------------------------------------------------------
# Controls
# ------------------------------------------------------------
st.title("Vendor Directory (Read-Only)")

col1, col2, col3 = st.columns([3,2,2], gap="large")
with col1:
    kw = st.text_input("Keyword (names, notes text, URLs)", value="")
with col2:
    category = st.selectbox("Category (optional)", get_distinct(conn, "category"))
with col3:
    service = st.selectbox("Service (optional)", get_distinct(conn, "service"))

# ------------------------------------------------------------
# Search
# ------------------------------------------------------------
def run_search(kw, category, service, use_fts_flag=True, limit=5000):
    base = (
        "SELECT id, category, service, business_name, contact_name, phone, "
        "address, website, notes, COALESCE(Keywords,'') AS Keywords "
        "FROM vendors"
    )
    conds, params = [], []
    if category:
        conds.append("category = ?"); params.append(category)
    if service:
        conds.append("service = ?"); params.append(service)

    if kw.strip():
        # Try FTS first if available and requested
        if use_fts_flag and fts_available(conn):
            conds_fts = conds + ["rowid IN (SELECT rowid FROM vendors_fts WHERE vendors_fts MATCH ?)"]
            params_fts = params + [kw.strip()]
            sql_fts = base + " WHERE " + " AND ".join(conds_fts) + " ORDER BY business_name COLLATE NOCASE LIMIT ?"
            rows = q(conn, sql_fts, tuple(params_fts + [limit]))
            if rows:
                return rows

        # Fallback: case-insensitive substring search across all fields + Keywords
        like_expr = (
            "LOWER("
            "COALESCE(category,'')||' '||"
            "COALESCE(service,'')||' '||"
            "COALESCE(business_name,'')||' '||"
            "COALESCE(contact_name,'')||' '||"
            "COALESCE(phone,'')||' '||"
            "COALESCE(address,'')||' '||"
            "COALESCE(website,'')||' '||"
            "COALESCE(notes,'')||' '||"
            "COALESCE(Keywords,'')"
            ") LIKE ?"
        )
        conds_like = conds + [like_expr]
        params_like = params + [f"%{kw.strip().lower()}%"]
        sql_like = base + " WHERE " + " AND ".join(conds_like) + " ORDER BY business_name COLLATE NOCASE LIMIT ?"
        return q(conn, sql_like, tuple(params_like + [limit]))
    else:
        sql = base + (" WHERE " + " AND ".join(conds) if conds else "") + " ORDER BY business_name COLLATE NOCASE LIMIT ?"
        return q(conn, sql, tuple(params + [limit]))

# ------------------------------------------------------------
# Table + CSV (CORRECT mapping: website=URL, notes=text)
# ------------------------------------------------------------
if rows:
    df = pd.DataFrame([dict(r) for r in rows])

    # Ensure Keywords exists even if empty (belt + suspenders)
    if "Keywords" not in df.columns:
        df["Keywords"] = ""

    # Build display copy with pretty columns/links
    df_display = df.copy()
    df_display["Website (URL)"] = df_display["website"].apply(render_link)
    df_display = df_display.rename(columns={"notes": "Notes (text)"})

    # Show/Hide Keywords in the table (search always includes it)
    show_kw = st.toggle("Show Keywords column", value=True)
    base_display_cols = [
        "category","service","business_name","contact_name","phone",
        "address","Website (URL)","Notes (text)"
    ]
    display_cols = base_display_cols + (["Keywords"] if show_kw else [])

    st.dataframe(df_display[display_cols], use_container_width=True, hide_index=True)

    # CSV export (include Keywords for QA)
    csv_cols = [
        "category","service","business_name","contact_name","phone",
        "address","website","notes","Keywords"
    ]
    st.download_button(
        "Download results as CSV",
        df[csv_cols].to_csv(index=False).encode("utf-8"),
        "vendors_results.csv",
        "text/csv"
    )
else:
    st.info("No matches. Try a shorter keyword or clear filters.")


st.caption("Public read-only app. DB normalized: website = URL, notes = free-text.")
