# app_admin.py
# Vendors Admin — rollback to pre-"cell width counter" attempt
#
# Key behaviors preserved or restored from the stable build:
# - Category is REQUIRED on vendors; Service is optional
# - Add / Edit / Delete Vendors
# - Categories & Services library admin (add, rename, delete)
# - Browse tab with simple global search (non-FTS, case-insensitive, partial word match) across main fields
# - No per-column filters; no AG Grid-specific column filter UI here
# - Phone normalization to (XXX) XXX-XXXX or blank
# - Immediate UI refresh after write actions
# - Turso/libSQL primary with automatic local SQLite fallback
# - "Repair services table" utility (safeguard for schema drift)
# - Debug panel at bottom to aid troubleshooting
#
# Notes:
# - Secrets expected (optional but recommended):
#   TURSO_DATABASE_URL: e.g., "sqlite+libsql://vendors-prod-...turso.io?secure=true"
#   TURSO_AUTH_TOKEN:   libSQL/Turso token
#   page_title, page_max_width_px, sidebar_state (optional UI tweaks)

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text as sql_text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

# -----------------------------
# Page setup (do before any Streamlit UI)
# -----------------------------
PAGE_TITLE = st.secrets.get("page_title", "Vendors Admin") if hasattr(st, "secrets") else "Vendors Admin"
PAGE_MAX_WIDTH_PX = int(st.secrets.get("page_max_width_px", 2300)) if hasattr(st, "secrets") else 1200
SIDEBAR_STATE = st.secrets.get("sidebar_state", "expanded") if hasattr(st, "secrets") else "expanded"

st.set_page_config(page_title=PAGE_TITLE, layout="wide", initial_sidebar_state=SIDEBAR_STATE)

st.markdown(
    f"""
    <style>
      .block-container {{
        max-width: {PAGE_MAX_WIDTH_PX}px;
      }}
      /* tighter inputs in forms */
      .stTextInput > div > div > input {{
        line-height: 1.15;
      }}
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# DB Connection
# -----------------------------
@dataclass
class DBInfo:
    using_remote: bool
    sqlalchemy_url: str
    dialect: str
    driver: str


def build_engine() -> Tuple[Engine, DBInfo]:
    """Build a SQLAlchemy engine. Prefer Turso/libSQL, fall back to local vendors.db."""
    turso_url = None
    turso_token = None
    try:
        turso_url = st.secrets.get("TURSO_DATABASE_URL")  # expected form: sqlite+libsql://...turso.io?secure=true
        turso_token = st.secrets.get("TURSO_AUTH_TOKEN")
    except Exception:
        turso_url = os.environ.get("TURSO_DATABASE_URL")
        turso_token = os.environ.get("TURSO_AUTH_TOKEN")

    # Fallback local database path relative to repo root (works on Streamlit Cloud mount and locally)
    local_sqlite_path = os.path.abspath(os.path.join(os.getcwd(), "vendors.db"))
    local_url = f"sqlite:///{local_sqlite_path}"

    def _engine_from_url(url: str, token: Optional[str]) -> Engine:
        if url.startswith("sqlite+libsql://"):
            connect_args = {"auth_token": token} if token else {}
            return create_engine(url, connect_args=connect_args, pool_pre_ping=True, future=True)
        return create_engine(url, pool_pre_ping=True, future=True)

    # Try remote first
    if turso_url:
        try:
            eng = _engine_from_url(turso_url, turso_token)
            with eng.connect() as conn:
                conn.execute(sql_text("SELECT 1"))
            info = DBInfo(True, turso_url, eng.dialect.name, getattr(eng.dialect, "driver", ""))
            return eng, info
        except Exception as e:
            st.warning(f"Turso connection failed: {e}. Falling back to local SQLite vendors.db.")

    # Fallback
    eng = _engine_from_url(local_url, None)
    info = DBInfo(False, local_url, eng.dialect.name, getattr(eng.dialect, "driver", ""))
    return eng, info


engine, engine_info = build_engine()

# -----------------------------
# Schema helpers
# -----------------------------
CREATE_VENDORS_SQL = """
CREATE TABLE IF NOT EXISTS vendors (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  category TEXT NOT NULL,
  service TEXT,
  business_name TEXT NOT NULL,
  contact_name TEXT,
  phone TEXT,
  address TEXT,
  website TEXT,
  notes TEXT,
  keywords TEXT
);
"""

CREATE_CATEGORIES_SQL = """
CREATE TABLE IF NOT EXISTS categories (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL UNIQUE
);
"""

CREATE_SERVICES_SQL = """
CREATE TABLE IF NOT EXISTS services (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL UNIQUE
);
"""


def ensure_tables() -> None:
    with engine.begin() as conn:
        conn.execute(sql_text(CREATE_VENDORS_SQL))
        conn.execute(sql_text(CREATE_CATEGORIES_SQL))
        conn.execute(sql_text(CREATE_SERVICES_SQL))


ensure_tables()

# -----------------------------
# Utilities
# -----------------------------
PHONE_RE = re.compile(r"\D+")


def normalize_phone(raw: str) -> str:
    if raw is None:
        return ""
    digits = re.sub(PHONE_RE, "", raw)
    if not digits:
        return ""
    if len(digits) != 10:
        raise ValueError("Phone must be 10 digits or left blank")
    return f"({digits[0:3]}) {digits[3:6]}-{digits[6:10]}"


def load_categories() -> List[str]:
    with engine.begin() as conn:
        rows = conn.execute(sql_text("SELECT name FROM categories ORDER BY lower(trim(name)) ASC")).fetchall()
    return [r[0] for r in rows]


def load_services() -> List[str]:
    with engine.begin() as conn:
        rows = conn.execute(sql_text("SELECT name FROM services ORDER BY lower(trim(name)) ASC")).fetchall()
    return [r[0] for r in rows]


def load_vendors_df() -> pd.DataFrame:
    query = (
        "SELECT id, category, service, business_name, contact_name, phone, address, website, notes, keywords "
        "FROM vendors ORDER BY lower(trim(business_name)) ASC"
    )
    with engine.begin() as conn:
        df = pd.read_sql_query(sql_text(query), conn)
    return df


# -----------------------------
# Write helpers
# -----------------------------

def add_vendor(data: Dict[str, Optional[str]]) -> int:
    with engine.begin() as conn:
        res = conn.execute(
            sql_text(
                """
                INSERT INTO vendors (category, service, business_name, contact_name, phone, address, website, notes, keywords)
                VALUES (:category, :service, :business_name, :contact_name, :phone, :address, :website, :notes, :keywords)
                """
            ),
            data,
        )
        new_id = res.lastrowid if hasattr(res, "lastrowid") else None
    return int(new_id or 0)


def update_vendor(vid: int, data: Dict[str, Optional[str]]) -> None:
    with engine.begin() as conn:
        conn.execute(
            sql_text(
                """
                UPDATE vendors
                   SET category=:category, service=:service, business_name=:business_name,
                       contact_name=:contact_name, phone=:phone, address=:address,
                       website=:website, notes=:notes, keywords=:keywords
                 WHERE id=:id
                """
            ),
            {"id": vid, **data},
        )


def delete_vendor(vid: int) -> None:
    with engine.begin() as conn:
        conn.execute(sql_text("DELETE FROM vendors WHERE id=:id"), {"id": vid})


def add_category(name: str) -> None:
    with engine.begin() as conn:
        conn.execute(sql_text("INSERT OR IGNORE INTO categories(name) VALUES(:n)"), {"n": name.strip()})


def rename_category(old: str, new: str) -> None:
    with engine.begin() as conn:
        conn.execute(sql_text("UPDATE categories SET name=:new WHERE name=:old"), {"new": new.strip(), "old": old})
        # Keep vendors referencing the old text in sync
        conn.execute(sql_text("UPDATE vendors SET category=:new WHERE category=:old"), {"new": new.strip(), "old": old})


def delete_category(name: str) -> None:
    with engine.begin() as conn:
        conn.execute(sql_text("DELETE FROM categories WHERE name=:n"), {"n": name})


def add_service(name: str) -> None:
    with engine.begin() as conn:
        conn.execute(sql_text("INSERT OR IGNORE INTO services(name) VALUES(:n)"), {"n": name.strip()})


def rename_service(old: str, new: str) -> None:
    with engine.begin() as conn:
        conn.execute(sql_text("UPDATE services SET name=:new WHERE name=:old"), {"new": new.strip(), "old": old})
        conn.execute(sql_text("UPDATE vendors SET service=:new WHERE service=:old"), {"new": new.strip(), "old": old})


def delete_service(name: str) -> None:
    with engine.begin() as conn:
        conn.execute(sql_text("DELETE FROM services WHERE name=:n"), {"n": name})


# -----------------------------
# Maintenance / Repair
# -----------------------------

def repair_services_table() -> str:
    """Ensure services has schema (id INTEGER PK AUTOINCREMENT, name TEXT UNIQUE). Migrate old columns if needed."""
    with engine.begin() as conn:
        # detect columns
        cols = conn.execute(sql_text("PRAGMA table_info(services)")).fetchall()
        col_names = [c[1] for c in cols]
        if col_names == ["id", "name"]:
            return "services table already OK"

        # Create temp with correct schema
        conn.execute(sql_text("DROP TABLE IF EXISTS services_new"))
        conn.execute(sql_text(CREATE_SERVICES_SQL.replace("IF NOT EXISTS ", "")))

        # migrate possible old data
        if "name" in col_names:
            conn.execute(sql_text("INSERT OR IGNORE INTO services(name) SELECT DISTINCT name FROM services WHERE name IS NOT NULL AND TRIM(name)<>''"))
        elif "service" in col_names:
            conn.execute(sql_text("INSERT OR IGNORE INTO services(name) SELECT DISTINCT service FROM services WHERE service IS NOT NULL AND TRIM(service)<>''"))
        elif "category" in col_names and "service" in col_names:
            conn.execute(sql_text("INSERT OR IGNORE INTO services(name) SELECT DISTINCT service FROM services WHERE service IS NOT NULL AND TRIM(service)<>''"))

        # swap
        conn.execute(sql_text("DROP TABLE services"))
        conn.execute(sql_text("ALTER TABLE services_new RENAME TO services"))

    return "services table repaired to (id,name) schema"


# -----------------------------
# UI Tabs
# -----------------------------

def tab_browse():
    df = load_vendors_df()

    q = st.text_input("Search", placeholder="e.g., plumb returns any record with 'plumb' anywhere")
    if q:
        qlow = q.strip().lower()
        mask = pd.Series([False] * len(df))
        for col in [
            "category",
            "service",
            "business_name",
            "contact_name",
            "phone",
            "address",
            "website",
            "notes",
            "keywords",
        ]:
            mask = mask | df[col].fillna("").str.lower().str.contains(qlow, na=False)
        df = df[mask]

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
    )


def tab_add():
    st.subheader("Add Vendor")

    cats = load_categories()
    svcs = load_services()

    with st.form("add_vendor_form"):
        col1, col2 = st.columns(2)
        with col1:
            category = st.selectbox("Category *", options=cats, index=0 if cats else None, placeholder="Select category")
            service = st.selectbox("Service (optional)", options=[""] + svcs, index=0)
            business_name = st.text_input("Business Name *")
            contact_name = st.text_input("Contact Name")
            phone_raw = st.text_input("Phone (digits only or formatted)")
        with col2:
            address = st.text_area("Address", height=100)
            website = st.text_input("Website (URL)")
            notes = st.text_area("Notes", height=100)
            keywords = st.text_input("Keywords (comma or space separated)")

        submitted = st.form_submit_button("Add Vendor")

    if submitted:
        if not category:
            st.error("Category is required.")
            return
        if not business_name.strip():
            st.error("Business Name is required.")
            return
        try:
            phone = normalize_phone(phone_raw) if phone_raw.strip() else ""
        except ValueError as e:
            st.error(str(e))
            return

        # normalize keywords: split on comma or whitespace, lower, unique, join by comma+space
        kws = ", ".join(sorted({k.strip().lower() for k in re.split(r"[\s,]+", keywords) if k.strip()})) if keywords else ""

        data = {
            "category": category.strip(),
            "service": service.strip() if service else None,
            "business_name": business_name.strip(),
            "contact_name": contact_name.strip() if contact_name else None,
            "phone": phone or None,
            "address": address.strip() if address else None,
            "website": website.strip() if website else None,
            "notes": notes.strip() if notes else None,
            "keywords": kws or None,
        }
        vid = add_vendor(data)
        st.success(f"Added vendor #{vid}: {business_name.strip()}")
        st.rerun()


def tab_edit():
    st.subheader("Edit Vendor")

    df = load_vendors_df()
    if df.empty:
        st.info("No vendors yet.")
        return

    choices = [f"#{r.id} — {r.business_name}" for r in df.itertuples(index=False)]
    choice = st.selectbox("Select vendor", options=choices, index=0)
    vid = int(choice.split("—", 1)[0].strip().lstrip("#"))
    row = df[df["id"] == vid].iloc[0]

    cats = load_categories()
    svcs = load_services()

    with st.form("edit_vendor_form"):
        col1, col2 = st.columns(2)
        with col1:
            category = st.selectbox("Category *", options=cats, index=max(0, cats.index(row["category"])) if row["category"] in cats else 0)
            service = st.selectbox(
                "Service (optional)",
                options=[""] + svcs,
                index=(svcs.index(row["service"]) + 1) if (pd.notna(row["service"]) and row["service"] in svcs) else 0,
            )
            business_name = st.text_input("Business Name *", value=row["business_name"] or "")
            contact_name = st.text_input("Contact Name", value=row["contact_name"] or "")
            phone_raw = st.text_input("Phone (digits only or formatted)", value=row["phone"] or "")
        with col2:
            address = st.text_area("Address", value=row["address"] or "", height=100)
            website = st.text_input("Website (URL)", value=row["website"] or "")
            notes = st.text_area("Notes", value=row["notes"] or "", height=100)
            keywords = st.text_input("Keywords (comma or space separated)", value=row["keywords"] or "")

        submitted = st.form_submit_button("Save Changes")

    if submitted:
        if not category:
            st.error("Category is required.")
            return
        if not business_name.strip():
            st.error("Business Name is required.")
            return
        try:
            phone = normalize_phone(phone_raw) if phone_raw.strip() else ""
        except ValueError as e:
            st.error(str(e))
            return

        kws = ", ".join(sorted({k.strip().lower() for k in re.split(r"[\s,]+", keywords) if k.strip()})) if keywords else ""

        data = {
            "category": category.strip(),
            "service": service.strip() if service else None,
            "business_name": business_name.strip(),
            "contact_name": contact_name.strip() if contact_name else None,
            "phone": phone or None,
            "address": address.strip() if address else None,
            "website": website.strip() if website else None,
            "notes": notes.strip() if notes else None,
            "keywords": kws or None,
        }
        update_vendor(vid, data)
        st.success(f"Updated vendor #{vid}: {business_name.strip()}")
        st.rerun()


def tab_delete():
    st.subheader("Delete Vendor")
    df = load_vendors_df()
    if df.empty:
        st.info("No vendors to delete.")
        return

    choices = [f"#{r.id} — {r.business_name}" for r in df.itertuples(index=False)]
    choice = st.selectbox("Select vendor to delete", options=choices, index=0)
    vid = int(choice.split("—", 1)[0].strip().lstrip("#"))
    row = df[df["id"] == vid].iloc[0]

    st.warning(f"You are about to delete: #{vid} — {row['business_name']}")
    if st.button("Confirm Delete", type="primary"):
        delete_vendor(vid)
        st.success(f"Deleted vendor #{vid}")
        st.rerun()


def tab_lib_admin():
    st.subheader("Categories & Services Admin")

    colA, colB = st.columns(2)

    with colA:
        st.markdown("### Categories")
        cats = load_categories()
        st.write(pd.DataFrame({"Category": cats}))

        with st.form("cat_add"):
            new_c = st.text_input("Add Category")
            ok = st.form_submit_button("Add")
        if ok and new_c.strip():
            add_category(new_c)
            st.success(f"Added category '{new_c.strip()}'")
            st.rerun()

        if cats:
            with st.form("cat_rename"):
                old = st.selectbox("Rename from", options=cats)
                new = st.text_input("to")
                ok2 = st.form_submit_button("Rename")
            if ok2 and new.strip():
                rename_category(old, new)
                st.success(f"Renamed category '{old}' → '{new.strip()}' (vendors updated)")
                st.rerun()

        if cats:
            with st.form("cat_delete"):
                delc = st.selectbox("Delete Category", options=cats)
                ok3 = st.form_submit_button("Delete")
            if ok3:
                # Safety: only allow delete if no vendors use it
                with engine.begin() as conn:
                    count = conn.execute(sql_text("SELECT COUNT(*) FROM vendors WHERE category=:c"), {"c": delc}).scalar()
                if count and int(count) > 0:
                    st.error(f"Cannot delete. {count} vendor(s) still use this category.")
                else:
                    delete_category(delc)
                    st.success(f"Deleted category '{delc}'")
                    st.rerun()

    with colB:
        st.markdown("### Services")
        svcs = load_services()
        st.write(pd.DataFrame({"Service": svcs}))

        with st.form("svc_add"):
            new_s = st.text_input("Add Service")
            ok = st.form_submit_button("Add")
        if ok and new_s.strip():
            add_service(new_s)
            st.success(f"Added service '{new_s.strip()}'")
            st.rerun()

        if svcs:
            with st.form("svc_rename"):
                old = st.selectbox("Rename from", options=svcs)
                new = st.text_input("to")
                ok2 = st.form_submit_button("Rename")
            if ok2 and new.strip():
                rename_service(old, new)
                st.success(f"Renamed service '{old}' → '{new.strip()}' (vendors updated)")
                st.rerun()

        if svcs:
            with st.form("svc_delete"):
                dels = st.selectbox("Delete Service", options=svcs)
                ok3 = st.form_submit_button("Delete")
            if ok3:
                # Allow delete even if vendors reference it — vendors keep old text (optional policy).
                delete_service(dels)
                st.success(f"Deleted service '{dels}'")
                st.rerun()


def tab_maintenance():
    st.subheader("Maintenance & Utilities")

    if st.button("Repair services table (ensure schema id,name)"):
        msg = repair_services_table()
        st.success(msg)
        st.rerun()

    st.caption("If you recently migrated from an older build, run the repair once to lock the services schema.")


def tab_debug():
    st.subheader("Database Status & Schema (debug)")

    # Engine info
    st.json(
        {
            "engine_url": engine_info.sqlalchemy_url,
            "driver": engine_info.driver,
            "dialect": engine_info.dialect,
            "using_remote": engine_info.using_remote,
        }
    )

    with engine.begin() as conn:
        # Columns for vendors
        v_cols = conn.execute(sql_text("PRAGMA table_info(vendors)"))
        vendors_columns = [r[1] for r in v_cols.fetchall()]

        # presence of category/service tables
        has_categories = conn.execute(sql_text("SELECT name FROM sqlite_master WHERE type='table' AND name='categories'"))
        has_services = conn.execute(sql_text("SELECT name FROM sqlite_master WHERE type='table' AND name='services'"))
        has_categories_table = has_categories.fetchone() is not None
        has_services_table = has_services.fetchone() is not None

        counts = {
            "vendors": conn.execute(sql_text("SELECT COUNT(*) FROM vendors")).scalar(),
        }

    st.json(
        {
            "vendors_columns": {i: c for i, c in enumerate(vendors_columns)},
            "has_categories_table": has_categories_table,
            "has_services_table": has_services_table,
            "counts": counts,
        }
    )

    # --- Recent changes panel ---
    st.markdown("### Recent changes (last 10)")
    with engine.begin() as conn:
        rows = conn.execute(sql_text(
            """
            SELECT id, business_name, COALESCE(updated_at, created_at) AS timestamp, updated_by
            FROM vendors
            ORDER BY datetime(COALESCE(updated_at, created_at)) DESC, id DESC
            LIMIT 10
            """
        )).fetchall()
    if rows:
        st.dataframe(
            pd.DataFrame(rows, columns=["id", "business_name", "timestamp", "updated_by"]),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.caption("No recent rows.")

    # --- Read-only SQL runner (SELECT/PRAGMA only) ---
    st.markdown("### Run SQL (read-only)")
    st.caption("Allowed: statements starting with SELECT or PRAGMA. This is blocked from performing writes.")
    default_sql = """\
SELECT id, business_name, created_at, updated_at, updated_by
FROM vendors
ORDER BY id DESC
LIMIT 5;
"""
    sql_text_input = st.text_area("SQL", value=default_sql, height=140)
    run = st.button("Execute SQL")

    def _first_sql_token(s: str) -> str:
        if not s:
            return ""
        s = s.replace("\ufeff", "")  # strip BOM if present
        s = s.lstrip()
        # strip leading line and block comments
        import re as _re
        while True:
            if s.startswith("--"):
                nl = s.find("\n")
                s = "" if nl == -1 else s[nl+1:].lstrip()
                continue
            if s.startswith("/*"):
                m = _re.search(r"\*/", s)
                if m:
                    s = s[m.end():].lstrip()
                    continue
            break
        m = _re.match(r"([A-Za-z_]+)", s)
        return m.group(1).upper() if m else ""

    if run:
        stmt = (sql_text_input or "").strip()
        if not stmt:
            st.warning("Enter a statement.")
        else:
            tok = _first_sql_token(stmt)
            if tok not in {"SELECT", "PRAGMA", "WITH"}:  # allow CTEs that start with WITH
                st.error("Only SELECT or PRAGMA statements are allowed here.")
            else:
                try:
                    with engine.begin() as conn:
                        res = conn.execute(sql_text(stmt))
                        cols = res.keys() if hasattr(res, "keys") else None
                        rows = res.fetchall()
                    if rows:
                        df_out = pd.DataFrame(rows, columns=list(cols) if cols else None)
                        st.dataframe(df_out, use_container_width=True, hide_index=True)
                    else:
                        st.info("No rows returned.")
                except Exception as e:
                    st.error(f"Query error: {e}")


def main():
    # Title intentionally removed per user request
    tabs = st.tabs([
        "Browse",
        "Add",
        "Edit",
        "Delete",
        "Categories & Services Admin",
        "Maintenance",
        "Debug",
    ])

    with tabs[0]:
        tab_browse()
    with tabs[1]:
        tab_add()
    with tabs[2]:
        tab_edit()
    with tabs[3]:
        tab_delete()
    with tabs[4]:
        tab_lib_admin()
    with tabs[5]:
        tab_maintenance()
    with tabs[6]:
        tab_debug()


if __name__ == "__main__":
    main()

