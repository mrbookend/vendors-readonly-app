# app_admin.py
# Vendors Admin — Turso (libSQL) first, SQLite fallback.
# SQLAlchemy engine, safe params, category→service flow, phone/URL normalization,
# optional Keywords support (only if vendors.keywords exists).

from __future__ import annotations

import os
import re
from typing import List, Tuple, Dict, Optional
from urllib.parse import urlparse
from pathlib import Path

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine
from sqlalchemy.sql import text as sql_text  # alias to avoid collisions
from sqlalchemy.engine import Engine

st.set_page_config(page_title="Vendors Admin", layout="wide")

# ========= Secrets / Config =========
LIBSQL_URL = (
    (st.secrets.get("LIBSQL_URL") if hasattr(st, "secrets") else None)
    or os.getenv("LIBSQL_URL")
    or os.getenv("TURSO_DATABASE_URL")
)
LIBSQL_AUTH_TOKEN = (
    (st.secrets.get("LIBSQL_AUTH_TOKEN") if hasattr(st, "secrets") else None)
    or os.getenv("LIBSQL_AUTH_TOKEN")
    or os.getenv("TURSO_AUTH_TOKEN")
)
VENDORS_DB_PATH = (
    (st.secrets.get("VENDORS_DB_PATH") if hasattr(st, "secrets") else None)
    or os.getenv("VENDORS_DB_PATH")
    or str((Path(__file__).resolve().parent / "vendors.db"))
)
ADMIN_PASSWORD = (
    (st.secrets.get("ADMIN_PASSWORD") if hasattr(st, "secrets") else None)
    or os.getenv("ADMIN_PASSWORD")
    or "ADMIN"
)

USE_TURSO = bool(LIBSQL_URL and LIBSQL_AUTH_TOKEN)

def _as_sqlalchemy_url(u: str) -> str:
    """Convert libsql://... to sqlite+libsql://... (SQLAlchemy dialect)."""
    if not u:
        return u
    if u.startswith("sqlite+libsql://"):
        return u
    if u.startswith("libsql://"):
        tail = u[len("libsql://"):]
        return f"sqlite+libsql://{tail}" + ("" if "?" in tail else "?secure=true")
    return u

def make_engine() -> Engine:
    if USE_TURSO:
        sa_url = _as_sqlalchemy_url(LIBSQL_URL)
        return create_engine(
            sa_url,
            connect_args={"auth_token": LIBSQL_AUTH_TOKEN},
            pool_pre_ping=True,
            pool_recycle=300,
        )
    return create_engine(f"sqlite:///{VENDORS_DB_PATH}")

engine: Engine = make_engine()
DB_SOURCE = f"(Turso) {_as_sqlalchemy_url(LIBSQL_URL) or ''}" if USE_TURSO else f"(SQLite) {VENDORS_DB_PATH}"

# ========= Small DB helpers (safe param handling) =========
def run_df(sql: str, params: Dict | None = None) -> pd.DataFrame:
    with engine.begin() as conn:
        return pd.read_sql_query(sql_text(sql), conn, params=dict(params or {}))

def run_exec(sql: str, params: Dict | None = None) -> None:
    with engine.begin() as conn:
        conn.execute(sql_text(sql), dict(params or {}))

def query_scalar(sql: str, params: Dict | None = None):
    with engine.begin() as conn:
        row = conn.execute(sql_text(sql), dict(params or {})).fetchone()
        return row[0] if row else None

def table_exists(name: str) -> bool:
    return query_scalar(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=:n LIMIT 1;", {"n": name}
    ) is not None

def table_columns(table: str) -> List[str]:
    with engine.begin() as conn:
        rows = conn.execute(sql_text(f"PRAGMA table_info({table});")).fetchall()
        return [r[1] for r in rows]  # name column

# ========= Schema detection =========
def detect_schema():
    if not table_exists("vendors"):
        st.error("Table 'vendors' not found. Check DB connection/path.")
        st.stop()
    vcols = table_columns("vendors")
    has_categories = table_exists("categories")
    has_services = table_exists("services")
    return {
        "vendors_columns": vcols,
        "has_categories_table": has_categories,
        "has_services_table": has_services,
        "uses_cat_id": "category_id" in vcols and has_categories,
        "uses_svc_id": "service_id" in vcols and has_services,
        "uses_cat_text": "category" in vcols,
        "uses_svc_text": "service" in vcols,
        "has_keywords": "keywords" in vcols,  # optional column
    }

SCHEMA = detect_schema()

# ========= Catalog loaders (with robust fallbacks) =========
def get_categories() -> List[str]:
    if SCHEMA["has_categories_table"]:
        df = run_df("SELECT name FROM categories ORDER BY name;")
        cats = df["name"].tolist()
        if cats:
            return cats
    if SCHEMA["uses_cat_text"]:
        df = run_df(
            "SELECT DISTINCT TRIM(category) AS name FROM vendors "
            "WHERE category IS NOT NULL AND TRIM(category) <> '' ORDER BY 1;"
        )
        return df["name"].tolist()
    return []

def get_services() -> List[str]:
    if SCHEMA["has_services_table"]:
        df = run_df("SELECT name FROM services ORDER BY name;")
        svcs = df["name"].tolist()
        if svcs:
            return svcs
    if SCHEMA["uses_svc_text"]:
        df = run_df(
            "SELECT DISTINCT TRIM(service) AS name FROM vendors "
            "WHERE service IS NOT NULL AND TRIM(service) <> '' ORDER BY 1;"
        )
        return df["name"].tolist()
    return []

def get_services_for_category(cat: str) -> List[str]:
    """Filter services by chosen category when possible; fallback to unfiltered."""
    if not cat:
        return get_services()
    if SCHEMA["uses_cat_text"] and SCHEMA["uses_svc_text"]:
        df = run_df(
            "SELECT DISTINCT TRIM(service) AS name FROM vendors "
            "WHERE category = :c AND service IS NOT NULL AND TRIM(service) <> '' "
            "ORDER BY 1;",
            {"c": cat},
        )
        lst = df["name"].tolist()
        return lst if lst else get_services()
    return get_services()

def get_or_create_category(name: str) -> Optional[int]:
    if not name or not SCHEMA["has_categories_table"]:
        return None
    name = name.strip()
    cid = query_scalar("SELECT id FROM categories WHERE name=:n;", {"n": name})
    if cid is not None:
        return cid
    run_exec("INSERT INTO categories(name) VALUES(:n);", {"n": name})
    return query_scalar("SELECT id FROM categories WHERE name=:n;", {"n": name})

def get_or_create_service(name: str) -> Optional[int]:
    if not name or not SCHEMA["has_services_table"]:
        return None
    name = name.strip()
    sid = query_scalar("SELECT id FROM services WHERE name=:n;", {"n": name})
    if sid is not None:
        return sid
    run_exec("INSERT INTO services(name) VALUES(:n);", {"n": name})
    return query_scalar("SELECT id FROM services WHERE name=:n;", {"n": name})

# ========= Format / Validate helpers =========
def format_us_phone(raw: str) -> str:
    d = re.sub(r"\D", "", raw or "")
    if len(d) == 11 and d.startswith("1"):
        d = d[1:]
    if len(d) != 10:
        return raw or ""
    return f"({d[:3]}) {d[3:6]}-{d[6:]}"

def normalize_and_validate_url(raw: str) -> tuple[str, bool, str]:
    if not raw:
        return "", True, ""
    s = raw.strip()
    if " " in s:
        return s, False, "URL cannot contain spaces."
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", s):
        s = "https://" + s
    p = urlparse(s)
    if not p.netloc or "." not in p.netloc:
        return s, False, "URL must include a valid host (e.g., example.com)."
    return s, True, ""

# ========= Vendors data (list/detail) =========
def load_vendors_df() -> pd.DataFrame:
    if SCHEMA["uses_cat_id"]:
        cat_expr = "c.name AS Category"
        cat_join = "LEFT JOIN categories c ON c.id = v.category_id"
    elif SCHEMA["uses_cat_text"]:
        cat_expr, cat_join = "v.category AS Category", ""
    else:
        cat_expr, cat_join = "NULL AS Category", ""

    if SCHEMA["uses_svc_id"]:
        svc_expr = "s.name AS Service"
        svc_join = "LEFT JOIN services s ON s.id = v.service_id"
    elif SCHEMA["uses_svc_text"]:
        svc_expr, svc_join = "v.service AS Service", ""
    else:
        svc_expr, svc_join = "NULL AS Service", ""

    kw_expr = "v.keywords AS Keywords" if SCHEMA["has_keywords"] else "NULL AS Keywords"

    sql = f"""
        SELECT
          v.id AS id,
          {cat_expr},
          {svc_expr},
          v.business_name AS "Business Name",
          v.contact_name  AS "Contact Name",
          v.phone         AS Phone,
          v.address       AS Address,
          v.notes         AS Notes,
          v.website       AS Website,
          {kw_expr}
        FROM vendors v
        {cat_join}
        {svc_join}
        ORDER BY "Business Name" COLLATE NOCASE ASC;
    """
    return run_df(sql)

def vendor_display_label(row: pd.Series) -> str:
    bn = row.get("Business Name") or "(no name)"
    cat = row.get("Category") or "Uncategorized"
    svc = row.get("Service") or "Unspecified"
    vid = row.get("id")
    return f"{bn} — {cat} / {svc} (id={vid})"

# ========= UI helpers =========
ADD_NEW_CAT = "➕ Add new category…"
ADD_NEW_SVC = "➕ Add new service…"
CUSTOM_SERVICE = "⟂ Custom…"

def category_selector(key_prefix: str, default_name: Optional[str]) -> str:
    options = [""] + get_categories() + [ADD_NEW_CAT]
    idx = options.index(default_name) if default_name in options else 0
    sel = st.selectbox("Category", options=options, index=idx, key=f"{key_prefix}_cat_select")
    if sel == ADD_NEW_CAT:
        return st.text_input("New category name", key=f"{key_prefix}_cat_new").strip()
    return sel.strip() if sel else ""

def service_selector_filtered(key_prefix: str, category_name: str, default_name: Optional[str]) -> str:
    base = get_services_for_category(category_name)
    options = [""] + base + [CUSTOM_SERVICE, ADD_NEW_SVC]
    idx = options.index(default_name) if default_name in options else 0
    sel = st.selectbox("Service", options=options, index=idx, key=f"{key_prefix}_svc_select")
    if sel == ADD_NEW_SVC:
        return st.text_input("New service name", key=f"{key_prefix}_svc_new").strip()
    if sel == CUSTOM_SERVICE:
        return st.text_input("Custom service", key=f"{key_prefix}_svc_custom").strip()
    return sel.strip() if sel else ""

# ========= CRUD =========
def insert_vendor(payload: Dict) -> int:
    with engine.begin() as conn:
        vals: Dict = {}
        sets: List[str] = []

        cat_name = (payload.get("Category") or "").strip()
        svc_name = (payload.get("Service") or "").strip()

        if SCHEMA["uses_cat_id"]:
            cid = None
            if cat_name:
                row = conn.execute(sql_text("SELECT id FROM categories WHERE name=:n;"), {"n": cat_name}).fetchone()
                cid = row[0] if row else None
                if cid is None:
                    conn.execute(sql_text("INSERT INTO categories(name) VALUES(:n);"), {"n": cat_name})
                    row = conn.execute(sql_text("SELECT id FROM categories WHERE name=:n;"), {"n": cat_name}).fetchone()
                    cid = row[0] if row else None
            sets.append("category_id = :category_id"); vals["category_id"] = cid
        elif SCHEMA["uses_cat_text"]:
            sets.append("category = :category"); vals["category"] = cat_name or None

        if SCHEMA["uses_svc_id"]:
            sid = None
            if svc_name:
                row = conn.execute(sql_text("SELECT id FROM services WHERE name=:n;"), {"n": svc_name}).fetchone()
                sid = row[0] if row else None
                if sid is None:
                    conn.execute(sql_text("INSERT INTO services(name) VALUES(:n);"), {"n": svc_name})
                    row = conn.execute(sql_text("SELECT id FROM services WHERE name=:n;"), {"n": svc_name}).fetchone()
                    sid = row[0] if row else None
            sets.append("service_id = :service_id"); vals["service_id"] = sid
        elif SCHEMA["uses_svc_text"]:
            sets.append("service = :service"); vals["service"] = svc_name or None

        base_mapping = {
            "Business Name": "business_name",
            "Contact Name": "contact_name",
            "Phone": "phone",
            "Address": "address",
            "Notes": "notes",
            "Website": "website",
        }
        if SCHEMA["has_keywords"]:
            base_mapping["Keywords"] = "keywords"

        for k_src, k_db in base_mapping.items():
            sets.append(f"{k_db} = :{k_db}")
            vals[k_db] = payload.get(k_src) or None

        cols = ", ".join([s.split("=")[0].strip() for s in sets])
        marks = ", ".join([f":{s.split(':')[-1]}" for s in sets])
        conn.execute(sql_text(f"INSERT INTO vendors ({cols}) VALUES ({marks});"), vals)
        new_id = conn.execute(sql_text("SELECT last_insert_rowid();")).fetchone()[0]
        return int(new_id)

def update_vendor(vendor_id: int, payload: Dict) -> None:
    with engine.begin() as conn:
        vals: Dict = {"id": vendor_id}
        sets: List[str] = []

        cat_name = (payload.get("Category") or "").strip()
        svc_name = (payload.get("Service") or "").strip()

        if SCHEMA["uses_cat_id"]:
            cid = None
            if cat_name:
                row = conn.execute(sql_text("SELECT id FROM categories WHERE name=:n;"), {"n": cat_name}).fetchone()
                cid = row[0] if row else None
                if cid is None:
                    conn.execute(sql_text("INSERT INTO categories(name) VALUES(:n);"), {"n": cat_name})
                    row = conn.execute(sql_text("SELECT id FROM categories WHERE name=:n;"), {"n": cat_name}).fetchone()
                    cid = row[0] if row else None
            sets.append("category_id = :category_id"); vals["category_id"] = cid
        elif SCHEMA["uses_cat_text"]:
            sets.append("category = :category"); vals["category"] = cat_name or None

        if SCHEMA["uses_svc_id"]:
            sid = None
            if svc_name:
                row = conn.execute(sql_text("SELECT id FROM services WHERE name=:n;"), {"n": svc_name}).fetchone()
                sid = row[0] if row else None
                if sid is None:
                    conn.execute(sql_text("INSERT INTO services(name) VALUES(:n);"), {"n": svc_name})
                    row = conn.execute(sql_text("SELECT id FROM services WHERE name=:n;"), {"n": svc_name}).fetchone()
                    sid = row[0] if row else None
            sets.append("service_id = :service_id"); vals["service_id"] = sid
        elif SCHEMA["uses_svc_text"]:
            sets.append("service = :service"); vals["service"] = svc_name or None

        base_mapping = {
            "Business Name": "business_name",
            "Contact Name": "contact_name",
            "Phone": "phone",
            "Address": "address",
            "Notes": "notes",
            "Website": "website",
        }
        if SCHEMA["has_keywords"]:
            base_mapping["Keywords"] = "keywords"

        for k_src, k_db in base_mapping.items():
            sets.append(f"{k_db} = :{k_db}")
            vals[k_db] = payload.get(k_src) or None

        conn.execute(sql_text(f"UPDATE vendors SET {', '.join(sets)} WHERE id = :id;"), vals)

def delete_vendor(vendor_id: int) -> None:
    run_exec("DELETE FROM vendors WHERE id = :id;", {"id": vendor_id})

# ========= Categories/Services admin =========
def rename_category(old_name: str, new_name: str) -> None:
    new_name = new_name.strip()
    if SCHEMA["has_categories_table"] and SCHEMA["uses_cat_id"]:
        run_exec("UPDATE OR IGNORE categories SET name = :n WHERE name = :o;", {"n": new_name, "o": old_name})
    elif SCHEMA["uses_cat_text"]:
        run_exec("UPDATE vendors SET category = :n WHERE category = :o;", {"n": new_name, "o": old_name})

def rename_service(old_name: str, new_name: str) -> None:
    new_name = new_name.strip()
    if SCHEMA["has_categories_table"] and SCHEMA["uses_svc_id"]:
        run_exec("UPDATE OR IGNORE services SET name = :n WHERE name = :o;", {"n": new_name, "o": old_name})
    elif SCHEMA["uses_svc_text"]:
        run_exec("UPDATE vendors SET service = :n WHERE service = :o;", {"n": new_name, "o": old_name})

def delete_category_val(name: str) -> Tuple[bool, str]:
    if SCHEMA["has_categories_table"] and SCHEMA["uses_cat_id"]:
        n = query_scalar(
            "SELECT COUNT(*) FROM vendors v JOIN categories c ON v.category_id=c.id WHERE c.name=:n;", {"n": name}
        )
        if n:
            return False, f"Cannot delete '{name}': it is in use by {n} vendor(s)."
        run_exec("DELETE FROM categories WHERE name=:n;", {"n": name})
        return True, f"Deleted category '{name}'."
    elif SCHEMA["uses_cat_text"]:
        n = query_scalar("SELECT COUNT(*) FROM vendors WHERE category=:n;", {"n": name})
        if n:
            return False, f"Cannot delete '{name}': it is in use by {n} vendor(s)."
        return True, f"'{name}' removed from catalog view."
    return False, "Category storage not found."

def delete_service_val(name: str) -> Tuple[bool, str]:
    if SCHEMA["has_categories_table"] and SCHEMA["uses_svc_id"]:
        n = query_scalar(
            "SELECT COUNT(*) FROM vendors v JOIN services s ON v.service_id=s.id WHERE s.name=:n;", {"n": name}
        )
        if n:
            return False, f"Cannot delete '{name}': it is in use by {n} vendor(s)."
        run_exec("DELETE FROM services WHERE name=:n;", {"n": name})
        return True, f"Deleted service '{name}'."
    elif SCHEMA["uses_svc_text"]:
        n = query_scalar("SELECT COUNT(*) FROM vendors WHERE service=:n;", {"n": name})
        if n:
            return False, f"Cannot delete '{name}': it is in use by {n} vendor(s)."
        return True, f"'{name}' removed from catalog view."
    return False, "Service storage not found."

# ========= Auth gate (Enter submits) =========
if "auth_ok" not in st.session_state:
    st.session_state.auth_ok = False

if not st.session_state.auth_ok:
    st.title("Vendors Admin – Sign in")
    with st.form("login_form", clear_on_submit=False):
        pwd = st.text_input("Admin password", type="password")
        submitted = st.form_submit_button("Sign in")
    if submitted:
        if (pwd or "").strip() == ADMIN_PASSWORD:
            st.session_state.auth_ok = True
            st.rerun()
        else:
            st.error("Wrong password.")
    st.stop()

# ========= App UI =========
st.title("Vendors Admin")

with st.sidebar:
    st.subheader("Navigation")
    page = st.radio("Go to", ["View", "Add", "Edit", "Delete", "Categories & Services Admin"], index=0)
    if st.button("Sign out"):
        st.session_state.auth_ok = False
        st.rerun()

# Debug/status expander
with st.expander("Database Status & Schema (debug)"):
    counts = {
        "vendors": query_scalar("SELECT COUNT(*) FROM vendors;") or 0,
        "categories": query_scalar("SELECT COUNT(*) FROM categories;") if table_exists("categories") else None,
        "services": query_scalar("SELECT COUNT(*) FROM services;") if table_exists("services") else None,
    }
    st.write({
        "db_source": DB_SOURCE,
        "vendors_columns": SCHEMA["vendors_columns"],
        "has_categories_table": SCHEMA["has_categories_table"],
        "has_services_table": SCHEMA["has_services_table"],
        "uses_cat_id": SCHEMA["uses_cat_id"],
        "uses_svc_id": SCHEMA["uses_svc_id"],
        "uses_cat_text": SCHEMA["uses_cat_text"],
        "uses_svc_text": SCHEMA["uses_svc_text"],
        "has_keywords": SCHEMA["has_keywords"],
        "counts": counts,
        "sample_categories": get_categories()[:15],
        "sample_services": get_services()[:15],
    })

def refresh_notice():
    st.caption("Lists refresh automatically. If values look stale, use **Rerun** from the menu.")

# Pages
if page == "View":
    st.header("All Vendors")
    df = load_vendors_df()
    if df.empty:
        st.info("No vendors found.")
    else:
        drop_cols = [c for c in ("id",) if c in df.columns]
        st.dataframe(df.drop(columns=drop_cols), use_container_width=True, hide_index=True)
    refresh_notice()

elif page == "Add":
    st.header("Add Vendor")
    with st.form("add_vendor_form", clear_on_submit=True):
        col1, col2 = st.columns(2)

        with col1:
            business_name = st.text_input("Business Name").strip()
            contact_name = st.text_input("Contact Name").strip()
            phone = st.text_input("Phone", key="add_phone", placeholder="(210) 555-1212").strip()

            category_name = category_selector("add", default_name=None)
            service_name = service_selector_filtered("add", category_name, default_name=None)

        with col2:
            address = st.text_area("Address").strip()
            notes = st.text_area("Notes").strip()

            website_raw = st.text_input("Website (URL)", key="add_website").strip()
            norm_url, url_ok, url_msg = normalize_and_validate_url(website_raw)
            if website_raw and not url_ok:
                st.error(url_msg)
            elif website_raw and url_ok:
                st.caption("Preview:")
                st.markdown(f"[Open link]({norm_url})")

            keywords = st.text_input("Keywords (comma-separated)").strip()

        errors = []
        if not business_name:
            errors.append("Business Name is required.")
        d = re.sub(r"\D", "", phone or "")
        if phone and not (len(d) == 10 or (len(d) == 11 and d.startswith("1"))):
            errors.append("Phone must have 10 digits (optionally leading 1).")
        if website_raw and not url_ok:
            errors.append("Website/URL is not valid.")

        submitted = st.form_submit_button("Add Vendor", type="primary", disabled=bool(errors))

    if submitted:
        payload = {
            "Business Name": business_name or None,
            "Contact Name": contact_name or None,
            "Phone": format_us_phone(phone) if phone else None,
            "Address": address or None,
            "Notes": notes or None,
            "Website": norm_url if website_raw else None,
            "Category": category_name or None,
            "Service": service_name or None,
        }
        if SCHEMA["has_keywords"]:
            payload["Keywords"] = keywords or None
        try:
            new_id = insert_vendor(payload)
            st.success(f"Vendor added (id={new_id}).")
        except Exception as e:
            st.error(f"Failed to add vendor: {e}")
    refresh_notice()

elif page == "Edit":
    st.header("Edit Vendor")
    df = load_vendors_df()
    if df.empty:
        st.info("No vendors found.")
    else:
        labels = df.apply(vendor_display_label, axis=1).tolist()
        id_by_label = {labels[i]: int(df.iloc[i]["id"]) for i in range(len(labels))}
        chosen_label = st.selectbox("Select Vendor", options=labels, key="edit_vendor_select")
        vid = id_by_label[chosen_label]
        row = df.loc[df["id"] == vid].iloc[0]

        with st.form("edit_vendor_form"):
            col1, col2 = st.columns(2)
            with col1:
                business_name = st.text_input("Business Name", value=row.get("Business Name") or "").strip()
                contact_name = st.text_input("Contact Name", value=row.get("Contact Name") or "").strip()

                phone_init = format_us_phone(row.get("Phone") or "")
                phone = st.text_input("Phone", key="edit_phone", value=phone_init).strip()

                category_name = category_selector("edit", default_name=row.get("Category"))
                service_name = service_selector_filtered("edit", category_name, default_name=row.get("Service"))

            with col2:
                address = st.text_area("Address", value=row.get("Address") or "").strip()
                notes = st.text_area("Notes", value=row.get("Notes") or "").strip()

                website_raw = st.text_input("Website (URL)", value=row.get("Website") or "").strip()
                norm_url, url_ok, url_msg = normalize_and_validate_url(website_raw)
                if website_raw and not url_ok:
                    st.error(url_msg)
                elif website_raw and url_ok:
                    st.caption("Preview:")
                    st.markdown(f"[Open link]({norm_url})")

                if SCHEMA["has_keywords"]:
                    keywords_init = row.get("Keywords") or ""
                    keywords = st.text_input("Keywords (comma-separated)", value=keywords_init).strip()
                else:
                    keywords = ""

            errors = []
            if not business_name:
                errors.append("Business Name is required.")
            d = re.sub(r"\D", "", phone or "")
            if phone and not (len(d) == 10 or (len(d) == 11 and d.startswith("1"))):
                errors.append("Phone must have 10 digits (optionally leading 1).")
            if website_raw and not url_ok:
                errors.append("Website/URL is not valid.")

            save = st.form_submit_button("Save Changes", disabled=bool(errors))
        if save:
            payload = {
                "Business Name": business_name or None,
                "Contact Name": contact_name or None,
                "Phone": format_us_phone(phone) if phone else None,
                "Address": address or None,
                "Notes": notes or None,
                "Website": norm_url if website_raw else None,
                "Category": category_name or None,
                "Service": service_name or None,
            }
            if SCHEMA["has_keywords"]:
                payload["Keywords"] = keywords or None
            try:
                update_vendor(int(vid), payload)
                st.success("Vendor updated.")
            except Exception as e:
                st.error(f"Failed to update vendor: {e}")
    refresh_notice()

elif page == "Delete":
    st.header("Delete Vendor")
    df = load_vendors_df()
    if df.empty:
        st.info("No vendors found.")
    else:
        labels = df.apply(vendor_display_label, axis=1).tolist()
        id_by_label = {labels[i]: int(df.iloc[i]["id"]) for i in range(len(labels))}
        chosen_label = st.selectbox("Select Vendor to delete", options=labels, key="del_vendor_select")
        vid = id_by_label[chosen_label]
        st.warning(f"This will permanently delete vendor id={vid}.")
        if st.button("Confirm Delete", type="primary"):
            try:
                delete_vendor(int(vid))
                st.success("Vendor deleted.")
            except Exception as e:
                st.error(f"Failed to delete vendor: {e}")
    refresh_notice()

elif page == "Categories & Services Admin":
    st.header("Categories & Services Admin")
    tab_cat, tab_svc = st.tabs(["Categories", "Services"])

    with tab_cat:
        st.subheader("Manage Categories")
        cats = get_categories()
        st.write("Existing categories:", cats or "(none)")

        with st.expander("Add Category"):
            new_cat = st.text_input("New category name", key="admin_cat_new").strip()
            if st.button("Add Category", key="admin_cat_add_btn"):
                if not new_cat:
                    st.error("Enter a category name.")
                else:
                    try:
                        if table_exists("categories"):
                            get_or_create_category(new_cat)
                        else:
                            st.info("No categories table; using denormalized vendors.category.")
                        st.success(f"Category '{new_cat}' added (or already existed).")
                    except Exception as e:
                        st.error(f"Failed to add category: {e}")

        with st.expander("Rename Category"):
            if cats:
                old = st.selectbox("Select category to rename", options=cats, key="admin_cat_old")
                new = st.text_input("New name", key="admin_cat_rename_to").strip()
                if st.button("Rename", key="admin_cat_rename_btn"):
                    if not new:
                        st.error("Enter the new name.")
                    else:
                        try:
                            rename_category(old, new)
                            st.success(f"Renamed '{old}' to '{new}'.")
                        except Exception as e:
                            st.error(f"Failed to rename category: {e}")
            else:
                st.info("No categories found.")

        with st.expander("Delete Category"):
            if cats:
                victim = st.selectbox("Select category to delete", options=cats, key="admin_cat_del_sel")
                if st.button("Delete Category", key="admin_cat_del_btn"):
                    ok, msg = delete_category_val(victim)
                    if ok:
                        st.success(msg)
                    else:
                        st.error(msg)
            else:
                st.info("No categories found.")

    with tab_svc:
        st.subheader("Manage Services")
        svcs = get_services()
        st.write("Existing services:", svcs or "(none)")

        with st.expander("Add Service"):
            new_svc = st.text_input("New service name", key="admin_svc_new").strip()
            if st.button("Add Service", key="admin_svc_add_btn"):
                if not new_svc:
                    st.error("Enter a service name.")
                else:
                    try:
                        if table_exists("services"):
                            get_or_create_service(new_svc)
                        else:
                            st.info("No services table; using denormalized vendors.service.")
                        st.success(f"Service '{new_svc}' added (or already existed).")
                    except Exception as e:
                        st.error(f"Failed to add service: {e}")

        with st.expander("Rename Service"):
            if svcs:
                old = st.selectbox("Select service to rename", options=svcs, key="admin_svc_old")
                new = st.text_input("New name", key="admin_svc_rename_to").strip()
                if st.button("Rename", key="admin_svc_rename_btn"):
                    if not new:
                        st.error("Enter the new name.")
                    else:
                        try:
                            rename_service(old, new)
                            st.success(f"Renamed '{old}' to '{new}'.")
                        except Exception as e:
                            st.error(f"Failed to rename service: {e}")
            else:
                st.info("No services found.")

        with st.expander("Delete Service"):
            if svcs:
                victim = st.selectbox("Select service to delete", options=svcs, key="admin_svc_del_sel")
                if st.button("Delete Service", key="admin_svc_del_btn"):
                    ok, msg = delete_service_val(victim)
                    if ok:
                        st.success(msg)
                    else:
                        st.error(msg)
            else:
                st.info("No services found.")
# EOF
