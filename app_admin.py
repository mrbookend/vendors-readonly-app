# app_admin.py
# Vendors Admin — Category REQUIRED, Service optional (blank display if missing)
# - Safe SQL with SQLAlchemy
# - Works with Turso/libSQL or local SQLite
# - Categories and Services admin split (no cross-requirements)
# - Add/Edit/Delete Vendor flows
# - Category is REQUIRED (non-blank) on Add + Edit
# - Service is optional in all places; blank when missing
# - Auto-detects presence of vendors.keywords
# - Phone normalization and Website normalization (lightweight)
# - Robust cache invalidation compatible with multiple Streamlit versions
# - Success notices persist across reruns for Add, Edit, and Delete

from __future__ import annotations
import os
import re
from typing import List, Dict, Optional

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text as sql_text
from sqlalchemy.engine import Engine

# -----------------------------
# Engine setup
# -----------------------------
def get_engine() -> Engine:
    # Prefer Turso/libSQL if provided, else SQLite vendors.db in repo root.
    libsql_url = os.environ.get("LIBSQL_URL") or st.secrets.get("LIBSQL_URL", None)
    libsql_token = os.environ.get("LIBSQL_AUTH_TOKEN") or st.secrets.get("LIBSQL_AUTH_TOKEN", None)

    if libsql_url:
        # Example: libsql://<host-or-db>
        connect_args = {}
        if libsql_token:
            connect_args["authToken"] = libsql_token
        return create_engine(libsql_url, connect_args=connect_args, pool_pre_ping=True, future=True)

    # Fallback local sqlite
    db_path = os.environ.get("SQLITE_PATH", "/mount/src/vendors-readonly-app/vendors.db")
    return create_engine(f"sqlite:///{db_path}", future=True)

engine = get_engine()

# -----------------------------
# Introspection & helpers
# -----------------------------
@st.cache_data(show_spinner=False, ttl=60)
def table_exists(name: str) -> bool:
    q = sql_text("""
        SELECT 1
        FROM sqlite_master
        WHERE type IN ('table','view') AND lower(name) = lower(:n)
        UNION ALL
        SELECT 1 FROM pragma_table_info(:n) LIMIT 1
    """)
    try:
        with engine.begin() as conn:
            r = conn.execute(q, {"n": name}).first()
        return bool(r)
    except Exception:
        # libSQL may not support sqlite_master; use ANSI fallback
        try:
            with engine.begin() as conn:
                conn.execute(sql_text(f"SELECT 1 FROM {name} WHERE 1=0"))
            return True
        except Exception:
            return False

@st.cache_data(show_spinner=False, ttl=60)
def get_columns(table: str) -> List[str]:
    # SQLite/LibSQL pragma works on both
    try:
        with engine.begin() as conn:
            rows = conn.execute(sql_text(f"PRAGMA table_info({table})")).fetchall()
            cols = [r[1] for r in rows]  # second field is 'name'
            return cols
    except Exception:
        # Fallback naive select
        try:
            with engine.begin() as conn:
                df = pd.read_sql_query(sql_text(f"SELECT * FROM {table} LIMIT 1"), conn)
            return list(df.columns)
        except Exception:
            return []

@st.cache_data(show_spinner=False, ttl=60)
def vendors_has_keywords() -> bool:
    return "keywords" in get_columns("vendors")

def normalize_phone(raw: str | None) -> str | None:
    if not raw:
        return None
    digits = re.sub(r"\D+", "", raw)
    if len(digits) != 10:
        # leave as-is; UI should not block save—admin can correct later
        return raw.strip()
    return f"({digits[0:3]}) {digits[3:6]}-{digits[6:10]}"

def normalize_url(url: str | None) -> str | None:
    if not url:
        return None
    u = url.strip()
    if not u:
        return None
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+\-.]*://", u):
        # add scheme if missing
        u = "https://" + u
    return u

# -----------------------------
# Reference data (categories/services)
# -----------------------------
@st.cache_data(show_spinner=False, ttl=30)
def list_categories() -> List[str]:
    # Prefer categories table, else distinct from vendors
    if table_exists("categories") and "name" in get_columns("categories"):
        with engine.begin() as conn:
            rows = conn.execute(sql_text("SELECT name FROM categories ORDER BY lower(name) ASC")).fetchall()
        return [r[0] for r in rows if r[0]]
    # fallback
    with engine.begin() as conn:
        rows = conn.execute(sql_text("SELECT DISTINCT category FROM vendors WHERE IFNULL(category,'')<>''")).fetchall()
    cats = sorted([r[0] for r in rows if r[0]], key=lambda x: x.lower())
    return cats

@st.cache_data(show_spinner=False, ttl=30)
def list_services() -> List[str]:
    # Prefer services table, else distinct from vendors
    if table_exists("services") and "name" in get_columns("services"):
        with engine.begin() as conn:
            rows = conn.execute(sql_text("SELECT name FROM services ORDER BY lower(name) ASC")).fetchall()
        return [r[0] for r in rows if r[0]]
    # fallback
    with engine.begin() as conn:
        rows = conn.execute(sql_text("SELECT DISTINCT service FROM vendors WHERE IFNULL(service,'')<>''")).fetchall()
    svcs = sorted([r[0] for r in rows if r[0]], key=lambda x: x.lower())
    return svcs

def invalidate_caches():
    """Best-effort cache invalidation that works across Streamlit versions."""
    for f in (list_categories, list_services, get_columns, table_exists, vendors_has_keywords):
        try:
            f.clear()  # type: ignore[attr-defined]
        except Exception:
            pass
    try:
        st.cache_data.clear()
    except Exception:
        pass

# -----------------------------
# Load vendors
# -----------------------------
@st.cache_data(show_spinner=False, ttl=30)
def load_vendors_df_cached(sel_cols: List[str]) -> pd.DataFrame:
    with engine.begin() as conn:
        return pd.read_sql_query(sql_text(f"SELECT {', '.join(sel_cols)} FROM vendors"), conn)

def load_vendors_df() -> pd.DataFrame:
    cols = get_columns("vendors")
    base_cols = ["id", "category", "service", "business_name", "contact_name",
                 "phone", "address", "website", "notes"]
    optional = []
    if "keywords" in cols:
        optional.append("keywords")
    sel_cols = [c for c in base_cols + optional if c in cols]
    if not sel_cols:
        return pd.DataFrame()
    df = load_vendors_df_cached(sel_cols)
    if "service" in df.columns:
        df["service"] = df["service"].apply(lambda v: v if (isinstance(v, str) and v.strip()) else "")
    if "business_name" in df.columns:
        df = df.sort_values("business_name", key=lambda s: s.str.lower()).reset_index(drop=True)
    return df

# -----------------------------
# Upserts & deletes
# -----------------------------
def insert_vendor(
    category: str,
    service: Optional[str],
    business_name: str,
    contact_name: Optional[str],
    phone: Optional[str],
    address: Optional[str],
    website: Optional[str],
    notes: Optional[str],
    keywords: Optional[str],
):
    if not category or not str(category).strip():
        raise ValueError("Category is required and cannot be blank.")
    cols = get_columns("vendors")
    has_kw = "keywords" in cols
    sql = """
        INSERT INTO vendors (category, service, business_name, contact_name, phone, address, website, notes{kwc})
        VALUES (:category, :service, :business_name, :contact_name, :phone, :address, :website, :notes{kwv})
    """.format(kwc=", keywords" if has_kw else "", kwv=", :keywords" if has_kw else "")
    params = {
        "category": category.strip(),
        "service": service or None,
        "business_name": business_name,
        "contact_name": contact_name or None,
        "phone": normalize_phone(phone),
        "address": address or None,
        "website": normalize_url(website),
        "notes": notes or None,
    }
    if has_kw:
        params["keywords"] = (keywords or None)
    with engine.begin() as conn:
        conn.execute(sql_text(sql), params)

def update_vendor(
    vid: int,
    category: str,
    service: Optional[str],
    business_name: str,
    contact_name: Optional[str],
    phone: Optional[str],
    address: Optional[str],
    website: Optional[str],
    notes: Optional[str],
    keywords: Optional[str],
):
    if not category or not str(category).strip():
        raise ValueError("Category is required and cannot be blank.")
    cols = get_columns("vendors")
    has_kw = "keywords" in cols
    sql = """
        UPDATE vendors
        SET category = :category,
            service = :service,
            business_name = :business_name,
            contact_name = :contact_name,
            phone = :phone,
            address = :address,
            website = :website,
            notes = :notes{kwset}
        WHERE id = :id
    """.format(kwset=", keywords = :keywords" if has_kw else "")
    params = {
        "id": int(vid),
        "category": category.strip(),
        "service": service or None,
        "business_name": business_name,
        "contact_name": contact_name or None,
        "phone": normalize_phone(phone),
        "address": address or None,
        "website": normalize_url(website),
        "notes": notes or None,
    }
    if has_kw:
        params["keywords"] = (keywords or None)
    with engine.begin() as conn:
        conn.execute(sql_text(sql), params)

def delete_vendor(vid: int):
    with engine.begin() as conn:
        conn.execute(sql_text("DELETE FROM vendors WHERE id = :id"), {"id": int(vid)})

# -----------------------------
# Category / Service admin
# -----------------------------
def add_category(name: str):
    if not name.strip():
        return
    if table_exists("categories") and "name" in get_columns("categories"):
        with engine.begin() as conn:
            conn.execute(sql_text("INSERT OR IGNORE INTO categories(name) VALUES(:n)"), {"n": name.strip()})

def add_service(name: str):
    if not name.strip():
        return
    if table_exists("services") and "name" in get_columns("services"):
        with engine.begin() as conn:
            conn.execute(sql_text("INSERT OR IGNORE INTO services(name) VALUES(:n)"), {"n": name.strip()})

# -----------------------------
# UI Helpers
# -----------------------------
def rerun():
    st.rerun()

def text_input_w(label: str, value: Optional[str], key: str) -> str:
    return st.text_input(label, value=value or "", key=key)

def select_category_required(label: str, value: Optional[str], key: str) -> str:
    """
    Required category selector.
    If categories list exists, force a selection from it (no blank option).
    If no categories are available (no table/rows), fall back to a required text input.
    """
    cats = list_categories()
    if cats:
        idx = 0
        if value and value in cats:
            idx = cats.index(value)
        selected = st.selectbox(label + " *", options=cats, index=idx, key=key)
        return selected.strip() if selected else ""
    else:
        typed = st.text_input(label + " * (type a category name)", value=value or "", key=key)
        return typed.strip()

def select_service_optional(label: str, value: Optional[str], key: str) -> Optional[str]:
    svcs = list_services()
    options = [""] + svcs  # "" means none/blank; display blank
    idx = 0
    if value and value in svcs:
        idx = options.index(value)
    chosen = st.selectbox(label, options=options, index=idx, key=key)
    return None if (chosen is None or str(chosen).strip() == "") else chosen

def load_vendor_by_id(vid: int) -> Optional[Dict]:
    with engine.begin() as conn:
        df = pd.read_sql_query(sql_text("SELECT * FROM vendors WHERE id = :id"), conn, params={"id": int(vid)})
    if df.empty:
        return None
    row = df.iloc[0].to_dict()
    if not row.get("service"):
        row["service"] = ""
    return row

# -----------------------------
# App Layout
# -----------------------------
st.set_page_config(page_title="Vendors Admin", layout="wide")
st.title("Vendors Admin")

with st.expander("Database Status & Schema (debug)", expanded=False):
    status = {}
    try:
        cols = get_columns("vendors")
        status["vendors_columns"] = cols
        status["has_categories_table"] = table_exists("categories")
        status["has_services_table"] = table_exists("services")
        status["has_keywords"] = vendors_has_keywords()
        with engine.begin() as conn:
            cnt = conn.execute(sql_text("SELECT COUNT(1) FROM vendors")).scalar_one()
        status["counts"] = {"vendors": int(cnt)}
    except Exception as e:
        status["error"] = str(e)
    st.json(status)

# Tabs
tab_view, tab_add, tab_edit, tab_delete, tab_cat, tab_svc = st.tabs(
    ["View", "Add", "Edit", "Delete", "Categories Admin", "Services Admin"]
)

# -----------------------------
# View Tab
# -----------------------------
with tab_view:
    st.subheader("Browse Vendors")
    df = load_vendors_df()
    desired = ["business_name", "category", "service", "contact_name",
               "phone", "address", "website", "notes", "keywords"]
    show_cols = [c for c in df.columns if c in desired]
    st.dataframe(
        df[["id"] + show_cols] if "id" in df.columns else df[show_cols],
        use_container_width=True,
        hide_index=True
    )

# -----------------------------
# Add Tab — persistent success
# -----------------------------
with tab_add:
    st.subheader("Add Vendor")
    col1, col2, col3 = st.columns(3)
    with col1:
        business_name = st.text_input("Business Name *", key="add_bn")
        contact_name  = st.text_input("Contact Name", key="add_cn")
        phone         = st.text_input("Phone", key="add_ph")
    with col2:
        address       = st.text_input("Address", key="add_addr")
        website       = st.text_input("Website (URL)", key="add_url")
        category      = select_category_required("Category", value=None, key="add_cat")
    with col3:
        service       = select_service_optional("Service (optional)", value=None, key="add_svc")
        notes         = st.text_area("Notes", key="add_notes", height=100)
        keywords_val  = st.text_input("Keywords (comma/space OK)" if vendors_has_keywords() else "Keywords (not in DB)", key="add_kw")

    colA, _ = st.columns([1,1])
    with colA:
        add_feedback = st.empty()
        add_clicked = st.button("Add Vendor", type="primary", key="btn_add_vendor")
        if add_clicked:
            errs = []
            if not business_name.strip():
                errs.append("Business Name is required.")
            if not category.strip():
                errs.append("Category is required.")
            if errs:
                for e in errs:
                    add_feedback.error(e)
            else:
                try:
                    insert_vendor(
                        category=category,
                        service=service,
                        business_name=business_name.strip(),
                        contact_name=contact_name.strip() if contact_name else None,
                        phone=phone.strip() if phone else None,
                        address=address.strip() if address else None,
                        website=website.strip() if website else None,
                        notes=notes.strip() if notes else None,
                        keywords=keywords_val.strip() if vendors_has_keywords() and keywords_val else None,
                    )
                    st.session_state["add_success_msg"] = f"Vendor successfully added: {business_name.strip()}"
                    invalidate_caches()
                    rerun()
                except Exception as ex:
                    add_feedback.error(f"Failed to add vendor: {ex}")
        if "add_success_msg" in st.session_state:
            add_feedback.success(st.session_state.pop("add_success_msg"))

# -----------------------------
# Edit Tab — persistent success
# -----------------------------
with tab_edit:
    st.subheader("Edit Vendor")
    df_all = load_vendors_df()
    if df_all.empty:
        st.info("No vendors found.")
    else:
        id_to_label = {int(r["id"]): f'{r.get("business_name","(no name)")} — #{int(r["id"])}' for _, r in df_all.iterrows()}
        sort_ids = sorted(id_to_label.keys(), key=lambda i: id_to_label[i].lower())
        chosen_id = st.selectbox("Select Vendor", options=sort_ids, format_func=lambda i: id_to_label[i], key="edit_sel")

        if chosen_id:
            row = load_vendor_by_id(int(chosen_id))
            if not row:
                st.error("Selected vendor not found.")
            else:
                col1, col2, col3 = st.columns(3)
                with col1:
                    business_name_e = text_input_w("Business Name *", row.get("business_name"), key="e_bn")
                    contact_name_e  = text_input_w("Contact Name", row.get("contact_name"), key="e_cn")
                    phone_e         = text_input_w("Phone", row.get("phone"), key="e_ph")
                with col2:
                    address_e       = text_input_w("Address", row.get("address"), key="e_addr")
                    website_e       = text_input_w("Website (URL)", row.get("website"), key="e_url")
                    category_e      = select_category_required("Category", row.get("category"), key="e_cat")
                with col3:
                    service_val = row.get("service") or ""
                    service_e   = select_service_optional("Service (optional)", value=service_val if service_val else None, key="e_svc")
                    notes_e     = st.text_area("Notes", value=row.get("notes") or "", key="e_notes", height=100)
                    if vendors_has_keywords():
                        keywords_e = text_input_w("Keywords (comma/space OK)", row.get("keywords"), key="e_kw")
                    else:
                        keywords_e = None

                save_feedback = st.empty()
                if st.button("Save Changes", type="primary", key="btn_save_vendor"):
                    errs = []
                    if not business_name_e.strip():
                        errs.append("Business Name is required.")
                    if not category_e.strip():
                        errs.append("Category is required.")
                    if errs:
                        for e in errs:
                            save_feedback.error(e)
                    else:
                        try:
                            update_vendor(
                                vid=int(chosen_id),
                                category=category_e,
                                service=service_e,
                                business_name=business_name_e.strip(),
                                contact_name=contact_name_e.strip() if contact_name_e else None,
                                phone=phone_e.strip() if phone_e else None,
                                address=address_e.strip() if address_e else None,
                                website=website_e.strip() if website_e else None,
                                notes=notes_e.strip() if notes_e else None,
                                keywords=keywords_e.strip() if (vendors_has_keywords() and keywords_e) else None,
                            )
                            st.session_state["edit_success_msg"] = "Changes successfully applied."
                            invalidate_caches()
                            rerun()
                        except Exception as ex:
                            save_feedback.error(f"Failed to save changes: {ex}")
                if "edit_success_msg" in st.session_state:
                    save_feedback.success(st.session_state.pop("edit_success_msg"))

# -----------------------------
# Delete Tab — persistent success
# -----------------------------
with tab_delete:
    st.subheader("Delete Vendor")
    df_all = load_vendors_df()
    if df_all.empty:
        st.info("No vendors to delete.")
    else:
        id_to_label = {int(r["id"]): f'{r.get("business_name","(no name)")} — #{int(r["id"])}' for _, r in df_all.iterrows()}
        sort_ids = sorted(id_to_label.keys(), key=lambda i: id_to_label[i].lower())
        del_id = st.selectbox("Select Vendor to Delete", options=sort_ids, format_func=lambda i: id_to_label[i], key="del_sel")
        del_feedback = st.empty()
        if del_id and st.button("Delete", type="secondary", key="btn_delete_vendor"):
            try:
                delete_vendor(int(del_id))
                st.session_state["delete_success_msg"] = f"Vendor successfully deleted: {id_to_label[int(del_id)]}"
                invalidate_caches()
                rerun()
            except Exception as ex:
                del_feedback.error(f"Failed to delete vendor: {ex}")
        if "delete_success_msg" in st.session_state:
            del_feedback.success(st.session_state.pop("delete_success_msg"))

# -----------------------------
# Categories Admin (no service required)
# -----------------------------
with tab_cat:
    st.subheader("Categories Admin")
    if not table_exists("categories") or "name" not in get_columns("categories"):
        st.info("Table 'categories(name)' not found. You can still assign Category text directly in vendors.")
    new_cat = st.text_input("New Category Name")
    if st.button("Add Category", key="btn_add_cat"):
        if not new_cat.strip():
            st.error("Category name required.")
        else:
            add_category(new_cat.strip())
            st.success(f"Category added/kept: {new_cat.strip()}")
            invalidate_caches()
            rerun()

    st.markdown("**Existing Categories**")
    cats = list_categories()
    if cats:
        st.write(", ".join(cats))
    else:
        st.write("(none)")

# -----------------------------
# Services Admin (standalone, optional association not enforced here)
# -----------------------------
with tab_svc:
    st.subheader("Services Admin")
    if not table_exists("services") or "name" not in get_columns("services"):
        st.info("Table 'services(name)' not found. You can still leave Service blank for vendors, or type free-form in vendors.service if you later add the column.")

    new_svc = st.text_input("New Service Name")
    if st.button("Add Service", key="btn_add_svc"):
        if not new_svc.strip():
            st.error("Service name required.")
        else:
            add_service(new_svc.strip())
            st.success(f"Service added/kept: {new_svc.strip()}")
            invalidate_caches()
            rerun()

    st.markdown("**Existing Services**")
    svcs = list_services()
    if svcs:
        st.write(", ".join(svcs))
    else:
        st.write("(none)")
