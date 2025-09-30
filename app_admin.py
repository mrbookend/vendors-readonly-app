# app_admin.py — Vendors Admin (SQLite) with phone normalization
# Features:
# - Enforce/normalize phone format to (xxx) xxx-xxxx on add/edit. Accepts raw digits (e.g., 2108333141 or 2108333141.0) and formats.
# - "Add Vendor" uses Category -> Service cascade. Services are filtered by selected Category.
# - Dropdowns have a clear placeholder ("— select —"). A special "+ Add New…" entry lets you add a new Category/Service inline.
# - "Vendors Admin" includes Add / Edit / Delete flows. Business Name picker sorted A→Z.
# - "Categories & Services Admin" page to add new Categories and Services (creates tables if missing).
# - Works if you only have a vendors table; if categories/services tables exist, they are used; otherwise distinct values from vendors are used for picks.

from __future__ import annotations

import os
import re
import sqlite3
from contextlib import closing
from typing import List, Optional, Tuple

import pandas as pd
import streamlit as st

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
DB_PATH = os.getenv("VENDORS_DB", "/mount/src/vendors-readonly-app/vendors.db")
PLACEHOLDER = "— select —"
ADD_NEW = "+ Add New…"
REQUIRED_COLUMNS = [
    "id",
    "category",
    "service",
    "business_name",
    "contact_name",
    "phone",
    "address",
    "website",
    "notes",
    "keywords",
]

# -----------------------------------------------------------------------------
# Database helpers
# -----------------------------------------------------------------------------

def conn() -> sqlite3.Connection:
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def table_exists(c: sqlite3.Connection, table: str) -> bool:
    cur = c.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
    )
    return cur.fetchone() is not None


def ensure_aux_tables(c: sqlite3.Connection) -> None:
    # Create if missing
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS categories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL
        )
        """
    )
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS services (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category TEXT NOT NULL,
            service TEXT NOT NULL,
            UNIQUE(category, service)
        )
        """
    )
    c.commit()


# -----------------------------------------------------------------------------
# Data access
# -----------------------------------------------------------------------------

def get_vendors_df() -> pd.DataFrame:
    with closing(conn()) as c:
        df = pd.read_sql_query("SELECT * FROM vendors", c)
    # Ensure all required columns exist
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df


def get_categories() -> List[str]:
    with closing(conn()) as c:
        if table_exists(c, "categories"):
            rows = c.execute("SELECT name FROM categories ORDER BY name").fetchall()
            return [r[0] for r in rows]
        else:
            rows = c.execute(
                "SELECT DISTINCT category FROM vendors WHERE IFNULL(category,'')<>'' ORDER BY category"
            ).fetchall()
            return [r[0] for r in rows]


def get_services_for_category(category: str) -> List[str]:
    if not category:
        return []
    with closing(conn()) as c:
        if table_exists(c, "services"):
            rows = c.execute(
                "SELECT service FROM services WHERE category=? ORDER BY service",
                (category,),
            ).fetchall()
            return [r[0] for r in rows]
        else:
            rows = c.execute(
                """
                SELECT DISTINCT service FROM vendors
                WHERE IFNULL(service,'')<>'' AND category=?
                ORDER BY service
                """,
                (category,),
            ).fetchall()
            return [r[0] for r in rows]


def upsert_category(name: str) -> None:
    with closing(conn()) as c:
        ensure_aux_tables(c)
        c.execute("INSERT OR IGNORE INTO categories(name) VALUES(?)", (name.strip(),))
        c.commit()


def upsert_service(category: str, service: str) -> None:
    with closing(conn()) as c:
        ensure_aux_tables(c)
        c.execute(
            "INSERT OR IGNORE INTO services(category, service) VALUES(?,?)",
            (category.strip(), service.strip()),
        )
        c.commit()


def insert_vendor(rec: dict) -> int:
    with closing(conn()) as c:
        cur = c.execute(
            """
            INSERT INTO vendors (category, service, business_name, contact_name, phone, address, website, notes, keywords)
            VALUES (:category, :service, :business_name, :contact_name, :phone, :address, :website, :notes, :keywords)
            """,
            rec,
        )
        c.commit()
        return cur.lastrowid


def update_vendor(vid: int, rec: dict) -> None:
    with closing(conn()) as c:
        c.execute(
            """
            UPDATE vendors
            SET category=:category,
                service=:service,
                business_name=:business_name,
                contact_name=:contact_name,
                phone=:phone,
                address=:address,
                website=:website,
                notes=:notes,
                keywords=:keywords
            WHERE id=:id
            """,
            {**rec, "id": vid},
        )
        c.commit()


def delete_vendor(vid: int) -> None:
    with closing(conn()) as c:
        c.execute("DELETE FROM vendors WHERE id=?", (vid,))
        c.commit()


# -----------------------------------------------------------------------------
# Validation / Normalization
# -----------------------------------------------------------------------------

def normalize_phone(raw: str) -> str:
    """Return phone as (xxx) xxx-xxxx or '' if blank. Accepts 10 digits only.
    Handles float-looking inputs like '2108333141.0'.
    """
    if raw is None:
        return ""
    s = str(raw).strip()
    if not s:
        return ""
    # Strip all non-digits
    digits = re.sub(r"\D", "", s)
    # If it came from a float string like '2108333141.0', strip trailing .0 artifacts
    digits = digits.rstrip("0") if s.endswith(".0") and len(digits) > 10 else digits
    if len(digits) == 11 and digits.startswith("1"):
        digits = digits[1:]
    if len(digits) != 10:
        raise ValueError("Phone must be 10 digits or left blank")
    area, mid, last = digits[:3], digits[3:6], digits[6:]
    return f"({area}) {mid}-{last}"


def require(text: str, label: str) -> None:
    if not text or not text.strip():
        raise ValueError(f"{label} is required")


# -----------------------------------------------------------------------------
# UI Helpers
# -----------------------------------------------------------------------------

def header(title: str):
    st.markdown(f"## {title}")


def select_with_add(label: str, options: List[str], key: str) -> Tuple[str, Optional[str]]:
    """Show a selectbox with PLACEHOLDER and ADD_NEW option.
    Returns (value, new_value) where new_value is non-empty if user chose ADD_NEW and provided a value.
    """
    opts = [PLACEHOLDER] + options + [ADD_NEW]
    choice = st.selectbox(label, opts, index=0, key=key)
    new_val = None
    if choice == ADD_NEW:
        new_val = st.text_input(f"New {label}", key=f"{key}_new")
    return ("" if choice in (PLACEHOLDER, ADD_NEW) else choice, new_val)


# -----------------------------------------------------------------------------
# Main App
# -----------------------------------------------------------------------------

def page_vendors_admin():
    header("Vendors Admin")

    df = get_vendors_df()
    with st.expander("Database Status & Schema (debug)", expanded=False):
        st.json({
            "db_source": f"(SQLite) {DB_PATH}",
            "vendors_columns": list(df.columns),
            "has_categories_table": table_exists(conn(), "categories"),
            "has_services_table": table_exists(conn(), "services"),
            "uses_cat_text": True,
            "uses_svc_text": True,
            "has_keywords": "keywords" in df.columns,
            "counts": {"vendors": int(df.shape[0])},
        })

    tab_add, tab_edit, tab_delete = st.tabs(["Add", "Edit", "Delete"])

    # --------------------------- Add ----------------------------------------
    with tab_add:
        st.subheader("Add a Vendor")
        cats = get_categories()
        cat_val, cat_new = select_with_add("Category", cats, key="add_cat")
        if cat_new:
            try:
                require(cat_new, "Category")
                upsert_category(cat_new)
                cat_val = cat_new
                st.success(f"Category '{cat_new}' added.")
            except Exception as e:
                st.error(str(e))
        # Services depend on final category value
        svcs = get_services_for_category(cat_val) if cat_val else []
        svc_val, svc_new = select_with_add("Service", svcs, key="add_svc")
        if svc_new:
            try:
                require(cat_val or cat_new, "Category (before adding a Service)")
                require(svc_new, "Service")
                upsert_service(cat_val, svc_new)
                svc_val = svc_new
                st.success(f"Service '{svc_new}' added to '{cat_val}'.")
            except Exception as e:
                st.error(str(e))

        business_name = st.text_input("Business Name", key="add_biz")
        contact_name = st.text_input("Contact Name", key="add_contact")
        phone_raw = st.text_input("Phone (digits only or (xxx) xxx-xxxx)", key="add_phone")
        address = st.text_input("Address", key="add_addr")
        website = st.text_input("Website (URL)", key="add_site")
        notes = st.text_area("Notes", key="add_notes")
        keywords = st.text_input("Keywords (comma- or space-separated)", key="add_kw")

        if st.button("Add Vendor", type="primary"):
            try:
                require(cat_val, "Category")
                require(svc_val, "Service")
                require(business_name, "Business Name")
                phone = normalize_phone(phone_raw)
                rec = {
                    "category": cat_val,
                    "service": svc_val,
                    "business_name": business_name.strip(),
                    "contact_name": contact_name.strip(),
                    "phone": phone,
                    "address": address.strip(),
                    "website": website.strip(),
                    "notes": notes.strip(),
                    "keywords": re.sub(r"\s*,\s*", ", ", re.sub(r"\s+", ", ", keywords.strip())).strip(", ") if keywords else "",
                }
                vid = insert_vendor(rec)
                st.success(f"Added vendor #{vid} — {business_name}")
            except Exception as e:
                st.error(str(e))

    # --------------------------- Edit ---------------------------------------
    with tab_edit:
        st.subheader("Edit a Vendor")
        if df.empty:
            st.info("No vendors to edit.")
        else:
            df_sorted = df.sort_values("business_name", key=lambda s: s.str.lower())
            options = [(int(r["id"]), f"{r['business_name']} — {r['category']} / {r['service']}") for _, r in df_sorted.iterrows()]
            id_to_label = {vid: label for vid, label in options}
            vid = st.selectbox("Select Vendor", options=[vid for vid, _ in options], format_func=lambda i: id_to_label[i], key="edit_pick")

            row = df[df["id"] == vid].iloc[0]
            cats = get_categories()
            cat_val, cat_new = select_with_add("Category", cats, key="edit_cat")
            if not cat_val:
                cat_val = str(row["category"]) if row["category"] else ""
            if cat_new:
                try:
                    require(cat_new, "Category")
                    upsert_category(cat_new)
                    cat_val = cat_new
                    st.success(f"Category '{cat_new}' added.")
                except Exception as e:
                    st.error(str(e))

            svcs = get_services_for_category(cat_val) if cat_val else []
            svc_val, svc_new = select_with_add("Service", svcs, key="edit_svc")
            if not svc_val:
                svc_val = str(row["service"]) if row["service"] else ""
            if svc_new:
                try:
                    require(cat_val, "Category (before adding a Service)")
                    require(svc_new, "Service")
                    upsert_service(cat_val, svc_new)
                    svc_val = svc_new
                    st.success(f"Service '{svc_new}' added to '{cat_val}'.")
                except Exception as e:
                    st.error(str(e))

            business_name = st.text_input("Business Name", value=str(row["business_name"] or ""), key="edit_biz")
            contact_name = st.text_input("Contact Name", value=str(row["contact_name"] or ""), key="edit_contact")
            phone_raw = st.text_input("Phone (digits only or (xxx) xxx-xxxx)", value=str(row["phone"] or ""), key="edit_phone")
            address = st.text_input("Address", value=str(row["address"] or ""), key="edit_addr")
            website = st.text_input("Website (URL)", value=str(row["website"] or ""), key="edit_site")
            notes = st.text_area("Notes", value=str(row["notes"] or ""), key="edit_notes")
            keywords = st.text_input("Keywords (comma- or space-separated)", value=str(row["keywords"] or ""), key="edit_kw")

            if st.button("Save Changes", type="primary"):
                try:
                    require(cat_val, "Category")
                    require(svc_val, "Service")
                    require(business_name, "Business Name")
                    phone = normalize_phone(phone_raw)
                    rec = {
                        "category": cat_val,
                        "service": svc_val,
                        "business_name": business_name.strip(),
                        "contact_name": contact_name.strip(),
                        "phone": phone,
                        "address": address.strip(),
                        "website": website.strip(),
                        "notes": notes.strip(),
                        "keywords": re.sub(r"\s*,\s*", ", ", re.sub(r"\s+", ", ", keywords.strip())).strip(", ") if keywords else "",
                    }
                    update_vendor(int(vid), rec)
                    st.success(f"Updated vendor #{vid} — {business_name}")
                except Exception as e:
                    st.error(str(e))

    # --------------------------- Delete -------------------------------------
    with tab_delete:
        st.subheader("Delete a Vendor")
        if df.empty:
            st.info("No vendors to delete.")
        else:
            df_sorted = df.sort_values("business_name", key=lambda s: s.str.lower())
            options = [(int(r["id"]), f"{r['business_name']} — {r['category']} / {r['service']}") for _, r in df_sorted.iterrows()]
            id_to_label = {vid: label for vid, label in options}
            vid = st.selectbox("Select Vendor", options=[vid for vid, _ in options], format_func=lambda i: id_to_label[i], key="del_pick")
            if st.button("Delete", type="secondary"):
                delete_vendor(int(vid))
                st.success(f"Deleted vendor #{vid}")


def page_cat_svc_admin():
    header("Categories & Services Admin")
    st.caption("Add new categories or services here. These feed the dropdowns in Vendors Admin.")

    with closing(conn()) as c:
        ensure_aux_tables(c)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Add Category")
        new_cat = st.text_input("Category name", key="cs_new_cat")
        if st.button("Add Category"):
            try:
                require(new_cat, "Category")
                upsert_category(new_cat)
                st.success(f"Category '{new_cat}' added.")
            except Exception as e:
                st.error(str(e))

    with col2:
        st.subheader("Add Service")
        cats = get_categories()
        cat_val = st.selectbox("Category", [PLACEHOLDER] + cats, index=0, key="cs_cat_sel")
        new_svc = st.text_input("Service name", key="cs_new_svc")
        if st.button("Add Service"):
            try:
                require(cat_val if cat_val != PLACEHOLDER else "", "Category")
                require(new_svc, "Service")
                upsert_service(cat_val, new_svc)
                st.success(f"Service '{new_svc}' added to '{cat_val}'.")
            except Exception as e:
                st.error(str(e))

    st.divider()
    st.subheader("Reference Lists")
    st.write("Categories")
    st.dataframe(pd.DataFrame({"Category": get_categories()}), hide_index=True, use_container_width=True)

    st.write("Services (by Category)")
    rows = []
    for cat in get_categories():
        for svc in get_services_for_category(cat):
            rows.append({"Category": cat, "Service": svc})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


# -----------------------------------------------------------------------------
# Router
# -----------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="Vendors Admin", layout="wide")
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ("Vendors Admin", "Categories & Services Admin"))
    if page == "Vendors Admin":
        page_vendors_admin()
    else:
        page_cat_svc_admin()


if __name__ == "__main__":
    main()
