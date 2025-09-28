# app_admin.py
# Streamlit Admin for Vendors DB with intuitive Add/Edit and live-populated Category/Service dropdowns.
# Compatible with:
#   (A) Normalized schema: vendors(category_id, service_id) + categories(id,name), services(id,name)
#   (B) Denormalized schema: vendors(category TEXT, service TEXT)
#
# Safe SQL, no fragile globals, no syntax errors. Inline "Add new ..." for Category/Service.

import os
import sqlite3
from contextlib import closing
from typing import List, Tuple, Dict, Optional

import pandas as pd
import streamlit as st

# ---------- Configuration ----------
DB_PATH = os.environ.get("VENDORS_DB_PATH", "vendors.db")  # override via env if needed

st.set_page_config(page_title="Vendors Admin", layout="wide")

# ---------- Utility: DB connection ----------
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def table_exists(conn, name: str) -> bool:
    with closing(conn.cursor()) as c:
        c.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1;", (name,)
        )
        return c.fetchone() is not None

def get_table_columns(conn, table: str) -> List[str]:
    with closing(conn.cursor()) as c:
        c.execute(f"PRAGMA table_info({table});")
        rows = c.fetchall()
    return [row["name"] for row in rows]

# ---------- Schema detection ----------
@st.cache_data(show_spinner=False)
def detect_schema():
    with get_conn() as conn:
        if not table_exists(conn, "vendors"):
            raise RuntimeError("Table 'vendors' not found. Verify your database.")

        vcols = get_table_columns(conn, "vendors")
        has_cat_tbl = table_exists(conn, "categories")
        has_svc_tbl = table_exists(conn, "services")

        schema = {
            "vendors_columns": vcols,
            "has_categories_table": has_cat_tbl,
            "has_services_table": has_svc_tbl,
            "uses_cat_id": "category_id" in vcols,
            "uses_svc_id": "service_id" in vcols,
            "uses_cat_text": "category" in vcols,
            "uses_svc_text": "service" in vcols,
        }

        # Basic sanity checks
        if not (schema["uses_cat_id"] or schema["uses_cat_text"]):
            st.warning(
                "No 'category_id' or 'category' column found on vendors. Category will be unavailable."
            )
        if not (schema["uses_svc_id"] or schema["uses_svc_text"]):
            st.warning(
                "No 'service_id' or 'service' column found on vendors. Service will be unavailable."
            )

        return schema

SCHEMA = detect_schema()

# ---------- Catalog loaders ----------
def _fetchall(conn, sql: str, params: Tuple = ()) -> List[sqlite3.Row]:
    with closing(conn.cursor()) as c:
        c.execute(sql, params)
        return c.fetchall()

@st.cache_data(show_spinner=False)
def get_categories() -> List[str]:
    with get_conn() as conn:
        if SCHEMA["has_categories_table"]:
            rows = _fetchall(conn, "SELECT name FROM categories ORDER BY name;")
            return [r["name"] for r in rows]
        # Fallback to distinct from vendors (denormalized)
        if SCHEMA["uses_cat_text"]:
            rows = _fetchall(
                conn,
                "SELECT DISTINCT category AS name FROM vendors WHERE category IS NOT NULL AND TRIM(category) <> '' ORDER BY 1;"
            )
            return [r["name"] for r in rows]
        # If normalized columns exist but table missing (edge), return empty list
        return []

@st.cache_data(show_spinner=False)
def get_services() -> List[str]:
    with get_conn() as conn:
        if SCHEMA["has_services_table"]:
            rows = _fetchall(conn, "SELECT name FROM services ORDER BY name;")
            return [r["name"] for r in rows]
        # Fallback to distinct from vendors (denormalized)
        if SCHEMA["uses_svc_text"]:
            rows = _fetchall(
                conn,
                "SELECT DISTINCT service AS name FROM vendors WHERE service IS NOT NULL AND TRIM(service) <> '' ORDER BY 1;"
            )
            return [r["name"] for r in rows]
        return []

def get_category_id(conn, name: str) -> Optional[int]:
    if not SCHEMA["has_categories_table"]:
        return None
    with closing(conn.cursor()) as c:
        c.execute("SELECT id FROM categories WHERE name = ?;", (name.strip(),))
        row = c.fetchone()
        return row["id"] if row else None

def get_service_id(conn, name: str) -> Optional[int]:
    if not SCHEMA["has_services_table"]:
        return None
    with closing(conn.cursor()) as c:
        c.execute("SELECT id FROM services WHERE name = ?;", (name.strip(),))
        row = c.fetchone()
        return row["id"] if row else None

def get_or_create_category(conn, name: str) -> Optional[int]:
    if name is None or not name.strip():
        return None
    if not SCHEMA["has_categories_table"]:
        return None  # denormalized path uses text directly
    cid = get_category_id(conn, name)
    if cid is not None:
        return cid
    with closing(conn.cursor()) as c:
        c.execute("INSERT INTO categories(name) VALUES(?);", (name.strip(),))
        conn.commit()
        return c.lastrowid

def get_or_create_service(conn, name: str) -> Optional[int]:
    if name is None or not name.strip():
        return None
    if not SCHEMA["has_services_table"]:
        return None
    sid = get_service_id(conn, name)
    if sid is not None:
        return sid
    with closing(conn.cursor()) as c:
        c.execute("INSERT INTO services(name) VALUES(?);", (name.strip(),))
        conn.commit()
        return c.lastrowid

# ---------- Vendor data helpers ----------
def load_vendors_df() -> pd.DataFrame:
    with get_conn() as conn:
        vcols = SCHEMA["vendors_columns"]
        # Decide select expression to get Category and Service as NAMES for display
        if SCHEMA["uses_cat_id"] and SCHEMA["has_categories_table"]:
            cat_expr = "categories.name AS Category"
            cat_join = "LEFT JOIN categories ON categories.id = vendors.category_id"
        elif SCHEMA["uses_cat_text"]:
            cat_expr = "vendors.category AS Category"
            cat_join = ""
        else:
            cat_expr = "NULL AS Category"
            cat_join = ""

        if SCHEMA["uses_svc_id"] and SCHEMA["has_services_table"]:
            svc_expr = "services.name AS Service"
            svc_join = "LEFT JOIN services ON services.id = vendors.service_id"
        elif SCHEMA["uses_svc_text"]:
            svc_expr = "vendors.service AS Service"
            svc_join = ""
        else:
            svc_expr = "NULL AS Service"
            svc_join = ""

        select_cols = [
            "vendors.id AS id",
            cat_expr,
            svc_expr,
            "vendors.business_name AS Business_Name",
            "vendors.contact_name AS Contact_Name",
            "vendors.phone AS Phone",
            "vendors.address AS Address",
            "vendors.notes AS Notes",
            "vendors.website AS Website",
        ]
        sql = f"""
            SELECT {", ".join(select_cols)}
            FROM vendors
            {cat_join}
            {svc_join}
            ORDER BY Business_Name COLLATE NOCASE ASC;
        """
        df = pd.read_sql_query(sql, conn)
        # For display, harmonize column names with spaces
        df = df.rename(
            columns={
                "Business_Name": "Business Name",
                "Contact_Name": "Contact Name",
            }
        )
        return df

def vendor_display_label(row: pd.Series) -> str:
    parts = [
        f"{row.get('Business Name') or '(no name)'}",
        f"— {row.get('Category') or 'Uncategorized'}",
        f"/ {row.get('Service') or 'Unspecified'}",
        f"(id={row.get('id')})",
    ]
    return " ".join(parts)

# ---------- UI components for category/service with inline add ----------
ADD_NEW_CAT = "➕ Add new category…"
ADD_NEW_SVC = "➕ Add new service…"

def category_selector(key_prefix: str, default_name: Optional[str]) -> str:
    cats = get_categories()
    options = cats.copy()
    options.insert(0, "")  # allow blank
    options.append(ADD_NEW_CAT)
    sel = st.selectbox(
        "Category",
        options=options,
        index=(options.index(default_name) if default_name in options else 0),
        key=f"{key_prefix}_cat_select",
    )
    if sel == ADD_NEW_CAT:
        new_val = st.text_input(
            "New category name", value="", key=f"{key_prefix}_cat_new"
        ).strip()
        return new_val
    return sel.strip() if sel else ""

def service_selector(key_prefix: str, default_name: Optional[str]) -> str:
    svcs = get_services()
    options = svcs.copy()
    options.insert(0, "")  # allow blank
    options.append(ADD_NEW_SVC)
    sel = st.selectbox(
        "Service",
        options=options,
        index=(options.index(default_name) if default_name in options else 0),
        key=f"{key_prefix}_svc_select",
    )
    if sel == ADD_NEW_SVC:
        new_val = st.text_input(
            "New service name", value="", key=f"{key_prefix}_svc_new"
        ).strip()
        return new_val
    return sel.strip() if sel else ""

# ---------- CRUD operations ----------
def insert_vendor(payload: Dict):
    with get_conn() as conn, closing(conn.cursor()) as c:
        # Resolve category/service per schema
        cat_name = (payload.get("Category") or "").strip()
        svc_name = (payload.get("Service") or "").strip()

        if SCHEMA["uses_cat_id"] and SCHEMA["has_categories_table"]:
            cat_id = get_or_create_category(conn, cat_name) if cat_name else None
        else:
            cat_id = None

        if SCHEMA["uses_svc_id"] and SCHEMA["has_services_table"]:
            svc_id = get_or_create_service(conn, svc_name) if svc_name else None
        else:
            svc_id = None

        cols = []
        vals = []
        params = []

        # category/service mapping
        if SCHEMA["uses_cat_id"] and SCHEMA["has_categories_table"]:
            cols.append("category_id"); vals.append("?"); params.append(cat_id)
        elif SCHEMA["uses_cat_text"]:
            cols.append("category"); vals.append("?"); params.append(cat_name or None)
        # else: no-op if neither exists

        if SCHEMA["uses_svc_id"] and SCHEMA["has_services_table"]:
            cols.append("service_id"); vals.append("?"); params.append(svc_id)
        elif SCHEMA["uses_svc_text"]:
            cols.append("service"); vals.append("?"); params.append(svc_name or None)

        # other fields
        for k_src, k_db in [
            ("Business Name", "business_name"),
            ("Contact Name", "contact_name"),
            ("Phone", "phone"),
            ("Address", "address"),
            ("Notes", "notes"),
            ("Website", "website"),
        ]:
            cols.append(k_db); vals.append("?"); params.append(payload.get(k_src) or None)

        sql = f"INSERT INTO vendors ({', '.join(cols)}) VALUES ({', '.join(vals)});"
        c.execute(sql, tuple(params))
        conn.commit()
        return c.lastrowid

def update_vendor(vendor_id: int, payload: Dict):
    with get_conn() as conn, closing(conn.cursor()) as c:
        sets = []
        params = []

        cat_name = (payload.get("Category") or "").strip()
        svc_name = (payload.get("Service") or "").strip()

        if SCHEMA["uses_cat_id"] and SCHEMA["has_categories_table"]:
            cat_id = get_or_create_category(conn, cat_name) if cat_name else None
            sets.append("category_id = ?"); params.append(cat_id)
        elif SCHEMA["uses_cat_text"]:
            sets.append("category = ?"); params.append(cat_name or None)

        if SCHEMA["uses_svc_id"] and SCHEMA["has_services_table"]:
            svc_id = get_or_create_service(conn, svc_name) if svc_name else None
            sets.append("service_id = ?"); params.append(svc_id)
        elif SCHEMA["uses_svc_text"]:
            sets.append("service = ?"); params.append(svc_name or None)

        for k_src, k_db in [
            ("Business Name", "business_name"),
            ("Contact Name", "contact_name"),
            ("Phone", "phone"),
            ("Address", "address"),
            ("Notes", "notes"),
            ("Website", "website"),
        ]:
            sets.append(f"{k_db} = ?")
            params.append(payload.get(k_src) or None)

        params.append(vendor_id)
        sql = f"UPDATE vendors SET {', '.join(sets)} WHERE id = ?;"
        c.execute(sql, tuple(params))
        conn.commit()

def delete_vendor(vendor_id: int) -> int:
    with get_conn() as conn, closing(conn.cursor()) as c:
        c.execute("DELETE FROM vendors WHERE id = ?;", (vendor_id,))
        conn.commit()
        return c.rowcount

# ---------- Admin functions for categories/services ----------
def rename_category(old_name: str, new_name: str) -> None:
    new_name = new_name.strip()
    with get_conn() as conn, closing(conn.cursor()) as c:
        if SCHEMA["has_categories_table"] and SCHEMA["uses_cat_id"]:
            # Update lookup; unique constraint may exist
            c.execute("UPDATE OR IGNORE categories SET name = ? WHERE name = ?;", (new_name, old_name))
            conn.commit()
        elif SCHEMA["uses_cat_text"]:
            c.execute("UPDATE vendors SET category = ? WHERE category = ?;", (new_name, old_name))
            conn.commit()

def rename_service(old_name: str, new_name: str) -> None:
    new_name = new_name.strip()
    with get_conn() as conn, closing(conn.cursor()) as c:
        if SCHEMA["has_services_table"] and SCHEMA["uses_svc_id"]:
            c.execute("UPDATE OR IGNORE services SET name = ? WHERE name = ?;", (new_name, old_name))
            conn.commit()
        elif SCHEMA["uses_svc_text"]:
            c.execute("UPDATE vendors SET service = ? WHERE service = ?;", (new_name, old_name))
            conn.commit()

def delete_category(name: str) -> Tuple[bool, str]:
    with get_conn() as conn, closing(conn.cursor()) as c:
        if SCHEMA["has_categories_table"] and SCHEMA["uses_cat_id"]:
            # Check usage
            c.execute(
                "SELECT COUNT(*) AS n FROM vendors v JOIN categories c ON v.category_id=c.id WHERE c.name = ?;",
                (name,),
            )
            in_use = c.fetchone()["n"]
            if in_use:
                return False, f"Cannot delete '{name}': it is in use by {in_use} vendor(s)."
            c.execute("DELETE FROM categories WHERE name = ?;", (name,))
            conn.commit()
            return True, f"Deleted category '{name}'."
        elif SCHEMA["uses_cat_text"]:
            # If denormalized, check in use
            c.execute("SELECT COUNT(*) AS n FROM vendors WHERE category = ?;", (name,))
            in_use = c.fetchone()["n"]
            if in_use:
                return False, f"Cannot delete '{name}': it is in use by {in_use} vendor(s)."
            # No separate table to delete from
            return True, f"'{name}' removed from catalog view (it may still appear if vendors reference it)."
        return False, "Category storage not found."

def delete_service(name: str) -> Tuple[bool, str]:
    with get_conn() as conn, closing(conn.cursor()) as c:
        if SCHEMA["has_services_table"] and SCHEMA["uses_svc_id"]:
            c.execute(
                "SELECT COUNT(*) AS n FROM vendors v JOIN services s ON v.service_id=s.id WHERE s.name = ?;",
                (name,),
            )
            in_use = c.fetchone()["n"]
            if in_use:
                return False, f"Cannot delete '{name}': it is in use by {in_use} vendor(s)."
            c.execute("DELETE FROM services WHERE name = ?;", (name,))
            conn.commit()
            return True, f"Deleted service '{name}'."
        elif SCHEMA["uses_svc_text"]:
            c.execute("SELECT COUNT(*) AS n FROM vendors WHERE service = ?;", (name,))
            in_use = c.fetchone()["n"]
            if in_use:
                return False, f"Cannot delete '{name}': it is in use by {in_use} vendor(s)."
            return True, f"'{name}' removed from catalog view (it may still appear if vendors reference it)."
        return False, "Service storage not found."

# ---------- UI ----------
st.title("Vendors Admin")

with st.sidebar:
    st.subheader("Navigation")
    page = st.radio(
        "Go to",
        ["View", "Add", "Edit", "Delete", "Categories & Services Admin"],
        index=0,
    )

# Cache invalidation helper
def invalidate_catalog_caches():
    get_categories.clear()
    get_services.clear()
    load_vendors_df.clear()

# ---------- Pages ----------
if page == "View":
    st.header("All Vendors")
    df = load_vendors_df()
    st.dataframe(df.drop(columns=["id"]), use_container_width=True, hide_index=True)

elif page == "Add":
    st.header("Add Vendor")

    with st.form("add_vendor_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        with col1:
            business_name = st.text_input("Business Name", key="add_bname").strip()
            contact_name = st.text_input("Contact Name", key="add_cname").strip()
            phone = st.text_input("Phone", key="add_phone").strip()
            category_name = category_selector("add", default_name=None)
        with col2:
            address = st.text_area("Address", key="add_addr").strip()
            notes = st.text_area("Notes", key="add_notes").strip()
            website = st.text_input("Website (URL)", key="add_site").strip()
            service_name = service_selector("add", default_name=None)

        submitted = st.form_submit_button("Add Vendor")
        if submitted:
            if not business_name:
                st.error("Business Name is required.")
            else:
                payload = {
                    "Business Name": business_name or None,
                    "Contact Name": contact_name or None,
                    "Phone": phone or None,
                    "Address": address or None,
                    "Notes": notes or None,
                    "Website": website or None,
                    "Category": category_name or None,
                    "Service": service_name or None,
                }
                try:
                    new_id = insert_vendor(payload)
                    invalidate_catalog_caches()
                    st.success(f"Vendor added (id={new_id}).")
                except Exception as e:
                    st.error(f"Failed to add vendor: {e}")

elif page == "Edit":
    st.header("Edit Vendor")

    df = load_vendors_df()
    if df.empty:
        st.info("No vendors found.")
    else:
        # Build selection labels
        labels = df.apply(vendor_display_label, axis=1).tolist()
        id_by_label = {labels[i]: int(df.iloc[i]["id"]) for i in range(len(labels))}

        chosen_label = st.selectbox("Select Vendor", options=labels, key="edit_vendor_select")
        vid = id_by_label[chosen_label]

        row = df.loc[df["id"] == vid].iloc[0]

        with st.form("edit_vendor_form"):
            col1, col2 = st.columns(2)
            with col1:
                business_name = st.text_input("Business Name", value=row["Business Name"] or "", key="edit_bname").strip()
                contact_name = st.text_input("Contact Name", value=row["Contact Name"] or "", key="edit_cname").strip()
                phone = st.text_input("Phone", value=row["Phone"] or "", key="edit_phone").strip()
                category_name = category_selector("edit", default_name=row["Category"])
            with col2:
                address = st.text_area("Address", value=row["Address"] or "", key="edit_addr").strip()
                notes = st.text_area("Notes", value=row["Notes"] or "", key="edit_notes").strip()
                website = st.text_input("Website (URL)", value=row["Website"] or "", key="edit_site").strip()
                service_name = service_selector("edit", default_name=row["Service"])

            save = st.form_submit_button("Save Changes")
            if save:
                if not business_name:
                    st.error("Business Name is required.")
                else:
                    payload = {
                        "Business Name": business_name or None,
                        "Contact Name": contact_name or None,
                        "Phone": phone or None,
                        "Address": address or None,
                        "Notes": notes or None,
                        "Website": website or None,
                        "Category": category_name or None,
                        "Service": service_name or None,
                    }
                    try:
                        update_vendor(int(vid), payload)
                        invalidate_catalog_caches()
                        st.success("Vendor updated.")
                    except Exception as e:
                        st.error(f"Failed to update vendor: {e}")

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
                n = delete_vendor(int(vid))
                invalidate_catalog_caches()
                if n:
                    st.success("Vendor deleted.")
                else:
                    st.info("Nothing deleted (vendor may not exist).")
            except Exception as e:
                st.error(f"Failed to delete vendor: {e}")

elif page == "Categories & Services Admin":
    st.header("Categories & Services Admin")

    tab_cat, tab_svc = st.tabs(["Categories", "Services"])

    with tab_cat:
        st.subheader("Manage Categories")
        cats = get_categories()
        st.write("Existing categories:", cats)

        with st.expander("Add Category"):
            new_cat = st.text_input("New category name", key="admin_cat_new").strip()
            if st.button("Add Category", key="admin_cat_add_btn"):
                if not new_cat:
                    st.error("Enter a category name.")
                else:
                    try:
                        with get_conn() as conn:
                            if SCHEMA["has_categories_table"]:
                                get_or_create_category(conn, new_cat)
                            else:
                                st.info("No categories table; add via vendor form using denormalized path.")
                        invalidate_catalog_caches()
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
                            invalidate_catalog_caches()
                            st.success(f"Renamed '{old}' to '{new}'.")
                        except Exception as e:
                            st.error(f"Failed to rename category: {e}")
            else:
                st.info("No categories found.")

        with st.expander("Delete Category"):
            if cats:
                victim = st.selectbox("Select category to delete", options=cats, key="admin_cat_del_sel")
                if st.button("Delete Category", key="admin_cat_del_btn"):
                    ok, msg = delete_category(victim)
                    if ok:
                        invalidate_catalog_caches()
                        st.success(msg)
                    else:
                        st.error(msg)
            else:
                st.info("No categories found.")

    with tab_svc:
        st.subheader("Manage Services")
        svcs = get_services()
        st.write("Existing services:", svcs)

        with st.expander("Add Service"):
            new_svc = st.text_input("New service name", key="admin_svc_new").strip()
            if st.button("Add Service", key="admin_svc_add_btn"):
                if not new_svc:
                    st.error("Enter a service name.")
                else:
                    try:
                        with get_conn() as conn:
                            if SCHEMA["has_services_table"]:
                                get_or_create_service(conn, new_svc)
                            else:
                                st.info("No services table; add via vendor form using denormalized path.")
                        invalidate_catalog_caches()
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
                            invalidate_catalog_caches()
                            st.success(f"Renamed '{old}' to '{new}'.")
                        except Exception as e:
                            st.error(f"Failed to rename service: {e}")
            else:
                st.info("No services found.")

        with st.expander("Delete Service"):
            if svcs:
                victim = st.selectbox("Select service to delete", options=svcs, key="admin_svc_del_sel")
                if st.button("Delete Service", key="admin_svc_del_btn"):
                    ok, msg = delete_service(victim)
                    if ok:
                        invalidate_catalog_caches()
                        st.success(msg)
                    else:
                        st.error(msg)
            else:
                st.info("No services found.")
