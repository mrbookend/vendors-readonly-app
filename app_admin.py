"""
Admin UI for the Vendors directory (normalized schema preferred).

Goal: Provide a *full* admin capable of:
- Always listing vendors sorted ascending by Category → Service → Business Name.
- Add / Edit / Delete Vendors.
- Add new Categories and Services on the fly.
- Toggle Active status (soft delete) and hard delete.
- Works with normalized schema; falls back to legacy flat table if needed.
- Defensive against missing columns and minor schema drift.

Normalized schema (preferred):
  categories(id INTEGER PRIMARY KEY, name TEXT UNIQUE NOT NULL, key TEXT UNIQUE)
  services(id INTEGER PRIMARY KEY, category_id INTEGER NOT NULL, name TEXT NOT NULL,
           key TEXT, UNIQUE(category_id, name))
  vendors(id INTEGER PRIMARY KEY,
          category_id INTEGER NOT NULL,
          service_id  INTEGER NOT NULL,
          business_name TEXT NOT NULL,
          contact_name  TEXT,
          phone         TEXT,
          address       TEXT,
          notes         TEXT,
          website       TEXT,
          is_active     INTEGER DEFAULT 1,
          created_at    TEXT DEFAULT CURRENT_TIMESTAMP,
          updated_at    TEXT)

Legacy fallback: flat `vendors` table with text columns `category`, `service`, etc.

Usage:
  streamlit run app_admin.py

"""
from __future__ import annotations

import contextlib
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

APP_TITLE = "Vendors Admin — Full (Sorted Category → Service)"
DB_FILENAME = "vendors.db"  # expected at repo root

# ----------------------------- Utilities ------------------------------------

def _slugify(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s

@contextlib.contextmanager
def connect(dbfile: Path):
    conn = sqlite3.connect(str(dbfile))
    try:
        conn.row_factory = sqlite3.Row
        yield conn
    finally:
        conn.close()


def table_exists(conn: sqlite3.Connection, name: str) -> bool:
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,)
    )
    return cur.fetchone() is not None


def column_exists(conn: sqlite3.Connection, table: str, col: str) -> bool:
    with contextlib.suppress(Exception):
        cur = conn.execute(f"PRAGMA table_info({table})")
        return any(row[1] == col for row in cur.fetchall())
    return False


# ----------------------------- Schema / Migrations --------------------------

def ensure_normalized_schema(conn: sqlite3.Connection) -> None:
    """Create normalized tables if missing. Non-destructive (idempotent).
    Be *careful* if a legacy flat `vendors` table exists: skip creating
    normalized indexes that reference non-existent columns.
    """
    # Always ensure taxonomy tables exist (safe)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS categories (
            id   INTEGER PRIMARY KEY,
            name TEXT UNIQUE NOT NULL,
            key  TEXT UNIQUE
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS services (
            id          INTEGER PRIMARY KEY,
            category_id INTEGER NOT NULL,
            name        TEXT NOT NULL,
            key         TEXT,
            UNIQUE(category_id, name),
            FOREIGN KEY(category_id) REFERENCES categories(id) ON DELETE CASCADE
        )
        """
    )

    # If a vendors table already exists but is *legacy* (no category_id/service_id),
    # do NOT try to create normalized indexes that would fail.
    vendors_exists = table_exists(conn, "vendors")
    has_cat_id = column_exists(conn, "vendors", "category_id") if vendors_exists else False
    has_svc_id = column_exists(conn, "vendors", "service_id") if vendors_exists else False

    if not vendors_exists:
        # Create normalized vendors table fresh
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS vendors (
                id            INTEGER PRIMARY KEY,
                category_id   INTEGER NOT NULL,
                service_id    INTEGER NOT NULL,
                business_name TEXT NOT NULL,
                contact_name  TEXT,
                phone         TEXT,
                address       TEXT,
                notes         TEXT,
                website       TEXT,
                is_active     INTEGER DEFAULT 1,
                created_at    TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at    TEXT,
                FOREIGN KEY(category_id) REFERENCES categories(id),
                FOREIGN KEY(service_id)  REFERENCES services(id)
            )
            """
        )
        has_cat_id = has_svc_id = True

    # Create helpful indexes ONLY if normalized columns are present
    if has_cat_id and has_svc_id:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_vendors_cat_svc ON vendors(category_id, service_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_services_cat ON services(category_id)")

    conn.commit()


def has_normalized(conn: sqlite3.Connection) -> bool:
    """Return True only if vendors has foreign-key columns AND taxonomy tables exist."""
    if not (table_exists(conn, "vendors") and table_exists(conn, "categories") and table_exists(conn, "services")):
        return False
    return column_exists(conn, "vendors", "category_id") and column_exists(conn, "vendors", "service_id")

    return table_exists(conn, "vendors") and table_exists(conn, "categories") and table_exists(conn, "services")


# ----------------------------- Lookups --------------------------------------

def get_categories(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query("SELECT id, name, COALESCE(key, '') as key FROM categories ORDER BY name ASC", conn)


def get_services_for_category(conn: sqlite3.Connection, category_id: int) -> pd.DataFrame:
    q = "SELECT id, name, COALESCE(key, '') as key FROM services WHERE category_id=? ORDER BY name ASC"
    return pd.read_sql_query(q, conn, params=(category_id,))


def upsert_category(conn: sqlite3.Connection, name: str) -> int:
    key = _slugify(name)
    conn.execute("INSERT OR IGNORE INTO categories(name, key) VALUES(?, ?)", (name.strip(), key))
    conn.commit()
    cur = conn.execute("SELECT id FROM categories WHERE name=?", (name.strip(),))
    row = cur.fetchone()
    return int(row[0])


def upsert_service(conn: sqlite3.Connection, category_id: int, name: str) -> int:
    key = _slugify(name)
    conn.execute(
        "INSERT OR IGNORE INTO services(category_id, name, key) VALUES(?, ?, ?)",
        (category_id, name.strip(), key),
    )
    conn.commit()
    cur = conn.execute(
        "SELECT id FROM services WHERE category_id=? AND name=?",
        (category_id, name.strip()),
    )
    row = cur.fetchone()
    return int(row[0])


# ----------------------------- Fetch / List ---------------------------------

def fetch_vendors_sorted(conn: sqlite3.Connection, include_inactive: bool = False) -> pd.DataFrame:
    """Return vendors sorted by Category asc, Service asc, Business asc.
    Chooses normalized path only if vendors.category_id & vendors.service_id exist.
    Tolerates older normalized variants missing is_active/created_at/updated_at.
    Also tolerates `business` vs `business_name` column naming.
    """
    normalized = has_normalized(conn)

    if normalized:
        # Probe vendor columns
        has_is_active = column_exists(conn, "vendors", "is_active")
        has_created = column_exists(conn, "vendors", "created_at")
        has_updated = column_exists(conn, "vendors", "updated_at")
        vb_col = "business_name" if column_exists(conn, "vendors", "business_name") else ("business" if column_exists(conn, "vendors", "business") else None)

        where = ""
        if has_is_active and not include_inactive:
            where = "WHERE COALESCE(v.is_active,1)=1"

        select_bits = [
            "v.id as ID",
            "c.name AS Category",
            "s.name AS Service",
        ]
        if vb_col:
            select_bits.append(f"v.{vb_col} AS 'Business Name'")
        else:
            select_bits.append("'' AS 'Business Name'")
        select_bits += [
            "COALESCE(v.contact_name,'') AS 'Contact Name'" if column_exists(conn, "vendors", "contact_name") else "'' AS 'Contact Name'",
            "COALESCE(v.phone,'') AS Phone" if column_exists(conn, "vendors", "phone") else "'' AS Phone",
            "COALESCE(v.address,'') AS Address" if column_exists(conn, "vendors", "address") else "'' AS Address",
            "COALESCE(v.notes,'') AS Notes" if column_exists(conn, "vendors", "notes") else "'' AS Notes",
            "COALESCE(v.website,'') AS Website" if column_exists(conn, "vendors", "website") else "'' AS Website",
        ]
        select_bits.append("COALESCE(v.is_active,1) AS Active" if has_is_active else "1 AS Active")
        select_bits.append("COALESCE(v.created_at,'') AS Created" if has_created else "'' AS Created")
        select_bits.append("COALESCE(v.updated_at,'') AS Updated" if has_updated else "'' AS Updated")

        order_business = f", v.{vb_col} ASC" if vb_col else ""
        sql = f"""
            SELECT
                {', '.join(select_bits)}
            FROM vendors v
            LEFT JOIN categories c ON c.id = v.category_id
            LEFT JOIN services   s ON s.id = v.service_id
            {where}
            ORDER BY c.name ASC, s.name ASC{order_business}
        """
    else:
        # Legacy flat vendors table path
        # Try to discover likely column names for business/contact/etc
        vb_col = "business_name" if column_exists(conn, "vendors", "business_name") else ("business" if column_exists(conn, "vendors", "business") else None)
        cat_txt = "category" if column_exists(conn, "vendors", "category") else None
        svc_txt = "service" if column_exists(conn, "vendors", "service") else None

        select_bits = [
            "rowid as ID",
            f"COALESCE({cat_txt},'') AS Category" if cat_txt else "'' AS Category",
            f"COALESCE({svc_txt},'') AS Service" if svc_txt else "'' AS Service",
            f"COALESCE({vb_col},'') AS 'Business Name'" if vb_col else "'' AS 'Business Name'",
        ]
        for name, alias in [
            ("contact_name", "Contact Name"),
            ("phone", "Phone"),
            ("address", "Address"),
            ("notes", "Notes"),
            ("website", "Website"),
        ]:
            select_bits.append(f"COALESCE({name},'') AS '{alias}'" if column_exists(conn, "vendors", name) else f"'' AS '{alias}'")

        order_bits = []
        if cat_txt: order_bits.append("Category ASC")
        if svc_txt: order_bits.append("Service ASC")
        if vb_col: order_bits.append("'Business Name' ASC")
        order_clause = (" ORDER BY " + ", ".join(order_bits)) if order_bits else ""

        sql = f"SELECT {', '.join(select_bits)} FROM vendors{order_clause}"

    df = pd.read_sql_query(sql, conn)

    # Defensive post-sort in Python (stable mergesort)
    sort_cols = [c for c in ["Category", "Service", "Business Name"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols, kind="mergesort", ignore_index=True)
    return df

    # Defensive post-sort
    sort_cols = [c for c in ["Category", "Service", "Business Name"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols, kind="mergesort", ignore_index=True)
    return df


# ----------------------------- Mutations ------------------------------------

def insert_vendor(
    conn: sqlite3.Connection,
    category_name: str,
    service_name: str,
    business_name: str,
    contact_name: str = "",
    phone: str = "",
    address: str = "",
    notes: str = "",
    website: str = "",
    is_active: int = 1,
) -> int:
    if not has_normalized(conn):
        # Minimal legacy support: insert into flat vendors if columns exist
        cols = []
        if column_exists(conn, "vendors", "category"): cols.append("category")
        if column_exists(conn, "vendors", "service"): cols.append("service")
        if column_exists(conn, "vendors", "business_name"): cols.append("business_name")
        elif column_exists(conn, "vendors", "business"): cols.append("business")
        for c in ["contact_name","phone","address","notes","website"]:
            if column_exists(conn, "vendors", c): cols.append(c)
        values = {
            "category": category_name,
            "service": service_name,
            "business_name": business_name,
            "business": business_name,
            "contact_name": contact_name,
            "phone": phone,
            "address": address,
            "notes": notes,
            "website": website,
        }
        cols2 = [c for c in cols if c in values]
        placeholders = ",".join(["?"] * len(cols2))
        sql = f"INSERT INTO vendors({','.join(cols2)}) VALUES({placeholders})"
        conn.execute(sql, tuple(values[c] for c in cols2))
        conn.commit()
        return int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])

    # Normalized path
    cat_id = upsert_category(conn, category_name)
    svc_id = upsert_service(conn, cat_id, service_name)
    now = datetime.utcnow().isoformat(timespec="seconds")
    cur = conn.execute(
        """
        INSERT INTO vendors(
            category_id, service_id, business_name, contact_name, phone, address,
            notes, website, is_active, created_at, updated_at
        ) VALUES(?,?,?,?,?,?,?,?,?,?,?)
        """,
        (cat_id, svc_id, business_name.strip(), contact_name.strip(), phone.strip(), address.strip(),
         notes.strip(), website.strip(), int(bool(is_active)), now, now),
    )
    conn.commit()
    return int(cur.lastrowid)


def update_vendor(
    conn: sqlite3.Connection,
    vendor_id: int,
    category_name: str,
    service_name: str,
    business_name: str,
    contact_name: str,
    phone: str,
    address: str,
    notes: str,
    website: str,
    is_active: int,
) -> None:
    if not has_normalized(conn):
        # Best-effort legacy update if columns exist
        sets = []
        vals = []
        for col, val in [
            ("category", category_name), ("service", service_name),
            ("business_name", business_name), ("business", business_name),
            ("contact_name", contact_name), ("phone", phone),
            ("address", address), ("notes", notes), ("website", website),
        ]:
            if column_exists(conn, "vendors", col):
                sets.append(f"{col}=?")
                vals.append(val)
        if not sets:
            return
        vals.append(vendor_id)
        sql = f"UPDATE vendors SET {', '.join(sets)} WHERE rowid=?"
        conn.execute(sql, tuple(vals))
        conn.commit()
        return

    cat_id = upsert_category(conn, category_name)
    svc_id = upsert_service(conn, cat_id, service_name)
    now = datetime.utcnow().isoformat(timespec="seconds")
    conn.execute(
        """
        UPDATE vendors SET
            category_id=?, service_id=?, business_name=?, contact_name=?, phone=?,
            address=?, notes=?, website=?, is_active=?, updated_at=?
        WHERE id=?
        """,
        (cat_id, svc_id, business_name.strip(), contact_name.strip(), phone.strip(),
         address.strip(), notes.strip(), website.strip(), int(bool(is_active)), now, vendor_id),
    )
    conn.commit()


def soft_delete_vendor(conn: sqlite3.Connection, vendor_id: int) -> None:
    if has_normalized(conn) and column_exists(conn, "vendors", "is_active"):
        conn.execute("UPDATE vendors SET is_active=0, updated_at=? WHERE id=?", (datetime.utcnow().isoformat(timespec="seconds"), vendor_id))
    else:
        # legacy: best-effort hard delete
        conn.execute("DELETE FROM vendors WHERE rowid=?", (vendor_id,))
    conn.commit()


def hard_delete_vendor(conn: sqlite3.Connection, vendor_id: int) -> None:
    if has_normalized(conn):
        conn.execute("DELETE FROM vendors WHERE id=?", (vendor_id,))
    else:
        conn.execute("DELETE FROM vendors WHERE rowid=?", (vendor_id,))
    conn.commit()


# ----------------------------- UI Components --------------------------------

def select_category(conn: sqlite3.Connection, label: str, key: str, default: Optional[str] = None) -> Tuple[int, str]:
    cats = get_categories(conn)
    names = cats["name"].tolist()
    if default and default not in names:
        names.append(default)
    chosen = st.selectbox(label, options=sorted(set(names + ["(add new…)"])), index=(sorted(set(names + ["(add new…)"])).index(default) if default in names else 0), key=key)
    if chosen == "(add new…)":
        with st.popover("New Category"):
            new_name = st.text_input("Category name", key=f"new_cat_{key}")
            if st.button("Create Category", key=f"create_cat_{key}") and new_name.strip():
                cat_id = upsert_category(conn, new_name.strip())
                st.success(f"Created: {new_name}")
                return cat_id, new_name.strip()
        # fallthrough to any default selection
    # map to id
    if chosen and chosen != "(add new…)":
        row = cats.loc[cats["name"] == chosen]
        if not row.empty:
            return int(row.iloc[0]["id"]), chosen
    # default: create if missing
    if chosen and chosen not in (None, "(add new…)"):
        return upsert_category(conn, chosen), chosen
    return upsert_category(conn, "Uncategorized"), "Uncategorized"


def select_service(conn: sqlite3.Connection, category_id: int, label: str, key: str, default: Optional[str] = None) -> Tuple[int, str]:
    svcs = get_services_for_category(conn, category_id)
    names = svcs["name"].tolist()
    if default and default not in names:
        names.append(default)
    chosen = st.selectbox(label, options=sorted(set(names + ["(add new…)"])), index=(sorted(set(names + ["(add new…)"])).index(default) if default in names else 0), key=key)
    if chosen == "(add new…)":
        with st.popover("New Service"):
            new_name = st.text_input("Service name", key=f"new_svc_{key}")
            if st.button("Create Service", key=f"create_svc_{key}") and new_name.strip():
                svc_id = upsert_service(conn, category_id, new_name.strip())
                st.success(f"Created: {new_name}")
                return svc_id, new_name.strip()
    if chosen and chosen != "(add new…)":
        row = svcs.loc[svcs["name"] == chosen]
        if not row.empty:
            return int(row.iloc[0]["id"]), chosen
    if chosen and chosen not in (None, "(add new…)"):
        return upsert_service(conn, category_id, chosen), chosen
    return upsert_service(conn, category_id, "(Unspecified)"), "(Unspecified)"


# ----------------------------- Pages ----------------------------------------

def page_list(conn: sqlite3.Connection):
    st.subheader("Directory (sorted by Category → Service → Business)")

    include_inactive = st.toggle("Show inactive", value=False, help="Includes vendors with is_active=0 (normalized mode only)")
    df = fetch_vendors_sorted(conn, include_inactive=include_inactive)

    # Optional filters that DO NOT change sort order
    with st.expander("Filters (optional)"):
        c1, c2 = st.columns(2)
        cat = c1.selectbox("Category", ["(all, key="list_filter_category")"] + sorted([x for x in df.get("Category", pd.Series(dtype=str)).dropna().unique().tolist() if x != ""])) if "Category" in df.columns else "(all)"
        svc = c2.selectbox("Service", ["(all, key="list_filter_service")"] + sorted([x for x in df.get("Service", pd.Series(dtype=str)).dropna().unique().tolist() if x != ""])) if "Service" in df.columns else "(all)"
        if cat != "(all)":
            df = df[df["Category"] == cat]
        if svc != "(all)":
            df = df[df["Service"] == svc]
        # re-enforce sort
        df = df.sort_values([c for c in ["Category","Service","Business Name"] if c in df.columns], kind="mergesort", ignore_index=True)

    st.dataframe(df, use_container_width=True, hide_index=True)
    st.download_button("Download CSV (sorted view)", df.to_csv(index=False).encode("utf-8"), file_name="vendors_sorted.csv", mime="text/csv")


def page_add(conn: sqlite3.Connection):
    st.subheader("Add Vendor")
    ensure_normalized_schema(conn)

    cat_id, cat_name = select_category(conn, "Category", key="add_cat")
    svc_id, svc_name = select_service(conn, cat_id, "Service", key="add_svc")

    with st.form("add_vendor_form", clear_on_submit=True):
        business = st.text_input("Business Name", max_chars=200)
        contact = st.text_input("Contact Name", value="")
        phone   = st.text_input("Phone", value="")
        address = st.text_area("Address", value="", height=80)
        website = st.text_input("Website", value="")
        notes   = st.text_area("Notes", value="", height=120)
        active  = st.checkbox("Active", value=True)
        submitted = st.form_submit_button("Add Vendor")

    if submitted:
        if not business.strip():
            st.error("Business Name is required.")
            return
        vid = insert_vendor(conn, cat_name, svc_name, business.strip(), contact.strip(), phone.strip(), address.strip(), notes.strip(), website.strip(), 1 if active else 0)
        st.success(f"Added vendor #{vid}: {business.strip()}")


def page_edit(conn: sqlite3.Connection):
    st.subheader("Edit Vendor")
    df = fetch_vendors_sorted(conn, include_inactive=True)
    if df.empty:
        st.info("No vendors to edit.")
        return

    # Pick vendor by ID + label
    df["Label"] = df.apply(lambda r: f"#{r['ID']} — {r.get('Business Name','')} ({r.get('Category','')} → {r.get('Service','')})", axis=1)
    choices = df[["ID","Label"]].values.tolist()
    id_to_label = {int(i): lbl for i,lbl in choices}
    chosen_id = st.selectbox("Select Vendor", options=list(id_to_label.keys(, key="edit_vendor_select")), format_func=lambda i: id_to_label[i])

    row = df.loc[df["ID"] == chosen_id].iloc[0]

    ensure_normalized_schema(conn)
    # category/service pickers prefilled
    cat_id, cat_name = select_category(conn, "Category", key=f"edit_cat_{chosen_id}", default=str(row.get("Category","")))
    svc_id, svc_name = select_service(conn, cat_id, "Service", key=f"edit_svc_{chosen_id}", default=str(row.get("Service","")))

    with st.form(f"edit_vendor_form_{chosen_id}"):
        business = st.text_input("Business Name", value=str(row.get("Business Name","")), max_chars=200)
        contact  = st.text_input("Contact Name", value=str(row.get("Contact Name","")))
        phone    = st.text_input("Phone", value=str(row.get("Phone","")))
        address  = st.text_area("Address", value=str(row.get("Address","")), height=80)
        website  = st.text_input("Website", value=str(row.get("Website","")))
        notes    = st.text_area("Notes", value=str(row.get("Notes","")), height=120)
        active   = st.checkbox("Active", value=bool(int(row.get("Active",1))) if "Active" in row else True)
        submitted = st.form_submit_button("Save Changes")

    if submitted:
        if not business.strip():
            st.error("Business Name is required.")
            return
        update_vendor(conn, int(chosen_id), cat_name, svc_name, business.strip(), contact.strip(), phone.strip(), address.strip(), notes.strip(), website.strip(), 1 if active else 0)
        st.success(f"Updated vendor #{chosen_id}")


def page_delete(conn: sqlite3.Connection):
    st.subheader("Delete Vendor")
    df = fetch_vendors_sorted(conn, include_inactive=True)
    if df.empty:
        st.info("No vendors to delete.")
        return

    df["Label"] = df.apply(lambda r: f"#{r['ID']} — {r.get('Business Name','')} ({r.get('Category','')} → {r.get('Service','')})", axis=1)
    id_to_label = {int(r.ID): r.Label for _, r in df[['ID','Label']].iterrows()}
    chosen_id = st.selectbox("Select Vendor", options=list(id_to_label.keys(, key="delete_vendor_select")), format_func=lambda i: id_to_label[i])

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Soft Delete (mark inactive)"):
            soft_delete_vendor(conn, int(chosen_id))
            st.success(f"Vendor #{chosen_id} marked inactive")
    with col2:
        if st.button("Hard Delete (remove row)", type="secondary"):
            hard_delete_vendor(conn, int(chosen_id))
            st.warning(f"Vendor #{chosen_id} permanently deleted")


def page_manage_taxonomy(conn: sqlite3.Connection):
    st.subheader("Manage Categories & Services")
    ensure_normalized_schema(conn)

    # Add Category
    with st.expander("Add Category"):
        with st.form("add_cat_form", clear_on_submit=True):
            name = st.text_input("Category name")
            submitted = st.form_submit_button("Add Category")
        if submitted and name.strip():
            cat_id = upsert_category(conn, name.strip())
            st.success(f"Category added/exists (id={cat_id}): {name.strip()}")

    # Add Service under a Category
    with st.expander("Add Service"):
        cats = get_categories(conn)
        if cats.empty:
            st.info("No categories yet. Add a category first.")
        else:
            cat_name = st.selectbox("Category", options=cats["name"].tolist(), key="manage_svc_cat")
            cat_id = int(cats.loc[cats["name"]==cat_name].iloc[0]["id"])
            with st.form("add_svc_form", clear_on_submit=True):
                name = st.text_input("Service name")
                submitted = st.form_submit_button("Add Service")
            if submitted and name.strip():
                svc_id = upsert_service(conn, cat_id, name.strip())
                st.success(f"Service added/exists (id={svc_id}): {name.strip()} → {cat_name}")

    # Browse taxonomy tables
    col1, col2 = st.columns(2)
    with col1:
        st.write("Categories")
        st.dataframe(get_categories(conn), use_container_width=True, hide_index=True)
    with col2:
        st.write("Services (by category)")
        cats = get_categories(conn)
        if not cats.empty:
            cat_name = st.selectbox("Filter by Category", options=["(all)"] + cats["name"].tolist(), key="svc_filter")
            if cat_name == "(all)":
                svc_df = pd.read_sql_query(
                    "SELECT s.id, c.name as category, s.name, COALESCE(s.key,'') as key FROM services s JOIN categories c ON c.id=s.category_id ORDER BY c.name,s.name",
                    conn,
                )
            else:
                cat_id = int(cats.loc[cats["name"]==cat_name].iloc[0]["id"])
                svc_df = pd.read_sql_query(
                    "SELECT s.id, c.name as category, s.name, COALESCE(s.key,'') as key FROM services s JOIN categories c ON c.id=s.category_id WHERE c.id=? ORDER BY s.name",
                    conn,
                    params=(cat_id,),
                )
            st.dataframe(svc_df, use_container_width=True, hide_index=True)


# ----------------------------- App Shell ------------------------------------

def render_header():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption("Listing order is permanently enforced: Category → Service → Business Name (ascending).")


def render_dataset_info(conn: sqlite3.Connection):
    cols = st.columns(4)
    with cols[0]:
        try:
            total = conn.execute("SELECT COUNT(1) FROM vendors").fetchone()[0]
        except Exception:
            total = "?"
        st.metric("Vendors (rows)", total)
    with cols[1]:
        st.metric("DB file", Path(DB_FILENAME).name)
    with cols[2]:
        st.metric("Schema", "normalized" if has_normalized(conn) else "legacy")
    with cols[3]:
        st.metric("Sorted", "Category → Service → Business")


def main():
    render_header()
    path = Path(__file__).resolve().parent / DB_FILENAME
    if not path.exists():
        st.error(f"Database file not found at {path}. Place {DB_FILENAME} next to this script.")
        return

    with connect(path) as conn:
        # Make sure normalized tables exist for admin operations, but don't break legacy
        ensure_normalized_schema(conn)
        render_dataset_info(conn)

        tabs = st.tabs(["List", "Add", "Edit", "Delete", "Categories & Services"])
        with tabs[0]:
            page_list(conn)
        with tabs[1]:
            page_add(conn)
        with tabs[2]:
            page_edit(conn)
        with tabs[3]:
            page_delete(conn)
        with tabs[4]:
            page_manage_taxonomy(conn)


if __name__ == "__main__":
    main()
