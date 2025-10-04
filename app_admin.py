# app_admin.py
# Vendors Admin — stable Streamlit CRUD (no AG Grid), Turso (libSQL) first, SQLite fallback.
# Features:
# - Browse with global (non-FTS) filter across all text fields
# - Add / Edit / Delete Vendors with validation
# - Categories & Services Admin (add/delete when unused, reassign vendors on delete)
# - "Repair services table" button (ensures minimal expected schema)
# - Immediate refresh after writes (st.rerun())
# - Phone normalization: (XXX) XXX-XXXX or blank
# - URL normalization: ensure https:// scheme
# - Optional keywords support if vendors.keywords exists
#
# Expected DB schema (text-based linking):
# vendors(id INTEGER PK, category TEXT, service TEXT, business_name TEXT, contact_name TEXT,
#         phone TEXT, address TEXT, website TEXT, notes TEXT, keywords TEXT [optional])
# categories(id INTEGER PK, name TEXT UNIQUE)
# services(id INTEGER PK, name TEXT UNIQUE)
#
# Secrets (Streamlit):
#   TURSO_DATABASE_URL: "sqlite+libsql://<your-db-host>?secure=true"
#   TURSO_AUTH_TOKEN: "<jwt token from Turso>"
# Optional layout secrets:
#   page_title, page_max_width_px, sidebar_state

from __future__ import annotations

import os
import re
from typing import List, Dict, Optional, Tuple

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text as sql_text
from sqlalchemy.engine import Engine

# -----------------------------
# Page layout (do this first)
# -----------------------------
def _read_secret(name: str, default=None):
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.environ.get(name, default)

def _apply_page_width_css(max_width_px: Optional[int]):
    if not max_width_px:
        return
    st.markdown(
        f"""
        <style>
          .block-container {{
              max-width: {int(max_width_px)}px;
          }}
        </style>
        """,
        unsafe_allow_html=True,
    )

PAGE_TITLE = _read_secret("page_title", "Vendors Admin")
PAGE_MAX_WIDTH = _read_secret("page_max_width_px", 1400)
SIDEBAR_STATE = _read_secret("sidebar_state", "expanded")

st.set_page_config(page_title=PAGE_TITLE, layout="wide", initial_sidebar_state=SIDEBAR_STATE)
_apply_page_width_css(PAGE_MAX_WIDTH)

st.title(PAGE_TITLE)

# -----------------------------
# Engine / DB helpers
# -----------------------------
def build_engine() -> Tuple[Engine, Dict[str, str]]:
    # Try Turso/libSQL first
    turso_url = _read_secret("TURSO_DATABASE_URL")
    turso_token = _read_secret("TURSO_AUTH_TOKEN")
    debug = {}

    if turso_url:
        try:
            engine = create_engine(
                turso_url,
                connect_args={"auth_token": turso_token} if turso_token else {},
                pool_pre_ping=True,
                future=True,
            )
            # Smoke test
            with engine.connect() as conn:
                conn.execute(sql_text("SELECT 1"))
            debug["engine_url"] = turso_url
            debug["using_remote"] = True
            return engine, debug
        except Exception as e:
            debug["remote_error"] = str(e)

    # Fallback to local SQLite vendors.db (mounted in container/repo root)
    local_path = os.path.abspath(os.path.join(os.getcwd(), "vendors.db"))
    engine = create_engine(f"sqlite:///{local_path}", future=True)
    debug["engine_url"] = f"sqlite:///{local_path}"
    debug["using_remote"] = False
    return engine, debug

engine, engine_info = build_engine()

def table_exists(table: str) -> bool:
    with engine.connect() as conn:
        res = conn.execute(
            sql_text(
                "SELECT name FROM sqlite_master WHERE type='table' AND lower(name)=lower(:t)"
            ),
            {"t": table},
        ).fetchone()
    return res is not None

def ensure_min_schema():
    """Create minimal tables if missing. Do not drop/alter existing user data."""
    with engine.begin() as conn:
        if not table_exists("categories"):
            conn.execute(sql_text("""
                CREATE TABLE categories(
                    id INTEGER PRIMARY KEY,
                    name TEXT UNIQUE
                )
            """))
        if not table_exists("services"):
            conn.execute(sql_text("""
                CREATE TABLE services(
                    id INTEGER PRIMARY KEY,
                    name TEXT UNIQUE
                )
            """))
        if not table_exists("vendors"):
            conn.execute(sql_text("""
                CREATE TABLE vendors(
                    id INTEGER PRIMARY KEY,
                    category TEXT,
                    service TEXT,
                    business_name TEXT,
                    contact_name TEXT,
                    phone TEXT,
                    address TEXT,
                    website TEXT,
                    notes TEXT
                )
            """))
        # If keywords column needed, leave to admin to add; we detect it dynamically.

ensure_min_schema()

def get_columns(table: str) -> List[str]:
    with engine.connect() as conn:
        rows = conn.execute(sql_text(f"PRAGMA table_info({table})")).fetchall()
    return [r[1] for r in rows]  # name is 2nd column in pragma

def has_column(table: str, col: str) -> bool:
    return col in get_columns(table)

def read_df(table: str) -> pd.DataFrame:
    with engine.connect() as conn:
        df = pd.read_sql(sql_text(f"SELECT * FROM {table}"), conn)
    return df

def write_df(df: pd.DataFrame, table: str, if_exists: str = "append"):
    # Not used for row-by-row writes; we use SQL for atomicity
    with engine.begin() as conn:
        df.to_sql(table, conn, if_exists=if_exists, index=False)

# -----------------------------
# Normalization / Validation
# -----------------------------
_phone_digits = re.compile(r"\D+")

def normalize_phone(phone: str) -> str:
    if not phone:
        return ""
    digits = _phone_digits.sub("", phone)
    if len(digits) != 10:
        raise ValueError("Phone must be 10 digits or left blank")
    return f"({digits[0:3]}) {digits[3:6]}-{digits[6:10]}"

def normalize_url(url: str) -> str:
    if not url:
        return ""
    u = url.strip()
    if u.startswith("http://") or u.startswith("https://"):
        return u
    # Accept bare domains
    return f"https://{u}"

def nonempty_trim(s: Optional[str]) -> str:
    return (s or "").strip()

# -----------------------------
# Data access helpers (CRUD)
# -----------------------------
def list_categories() -> List[str]:
    if not table_exists("categories"):
        return []
    df = read_df("categories")
    if "name" in df.columns:
        vals = sorted({nonempty_trim(x) for x in df["name"].astype(str).tolist() if nonempty_trim(x)})
        return vals
    return []

def list_services() -> List[str]:
    if not table_exists("services"):
        return []
    df = read_df("services")
    if "name" in df.columns:
        vals = sorted({nonempty_trim(x) for x in df["name"].astype(str).tolist() if nonempty_trim(x)})
        return vals
    return []

def vendors_df() -> pd.DataFrame:
    if not table_exists("vendors"):
        return pd.DataFrame(columns=["id","category","service","business_name","contact_name","phone","address","website","notes"])
    df = read_df("vendors")
    # Ensure consistent column presence
    base_cols = ["id","category","service","business_name","contact_name","phone","address","website","notes"]
    for c in base_cols:
        if c not in df.columns:
            df[c] = ""
    # Keep optional keywords if present
    cols = df.columns.tolist()
    order = base_cols + ([c for c in cols if c not in base_cols] if cols else [])
    return df[order]

def add_vendor(row: Dict[str, str]) -> int:
    with engine.begin() as conn:
        res = conn.execute(sql_text("""
            INSERT INTO vendors(category, service, business_name, contact_name, phone, address, website, notes {kw_cols})
            VALUES(:category, :service, :business_name, :contact_name, :phone, :address, :website, :notes {kw_vals})
        """.format(
            kw_cols=(", keywords" if has_column("vendors","keywords") else ""),
            kw_vals=(", :keywords" if has_column("vendors","keywords") else "")
        )), row)
        new_id = res.lastrowid if hasattr(res, "lastrowid") else None
    return int(new_id or 0)

def update_vendor(vendor_id: int, row: Dict[str, str]) -> None:
    set_parts = ["category=:category","service=:service","business_name=:business_name","contact_name=:contact_name",
                 "phone=:phone","address=:address","website=:website","notes=:notes"]
    if has_column("vendors","keywords"):
        set_parts.append("keywords=:keywords")
    with engine.begin() as conn:
        conn.execute(sql_text(f"""
            UPDATE vendors
               SET {", ".join(set_parts)}
             WHERE id=:id
        """), {**row, "id": vendor_id})

def delete_vendor(vendor_id: int) -> None:
    with engine.begin() as conn:
        conn.execute(sql_text("DELETE FROM vendors WHERE id=:id"), {"id": vendor_id})

def add_category(name: str) -> None:
    with engine.begin() as conn:
        conn.execute(sql_text("INSERT OR IGNORE INTO categories(name) VALUES(:n)"), {"n": name})

def delete_category(name: str) -> None:
    # Only if unused OR you reassign first (handled in UI)
    with engine.begin() as conn:
        conn.execute(sql_text("DELETE FROM categories WHERE name=:n"), {"n": name})

def add_service(name: str) -> None:
    with engine.begin() as conn:
        conn.execute(sql_text("INSERT OR IGNORE INTO services(name) VALUES(:n)"), {"n": name})

def delete_service(name: str) -> None:
    with engine.begin() as conn:
        conn.execute(sql_text("DELETE FROM services WHERE name=:n"), {"n": name})

def reassign_category(old: str, new: str) -> int:
    with engine.begin() as conn:
        res = conn.execute(sql_text("""
            UPDATE vendors SET category=:new WHERE lower(trim(category))=lower(trim(:old))
        """), {"new": new, "old": old})
        return res.rowcount or 0

def reassign_service(old: str, new: str) -> int:
    with engine.begin() as conn:
        res = conn.execute(sql_text("""
            UPDATE vendors SET service=:new WHERE lower(trim(service))=lower(trim(:old))
        """), {"new": new, "old": old})
        return res.rowcount or 0

def usage_counts(col: str, val: str) -> int:
    with engine.connect() as conn:
        res = conn.execute(sql_text(f"""
            SELECT COUNT(*) FROM vendors WHERE lower(trim({col}))=lower(trim(:v))
        """), {"v": val}).fetchone()
    return int(res[0]) if res else 0

# -----------------------------
# Repair button
# -----------------------------
def repair_services_table():
    """Ensure 'services' has minimal expected schema; add UNIQUE index if missing."""
    with engine.begin() as conn:
        # Create if missing
        conn.execute(sql_text("""
            CREATE TABLE IF NOT EXISTS services(
                id INTEGER PRIMARY KEY,
                name TEXT UNIQUE
            )
        """))
        # Try to add unique index (ignore if exists)
        try:
            conn.execute(sql_text("CREATE UNIQUE INDEX IF NOT EXISTS idx_services_name ON services(name)"))
        except Exception:
            pass

# -----------------------------
# Sidebar Nav
# -----------------------------
st.sidebar.header("Navigation")
nav = st.sidebar.radio(
    "Go to",
    ["Browse", "Add Vendor", "Edit Vendor", "Delete Vendor", "Categories & Services Admin", "Database Status & Schema (debug)"],
    index=0,
)

# -----------------------------
# Browse
# -----------------------------
if nav == "Browse":
    st.subheader("Browse Vendors")

    df = vendors_df()
    text_cols = [c for c in df.columns if c != "id"]
    st.caption("Global search across all fields (non-FTS, case-insensitive; matches partial words).")
    q = st.text_input("Search", value="", placeholder="e.g., plumb returns any record with 'plumb' anywhere")

    if q.strip():
        needle = q.strip().lower()
        mask = pd.Series([False] * len(df))
        for c in text_cols:
            mask = mask | df[c].astype(str).str.lower().str.contains(needle, na=False)
        df_view = df[mask].copy()
    else:
        df_view = df.copy()

    # Show selected columns in a logical order
    show_cols = [c for c in ["id","category","service","business_name","contact_name","phone","address","website","notes","keywords"] if c in df_view.columns]
    st.dataframe(df_view[show_cols], use_container_width=True, hide_index=True)

# -----------------------------
# Add Vendor
# -----------------------------
elif nav == "Add Vendor":
    st.subheader("Add Vendor")
    cats = list_categories()
    svcs = list_services()

    with st.form("add_vendor_form", clear_on_submit=False):
        category = st.selectbox("Category (required)", options=cats, index=0 if cats else None, placeholder="Select category")
        service  = st.selectbox("Service (optional; may be blank if you allow)", options=[""] + svcs, index=0)
        business_name = st.text_input("Business Name (required)")
        contact_name = st.text_input("Contact Name")
        phone = st.text_input("Phone — stored as (XXX) XXX-XXXX or left blank")
        address = st.text_input("Address")
        website = st.text_input("Website (normalize to https://...)")
        notes = st.text_area("Notes")
        kw_val = ""
        if has_column("vendors", "keywords"):
            kw_val = st.text_input("Keywords (comma or space separated allowed)")

        submitted = st.form_submit_button("Add Vendor")

    if submitted:
        try:
            if not business_name.strip():
                st.error("Business Name is required.")
                st.stop()
            if not category:
                st.error("Category is required.")
                st.stop()

            phone_norm = normalize_phone(phone.strip()) if phone.strip() else ""
            url_norm = normalize_url(website.strip()) if website.strip() else ""

            # Accept keywords with comma or space separation; store as comma-separated
            if has_column("vendors", "keywords"):
                raw = kw_val.strip()
                if raw:
                    parts = [p.strip() for p in re.split(r"[,\s]+", raw) if p.strip()]
                    kw_final = ", ".join(sorted(set(parts), key=str.lower))
                else:
                    kw_final = ""
            else:
                kw_final = None

            payload = {
                "category": nonempty_trim(category),
                "service": nonempty_trim(service),
                "business_name": nonempty_trim(business_name),
                "contact_name": nonempty_trim(contact_name),
                "phone": phone_norm,
                "address": nonempty_trim(address),
                "website": url_norm,
                "notes": nonempty_trim(notes),
            }
            if kw_final is not None:
                payload["keywords"] = kw_final

            new_id = add_vendor(payload)
            st.success(f"Added vendor #{new_id}: {payload['business_name']}")
            st.rerun()
        except Exception as e:
            st.exception(e)

# -----------------------------
# Edit Vendor
# -----------------------------
elif nav == "Edit Vendor":
    st.subheader("Edit Vendor")

    df = vendors_df()
    if df.empty:
        st.info("No vendors yet.")
    else:
        # Sort selection by Business Name ascending
        df_sel = df.copy()
        df_sel["bn_norm"] = df_sel["business_name"].astype(str).str.lower()
        df_sel = df_sel.sort_values(["bn_norm","id"])
        choices = [f"#{row.id} — {row.business_name}" for row in df_sel.itertuples(index=False)]

        sel = st.selectbox("Select vendor (Business Name ascending)", options=choices)
        if sel:
            # Parse id
            vid = int(sel.split("—")[0].strip().lstrip("#").strip())
            row = df[df["id"] == vid].iloc[0].to_dict()

            cats = list_categories()
            svcs = list_services()

            with st.form("edit_vendor_form"):
                category = st.selectbox("Category (required)", options=cats, index=(cats.index(row["category"]) if row["category"] in cats else 0 if cats else None))
                service  = st.selectbox("Service (optional)", options=[""] + svcs, index=( ([""]+svcs).index(row["service"]) if row["service"] in svcs else 0 ))
                business_name = st.text_input("Business Name (required)", value=row.get("business_name",""))
                contact_name = st.text_input("Contact Name", value=row.get("contact_name","") or "")
                phone = st.text_input("Phone", value=row.get("phone","") or "")
                address = st.text_input("Address", value=row.get("address","") or "")
                website = st.text_input("Website", value=row.get("website","") or "")
                notes = st.text_area("Notes", value=row.get("notes","") or "")
                kw_val = None
                if has_column("vendors","keywords"):
                    kw_val = st.text_input("Keywords", value=row.get("keywords","") or "")

                save = st.form_submit_button("Save Changes")

            if save:
                try:
                    if not business_name.strip():
                        st.error("Business Name is required.")
                        st.stop()
                    if not category:
                        st.error("Category is required.")
                        st.stop()

                    phone_norm = normalize_phone(phone.strip()) if phone.strip() else ""
                    url_norm = normalize_url(website.strip()) if website.strip() else ""

                    payload = {
                        "category": nonempty_trim(category),
                        "service": nonempty_trim(service),
                        "business_name": nonempty_trim(business_name),
                        "contact_name": nonempty_trim(contact_name),
                        "phone": phone_norm,
                        "address": nonempty_trim(address),
                        "website": url_norm,
                        "notes": nonempty_trim(notes),
                    }
                    if kw_val is not None:
                        raw = kw_val.strip()
                        if raw:
                            parts = [p.strip() for p in re.split(r"[,\s]+", raw) if p.strip()]
                            payload["keywords"] = ", ".join(sorted(set(parts), key=str.lower))
                        else:
                            payload["keywords"] = ""

                    update_vendor(vid, payload)
                    st.success(f"Updated vendor #{vid}.")
                    st.rerun()
                except Exception as e:
                    st.exception(e)

# -----------------------------
# Delete Vendor
# -----------------------------
elif nav == "Delete Vendor":
    st.subheader("Delete Vendor")

    df = vendors_df()
    if df.empty:
        st.info("No vendors to delete.")
    else:
        df_sel = df.copy()
        df_sel["bn_norm"] = df_sel["business_name"].astype(str).str.lower()
        df_sel = df_sel.sort_values(["bn_norm","id"])
        choices = [f"#{row.id} — {row.business_name}" for row in df_sel.itertuples(index=False)]

        sel = st.selectbox("Select vendor to delete", options=choices)
        if sel:
            vid = int(sel.split("—")[0].strip().lstrip("#").strip())
            row = df[df["id"] == vid].iloc[0].to_dict()
            st.warning(f"Delete vendor #{vid}: {row.get('business_name','(no name)')}?")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Delete", type="primary"):
                    try:
                        delete_vendor(vid)
                        st.success(f"Deleted vendor #{vid}.")
                        st.rerun()
                    except Exception as e:
                        st.exception(e)
            with col2:
                st.info("Action is permanent.")

# -----------------------------
# Categories & Services Admin
# -----------------------------
elif nav == "Categories & Services Admin":
    st.subheader("Categories & Services Admin")

    st.caption("Tips: You can reassign all vendors from one value to another, then delete the now-unused value. "
               "Category is required on vendors; Service is optional (if you allow blank).")

    tab_cat, tab_svc, tab_repair = st.tabs(["Categories", "Services", "Repair/Utilities"])

    with tab_cat:
        st.markdown("#### Categories")
        cats = list_categories()
        st.write(f"Existing categories: {', '.join(cats) if cats else '(none)'}")

        st.markdown("**Add Category**")
        newc = st.text_input("New category name", key="new_cat_name")
        if st.button("Add Category"):
            if not newc.strip():
                st.error("Enter a name.")
            else:
                try:
                    add_category(nonempty_trim(newc))
                    st.success(f"Added category: {newc}")
                    st.rerun()
                except Exception as e:
                    st.exception(e)

        st.markdown("---")
        st.markdown("**Reassign Category**")
        if cats:
            oldc = st.selectbox("Reassign from", options=cats, key="oldc")
            newc2 = st.selectbox("Reassign to", options=cats, key="newc")
            if st.button("Reassign All Vendors to New Category"):
                if oldc == newc2:
                    st.warning("Old and new categories are the same.")
                else:
                    n = reassign_category(oldc, newc2)
                    st.success(f"Reassigned {n} vendors from '{oldc}' to '{newc2}'.")
                    st.rerun()

        st.markdown("---")
        st.markdown("**Delete Category**")
        if cats:
            delc = st.selectbox("Delete which category (must be unused)?", options=cats, key="delc")
            used = usage_counts("category", delc)
            st.write(f"Usage count: {used}")
            if st.button("Delete Category"):
                if used > 0:
                    st.error("Cannot delete a category still in use. Reassign vendors first.")
                else:
                    try:
                        delete_category(delc)
                        st.success(f"Deleted category '{delc}'.")
                        st.rerun()
                    except Exception as e:
                        st.exception(e)

    with tab_svc:
        st.markdown("#### Services")
        svcs = list_services()
        st.write(f"Existing services: {', '.join(svcs) if svcs else '(none)'}")

        st.markdown("**Add Service**")
        news = st.text_input("New service name", key="new_svc_name")
        if st.button("Add Service"):
            if not news.strip():
                st.error("Enter a name.")
            else:
                try:
                    add_service(nonempty_trim(news))
                    st.success(f"Added service: {news}")
                    st.rerun()
                except Exception as e:
                    st.exception(e)

        st.markdown("---")
        st.markdown("**Reassign Service**")
        if svcs:
            olds = st.selectbox("Reassign from", options=svcs, key="olds")
            news2 = st.selectbox("Reassign to", options=svcs, key="news2")
            if st.button("Reassign All Vendors to New Service"):
                if olds == news2:
                    st.warning("Old and new services are the same.")
                else:
                    n = reassign_service(olds, news2)
                    st.success(f"Reassigned {n} vendors from '{olds}' to '{news2}'.")
                    st.rerun()

        st.markdown("---")
        st.markdown("**Delete Service**")
        if svcs:
            dels = st.selectbox("Delete which service (must be unused)?", options=svcs, key="dels")
            used = usage_counts("service", dels)
            st.write(f"Usage count: {used}")
            if st.button("Delete Service"):
                if used > 0:
                    st.error("Cannot delete a service still in use. Reassign vendors first.")
                else:
                    try:
                        delete_service(dels)
                        st.success(f"Deleted service '{dels}'.")
                        st.rerun()
                    except Exception as e:
                        st.exception(e)

    with tab_repair:
        st.markdown("#### Repair / Utilities")
        st.write("Use this if the **services** table gets into a weird state.")
        if st.button("Repair services table (create/ensure UNIQUE name)"):
            try:
                repair_services_table()
                st.success("Services table repaired/ensured.")
            except Exception as e:
                st.exception(e)
        st.caption("No destructive changes. It only creates the minimal schema if missing and ensures a unique index on `services.name`.")

# -----------------------------
# Debug
# -----------------------------
elif nav == "Database Status & Schema (debug)":
    st.subheader("Database Status & Schema (debug)")
    st.json({
        "engine_url": engine_info.get("engine_url"),
        "using_remote": engine_info.get("using_remote"),
        "vendors_columns": get_columns("vendors") if table_exists("vendors") else "(missing)",
        "has_categories_table": table_exists("categories"),
        "has_services_table": table_exists("services"),
        "categories_columns": get_columns("categories") if table_exists("categories") else "(missing)",
        "services_columns": get_columns("services") if table_exists("services") else "(missing)",
        "counts": {
            "vendors": int(len(vendors_df())) if table_exists("vendors") else 0
        }
    })

    st.markdown("##### Sample rows (first 50)")
    if table_exists("vendors"):
        df = vendors_df().head(50)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("vendors table missing")
