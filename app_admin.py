# app_admin.py
# Vendors Admin — streamlined:
# - Wide layout via secrets (page_title, page_max_width_px, sidebar_state)
# - DB: Turso/libSQL via sqlite+libsql://… with auth token; fallback to local SQLite vendors.db
# - Browse Vendors: single global, case-insensitive, partial-word filter across all fields (non-FTS)
# - Add / Edit / Delete Vendor:
#       * Business Name REQUIRED
#       * Category REQUIRED (must exist in categories lib)
#       * Service OPTIONAL (must exist in services lib if provided)
#       * Phone must be 10 digits (US) or blank; normalized to ########## on save
#       * Immediate page refresh after any mutation
# - Category Admin & Service Admin:
#       * Add, Rename, Delete (with guardrails)
#       * Orphan surfacing (values used by vendors but missing from libs)
#       * Category/service usage counts
# - Maintenance tab:
#       * "Repair services table" one-click safety (ensures schema=id,name; migrates from old shapes)
#       * Normalize phones, trim spaces, title-case business names (optional)
# - Debug tab at bottom with engine status and schema snapshot
#
# Columns (vendors):
#   id (INT PK), category (TEXT), service (TEXT, nullable), business_name (TEXT),
#   contact_name (TEXT), phone (TEXT), address (TEXT), website (TEXT), notes (TEXT), keywords (TEXT)
#
# Columns (categories): id (INT PK), name (TEXT UNIQUE)
# Columns (services):   id (INT PK), name (TEXT UNIQUE)
#
from __future__ import annotations

import os
import re
from typing import List, Dict, Optional, Tuple

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text as sql_text
from sqlalchemy.engine import Engine

# =========================
# Page Layout / Secrets
# =========================

def _read_secret_early(name: str, default=None):
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.environ.get(name, default)

PAGE_TITLE = _read_secret_early("page_title", "Vendors Admin")
PAGE_MAX_WIDTH_PX = int(_read_secret_early("page_max_width_px", 2300))
SIDEBAR_STATE = _read_secret_early("sidebar_state", "expanded")

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

# =========================
# DB Engine: libSQL -> SQLite fallback
# =========================

def build_engine() -> Tuple[Engine, Dict[str, str]]:
    # Prefer libSQL (Turso) if secrets present; else local SQLite
    turso_url = _read_secret_early("TURSO_DATABASE_URL", "")
    turso_token = _read_secret_early("TURSO_AUTH_TOKEN", "")
    if turso_url and turso_token:
        # SQLAlchemy URL for libsql dialect
        sqlalchemy_url = f"sqlite+libsql://{turso_url.split('://')[-1]}?secure=true"
        engine = create_engine(
            sqlalchemy_url,
            connect_args={"auth_token": turso_token},
            pool_pre_ping=True,
        )
        return engine, {"using_remote": True, "sqlalchemy_url": sqlalchemy_url, "dialect": "sqlite", "driver": "libsql"}

    # Fallback to local SQLite vendors.db
    sqlite_path = os.path.join(os.path.dirname(__file__), "vendors.db")
    sqlalchemy_url = f"sqlite:///{sqlite_path}"
    engine = create_engine(sqlalchemy_url)
    return engine, {"using_remote": False, "sqlalchemy_url": sqlalchemy_url, "dialect": "sqlite", "driver": "sqlite"}

engine, engine_info = build_engine()

# =========================
# Schema helpers
# =========================

def ensure_tables():
    with engine.begin() as conn:
        conn.execute(sql_text("""
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
        );"""))
        conn.execute(sql_text("""
        CREATE TABLE IF NOT EXISTS categories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL
        );"""))
        conn.execute(sql_text("""
        CREATE TABLE IF NOT EXISTS services (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL
        );"""))

def snapshot_schema() -> Dict:
    out: Dict = {}
    with engine.begin() as conn:
        # vendors columns
        cols = conn.execute(sql_text("PRAGMA table_info(vendors);")).mappings().all()
        out["vendors_columns"] = [c["name"] for c in cols] if cols else []
        # categories
        cols = conn.execute(sql_text("PRAGMA table_info(categories);")).mappings().all()
        out["categories_columns"] = [c["name"] for c in cols] if cols else []
        # services
        cols = conn.execute(sql_text("PRAGMA table_info(services);")).mappings().all()
        out["services_columns"] = [c["name"] for c in cols] if cols else []
        # counts
        def count_of(tbl):
            try:
                return conn.execute(sql_text(f"SELECT COUNT(*) AS c FROM {tbl};")).scalar() or 0
            except Exception:
                return None
        out["counts"] = {
            "vendors": count_of("vendors"),
            "categories": count_of("categories"),
            "services": count_of("services"),
        }
    return out

ensure_tables()

# =========================
# Data access
# =========================

def fetch_vendors_df() -> pd.DataFrame:
    with engine.begin() as conn:
        df = pd.read_sql(sql_text("""
            SELECT id, category, service, business_name, contact_name, phone, address, website, notes, keywords
            FROM vendors
            ORDER BY business_name COLLATE NOCASE ASC
        """), conn)
    return df

def fetch_categories() -> List[str]:
    with engine.begin() as conn:
        rows = conn.execute(sql_text("SELECT name FROM categories ORDER BY name COLLATE NOCASE ASC;")).fetchall()
    return [r[0] for r in rows]

def fetch_services() -> List[str]:
    with engine.begin() as conn:
        rows = conn.execute(sql_text("SELECT name FROM services ORDER BY name COLLATE NOCASE ASC;")).fetchall()
    return [r[0] for r in rows]

def add_category(name: str) -> Tuple[bool, str]:
    name = name.strip()
    if not name:
        return False, "Category name cannot be blank."
    try:
        with engine.begin() as conn:
            conn.execute(sql_text("INSERT OR IGNORE INTO categories(name) VALUES(:n);"), {"n": name})
        return True, f"Category '{name}' added (or already existed)."
    except Exception as e:
        return False, f"Error adding category: {e}"

def rename_category(old: str, new: str) -> Tuple[bool, str]:
    old, new = old.strip(), new.strip()
    if not new:
        return False, "New category name cannot be blank."
    try:
        with engine.begin() as conn:
            # Update references in vendors
            conn.execute(sql_text("UPDATE vendors SET category=:new WHERE lower(trim(category))=lower(trim(:old));"),
                         {"new": new, "old": old})
            # Upsert new into categories, then delete old if different
            conn.execute(sql_text("INSERT OR IGNORE INTO categories(name) VALUES(:n);"), {"n": new})
            if old.lower() != new.lower():
                conn.execute(sql_text("DELETE FROM categories WHERE lower(trim(name))=lower(trim(:old));"),
                             {"old": old})
        return True, f"Category '{old}' renamed to '{new}'."
    except Exception as e:
        return False, f"Error renaming category: {e}"

def delete_category(name: str) -> Tuple[bool, str]:
    try:
        with engine.begin() as conn:
            count = conn.execute(sql_text("SELECT COUNT(*) FROM vendors WHERE lower(trim(category))=lower(trim(:n));"),
                                 {"n": name}).scalar()
            if count and count > 0:
                return False, f"Cannot delete '{name}': {count} vendor(s) still use it."
            conn.execute(sql_text("DELETE FROM categories WHERE lower(trim(name))=lower(trim(:n));"), {"n": name})
        return True, f"Category '{name}' deleted."
    except Exception as e:
        return False, f"Error deleting category: {e}"

def add_service(name: str) -> Tuple[bool, str]:
    name = name.strip()
    if not name:
        return False, "Service name cannot be blank."
    try:
        with engine.begin() as conn:
            conn.execute(sql_text("INSERT OR IGNORE INTO services(name) VALUES(:n);"), {"n": name})
        return True, f"Service '{name}' added (or already existed)."
    except Exception as e:
        return False, f"Error adding service: {e}"

def rename_service(old: str, new: str) -> Tuple[bool, str]:
    old, new = old.strip(), new.strip()
    if not new:
        return False, "New service name cannot be blank."
    try:
        with engine.begin() as conn:
            conn.execute(sql_text("UPDATE vendors SET service=:new WHERE lower(trim(service))=lower(trim(:old));"),
                         {"new": new, "old": old})
            conn.execute(sql_text("INSERT OR IGNORE INTO services(name) VALUES(:n);"), {"n": new})
            if old.lower() != new.lower():
                conn.execute(sql_text("DELETE FROM services WHERE lower(trim(name))=lower(trim(:old));"),
                             {"old": old})
        return True, f"Service '{old}' renamed to '{new}'."
    except Exception as e:
        return False, f"Error renaming service: {e}"

def delete_service(name: str) -> Tuple[bool, str]:
    try:
        with engine.begin() as conn:
            count = conn.execute(sql_text("SELECT COUNT(*) FROM vendors WHERE lower(trim(service))=lower(trim(:n));"),
                                 {"n": name}).scalar()
            if count and count > 0:
                return False, f"Cannot delete '{name}': {count} vendor(s) still use it."
            conn.execute(sql_text("DELETE FROM services WHERE lower(trim(name))=lower(trim(:n));"), {"n": name})
        return True, f"Service '{name}' deleted."
    except Exception as e:
        return False, f"Error deleting service: {e}"

def normalize_phone(p: str) -> str:
    if not p:
        return ""
    digits = re.sub(r"\D+", "", p)
    if len(digits) == 10:
        return digits
    return ""  # treat invalid as blank

def insert_vendor(row: Dict[str, str]) -> Tuple[bool, str]:
    try:
        with engine.begin() as conn:
            conn.execute(sql_text("""
                INSERT INTO vendors (category, service, business_name, contact_name, phone, address, website, notes, keywords)
                VALUES (:category, :service, :business_name, :contact_name, :phone, :address, :website, :notes, :keywords)
            """), row)
        return True, "Vendor added."
    except Exception as e:
        return False, f"Error adding vendor: {e}"

def update_vendor(vid: int, row: Dict[str, str]) -> Tuple[bool, str]:
    try:
        with engine.begin() as conn:
            conn.execute(sql_text("""
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
            """), {**row, "id": vid})
        return True, "Vendor updated."
    except Exception as e:
        return False, f"Error updating vendor: {e}"

def delete_vendor(vid: int) -> Tuple[bool, str]:
    try:
        with engine.begin() as conn:
            conn.execute(sql_text("DELETE FROM vendors WHERE id=:id;"), {"id": vid})
        return True, "Vendor deleted."
    except Exception as e:
        return False, f"Error deleting vendor: {e}"

# =========================
# Orphans / Usage
# =========================

def category_usage_counts() -> List[Tuple[str, int]]:
    with engine.begin() as conn:
        rows = conn.execute(sql_text("""
            SELECT v.category AS cat, COUNT(*) AS c
            FROM vendors v
            GROUP BY v.category
            ORDER BY lower(trim(v.category)) ASC
        """)).fetchall()
    return [(r[0], r[1]) for r in rows]

def service_usage_counts() -> List[Tuple[str, int]]:
    with engine.begin() as conn:
        rows = conn.execute(sql_text("""
            SELECT v.service AS svc, COUNT(*) AS c
            FROM vendors v
            WHERE v.service IS NOT NULL AND trim(v.service) <> ''
            GROUP BY v.service
            ORDER BY lower(trim(v.service)) ASC
        """)).fetchall()
    return [(r[0], r[1]) for r in rows]

def find_orphan_categories() -> List[str]:
    # Values used by vendors but missing from categories.name
    all_cats = set([c.lower().strip() for c in fetch_categories()])
    used = set([c.lower().strip() for c, _ in category_usage_counts() if c])
    return sorted([u for u in used if u not in all_cats])

def find_orphan_services() -> List[str]:
    all_svcs = set([s.lower().strip() for s in fetch_services()])
    used = set([s.lower().strip() for s, _ in service_usage_counts() if s])
    return sorted([u for u in used if u not in all_svcs])

# =========================
# Maintenance ops
# =========================

def repair_services_table() -> Tuple[bool, str]:
    """
    Ensure services has schema (id INTEGER PK AUTOINCREMENT, name TEXT UNIQUE NOT NULL).
    If older/incorrect schema exists, migrate distinct names.
    """
    try:
        with engine.begin() as conn:
            # Snapshot existing
            cols = conn.execute(sql_text("PRAGMA table_info(services);")).mappings().all()
            existing_cols = [c["name"] for c in cols]
            if existing_cols == ["id", "name"]:
                return True, "Services table already correct."

            # Create temp table with correct schema
            conn.execute(sql_text("""
                CREATE TABLE IF NOT EXISTS services_tmp (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL
                );
            """))

            # Try to extract plausible names from existing services table
            # Heuristics: if column 'name' exists, take it; else union of any text columns
            names = []
            if "name" in existing_cols:
                rows = conn.execute(sql_text("SELECT name FROM services;")).fetchall()
                names = [r[0] for r in rows if r and r[0]]
            else:
                # take any text-looking columns
                rows = conn.execute(sql_text("SELECT * FROM services;")).fetchall()
                for row in rows:
                    for val in row:
                        if isinstance(val, str) and val.strip():
                            names.append(val.strip())
            # Deduplicate/normalize
            uniq = sorted(set([n.strip() for n in names if n and n.strip()]))
            for n in uniq:
                try:
                    conn.execute(sql_text("INSERT OR IGNORE INTO services_tmp(name) VALUES(:n);"), {"n": n})
                except Exception:
                    pass

            # Drop old and rename
            conn.execute(sql_text("DROP TABLE services;"))
            conn.execute(sql_text("ALTER TABLE services_tmp RENAME TO services;"))

        return True, f"Services table repaired. Migrated {len(uniq)} distinct name(s)."
    except Exception as e:
        return False, f"Repair failed: {e}"

def normalize_phones_all() -> Tuple[int, int]:
    """
    Convert all vendor phone values to ########## if exactly 10 digits; else blank.
    Returns (updated_count, total_rows)
    """
    with engine.begin() as conn:
        rows = conn.execute(sql_text("SELECT id, phone FROM vendors;")).fetchall()
        total = len(rows)
        updated = 0
        for vid, p in rows:
            np = normalize_phone(p or "")
            if (p or "") != np:
                conn.execute(sql_text("UPDATE vendors SET phone=:p WHERE id=:id;"), {"p": np, "id": vid})
                updated += 1
    return updated, total

def trim_and_title_business_names() -> Tuple[int, int]:
    with engine.begin() as conn:
        rows = conn.execute(sql_text("SELECT id, business_name FROM vendors;")).fetchall()
        total = len(rows)
        updated = 0
        for vid, n in rows:
            if not n:
                continue
            newn = re.sub(r"\s+", " ", n).strip()
            # Prefer not to over-title-case acronyms; keep simple capitalization
            # Use title() but preserve all-caps segments >= 3 chars
            parts = []
            for tok in newn.split(" "):
                if len(tok) >= 3 and tok.isupper():
                    parts.append(tok)
                else:
                    parts.append(tok.title())
            newn2 = " ".join(parts)
            if newn2 != n:
                conn.execute(sql_text("UPDATE vendors SET business_name=:n WHERE id=:id;"), {"n": newn2, "id": vid})
                updated += 1
    return updated, total

# =========================
# UI Helpers
# =========================

def rerun():
    st.rerun()

def text_input(label: str, key: str, value: str = "", placeholder: str = "") -> str:
    return st.text_input(label, value=value or "", key=key, placeholder=placeholder)

def selectbox(label: str, options: List[str], key: str, index: Optional[int] = None) -> str:
    return st.selectbox(label, options, index=index, key=key)

# =========================
# Tabs
# =========================

def tab_browse():
    st.subheader("Browse Vendors")

    df = fetch_vendors_df()

    # Global non-FTS filter (case-insensitive, partial match across all string columns)
    q = st.text_input(
        "Global search across all fields (non-FTS, case-insensitive; matches partial words).",
        placeholder="e.g., plumb returns any record with 'plumb' anywhere",
        key="browse_query",
    ).strip()

    if q:
        q_lower = q.lower()
        mask = pd.Series([False] * len(df))
        for col in df.columns:
            if df[col].dtype == object:
                mask = mask | df[col].fillna("").str.lower().str.contains(q_lower, na=False)
        df = df[mask]

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
    )

def tab_vendor_crud():
    st.subheader("Add / Edit / Delete Vendor")

    # Load libs fresh each draw
    cats = fetch_categories()
    svcs = fetch_services()

    df = fetch_vendors_df()
    # -------- Add --------
    with st.expander("➕ Add Vendor", expanded=True):
        colA, colB, colC = st.columns([1, 1, 1.2])

        with colA:
            business_name = st.text_input("Business Name *", key="add_business_name").strip()
            category = st.selectbox("Category *", cats, key="add_category") if cats else st.text_input("Category * (no categories yet — type to create later)")
            service = st.selectbox("Service (optional)", [""] + svcs, key="add_service") if svcs else st.text_input("Service (optional)")

        with colB:
            contact_name = st.text_input("Contact Name", key="add_contact_name")
            phone_in = st.text_input("Phone (10 digits or blank)", key="add_phone")

        with colC:
            address = st.text_input("Address", key="add_address")
            website = st.text_input("Website", key="add_website")
            notes = st.text_area("Notes", key="add_notes")
            keywords = st.text_input("Keywords (comma-separated)", key="add_keywords")

        if st.button("Save New Vendor", type="primary", key="btn_add_vendor"):
            # Validate requireds
            if not business_name:
                st.error("Business Name is required.")
                return
            if not category or (isinstance(category, str) and not category.strip()):
                st.error("Category is required.")
                return

            # Enforce category existence
            cat_name = category.strip()
            if cat_name not in cats:
                ok, msg = add_category(cat_name)
                if not ok:
                    st.error(msg)
                    return
                cats.append(cat_name)

            # Service optional, but if provided, enforce existence
            svc_name = (service or "").strip()
            if svc_name:
                if svc_name not in svcs:
                    ok, msg = add_service(svc_name)
                    if not ok:
                        st.error(msg)
                        return

            phone = normalize_phone(phone_in.strip() if phone_in else "")
            if phone_in and not phone:
                st.error("Phone must be exactly 10 digits or left blank.")
                return

            row = {
                "category": cat_name,
                "service": svc_name or None,
                "business_name": business_name.strip(),
                "contact_name": (contact_name or "").strip(),
                "phone": phone,
                "address": (address or "").strip(),
                "website": (website or "").strip(),
                "notes": (notes or "").strip(),
                "keywords": (keywords or "").strip(),
            }
            ok, msg = insert_vendor(row)
            if ok:
                st.success(msg)
                rerun()
            else:
                st.error(msg)

    # -------- Edit/Delete --------
    with st.expander("✏️ Edit or 🗑️ Delete Vendor", expanded=False):
        if df.empty:
            st.info("No vendors to edit.")
            return
        # Sorted by business_name (already sorted in query)
        options = df["business_name"].tolist()
        sel_name = st.selectbox("Select Vendor by Business Name", options, key="edit_select_name")

        vrow = df.loc[df["business_name"] == sel_name].iloc[0]
        vid = int(vrow["id"])

        colA, colB, colC = st.columns([1, 1, 1.2])

        with colA:
            business_name_e = st.text_input("Business Name *", value=vrow["business_name"], key="edit_business_name").strip()
            category_e = st.selectbox("Category *", cats, index=(cats.index(vrow["category"]) if vrow["category"] in cats else 0), key="edit_category")
            # Service optional; allow blank choice at index 0
            svc_options = [""] + svcs
            svc_val = vrow["service"] if isinstance(vrow["service"], str) else ""
            idx_svc = svc_options.index(svc_val) if svc_val in svc_options else 0
            service_e = st.selectbox("Service (optional)", svc_options, index=idx_svc, key="edit_service")

        with colB:
            contact_name_e = st.text_input("Contact Name", value=vrow["contact_name"] or "", key="edit_contact_name")
            phone_e_in = st.text_input("Phone (10 digits or blank)", value=vrow["phone"] or "", key="edit_phone")

        with colC:
            address_e = st.text_input("Address", value=vrow["address"] or "", key="edit_address")
            website_e = st.text_input("Website", value=vrow["website"] or "", key="edit_website")
            notes_e = st.text_area("Notes", value=vrow["notes"] or "", key="edit_notes")
            keywords_e = st.text_input("Keywords (comma-separated)", value=vrow["keywords"] or "", key="edit_keywords")

        col1, col2 = st.columns([0.25, 0.25])
        with col1:
            if st.button("Save Changes", type="primary", key="btn_save_edit"):
                if not business_name_e:
                    st.error("Business Name is required.")
                    return
                if not category_e:
                    st.error("Category is required.")
                    return

                # ensure category exists
                if category_e not in cats:
                    ok, msg = add_category(category_e)
                    if not ok:
                        st.error(msg)
                        return

                # service optional; ensure existence if provided
                svc_e = (service_e or "").strip()
                if svc_e and svc_e not in svcs:
                    ok, msg = add_service(svc_e)
                    if not ok:
                        st.error(msg)
                        return

                phone_e = normalize_phone(phone_e_in.strip() if phone_e_in else "")
                if phone_e_in and not phone_e:
                    st.error("Phone must be exactly 10 digits or left blank.")
                    return

                row_e = {
                    "category": category_e.strip(),
                    "service": (svc_e or None),
                    "business_name": business_name_e.strip(),
                    "contact_name": (contact_name_e or "").strip(),
                    "phone": phone_e,
                    "address": (address_e or "").strip(),
                    "website": (website_e or "").strip(),
                    "notes": (notes_e or "").strip(),
                    "keywords": (keywords_e or "").strip(),
                }
                ok, msg = update_vendor(vid, row_e)
                if ok:
                    st.success(msg)
                    rerun()
                else:
                    st.error(msg)

        with col2:
            if st.button("Delete Vendor", type="secondary", key="btn_delete_vendor"):
                ok, msg = delete_vendor(vid)
                if ok:
                    st.success(msg)
                    rerun()
                else:
                    st.error(msg)

def tab_categories():
    st.subheader("Categories Admin")

    cats = fetch_categories()
    usage = category_usage_counts()
    usage_map = {c: n for c, n in usage}

    if cats:
        st.write("**Existing Categories (with usage counts):**")
        disp = pd.DataFrame({"Category": cats, "Used by (vendors)": [usage_map.get(c, 0) for c in cats]})
        st.dataframe(disp, use_container_width=True, hide_index=True)
    else:
        st.info("No categories yet. Add one below.")

    orphans = find_orphan_categories()
    if orphans:
        st.warning("**Orphan categories detected (used in vendors but missing from library):** " + ", ".join(orphans))

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        new_cat = st.text_input("Add Category", key="add_category_name")
        if st.button("Add Category", key="btn_add_category"):
            ok, msg = add_category(new_cat)
            (st.success if ok else st.error)(msg)
            if ok:
                rerun()
    with col2:
        if cats:
            old = st.selectbox("Rename: pick existing", cats, key="rename_cat_old")
            new = st.text_input("New name", key="rename_cat_new")
            if st.button("Rename Category", key="btn_rename_category"):
                ok, msg = rename_category(old, new)
                (st.success if ok else st.error)(msg)
                if ok:
                    rerun()
    with col3:
        if cats:
            delc = st.selectbox("Delete: pick category (must have zero usage)", cats, key="delete_cat_pick")
            if st.button("Delete Category", key="btn_delete_category"):
                ok, msg = delete_category(delc)
                (st.success if ok else st.error)(msg)
                if ok:
                    rerun()

    st.markdown("---")
    st.markdown("**How to edit Categories:**")
    st.markdown(
        "- Add new names here, or rename to merge history.  \n"
        "- You can’t delete a category that’s still used by any vendor; reassign or rename first."
    )

def tab_services():
    st.subheader("Services Admin")

    svcs = fetch_services()
    usage = service_usage_counts()
    usage_map = {s: n for s, n in usage}

    if svcs:
        st.write("**Existing Services (with usage counts):**")
        disp = pd.DataFrame({"Service": svcs, "Used by (vendors)": [usage_map.get(s, 0) for s in svcs]})
        st.dataframe(disp, use_container_width=True, hide_index=True)
    else:
        st.info("No services yet. Add one below.")

    orphans = find_orphan_services()
    if orphans:
        st.warning("**Orphan services detected (used in vendors but missing from library):** " + ", ".join(orphans))

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        new_svc = st.text_input("Add Service", key="add_service_name")
        if st.button("Add Service", key="btn_add_service"):
            ok, msg = add_service(new_svc)
            (st.success if ok else st.error)(msg)
            if ok:
                rerun()
    with col2:
        if svcs:
            old = st.selectbox("Rename: pick existing", svcs, key="rename_svc_old")
            new = st.text_input("New name", key="rename_svc_new")
            if st.button("Rename Service", key="btn_rename_service"):
                ok, msg = rename_service(old, new)
                (st.success if ok else st.error)(msg)
                if ok:
                    rerun()
    with col3:
        if svcs:
            dels = st.selectbox("Delete: pick service (must have zero usage)", svcs, key="delete_svc_pick")
            if st.button("Delete Service", key="btn_delete_service"):
                ok, msg = delete_service(dels)
                (st.success if ok else st.error)(msg)
                if ok:
                    rerun()

    st.markdown("---")
    st.markdown("**How to edit Services:**")
    st.markdown(
        "- Add new names here, or rename to merge history.  \n"
        "- You can’t delete a service that’s still used by any vendor; reassign or rename first."
    )

def tab_maintenance():
    st.subheader("Maintenance & Safety")

    st.caption("Quick operations for schema integrity and light data hygiene.")

    col1, col2, col3 = st.columns([0.6, 0.6, 0.6])

    with col1:
        if st.button("Repair services table (one-click)", key="btn_repair_services"):
            ok, msg = repair_services_table()
            (st.success if ok else st.error)(msg)
            if ok:
                rerun()

    with col2:
        if st.button("Normalize all phone numbers", key="btn_norm_phones"):
            updated, total = normalize_phones_all()
            st.success(f"Normalized {updated} of {total} phone entries.")
            rerun()

    with col3:
        if st.button("Trim + title-case all business names", key="btn_title_names"):
            updated, total = trim_and_title_business_names()
            st.success(f"Updated {updated} of {total} business names.")
            rerun()

def tab_debug():
    st.subheader("Status & Secrets (debug)")

    dbg = {
        "DB (resolved)": engine_info,
    }
    st.json(dbg)

    st.markdown("**DB Probe**")
    st.json(snapshot_schema())

# =========================
# Main
# =========================

def main():
    tabs = st.tabs(["Browse", "Vendors", "Categories", "Services", "Maintenance", "Debug"])
    with tabs[0]:
        tab_browse()
    with tabs[1]:
        tab_vendor_crud()
    with tabs[2]:
        tab_categories()
    with tabs[3]:
        tab_services()
    with tabs[4]:
        tab_maintenance()
    with tabs[5]:
        tab_debug()

if __name__ == "__main__":
    main()
