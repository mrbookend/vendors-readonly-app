# app_admin.py
# Vendors Admin — streamlined:
# - Wide layout via secrets (page_title, page_max_width_px=2300 default, sidebar_state)
# - DB: Turso/libSQL via sqlite+libsql://… with auth token; guarded fallback to local SQLite vendors.db
# - Browse Vendors: AgGrid with explicit column widths, wrapping, and stable behavior
# - Add / Edit / Delete Vendor:
#       * Business Name REQUIRED
#       * Category REQUIRED (must exist in categories lib)
#       * Service OPTIONAL (must exist in services lib if provided)
#       * Phone must be 10 digits (US) or blank; normalized to ########## on save
#       * Immediate page refresh after any mutation
# - Category Admin & Service Admin: add/rename/delete, usage counts, orphan surfacing
# - Maintenance tab: repair services table, normalize phones, title-case names, backfill audit cols
# - Debug tab with engine status and schema snapshot

from __future__ import annotations

import os
import re
from typing import List, Dict, Optional, Tuple

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text as sql_text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import NoSuchModuleError, SQLAlchemyError
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse

# AgGrid (table with column width control & wrapping)
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

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
# Default width 2300 (overrides previous 1200 default unless secret provided)
PAGE_MAX_WIDTH_PX = int(_read_secret_early("page_max_width_px", 2300))
SIDEBAR_STATE = _read_secret_early("sidebar_state", "expanded")

st.set_page_config(page_title=PAGE_TITLE, layout="wide", initial_sidebar_state=SIDEBAR_STATE)

st.markdown(
    f"""
    <style>
      .block-container {{
        max-width: {PAGE_MAX_WIDTH_PX}px;
      }}
      .stTextInput > div > div > input {{
        line-height: 1.15;
      }}
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# DB Engine: libSQL -> SQLite guarded fallback
# =========================

def _normalize_turso_sqlalchemy_url(raw: str) -> str:
    """
    Accepts any of:
      - 'libsql://host' (Turso style)
      - 'sqlite+libsql://host' (SQLAlchemy style)
      - with/without '?secure=true'
    Returns a clean 'sqlite+libsql://host?...' with secure=true exactly once.
    """
    if not raw:
        return ""

    raw = raw.strip()

    # If bare host, add Turso scheme
    if "://" not in raw:
        raw = "libsql://" + raw

    # Only convert scheme if it starts with libsql:// (avoid double 'sqlite+')
    if raw.startswith("libsql://"):
        raw = raw.replace("libsql://", "sqlite+libsql://", 1)
    elif raw.startswith("sqlite+libsql://"):
        pass  # already correct
    else:
        # Unknown scheme (e.g., http://) — treat as host and rewrite
        parsed_tmp = urlparse(raw)
        host = parsed_tmp.netloc or parsed_tmp.path
        raw = "sqlite+libsql://" + host

    parsed = urlparse(raw)
    q = dict(parse_qsl(parsed.query, keep_blank_values=True))
    q["secure"] = "true"  # normalize once
    new_query = urlencode(q, doseq=True)

    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, new_query, parsed.fragment))

def build_engine() -> Tuple[Engine, Dict[str, str]]:
    """
    Try Turso/libSQL first. If the dialect is missing or URL/auth are bad,
    warn and fall back to local SQLite vendors.db so the app stays up.
    """
    turso_url = _read_secret_early("TURSO_DATABASE_URL", "") or ""
    turso_token = _read_secret_early("TURSO_AUTH_TOKEN", "") or ""

    if turso_url and turso_token:
        sqlalchemy_url = _normalize_turso_sqlalchemy_url(turso_url)
        try:
            eng = create_engine(
                sqlalchemy_url,
                connect_args={"auth_token": turso_token},
                pool_pre_ping=True,
            )
            return eng, {
                "using_remote": True,
                "sqlalchemy_url": sqlalchemy_url,
                "dialect": "sqlite",
                "driver": "libsql",
            }
        except (NoSuchModuleError, ValueError, SQLAlchemyError) as e:
            # Expanded warning includes the actual error message for easier diagnosis
            st.warning(f"Turso/libSQL unavailable ({type(e).__name__}: {e}); falling back to local vendors.db.")

    # Local fallback
    sqlite_path = os.path.join(os.path.dirname(__file__), "vendors.db")
    sqlalchemy_url = f"sqlite:///{sqlite_path}"
    eng = create_engine(sqlalchemy_url)
    return eng, {
        "using_remote": False,
        "sqlalchemy_url": sqlalchemy_url,
        "dialect": "sqlite",
        "driver": "sqlite",
    }

# Instantiate the engine BEFORE any DB calls.
engine, engine_info = build_engine()

# =========================
# Schema helpers
# =========================

def ensure_tables(db: Engine):
    with db.begin() as conn:
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

def snapshot_schema(db: Engine) -> Dict:
    out: Dict = {}
    with db.begin() as conn:
        cols = conn.execute(sql_text("PRAGMA table_info(vendors);")).mappings().all()
        out["vendors_columns"] = [c["name"] for c in cols] if cols else []
        cols = conn.execute(sql_text("PRAGMA table_info(categories);")).mappings().all()
        out["categories_columns"] = [c["name"] for c in cols] if cols else []
        cols = conn.execute(sql_text("PRAGMA table_info(services);")).mappings().all()
        out["services_columns"] = [c["name"] for c in cols] if cols else []
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

# Create tables now that engine exists
ensure_tables(engine)

# =========================
# Data access
# =========================

def fetch_vendors_df(db: Engine) -> pd.DataFrame:
    with db.begin() as conn:
        df = pd.read_sql(sql_text("""
            SELECT id, category, service, business_name, contact_name, phone, address, website, notes, keywords
            FROM vendors
            ORDER BY business_name COLLATE NOCASE ASC
        """), conn)
    return df

def fetch_categories(db: Engine) -> List[str]:
    with db.begin() as conn:
        rows = conn.execute(sql_text("SELECT name FROM categories ORDER BY name COLLATE NOCASE ASC;")).fetchall()
    return [r[0] for r in rows]

def fetch_services(db: Engine) -> List[str]:
    with db.begin() as conn:
        rows = conn.execute(sql_text("SELECT name FROM services ORDER BY name COLLATE NOCASE ASC;")).fetchall()
    return [r[0] for r in rows]

# =========================
# Audit-aware helpers
# =========================

def _table_has_column(db: Engine, table: str, col: str) -> bool:
    with db.begin() as conn:
        cols = conn.execute(sql_text(f"PRAGMA table_info({table});")).mappings().all()
    return any(c["name"] == col for c in cols)

def _vendor_audit_caps(db: Engine) -> Dict[str, bool]:
    # Detect audit columns on vendors table
    return {
        "created_at": _table_has_column(db, "vendors", "created_at"),
        "updated_at": _table_has_column(db, "vendors", "updated_at"),
        "updated_by": _table_has_column(db, "vendors", "updated_by"),
    }

def _admin_user() -> str:
    # Change source of actor if desired (cookie, auth, etc.). Defaults to 'admin'.
    return str(_read_secret_early("ADMIN_USERNAME", "admin")).strip() or "admin"

# =========================
# CRUD
# =========================

def normalize_phone(p: str) -> str:
    if not p:
        return ""
    digits = re.sub(r"\D+", "", p)
    if len(digits) == 10:
        return digits
    return ""  # invalid => blank

def insert_vendor(db: Engine, row: Dict[str, str]) -> Tuple[bool, str]:
    try:
        caps = _vendor_audit_caps(db)
        # Base column/value sets
        cols = [
            "category", "service", "business_name", "contact_name",
            "phone", "address", "website", "notes", "keywords"
        ]
        params = {k: row.get(k) for k in cols}

        # Conditionally append audit fields
        if caps["created_at"]:
            cols.append("created_at")
        if caps["updated_at"]:
            cols.append("updated_at")
        if caps["updated_by"]:
            cols.append("updated_by")
            params["updated_by"] = _admin_user()

        # Build SQL with server-side timestamps to avoid client clock issues
        values_sql = []
        for c in cols:
            if c in ("created_at", "updated_at"):
                values_sql.append("CURRENT_TIMESTAMP")
            else:
                values_sql.append(f":{c}")

        sql = f"""
            INSERT INTO vendors ({", ".join(cols)})
            VALUES ({", ".join(values_sql)})
        """

        with db.begin() as conn:
            conn.execute(sql_text(sql), params)
        return True, "Vendor added."
    except Exception as e:
        return False, f"Error adding vendor: {e}"

def update_vendor(db: Engine, vid: int, row: Dict[str, str]) -> Tuple[bool, str]:
    try:
        caps = _vendor_audit_caps(db)
        sets = [
            "category=:category",
            "service=:service",
            "business_name=:business_name",
            "contact_name=:contact_name",
            "phone=:phone",
            "address=:address",
            "website=:website",
            "notes=:notes",
            "keywords=:keywords",
        ]
        params = {
            "category": row.get("category"),
            "service": row.get("service"),
            "business_name": row.get("business_name"),
            "contact_name": row.get("contact_name"),
            "phone": row.get("phone"),
            "address": row.get("address"),
            "website": row.get("website"),
            "notes": row.get("notes"),
            "keywords": row.get("keywords"),
            "id": vid,
        }

        if caps["updated_at"]:
            sets.append("updated_at=CURRENT_TIMESTAMP")
        if caps["updated_by"]:
            sets.append("updated_by=:updated_by")
            params["updated_by"] = _admin_user()

        sql = f"""
            UPDATE vendors
            SET {", ".join(sets)}
            WHERE id=:id
        """

        with db.begin() as conn:
            conn.execute(sql_text(sql), params)
        return True, "Vendor updated."
    except Exception as e:
        return False, f"Error updating vendor: {e}"

def delete_vendor(db: Engine, vid: int) -> Tuple[bool, str]:
    try:
        with db.begin() as conn:
            conn.execute(sql_text("DELETE FROM vendors WHERE id=:id;"), {"id": vid})
        return True, "Vendor deleted."
    except Exception as e:
        return False, f"Error deleting vendor: {e}"

# =========================
# Orphans / Usage
# =========================

def category_usage_counts(db: Engine) -> List[Tuple[str, int]]:
    with db.begin() as conn:
        rows = conn.execute(sql_text("""
            SELECT v.category AS cat, COUNT(*) AS c
            FROM vendors v
            GROUP BY v.category
            ORDER BY lower(trim(v.category)) ASC
        """)).fetchall()
    return [(r[0], r[1]) for r in rows]

def service_usage_counts(db: Engine) -> List[Tuple[str, int]]:
    with db.begin() as conn:
        rows = conn.execute(sql_text("""
            SELECT v.service AS svc, COUNT(*) AS c
            FROM vendors v
            WHERE v.service IS NOT NULL AND trim(v.service) <> ''
            GROUP BY v.service
            ORDER BY lower(trim(v.service)) ASC
        """)).fetchall()
    return [(r[0], r[1]) for r in rows]

def find_orphan_categories(db: Engine) -> List[str]:
    all_cats = set([c.lower().strip() for c in fetch_categories(db)])
    used = set([c.lower().strip() for c, _ in category_usage_counts(db) if c])
    return sorted([u for u in used if u not in all_cats])

def find_orphan_services(db: Engine) -> List[str]:
    all_svcs = set([s.lower().strip() for s in fetch_services(db)])
    used = set([s.lower().strip() for s, _ in service_usage_counts(db) if s])
    return sorted([u for u in used if u not in all_svcs])

# =========================
# Maintenance ops
# =========================

def repair_services_table(db: Engine) -> Tuple[bool, str]:
    """
    Ensure services has schema (id INTEGER PK AUTOINCREMENT, name TEXT UNIQUE NOT NULL).
    If older/incorrect schema exists, migrate distinct names.
    """
    try:
        with db.begin() as conn:
            cols = conn.execute(sql_text("PRAGMA table_info(services);")).mappings().all()
            existing_cols = [c["name"] for c in cols]
            if existing_cols == ["id", "name"]:
                return True, "Services table already correct."

            conn.execute(sql_text("""
                CREATE TABLE IF NOT EXISTS services_tmp (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL
                );
            """))

            names = []
            if "name" in existing_cols:
                rows = conn.execute(sql_text("SELECT name FROM services;")).fetchall()
                names = [r[0] for r in rows if r and r[0]]
            else:
                rows = conn.execute(sql_text("SELECT * FROM services;")).fetchall()
                for row in rows:
                    for val in row:
                        if isinstance(val, str) and val.strip():
                            names.append(val.strip())

            uniq = sorted(set([n.strip() for n in names if n and n.strip()]))
            for n in uniq:
                try:
                    conn.execute(sql_text("INSERT OR IGNORE INTO services_tmp(name) VALUES(:n);"), {"n": n})
                except Exception:
                    pass

            conn.execute(sql_text("DROP TABLE services;"))
            conn.execute(sql_text("ALTER TABLE services_tmp RENAME TO services;"))

        return True, f"Services table repaired. Migrated {len(uniq)} distinct name(s)."
    except Exception as e:
        return False, f"Repair failed: {e}"

def normalize_phones_all(db: Engine) -> Tuple[int, int]:
    with db.begin() as conn:
        rows = conn.execute(sql_text("SELECT id, phone FROM vendors;")).fetchall()
        total = len(rows)
        updated = 0
        for vid, p in rows:
            np = normalize_phone(p or "")
            if (p or "") != np:
                conn.execute(sql_text("UPDATE vendors SET phone=:p WHERE id=:id;"), {"p": np, "id": vid})
                updated += 1
    return updated, total

def trim_and_title_business_names(db: Engine) -> Tuple[int, int]:
    with db.begin() as conn:
        rows = conn.execute(sql_text("SELECT id, business_name FROM vendors;")).fetchall()
        total = len(rows)
        updated = 0
        for vid, n in rows:
            if not n:
                continue
            newn = re.sub(r"\s+", " ", n).strip()
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

def backfill_vendor_audit(db: Engine) -> Tuple[int, int]:
    caps = _vendor_audit_caps(db)
    if not any(caps.values()):
        return 0, 0
    updates = 0
    with db.begin() as conn:
        total = conn.execute(sql_text("SELECT COUNT(*) FROM vendors;")).scalar() or 0
        if caps["created_at"]:
            updates += conn.execute(sql_text("""
                UPDATE vendors SET created_at = COALESCE(created_at, CURRENT_TIMESTAMP)
            """)).rowcount or 0
        if caps["updated_at"]:
            updates += conn.execute(sql_text("""
                UPDATE vendors SET updated_at = COALESCE(updated_at, created_at, CURRENT_TIMESTAMP)
            """)).rowcount or 0
        if caps["updated_by"]:
            updates += conn.execute(sql_text("""
                UPDATE vendors SET updated_by = COALESCE(updated_by, 'system')
            """)).rowcount or 0
    return updates, total

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

def tab_browse(db: Engine):
    st.subheader("Browse Vendors")
    df = fetch_vendors_df(db)

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

    # --- Sidebar controls for column widths ---
    with st.sidebar.expander("Browse table layout", expanded=False):
        id_w     = st.number_input("ID width",              value=80,  min_value=50,  max_value=400,  step=10)
        cat_w    = st.number_input("Category width",        value=140, min_value=80,  max_value=600,  step=10)
        svc_w    = st.number_input("Service width",         value=160, min_value=80,  max_value=600,  step=10)
        name_w   = st.number_input("Business name width",   value=220, min_value=120, max_value=800,  step=10)
        contact_w= st.number_input("Contact name width",    value=160, min_value=100, max_value=600,  step=10)
        phone_w  = st.number_input("Phone width",           value=120, min_value=100, max_value=300,  step=10)
        addr_w   = st.number_input("Address width",         value=260, min_value=120, max_value=900,  step=10)
        site_w   = st.number_input("Website width",         value=200, min_value=120, max_value=700,  step=10)
        notes_w  = st.number_input("Notes width",           value=520, min_value=200, max_value=1600, step=20)
        keys_w   = st.number_input("Keywords width",        value=420, min_value=200, max_value=1600, step=20)

        wrap_notes = st.checkbox("Wrap Notes", value=True)
        wrap_keys  = st.checkbox("Wrap Keywords", value=True)

    # --- Build AgGrid options ---
    gob = GridOptionsBuilder.from_dataframe(df)

    # General defaults
    gob.configure_grid_options(
        domLayout="autoHeight",                 # grid grows with rows
        ensureDomOrder=True,
        suppressFieldDotNotation=True,
        suppressMovableColumns=False,
    )
    gob.configure_default_column(
        resizable=True,
        sortable=True,
        filter=True,
        wrapHeaderText=True,
        autoHeaderHeight=True,
        minWidth=90,
    )

    # Column-specific configs (stable ordering)
    col_order = [
        "id","category","service","business_name","contact_name","phone",
        "address","website","notes","keywords"
    ]
    existing = [c for c in col_order if c in df.columns] + [c for c in df.columns if c not in col_order]

    if "id" in df:
        gob.configure_column("id", header_name="ID", width=id_w, pinned=None)
    if "category" in df:
        gob.configure_column("category", width=cat_w)
    if "service" in df:
        gob.configure_column("service", width=svc_w)
    if "business_name" in df:
        gob.configure_column("business_name", width=name_w)
    if "contact_name" in df:
        gob.configure_column("contact_name", width=contact_w)
    if "phone" in df:
        gob.configure_column("phone", width=phone_w)
    if "address" in df:
        gob.configure_column("address", width=addr_w)
    if "website" in df:
        gob.configure_column("website", width=site_w)

    # Notes & Keywords: wide + wrapping
    if "notes" in df:
        gob.configure_column(
            "notes",
            width=notes_w,
            cellStyle={"whiteSpace": "normal"} if wrap_notes else {"whiteSpace": "nowrap"},
            autoHeight=wrap_notes,  # grows row height if wrapping
        )
    if "keywords" in df:
        gob.configure_column(
            "keywords",
            width=keys_w,
            cellStyle={"whiteSpace": "normal"} if wrap_keys else {"whiteSpace": "nowrap"},
            autoHeight=wrap_keys,
        )

    grid_options = gob.build()
    # Enforce stable field order
    grid_options["columnDefs"] = sorted(
        grid_options["columnDefs"],
        key=lambda d: existing.index(d.get("field")) if d.get("field") in existing else 1e9
    )

    AgGrid(
        df,
        gridOptions=grid_options,
        theme="balham",
        fit_columns_on_grid_load=True,  # initial fit only; manual resizes persist
        enable_enterprise_modules=False,
        allow_unsafe_jscode=False,
        reload_data=False,
        update_mode=GridUpdateMode.NO_UPDATE,
        height=None,                     # auto via domLayout
    )

def tab_vendor_crud(db: Engine):
    st.subheader("Add / Edit / Delete Vendor")

    cats = fetch_categories(db)
    svcs = fetch_services(db)
    df = fetch_vendors_df(db)

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
            if not business_name:
                st.error("Business Name is required.")
                return
            if not category or (isinstance(category, str) and not category.strip()):
                st.error("Category is required.")
                return

            cat_name = category.strip()
            if cat_name not in cats:
                ok, msg = add_category(db, cat_name)
                if not ok:
                    st.error(msg)
                    return
                cats.append(cat_name)

            svc_name = (service or "").strip()
            if svc_name:
                if svc_name not in svcs:
                    ok, msg = add_service(db, svc_name)
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
            ok, msg = insert_vendor(db, row)
            if ok:
                st.success(msg)
                rerun()
            else:
                st.error(msg)

    with st.expander("✏️ Edit or 🗑️ Delete Vendor", expanded=False):
        if df.empty:
            st.info("No vendors to edit.")
            return
        options = df["business_name"].tolist()
        sel_name = st.selectbox("Select Vendor by Business Name", options, key="edit_select_name")

        vrow = df.loc[df["business_name"] == sel_name].iloc[0]
        vid = int(vrow["id"])

        colA, colB, colC = st.columns([1, 1, 1.2])

        with colA:
            business_name_e = st.text_input("Business Name *", value=vrow["business_name"], key="edit_business_name").strip()
            category_e = st.selectbox("Category *", cats, index=(cats.index(vrow["category"]) if vrow["category"] in cats else 0), key="edit_category")
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

                if category_e not in cats:
                    ok, msg = add_category(db, category_e)
                    if not ok:
                        st.error(msg)
                        return

                svc_e = (service_e or "").strip()
                if svc_e and svc_e not in svcs:
                    ok, msg = add_service(db, svc_e)
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
                ok, msg = update_vendor(db, vid, row_e)
                if ok:
                    st.success(msg)
                    rerun()
                else:
                    st.error(msg)

        with col2:
            if st.button("Delete Vendor", type="secondary", key="btn_delete_vendor"):
                ok, msg = delete_vendor(db, vid)
                if ok:
                    st.success(msg)
                    rerun()
                else:
                    st.error(msg)

def tab_categories(db: Engine):
    st.subheader("Categories Admin")

    cats = fetch_categories(db)
    usage = category_usage_counts(db)
    usage_map = {c: n for c, n in usage}

    if cats:
        st.write("**Existing Categories (with usage counts):**")
        disp = pd.DataFrame({"Category": cats, "Used by (vendors)": [usage_map.get(c, 0) for c in cats]})
        st.dataframe(disp, use_container_width=True, hide_index=True)
    else:
        st.info("No categories yet. Add one below.")

    orphans = find_orphan_categories(db)
    if orphans:
        st.warning("**Orphan categories detected (used in vendors but missing from library):** " + ", ".join(orphans))

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        new_cat = st.text_input("Add Category", key="add_category_name")
        if st.button("Add Category", key="btn_add_category"):
            ok, msg = add_category(db, new_cat)
            (st.success if ok else st.error)(msg)
            if ok:
                rerun()
    with col2:
        if cats:
            old = st.selectbox("Rename: pick existing", cats, key="rename_cat_old")
            new = st.text_input("New name", key="rename_cat_new")
            if st.button("Rename Category", key="btn_rename_category"):
                ok, msg = rename_category(db, old, new)
                (st.success if ok else st.error)(msg)
                if ok:
                    rerun()
    with col3:
        if cats:
            delc = st.selectbox("Delete: pick category (must have zero usage)", cats, key="delete_cat_pick")
            if st.button("Delete Category", key="btn_delete_category"):
                ok, msg = delete_category(db, delc)
                (st.success if ok else st.error)(msg)
                if ok:
                    rerun()

    st.markdown("---")
    st.markdown("**How to edit Categories:**")
    st.markdown(
        "- Add new names here, or rename to merge history.  \n"
        "- You can’t delete a category that’s still used by any vendor; reassign or rename first."
    )

def tab_services(db: Engine):
    st.subheader("Services Admin")

    svcs = fetch_services(db)
    usage = service_usage_counts(db)
    usage_map = {s: n for s, n in usage}

    if svcs:
        st.write("**Existing Services (with usage counts):**")
        disp = pd.DataFrame({"Service": svcs, "Used by (vendors)": [usage_map.get(s, 0) for s in svcs]})
        st.dataframe(disp, use_container_width=True, hide_index=True)
    else:
        st.info("No services yet. Add one below.")

    orphans = find_orphan_services(db)
    if orphans:
        st.warning("**Orphan services detected (used in vendors but missing from library):** " + ", ".join(orphans))

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        new_svc = st.text_input("Add Service", key="add_service_name")
        if st.button("Add Service", key="btn_add_service"):
            ok, msg = add_service(db, new_svc)
            (st.success if ok else st.error)(msg)
            if ok:
                rerun()
    with col2:
        if svcs:
            old = st.selectbox("Rename: pick existing", svcs, key="rename_svc_old")
            new = st.text_input("New name", key="rename_svc_new")
            if st.button("Rename Service", key="btn_rename_service"):
                ok, msg = rename_service(db, old, new)
                (st.success if ok else st.error)(msg)
                if ok:
                    rerun()
    with col3:
        if svcs:
            dels = st.selectbox("Delete: pick service (must have zero usage)", svcs, key="delete_svc_pick")
            if st.button("Delete Service", key="btn_delete_service"):
                ok, msg = delete_service(db, dels)
                (st.success if ok else st.error)(msg)
                if ok:
                    rerun()

    st.markdown("---")
    st.markdown("**How to edit Services:**")
    st.markdown(
        "- Add new names here, or rename to merge history.  \n"
        "- You can’t delete a service that’s still used by any vendor; reassign or rename first."
    )

def tab_maintenance(db: Engine):
    st.subheader("Maintenance & Safety")
    st.caption("Quick operations for schema integrity and light data hygiene.")

    col1, col2, col3 = st.columns([0.6, 0.6, 0.6])

    with col1:
        if st.button("Repair services table (one-click)", key="btn_repair_services"):
            ok, msg = repair_services_table(db)
            (st.success if ok else st.error)(msg)
            if ok:
                rerun()

    with col2:
        if st.button("Normalize all phone numbers", key="btn_norm_phones"):
            updated, total = normalize_phones_all(db)
            st.success(f"Normalized {updated} of {total} phone entries.")
            rerun()

    with col3:
        if st.button("Trim + title-case all business names", key="btn_title_names"):
            updated, total = trim_and_title_business_names(db)
            st.success(f"Updated {updated} of {total} business names.")
            rerun()

    st.markdown("---")
    if st.button("Backfill vendor audit columns (created/updated/by)", key="btn_backfill_audit"):
        upd, total = backfill_vendor_audit(db)
        st.success(f"Backfilled {upd} fields across {total} rows.")
        rerun()

def tab_debug(db: Engine):
    st.subheader("Status & Secrets (debug)")
    dbg = {"DB (resolved)": engine_info}
    st.json(dbg)
    st.markdown("**DB Probe**")
    st.json(snapshot_schema(db))

# =========================
# Main
# =========================

def main():
    tabs = st.tabs(["Browse", "Vendors", "Categories", "Services", "Maintenance", "Debug"])
    with tabs[0]:
        tab_browse(engine)
    with tabs[1]:
        tab_vendor_crud(engine)
    with tabs[2]:
        tab_categories(engine)
    with tabs[3]:
        tab_services(engine)
    with tabs[4]:
        tab_maintenance(engine)
    with tabs[5]:
        tab_debug(engine)

if __name__ == "__main__":
    main()
