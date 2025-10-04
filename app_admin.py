# app_admin.py
# Vendors Admin — URL-only column in grid
# - Wide layout via secrets (page_title, page_max_width_px, sidebar_state)
# - DB: Turso/libSQL via sqlite+libsql://… with auth token; guarded fallback to local SQLite vendors.db
# - Browse Vendors: AgGrid with explicit column widths, wrapping; **shows only one column named 'URL'** (no 'website'/'Website')
# - Add / Edit / Delete Vendor:
#       * Business Name REQUIRED
#       * Category REQUIRED (must exist in categories)
#       * Service OPTIONAL (must exist in services if provided)
#       * Phone must be 10 digits (US) or blank; normalized to ########## on save
#       * URL saved to DB column 'website' (but displayed/edited as 'URL')
#       * Immediate page refresh after any mutation
# - Category Admin & Service Admin: add/rename/delete, usage counts, orphan surfacing
# - Maintenance tab: normalize phones, title-case names, backfill audit cols
# - Debug tab with engine status and schema snapshot

from __future__ import annotations

import os
import re
from typing import List, Dict, Optional, Tuple

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text as sql_text
from sqlalchemy.engine import Engine

# Optional import of st_aggrid; provide a clear error if missing
try:
    from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
except Exception:
    AgGrid = None
    GridOptionsBuilder = None
    GridUpdateMode = None

# -----------------------------
# Page layout (set early)
# -----------------------------
PAGE_TITLE = st.secrets.get("page_title", "Vendors Admin") if hasattr(st, "secrets") else "Vendors Admin"
PAGE_MAX_WIDTH_PX = int(st.secrets.get("page_max_width_px", 1200)) if hasattr(st, "secrets") else 1200
SIDEBAR_STATE = st.secrets.get("sidebar_state", "expanded") if hasattr(st, "secrets") else "expanded"

st.set_page_config(page_title=PAGE_TITLE, layout="wide", initial_sidebar_state=SIDEBAR_STATE)

st.markdown(
    f"""
    <style>
      .block-container {{
        max-width: {PAGE_MAX_WIDTH_PX}px;
      }}
      /* Tighter inputs in forms */
      .stTextInput > div > div > input {{
        line-height: 1.15;
      }}
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# DB helpers
# -----------------------------

def build_engine() -> Tuple[Engine, Dict[str, str]]:
    """Build a SQLAlchemy engine for Turso libSQL with fallback to local SQLite vendors.db."""
    # Prefer Streamlit secrets if set
    url = None
    auth_token = None

    try:
        url = st.secrets.get("TURSO_DATABASE_URL", None)
        auth_token = st.secrets.get("TURSO_AUTH_TOKEN", None)
    except Exception:
        url = os.environ.get("TURSO_DATABASE_URL")
        auth_token = os.environ.get("TURSO_AUTH_TOKEN")

    engine = None
    info = {
        "using_remote": False,
        "sqlalchemy_url": "",
        "dialect": "",
        "driver": "",
    }

    if url:
        # Ensure the URL is sqlalchemy-compatible for libsql
        if url.startswith("libsql://"):
            sa_url = "sqlite+libsql://" + url[len("libsql://"):]
        elif url.startswith("sqlite+libsql://"):
            sa_url = url
        else:
            # Allow passing full SQLAlchemy URL
            sa_url = url
        try:
            connect_args = {"auth_token": auth_token} if auth_token else {}
            engine = create_engine(sa_url, connect_args=connect_args, pool_pre_ping=True)
            # Lightweight probe
            with engine.connect() as conn:
                conn.execute(sql_text("SELECT 1"))
            info.update({"using_remote": True, "sqlalchemy_url": sa_url})
        except Exception as e:
            st.warning(f"Turso connection failed ({e}). Falling back to local SQLite vendors.db.")

    if engine is None:
        local_path = os.path.abspath(os.path.join(os.getcwd(), "vendors.db"))
        sa_url = f"sqlite:///{local_path}"
        engine = create_engine(sa_url, pool_pre_ping=True)
        info.update({"using_remote": False, "sqlalchemy_url": sa_url})

    # Fill dialect/driver info
    try:
        info["dialect"] = getattr(engine.dialect, "name", "")
        info["driver"] = getattr(engine.dialect, "driver", "")
    except Exception:
        pass

    return engine, info


def read_table(engine: Engine, name: str) -> pd.DataFrame:
    try:
        return pd.read_sql(sql_text(f"SELECT * FROM {name}"), engine)
    except Exception:
        return pd.DataFrame()


def norm_phone(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    digits = re.sub(r"\D", "", s)
    return digits if len(digits) == 10 else None


def title_case_name(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    return " ".join([w.capitalize() for w in s.split()])


def _safe_index(options: List[str], value: str) -> int:
    """Return index of value in options or 0 if missing."""
    try:
        return options.index(value)
    except ValueError:
        return 0


def _rerun():
    try:
        st.rerun()
    except Exception:
        st.experimental_rerun()

# -----------------------------
# Data accessors
# -----------------------------

@st.cache_data(ttl=60)
def load_all(engine: Engine) -> Dict[str, pd.DataFrame]:
    vendors = read_table(engine, "vendors")
    categories = read_table(engine, "categories")
    services = read_table(engine, "services")

    # Ensure expected columns exist (soft checks)
    for df, cols in [
        (vendors, ["id","category","service","business_name","contact_name","phone","address","website","notes","keywords","created_at","updated_at","updated_by"]),
        (categories, ["id","name"]),
        (services, ["id","name"]),
    ]:
        for c in cols:
            if c not in df.columns:
                df[c] = None

    return {"vendors": vendors, "categories": categories, "services": services}


def refresh_data_cache():
    load_all.clear()

# -----------------------------
# UI helpers
# -----------------------------

def info_badge(text: str):
    st.caption(text)


def quick_filter(df: pd.DataFrame, placeholder: str = "Global search (partial, case-insensitive)") -> pd.DataFrame:
    q = st.text_input("Search", "", placeholder=placeholder, key="global_search_input")
    if not q:
        return df
    ql = q.strip().lower()
    mask = pd.Series(False, index=df.index)
    for c in df.columns:
        try:
            mask = mask | df[c].astype(str).str.lower().str.contains(ql, na=False)
        except Exception:
            continue
    return df[mask]

# -----------------------------
# Browse tab (AgGrid)
# -----------------------------

def tab_browse(engine: Engine, data: Dict[str, pd.DataFrame]):
    st.subheader("Browse Vendors")
    info_badge("Global search across all fields (non-FTS, case-insensitive; matches partial words).")

    df = data["vendors"].copy()

    # ---- Enforce single 'URL' display column derived from 'website' ----
    if "URL" not in df.columns:
        if "Website" in df.columns:
            df["URL"] = df["Website"]
        elif "website" in df.columns:
            df["URL"] = df["website"]
        else:
            df["URL"] = ""
    # Drop any other variants from the grid
    for _c in ("website", "Website"):
        if _c in df.columns:
            df.drop(columns=[_c], inplace=True)

    # Column order / whitelist
    default_order = [
        "id","category","service","business_name","contact_name","phone","address","URL","notes","keywords",
        "created_at","updated_at","updated_by",
    ]
    cols_present = [c for c in default_order if c in df.columns]
    if cols_present:
        df = df[cols_present]

    # Global quick filter
    df_filtered = quick_filter(df)

    # Column width config via secrets (optional)
    col_widths = {}
    try:
        col_widths = dict(st.secrets.get("COLUMN_WIDTHS_PX_ADMIN", {}))
    except Exception:
        pass

    if AgGrid is None:
        st.error("st_aggrid is not installed. Add 'st-aggrid==0.3.5' to requirements.txt.")
        st.dataframe(df_filtered, use_container_width=True, hide_index=True)
        return

    gob = GridOptionsBuilder.from_dataframe(df_filtered)
    # Disable per-column filters; rely on our global search
    gob.configure_default_column(filter=False, sortable=True, wrapText=True, autoHeight=True)

    # Apply explicit widths if given
    for c in df_filtered.columns:
        try:
            width = int(col_widths.get(c, 0)) if col_widths else 0
        except Exception:
            width = 0
        if width > 0:
            gob.configure_column(c, width=width)

    grid_options = gob.build()

    AgGrid(
        df_filtered,
        gridOptions=grid_options,
        update_mode=GridUpdateMode.NO_UPDATE,
        height=520,
        fit_columns_on_grid_load=False,
        allow_unsafe_jscode=False,  # avoid React #31 issues from HTML renderers
    )

# -----------------------------
# Add / Edit / Delete tab
# -----------------------------

def tab_edit(engine: Engine, data: Dict[str, pd.DataFrame]):
    st.subheader("Add / Edit / Delete Vendor")

    vendors = data["vendors"].copy()
    categories = sorted([x for x in data["categories"]["name"].dropna().astype(str).unique()])
    services = sorted([x for x in data["services"]["name"].dropna().astype(str).unique()])

    with st.expander("Add Vendor", expanded=True):
        c1, c2, c3 = st.columns([1,1,1])
        with c1:
            category = st.selectbox("Category (required)", options=[""] + categories, index=0, key="add_category")
        with c2:
            service = st.selectbox("Service (optional)", options=[""] + services, index=0, key="add_service")
        with c3:
            business_name = st.text_input("Business Name (required)", "", key="add_business_name")
        c4, c5, c6 = st.columns([1,1,1])
        with c4:
            contact_name = st.text_input("Contact Name", "", key="add_contact_name")
        with c5:
            phone = st.text_input("Phone (10 digits)", "", key="add_phone")
        with c6:
            url_val = st.text_input("URL", "", key="add_url")
        address = st.text_input("Address", "", key="add_address")
        notes = st.text_area("Notes", "", key="add_notes")
        keywords = st.text_input("Keywords", "", key="add_keywords")

        if st.button("Add Vendor", type="primary", key="btn_add_vendor"):
            # Validations
            if not business_name.strip():
                st.error("Business Name is required.")
                st.stop()
            if not category.strip():
                st.error("Category is required.")
                st.stop()
            if phone.strip():
                p = norm_phone(phone)
                if not p:
                    st.error("Phone must be 10 digits (numbers only) or left blank.")
                    st.stop()
            else:
                p = None

            # Category/Service existence checks
            if category and category not in categories:
                st.error("Category must exist in the Categories library.")
                st.stop()
            if service and service not in services:
                st.error("Service must exist in the Services library (or leave blank).")
                st.stop()

            with engine.begin() as conn:
                conn.execute(
                    sql_text(
                        """
                        INSERT INTO vendors (category, service, business_name, contact_name, phone, address, website, notes, keywords, created_at, updated_at, updated_by)
                        VALUES (:category, :service, :business_name, :contact_name, :phone, :address, :website, :notes, :keywords, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, :updated_by)
                        """
                    ),
                    {
                        "category": category.strip() or None,
                        "service": service.strip() or None,
                        "business_name": business_name.strip(),
                        "contact_name": title_case_name(contact_name.strip()) if contact_name else None,
                        "phone": p,
                        "address": address.strip() or None,
                        "website": url_val.strip() or None,  # save under DB column 'website'
                        "notes": notes.strip() or None,
                        "keywords": keywords.strip() or None,
                        "updated_by": "admin",
                    },
                )
            st.success("Vendor added.")
            refresh_data_cache()
            _rerun()

    with st.expander("Edit / Delete Vendor", expanded=False):
        # Select a vendor by ID
        id_list = vendors["id"].tolist() if "id" in vendors.columns else []
        id_selected = st.selectbox("Select Vendor ID", options=[""] + [str(x) for x in id_list], index=0, key="edit_pick_id")
        if id_selected:
            try:
                vid = int(id_selected)
            except Exception:
                st.stop()

            row = vendors.loc[vendors["id"] == vid]
            if row.empty:
                st.warning("Selected vendor not found.")
                st.stop()
            row = row.iloc[0].to_dict()

            categories_opts = [""] + categories
            services_opts = [""] + services

            c1, c2, c3 = st.columns([1,1,1])
            with c1:
                category = st.selectbox("Category (required)", options=categories_opts,
                                        index=_safe_index(categories_opts, str(row.get("category") or "")),
                                        key="edit_category")
            with c2:
                service = st.selectbox("Service (optional)", options=services_opts,
                                       index=_safe_index(services_opts, str(row.get("service") or "")),
                                       key="edit_service")
            with c3:
                business_name = st.text_input("Business Name (required)", str(row.get("business_name") or ""), key="edit_business_name")
            c4, c5, c6 = st.columns([1,1,1])
            with c4:
                contact_name = st.text_input("Contact Name", str(row.get("contact_name") or ""), key="edit_contact_name")
            with c5:
                phone = st.text_input("Phone (10 digits)", str(row.get("phone") or ""), key="edit_phone")
            with c6:
                url_val = st.text_input("URL", str(row.get("website") or ""), key="edit_url")  # read from DB 'website'
            address = st.text_input("Address", str(row.get("address") or ""), key="edit_address")
            notes = st.text_area("Notes", str(row.get("notes") or ""), key="edit_notes")
            keywords = st.text_input("Keywords", str(row.get("keywords") or ""), key="edit_keywords")

            cA, cB, cC = st.columns([1,1,1])
            with cA:
                if st.button("Save Changes", type="primary", key="btn_save_vendor"):
                    if not business_name.strip():
                        st.error("Business Name is required.")
                        st.stop()
                    if not category.strip():
                        st.error("Category is required.")
                        st.stop()
                    if phone.strip():
                        p = norm_phone(phone)
                        if not p:
                            st.error("Phone must be 10 digits (numbers only) or left blank.")
                            st.stop()
                    else:
                        p = None

                    if category and category not in categories:
                        st.error("Category must exist in the Categories library.")
                        st.stop()
                    if service and service not in services:
                        st.error("Service must exist in the Services library (or leave blank).")
                        st.stop()

                    with engine.begin() as conn:
                        conn.execute(
                            sql_text(
                                """
                                UPDATE vendors
                                SET category=:category, service=:service, business_name=:business_name,
                                    contact_name=:contact_name, phone=:phone, address=:address, website=:website,
                                    notes=:notes, keywords=:keywords, updated_at=CURRENT_TIMESTAMP, updated_by=:updated_by
                                WHERE id=:id
                                """
                            ),
                            {
                                "id": vid,
                                "category": category.strip() or None,
                                "service": service.strip() or None,
                                "business_name": business_name.strip(),
                                "contact_name": title_case_name(contact_name.strip()) if contact_name else None,
                                "phone": p,
                                "address": address.strip() or None,
                                "website": url_val.strip() or None,  # save under DB column 'website'
                                "notes": notes.strip() or None,
                                "keywords": keywords.strip() or None,
                                "updated_by": "admin",
                            },
                        )
                    st.success("Vendor updated.")
                    refresh_data_cache()
                    _rerun()

            with cB:
                if st.button("Delete Vendor", type="secondary", key="btn_delete_vendor"):
                    with engine.begin() as conn:
                        conn.execute(sql_text("DELETE FROM vendors WHERE id=:id"), {"id": vid})
                    st.success("Vendor deleted.")
                    refresh_data_cache()
                    _rerun()

# -----------------------------
# Category & Service Admin
# -----------------------------

def usage_counts(vendors: pd.DataFrame, field: str) -> Dict[str, int]:
    s = vendors[field].dropna().astype(str)
    return dict(s.value_counts())


def tab_refdata(engine: Engine, data: Dict[str, pd.DataFrame]):
    st.subheader("Category & Service Admin")

    vendors = data["vendors"].copy()
    cat_df = data["categories"].copy()
    svc_df = data["services"].copy()

    cat_counts = usage_counts(vendors, "category")
    svc_counts = usage_counts(vendors, "service")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Categories**")
        existing_cats = sorted(cat_df["name"].dropna().astype(str).unique())
        new_cat = st.text_input("Add Category", "", key="add_cat")
        if st.button("Add", key="btn_add_cat"):
            if not new_cat.strip():
                st.error("Category name required.")
            elif new_cat in existing_cats:
                st.warning("Category already exists.")
            else:
                with engine.begin() as conn:
                    conn.execute(sql_text("INSERT OR IGNORE INTO categories(name) VALUES(:n)"), {"n": new_cat.strip()})
                st.success("Category added.")
                refresh_data_cache()
                _rerun()

        cat_to_rename = st.selectbox("Rename Category", options=[""] + existing_cats, index=0, key="rename_cat_from")
        cat_new_name = st.text_input("New name", "", key="rename_cat_to")
        if st.button("Rename", key="btn_rename_cat"):
            if not cat_to_rename:
                st.error("Pick a category to rename.")
            elif not cat_new_name.strip():
                st.error("New name required.")
            else:
                with engine.begin() as conn:
                    conn.execute(sql_text("UPDATE categories SET name=:to WHERE name=:frm"), {"to": cat_new_name.strip(), "frm": cat_to_rename})
                    conn.execute(sql_text("UPDATE vendors SET category=:to WHERE category=:frm"), {"to": cat_new_name.strip(), "frm": cat_to_rename})
                st.success("Category renamed (and vendor rows updated).")
                refresh_data_cache()
                _rerun()

        cat_to_delete = st.selectbox("Delete Category (only if unused)", options=[""] + existing_cats, index=0, key="delete_cat")
        if st.button("Delete Category", key="btn_delete_cat"):
            if not cat_to_delete:
                st.error("Pick a category to delete.")
            elif cat_counts.get(cat_to_delete, 0) > 0:
                st.error("Cannot delete: category is in use by vendors.")
            else:
                with engine.begin() as conn:
                    conn.execute(sql_text("DELETE FROM categories WHERE name=:n"), {"n": cat_to_delete})
                st.success("Category deleted.")
                refresh_data_cache()
                _rerun()

        # Orphans used by vendors but missing in categories table
        used_cats = set(vendors["category"].dropna().astype(str).unique())
        known_cats = set(existing_cats)
        cat_orphans = sorted(list(used_cats - known_cats))
        if cat_orphans:
            st.info("Orphan categories in vendors (not in categories table): " + ", ".join(cat_orphans))

    with c2:
        st.markdown("**Services**")
        existing_svcs = sorted(svc_df["name"].dropna().astype(str).unique())
        new_svc = st.text_input("Add Service", "", key="add_svc")
        if st.button("Add", key="btn_add_svc"):
            if not new_svc.strip():
                st.error("Service name required.")
            elif new_svc in existing_svcs:
                st.warning("Service already exists.")
            else:
                with engine.begin() as conn:
                    conn.execute(sql_text("INSERT OR IGNORE INTO services(name) VALUES(:n)"), {"n": new_svc.strip()})
                st.success("Service added.")
                refresh_data_cache()
                _rerun()

        svc_to_rename = st.selectbox("Rename Service", options=[""] + existing_svcs, index=0, key="rename_svc_from")
        svc_new_name = st.text_input("New name", "", key="rename_svc_to")
        if st.button("Rename", key="btn_rename_svc"):
            if not svc_to_rename:
                st.error("Pick a service to rename.")
            elif not svc_new_name.strip():
                st.error("New name required.")
            else:
                with engine.begin() as conn:
                    conn.execute(sql_text("UPDATE services SET name=:to WHERE name=:frm"), {"to": svc_new_name.strip(), "frm": svc_to_rename})
                    conn.execute(sql_text("UPDATE vendors SET service=:to WHERE service=:frm"), {"to": svc_new_name.strip(), "frm": svc_to_rename})
                st.success("Service renamed (and vendor rows updated).")
                refresh_data_cache()
                _rerun()

        svc_to_delete = st.selectbox("Delete Service (only if unused)", options=[""] + existing_svcs, index=0, key="delete_svc")
        if st.button("Delete Service", key="btn_delete_svc"):
            if not svc_to_delete:
                st.error("Pick a service to delete.")
            elif svc_counts.get(svc_to_delete, 0) > 0:
                st.error("Cannot delete: service is in use by vendors.")
            else:
                with engine.begin() as conn:
                    conn.execute(sql_text("DELETE FROM services WHERE name=:n"), {"n": svc_to_delete})
                st.success("Service deleted.")
                refresh_data_cache()
                _rerun()

        used_svcs = set(vendors["service"].dropna().astype(str).unique())
        known_svcs = set(existing_svcs)
        svc_orphans = sorted(list(used_svcs - known_svcs))
        if svc_orphans:
            st.info("Orphan services in vendors (not in services table): " + ", ".join(svc_orphans))

# -----------------------------
# Maintenance tab
# -----------------------------

def tab_maint(engine: Engine, data: Dict[str, pd.DataFrame]):
    st.subheader("Maintenance")

    c1, c2, c3 = st.columns(3)

    with c1:
        if st.button("Normalize phones (10 digits)", key="btn_norm_phone"):
            vendors = data["vendors"].copy()
            updated = 0
            with engine.begin() as conn:
                for _, r in vendors.iterrows():
                    p = norm_phone(str(r.get("phone") or ""))
                    if p and p != r.get("phone"):
                        conn.execute(sql_text("UPDATE vendors SET phone=:p WHERE id=:id"), {"p": p, "id": int(r["id"])})
                        updated += 1
            st.success(f"Normalized {updated} phone numbers.")
            refresh_data_cache()

    with c2:
        if st.button("Title-case contact names", key="btn_tc_names"):
            vendors = data["vendors"].copy()
            updated = 0
            with engine.begin() as conn:
                for _, r in vendors.iterrows():
                    cname = str(r.get("contact_name") or "").strip()
                    tc = title_case_name(cname) if cname else None
                    if tc and tc != r.get("contact_name"):
                        conn.execute(sql_text("UPDATE vendors SET contact_name=:c WHERE id=:id"), {"c": tc, "id": int(r["id"])})
                        updated += 1
            st.success(f"Updated {updated} contact names.")
            refresh_data_cache()

    with c3:
        if st.button("Backfill audit columns", key="btn_backfill_audit"):
            with engine.begin() as conn:
                conn.execute(sql_text("UPDATE vendors SET updated_at=COALESCE(updated_at, CURRENT_TIMESTAMP)"))
                conn.execute(sql_text("UPDATE vendors SET created_at=COALESCE(created_at, updated_at, CURRENT_TIMESTAMP)"))
                conn.execute(sql_text("UPDATE vendors SET updated_by=COALESCE(updated_by, 'admin')"))
            st.success("Backfilled created_at/updated_at/updated_by.")
            refresh_data_cache()

# -----------------------------
# Debug tab
# -----------------------------

def tab_debug(engine: Engine, engine_info: Dict[str, str], data: Dict[str, pd.DataFrame]):
    st.subheader("Status & Secrets (debug)")

    try:
        debug_flag = bool(int(st.secrets.get("ADMIN_DEBUG", 1)))
    except Exception:
        debug_flag = True

    if not debug_flag:
        st.info("Debug disabled via ADMIN_DEBUG secret.")
        return

    st.json(engine_info)

    # Schema snapshot
    vendors = data["vendors"]
    categories = data["categories"]
    services = data["services"]

    schema = {
        "vendors_columns": list(vendors.columns),
        "categories_columns": list(categories.columns),
        "services_columns": list(services.columns),
        "counts": {
            "vendors": int(vendors.shape[0]),
            "categories": int(categories.shape[0]),
            "services": int(services.shape[0]),
        },
    }
    st.json(schema)

# -----------------------------
# Main
# -----------------------------

def main():
    engine, engine_info = build_engine()
    data = load_all(engine)

    tabs = st.tabs(["Browse", "Add/Edit/Delete", "Ref Data", "Maintenance", "Debug"])

    with tabs[0]:
        tab_browse(engine, data)
    with tabs[1]:
        tab_edit(engine, data)
    with tabs[2]:
        tab_refdata(engine, data)
    with tabs[3]:
        tab_maint(engine, data)
    with tabs[4]:
        tab_debug(engine, engine_info, data)


if __name__ == "__main__":
    main()
