# app_admin.py — Vendors Admin (v3.4) with Audit Trail + Changelog
# Tabs: View | Add | Edit | Delete | Categories Admin | Services Admin | Maintenance | Changelog
# - Same UX as v3.3 (validators, widths/labels/help, website link + URL col, CSV, debug-at-end)
# - NEW: Audit trail (created_at, updated_at, updated_by) on vendors + vendor_changes history table
# - Auto-migration: adds missing vendors audit columns; creates vendor_changes if absent

from __future__ import annotations

import os
import re
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine
from sqlalchemy.sql import text as sql_text
from urllib.parse import urlparse
from datetime import datetime, timezone

# -----------------------------
# Secrets / env helpers
# -----------------------------
def _read_secret(name: str, default=None):
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.environ.get(name, default)

def _get_bool(val, default=False):
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() in ("1", "true", "yes", "on")
    return bool(default)

def _now_iso() -> str:
    # Store as UTC ISO 8601 with seconds precision
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

# -----------------------------
# Page config
# -----------------------------
def _apply_layout():
    st.set_page_config(
        page_title=_read_secret("page_title", "HCR Vendors Admin"),
        layout="wide",
        initial_sidebar_state=_read_secret("sidebar_state", "expanded"),
    )
    maxw = _read_secret("page_max_width_px", 2300)
    try:
        maxw = int(maxw)
    except Exception:
        maxw = 2300
    st.markdown(
        f"""
        <style>
        .block-container {{ max-width: {maxw}px; }}
        </style>
        """,
        unsafe_allow_html=True,
    )
_apply_layout()

# -----------------------------
# Global config from secrets
# -----------------------------
LABEL_OVERRIDES: Dict[str, str] = _read_secret("READONLY_COLUMN_LABELS", {}) or {}
COLUMN_WIDTHS: Dict[str, int] = _read_secret("COLUMN_WIDTHS_PX_READONLY", {}) or {}
STICKY_FIRST: bool = _get_bool(_read_secret("READONLY_STICKY_FIRST_COL", False), False)

HELP_TITLE: str = (
    _read_secret("READONLY_HELP_TITLE")
    or LABEL_OVERRIDES.get("readonly_help_title")
    or "Providers Help / Tips"
)
HELP_MD: str = (
    _read_secret("READONLY_HELP_MD")
    or LABEL_OVERRIDES.get("readonly_help_md")
    or ""
)
HELP_DEBUG: bool = _get_bool(
    _read_secret("READONLY_HELP_DEBUG"),
    _get_bool(LABEL_OVERRIDES.get("readonly_help_debug"), False),
)

ADMIN_DISPLAY_NAME: str = _read_secret("ADMIN_DISPLAY_NAME", "Admin")

RAW_COLS = [
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
    # audit columns are handled separately; they may or may not exist initially
]

AUDIT_COLS = ["created_at", "updated_at", "updated_by"]

# -----------------------------
# Engine
# -----------------------------
def _engine():
    url = _read_secret(
        "TURSO_DATABASE_URL",
        "sqlite+libsql://vendors-prod-mrbookend.aws-us-west-2.turso.io?secure=true",
    )
    token = _read_secret("TURSO_AUTH_TOKEN", None)
    if isinstance(url, str) and url.startswith("sqlite+libsql://"):
        try:
            eng = create_engine(url, connect_args={"auth_token": token} if token else {}, pool_pre_ping=True)
            with eng.connect() as c:
                c.execute(sql_text("SELECT 1"))
            return eng
        except Exception as e:
            st.warning(f"Turso connection failed ({e}). Falling back to local SQLite vendors.db.")
    local = "/mount/src/vendors-readonly-app/vendors.db"
    if not os.path.exists(local):
        local = "vendors.db"
    return create_engine(f"sqlite:///{local}")
eng = _engine()

# -----------------------------
# DB migration (adds audit columns & change log table)
# -----------------------------
def _ensure_schema():
    with eng.begin() as c:
        # Add audit columns to vendors if missing
        # Note: simple ADD COLUMN is supported by SQLite/libsql
        # created_at
        c.execute(sql_text(
            "ALTER TABLE vendors ADD COLUMN created_at TEXT"
        )) if not _column_exists(c, "vendors", "created_at") else None
        # updated_at
        c.execute(sql_text(
            "ALTER TABLE vendors ADD COLUMN updated_at TEXT"
        )) if not _column_exists(c, "vendors", "updated_at") else None
        # updated_by
        c.execute(sql_text(
            "ALTER TABLE vendors ADD COLUMN updated_by TEXT"
        )) if not _column_exists(c, "vendors", "updated_by") else None

        # vendor_changes table
        c.execute(sql_text("""
            CREATE TABLE IF NOT EXISTS vendor_changes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                vendor_id INTEGER NOT NULL,
                changed_at TEXT NOT NULL,
                changed_by TEXT NOT NULL,
                action TEXT NOT NULL,             -- 'insert' | 'update' | 'delete'
                field TEXT,                       -- null for insert/delete; otherwise changed column
                old_value TEXT,
                new_value TEXT
            )
        """))

def _column_exists(conn, table: str, col: str) -> bool:
    try:
        res = conn.execute(sql_text(f"PRAGMA table_info({table})")).fetchall()
        cols = [r[1] for r in res]  # (cid, name, type, notnull, dflt, pk)
        return col in cols
    except Exception:
        return False

_ensure_schema()

# -----------------------------
# CSS widths / sticky (same as Read-Only)
# -----------------------------
def _apply_css(field_order: List[str]):
    rules = []
    for idx, col in enumerate(field_order, start=1):
        px = COLUMN_WIDTHS.get(col)
        if not px:
            continue
        rules.append(
            f"""
            div[data-testid='stDataFrame'] table thead tr th:nth-child({idx}),
            div[data-testid='stDataFrame'] table tbody tr td:nth-child({idx}) {{
                min-width:{px}px !important; max-width:{px}px !important; width:{px}px !important;
                white-space: normal !important; overflow-wrap:anywhere !important; word-break:break-word !important;
            }}
            """
        )
    sticky = ""
    if STICKY_FIRST and field_order:
        sticky = """
        div[data-testid='stDataFrame'] table thead tr th:nth-child(1),
        div[data-testid='stDataFrame'] table tbody tr td:nth-child(1){
            position:sticky;left:0;z-index:2;background:var(--background-color,white);box-shadow:1px 0 0 rgba(0,0,0,0.06);
        }
        div[data-testid='stDataFrame'] table thead tr th:nth-child(1){z-index:3;}
        """
    st.markdown(
        f"""
        <style>
        div[data-testid='stDataFrame'] table {{ table-layout: fixed !important; }}
        div[data-testid='stDataFrame'] table td, div[data-testid='stDataFrame'] table th {{
            white-space: normal !important; overflow-wrap:anywhere !important; word-break:break-word !important;
        }}
        {''.join(rules)}
        {sticky}
        </style>
        """,
        unsafe_allow_html=True,
    )

# -----------------------------
# URL + phone helpers (validators/formatters)
# -----------------------------
def _normalize_url(u: str) -> str:
    """Return normalized URL or empty string if invalid."""
    s = (u or "").strip()
    if not s:
        return ""
    if not s.lower().startswith(("http://", "https://")):
        s = "http://" + s
    try:
        p = urlparse(s)
        if p.scheme not in ("http", "https"):
            return ""
        if not p.netloc or "." not in p.netloc:
            return ""
        return s
    except Exception:
        return ""

def _digits_only(s: str) -> str:
    return re.sub(r"[^0-9]", "", s or "")

def _format_phone_10(digits: str) -> str:
    d = _digits_only(digits)
    if len(d) != 10:
        return ""
    return f"({d[0:3]}) {d[3:6]}-{d[6:10]}"

def _validate_vendor_fields(
    business_name: str,
    category: str,
    service: str,
    phone: str,
    website: str,
) -> Tuple[bool, Dict[str, str], Dict[str, str]]:
    errors: Dict[str, str] = {}
    cleaned: Dict[str, str] = {}

    # Business name
    if not (business_name or "").strip():
        errors["business_name"] = "Business name is required."
    else:
        cleaned["business_name"] = business_name.strip()

    # Category
    if not (category or "").strip():
        errors["category"] = "Category is required."
    else:
        cleaned["category"] = category.strip()

    # Service required unless Home Repair
    if (category or "").strip().lower() != "home repair":
        if not (service or "").strip():
            errors["service"] = "Service is required unless Category is 'Home Repair'."
        else:
            cleaned["service"] = service.strip()
    else:
        cleaned["service"] = (service or "").strip()

    # Phone
    fmt_phone = _format_phone_10(phone)
    if not fmt_phone:
        errors["phone"] = "Phone must have exactly 10 digits."
    else:
        cleaned["phone"] = fmt_phone

    # Website
    web_norm = _normalize_url(website)
    if (website or "").strip() and not web_norm:
        errors["website"] = "Website must start with http/https and include a valid host (e.g., example.com)."
    else:
        cleaned["website"] = web_norm  # may be ""

    return (len(errors) == 0), cleaned, errors

# -----------------------------
# Help (single expander; Markdown)
# -----------------------------
def render_help():
    with st.expander(HELP_TITLE, expanded=False):
        if HELP_MD:
            st.markdown(HELP_MD)
            if HELP_DEBUG:
                st.caption("Raw help (debug preview)")
                st.code(HELP_MD, language=None)
        else:
            st.info("No help content has been configured yet.")

# -----------------------------
# Data access + Audit helpers
# -----------------------------
def _fetch_df() -> pd.DataFrame:
    with eng.connect() as c:
        df = pd.read_sql_query(sql_text("SELECT * FROM vendors"), c)
    # Ensure columns exist
    for col in RAW_COLS + AUDIT_COLS:
        if col not in df.columns:
            df[col] = ""
    return df[RAW_COLS + AUDIT_COLS]

def _cats() -> pd.DataFrame:
    with eng.connect() as c:
        try:
            return pd.read_sql_query(sql_text("SELECT id, name FROM categories ORDER BY name COLLATE NOCASE"), c)
        except Exception:
            return pd.DataFrame(columns=["id", "name"]).astype({"id": int, "name": str})

def _svcs() -> pd.DataFrame:
    with eng.connect() as c:
        try:
            return pd.read_sql_query(sql_text("SELECT id, name FROM services ORDER BY name COLLATE NOCASE"), c)
        except Exception:
            return pd.DataFrame(columns=["id", "name"]).astype({"id": int, "name": str})

def _insert_vendor(row: Dict[str, str]):
    now = _now_iso()
    row2 = dict(row)
    row2["created_at"] = now
    row2["updated_at"] = now
    row2["updated_by"] = ADMIN_DISPLAY_NAME
    q = sql_text("""
        INSERT INTO vendors(category, service, business_name, contact_name, phone, address, website, notes, keywords, created_at, updated_at, updated_by)
        VALUES(:category,:service,:business_name,:contact_name,:phone,:address,:website,:notes,:keywords,:created_at,:updated_at,:updated_by)
    """)
    with eng.begin() as c:
        res = c.execute(q, row2)
        # vendor id
        vid = c.execute(sql_text("SELECT last_insert_rowid()")).scalar()
        # changelog insert
        c.execute(sql_text("""
            INSERT INTO vendor_changes(vendor_id, changed_at, changed_by, action, field, old_value, new_value)
            VALUES(:vendor_id, :changed_at, :changed_by, 'insert', NULL, NULL, NULL)
        """), {"vendor_id": vid, "changed_at": now, "changed_by": ADMIN_DISPLAY_NAME})

def _update_vendor(vid: int, new_row: Dict[str, str]):
    # compute diffs against current
    with eng.begin() as c:
        cur = c.execute(sql_text("SELECT * FROM vendors WHERE id = :id"), {"id": vid}).mappings().first()
        if not cur:
            return
        now = _now_iso()
        # prepare updates and audit fields
        upd = dict(new_row)
        upd["updated_at"] = now
        upd["updated_by"] = ADMIN_DISPLAY_NAME
        sets = ", ".join([f"{k} = :{k}" for k in upd.keys()])
        c.execute(sql_text(f"UPDATE vendors SET {sets} WHERE id = :id"), {**upd, "id": vid})
        # write change rows for fields that changed
        for k, newv in new_row.items():
            oldv = cur.get(k, "")
            if (oldv or "") != (newv or ""):
                c.execute(sql_text("""
                    INSERT INTO vendor_changes(vendor_id, changed_at, changed_by, action, field, old_value, new_value)
                    VALUES(:vendor_id, :changed_at, :changed_by, 'update', :field, :old_value, :new_value)
                """), {
                    "vendor_id": vid,
                    "changed_at": now,
                    "changed_by": ADMIN_DISPLAY_NAME,
                    "field": k,
                    "old_value": str(oldv) if oldv is not None else "",
                    "new_value": str(newv) if newv is not None else "",
                })

def _delete_vendor(vid: int):
    now = _now_iso()
    with eng.begin() as c:
        c.execute(sql_text("""
            INSERT INTO vendor_changes(vendor_id, changed_at, changed_by, action, field, old_value, new_value)
            VALUES(:vendor_id, :changed_at, :changed_by, 'delete', NULL, NULL, NULL)
        """), {"vendor_id": vid, "changed_at": now, "changed_by": ADMIN_DISPLAY_NAME})
        c.execute(sql_text("DELETE FROM vendors WHERE id = :id"), {"id": vid})

def _upsert_category(name: str):
    with eng.begin() as c:
        c.execute(sql_text("INSERT OR IGNORE INTO categories(name) VALUES(:n)"), {"n": name.strip()})

def _delete_category(name: str):
    with eng.begin() as c:
        c.execute(sql_text("DELETE FROM categories WHERE name = :n"), {"n": name.strip()})

def _upsert_service(name: str):
    with eng.begin() as c:
        c.execute(sql_text("INSERT OR IGNORE INTO services(name) VALUES(:n)"), {"n": name.strip()})

def _delete_service(name: str):
    with eng.begin() as c:
        c.execute(sql_text("DELETE FROM services WHERE name = :n"), {"n": name.strip()})

# -----------------------------
# Label mapping (display only)
# -----------------------------
def _apply_labels(df: pd.DataFrame) -> pd.DataFrame:
    if not LABEL_OVERRIDES:
        return df
    mapping = {k: v for k, v in LABEL_OVERRIDES.items() if k in df.columns and isinstance(v, str) and v}
    return df.rename(columns=mapping)

# -----------------------------
# VIEW tab
# -----------------------------
def _apply_table_css_for_view(df_raw_cols: List[str]):
    _apply_css(df_raw_cols)

def tab_view():
    df = _fetch_df()
    df_show = _apply_labels(df)

    _apply_table_css_for_view([c for c in df.columns if c in COLUMN_WIDTHS or c in RAW_COLS])

    # Website link with adjacent full URL column
    website_key = LABEL_OVERRIDES.get("website", "website")
    _df = df_show.copy()
    if website_key in _df.columns:
        try:
            base_col = "website" if "website" in df.columns else website_key
            norm = df[base_col].apply(_normalize_url) if base_col in df.columns else _df[website_key].apply(_normalize_url)
            url_col = f"{website_key} URL" if website_key.lower() != "website" else "Website URL"
            w_idx = _df.columns.get_loc(website_key)
            _df.insert(w_idx + 1, url_col, norm)
            st.dataframe(
                _df.assign(**{website_key: norm}),
                use_container_width=True,
                hide_index=True,
                column_config={website_key: st.column_config.LinkColumn(display_text="Website")},
            )
        except Exception:
            st.dataframe(_df, use_container_width=True, hide_index=True)
    else:
        st.dataframe(_df, use_container_width=True, hide_index=True)

    st.download_button(
        label="Download Providers (CSV)",
        data=df_show.to_csv(index=False).encode("utf-8"),
        file_name="providers_admin_view.csv",
        mime="text/csv",
    )

# -----------------------------
# ADD tab (validators + audit)
# -----------------------------
def tab_add():
    cats = _cats()["name"].tolist()
    svcs = _svcs()["name"].tolist()

    st.info("Fields in **bold** are required. Service is optional only when Category is 'Home Repair'.")
    with st.form("add_vendor_form", clear_on_submit=True):
        col = st.columns(2)
        with col[0]:
            business_name = st.text_input(f"**{LABEL_OVERRIDES.get('business_name','business_name')}** *", placeholder="Acme Plumbing")
            category = st.selectbox(LABEL_OVERRIDES.get("category", "category"), options=sorted(cats))
            service = st.selectbox(LABEL_OVERRIDES.get("service", "service"), options=[""] + sorted(svcs))
            contact_name = st.text_input(LABEL_OVERRIDES.get("contact_name", "contact_name"))
            phone = st.text_input(LABEL_OVERRIDES.get("phone", "phone"), placeholder="(210) 555-0123 or 210-555-0123")
        with col[1]:
            address = st.text_input(LABEL_OVERRIDES.get("address", "address"))
            website = st.text_input(LABEL_OVERRIDES.get("website", "website"), placeholder="https://example.com")
            notes = st.text_area(LABEL_OVERRIDES.get("notes", "notes"))
            keywords = st.text_input(LABEL_OVERRIDES.get("keywords", "keywords"))

        submitted = st.form_submit_button("Add")
        if submitted:
            ok, cleaned, errors = _validate_vendor_fields(
                business_name=business_name,
                category=category,
                service=service,
                phone=phone,
                website=website,
            )
            if not ok:
                for field, msg in errors.items():
                    st.error(f"{field}: {msg}")
                return

            row = {
                "business_name": cleaned["business_name"],
                "category": cleaned["category"],
                "service": cleaned["service"],
                "contact_name": (contact_name or "").strip(),
                "phone": cleaned["phone"],
                "address": (address or "").strip(),
                "website": cleaned["website"],
                "notes": (notes or "").strip(),
                "keywords": (keywords or "").strip(),
            }
            _insert_vendor(row)
            st.success("Vendor added.")

# -----------------------------
# EDIT tab (validators + audit)
# -----------------------------
def tab_edit():
    df = _fetch_df()
    if df.empty:
        st.info("No vendors available.")
        return

    df_sel = df.copy()
    df_sel["label"] = (
        df_sel["business_name"].fillna("")
        + " ("
        + df_sel["category"].fillna("")
        + ") — id#"
        + df_sel["id"].astype(str)
    )
    choice = st.selectbox("Select Vendor", options=df_sel["label"].tolist())
    vid = int(choice.split("id#")[-1])
    row = df[df["id"] == vid].iloc[0].to_dict()

    cats = _cats()["name"].tolist()
    svcs = _svcs()["name"].tolist()

    with st.form("edit_vendor_form"):
        col = st.columns(2)
        with col[0]:
            row["business_name"] = st.text_input(LABEL_OVERRIDES.get("business_name", "business_name"), value=row["business_name"])
            row["category"] = st.selectbox(
                LABEL_OVERRIDES.get("category", "category"),
                options=sorted(cats),
                index=max(0, sorted(cats).index(row["category"]) if row["category"] in cats else 0),
            )
            row["service"] = st.selectbox(
                LABEL_OVERRIDES.get("service", "service"),
                options=[""] + sorted(svcs),
                index=0 if not row.get("service") else ([""] + sorted(svcs)).index(row["service"]) if row["service"] in svcs else 0,
            )
            row["contact_name"] = st.text_input(LABEL_OVERRIDES.get("contact_name", "contact_name"), value=row.get("contact_name", ""))
            row["phone"] = st.text_input(LABEL_OVERRIDES.get("phone", "phone"), value=row.get("phone", ""))
        with col[1]:
            row["address"] = st.text_input(LABEL_OVERRIDES.get("address", "address"), value=row.get("address", ""))
            row["website"] = st.text_input(LABEL_OVERRIDES.get("website", "website"), value=row.get("website", ""))
            row["notes"] = st.text_area(LABEL_OVERRIDES.get("notes", "notes"), value=row.get("notes", ""))
            row["keywords"] = st.text_input(LABEL_OVERRIDES.get("keywords", "keywords"), value=row.get("keywords", ""))

        col2 = st.columns(2)
        with col2[0]:
            if st.form_submit_button("Save Changes"):
                ok, cleaned, errors = _validate_vendor_fields(
                    business_name=row.get("business_name", ""),
                    category=row.get("category", ""),
                    service=row.get("service", ""),
                    phone=row.get("phone", ""),
                    website=row.get("website", ""),
                )
                if not ok:
                    for field, msg in errors.items():
                        st.error(f"{field}: {msg}")
                    return

                upd = {
                    "business_name": cleaned["business_name"],
                    "category": cleaned["category"],
                    "service": cleaned["service"],
                    "contact_name": (row.get("contact_name", "")).strip(),
                    "phone": cleaned["phone"],
                    "address": (row.get("address", "")).strip(),
                    "website": cleaned["website"],
                    "notes": (row.get("notes", "")).strip(),
                    "keywords": (row.get("keywords", "")).strip(),
                }
                _update_vendor(vid, upd)
                st.success("Saved.")
        with col2[1]:
            if st.form_submit_button("Cancel"):
                st.stop()

# -----------------------------
# DELETE tab (audited delete)
# -----------------------------
def tab_delete():
    df = _fetch_df()
    if df.empty:
        st.info("No vendors to delete.")
        return
    df_sel = df.copy()
    df_sel["label"] = (
        df_sel["business_name"].fillna("")
        + " ("
        + df_sel["category"].fillna("")
        + ") — id#"
        + df_sel["id"].astype(str)
    )
    choice = st.selectbox("Select Vendor to Delete", options=df_sel["label"].tolist())
    vid = int(choice.split("id#")[-1])
    if st.button("Delete", type="primary"):
        _delete_vendor(vid)
        st.success("Deleted.")

# -----------------------------
# CATEGORIES ADMIN
# -----------------------------
def tab_categories():
    st.subheader("Categories Admin")
    cats = _cats()
    st.dataframe(cats, use_container_width=True, hide_index=True)
    with st.form("add_cat"):
        new = st.text_input("Add Category")
        if st.form_submit_button("Add") and (new or "").strip():
            _upsert_category(new)
            st.success("Category added (or already existed). Reload to see it in the list.")
    with st.form("del_cat"):
        delname = st.text_input("Delete Category (exact name)")
        if st.form_submit_button("Delete") and (delname or "").strip():
            _delete_category(delname)
            st.success("Category deleted if it existed. Reassign vendors first if needed.")

# -----------------------------
# SERVICES ADMIN
# -----------------------------
def tab_services():
    st.subheader("Services Admin")
    svcs = _svcs()
    st.dataframe(svcs, use_container_width=True, hide_index=True)
    with st.form("add_svc"):
        new = st.text_input("Add Service")
        if st.form_submit_button("Add") and (new or "").strip():
            _upsert_service(new)
            st.success("Service added (or already existed). Reload to see it in the list.")
    with st.form("del_svc"):
        delname = st.text_input("Delete Service (exact name)")
        if st.form_submit_button("Delete") and (delname or "").strip():
            _delete_service(delname)
            st.success("Service deleted if it existed. Reassign vendors first if needed.")

# -----------------------------
# MAINTENANCE
# -----------------------------
def tab_maintenance():
    st.subheader("Maintenance")
    st.caption("Quick utilities for data hygiene.")
    col = st.columns(2)
    with col[0]:
        if st.button("Normalize phone format to (xxx) xxx-xxxx"):
            with eng.begin() as c:
                dfp = pd.read_sql_query(sql_text("SELECT id, phone FROM vendors"), c.connection)
                for _, r in dfp.iterrows():
                    fmt = _format_phone_10(str(r.get("phone", "")))
                    if fmt and fmt != (r.get("phone") or ""):
                        c.execute(sql_text(
                            "UPDATE vendors SET phone = :phone, updated_at = :ua, updated_by = :ub WHERE id = :id"
                        ), {"phone": fmt, "ua": _now_iso(), "ub": ADMIN_DISPLAY_NAME, "id": int(r["id"])})
                        c.execute(sql_text("""
                            INSERT INTO vendor_changes(vendor_id, changed_at, changed_by, action, field, old_value, new_value)
                            VALUES(:vendor_id, :changed_at, :changed_by, 'update', 'phone', :old_value, :new_value)
                        """), {"vendor_id": int(r["id"]), "changed_at": _now_iso(), "changed_by": ADMIN_DISPLAY_NAME,
                               "old_value": str(r.get("phone") or ""), "new_value": fmt})
            st.success("Phone numbers normalized where possible.")
    with col[1]:
        if st.button("Trim whitespace in text fields"):
            with eng.begin() as c:
                for k in ["category", "service", "business_name", "contact_name", "address", "website", "notes", "keywords"]:
                    c.execute(sql_text(f"UPDATE vendors SET {k} = TRIM({k})"))
                c.execute(sql_text("UPDATE vendors SET updated_at = :ua, updated_by = :ub"), {"ua": _now_iso(), "ub": ADMIN_DISPLAY_NAME})
            st.success("Whitespace trimmed.")

# -----------------------------
# CHANGELOG tab
# -----------------------------
def _fmt_when(s: str) -> str:
    # Show local-ish friendly format; leave as ISO if parsing fails
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return dt.astimezone().strftime("%b %-d, %Y, %-I:%M %p")
    except Exception:
        return s

def tab_changelog():
    st.subheader("Changelog")
    # Optional filters
    fcol1, fcol2 = st.columns(2)
    with fcol1:
        vendor_filter = st.text_input("Filter by vendor name contains", "")
    with fcol2:
        user_filter = st.text_input("Filter by edited-by contains", "")

    with eng.connect() as c:
        q = """
            SELECT vc.id, vc.vendor_id, vc.changed_at, vc.changed_by, vc.action, vc.field, vc.old_value, vc.new_value,
                   v.business_name
            FROM vendor_changes vc
            LEFT JOIN vendors v ON v.id = vc.vendor_id
            ORDER BY vc.changed_at DESC, vc.id DESC
        """
        df = pd.read_sql_query(sql_text(q), c)

    if vendor_filter.strip():
        df = df[df["business_name"].fillna("").str.contains(vendor_filter.strip(), case=False, na=False)]
    if user_filter.strip():
        df = df[df["changed_by"].fillna("").str.contains(user_filter.strip(), case=False, na=False)]

    # Human-readable lines
    lines = []
    for _, r in df.iterrows():
        when = _fmt_when(str(r["changed_at"]))
        by = r.get("changed_by") or "unknown"
        name = r.get("business_name") or f"Vendor #{r.get('vendor_id')}"
        action = r.get("action")
        if action == "insert":
            lines.append(f"{when} — Added **{name}** (by {by})")
        elif action == "delete":
            lines.append(f"{when} — Deleted **{name}** (by {by})")
        elif action == "update":
            field = r.get("field") or "field"
            oldv = r.get("old_value") or ""
            newv = r.get("new_value") or ""
            # Compact old→new
            arrow = f"`{oldv}` → `{newv}`" if oldv or newv else ""
            lines.append(f"{when} — Updated **{field}** for **{name}** (by {by}) {arrow}")
        else:
            lines.append(f"{when} — Change on **{name}** (by {by})")

    if not lines:
        st.info("No changes recorded yet.")
    else:
        st.markdown("\n\n".join(f"- {ln}" for ln in lines))

# -----------------------------
# Status & Secrets (debug) — END
# -----------------------------
def render_status_debug():
    with st.expander("Status & Secrets (debug)", expanded=False):
        backend = "libsql" if str(_read_secret("TURSO_DATABASE_URL", "")).startswith("sqlite+libsql://") else "sqlite"
        st.write("DB")
        st.code({"backend": backend, "dsn": _read_secret("TURSO_DATABASE_URL", ""), "auth": "token_set" if bool(_read_secret("TURSO_AUTH_TOKEN")) else "none"})
        try:
            keys = list(st.secrets.keys())
        except Exception:
            keys = []
        st.write("Secrets keys (present)"); st.code(keys)
        st.write("Help MD:", "present" if bool(HELP_MD) else "(missing or empty)")
        st.write("Sticky first col enabled:", STICKY_FIRST)
        st.write("Raw COLUMN_WIDTHS_PX_READONLY (type)", type(COLUMN_WIDTHS).__name__); st.code(COLUMN_WIDTHS)
        st.write("Column label overrides (if any)"); st.code(LABEL_OVERRIDES)
        st.write("Admin display name:", ADMIN_DISPLAY_NAME)

# -----------------------------
# Main
# -----------------------------
def main():
    render_help()

    tabs = st.tabs([
        "View", "Add", "Edit", "Delete", "Categories Admin", "Services Admin", "Maintenance", "Changelog"
    ])

    with tabs[0]: tab_view()
    with tabs[1]: tab_add()
    with tabs[2]: tab_edit()
    with tabs[3]: tab_delete()
    with tabs[4]: tab_categories()
    with tabs[5]: tab_services()
    with tabs[6]: tab_maintenance()
    with tabs[7]: tab_changelog()

    # Debug LAST
    render_status_debug()

if __name__ == "__main__":
    main()
