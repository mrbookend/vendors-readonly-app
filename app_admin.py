# app_admin.py — Vendors Admin (Single page: Browse, Add, Edit, Delete; Debug at bottom)
# Lean, robust UI (no sidebar). Immediate refresh on add/edit/delete.
# Non-FTS AND search across all fields. Optional "keywords" column supported.

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.parse import urlparse

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine
from sqlalchemy.sql import text as sql_text

APP_TITLE = "Vendors Admin"
st.set_page_config(page_title=APP_TITLE, layout="wide")

# ---- Database ----
DEFAULT_DB_PATH = Path(__file__).resolve().parent / "vendors.db"
DB_URL = os.getenv("DB_URL") or f"sqlite:///{DEFAULT_DB_PATH}"

@st.cache_resource(show_spinner=False)
def get_engine():
    return create_engine(DB_URL, future=True)

def get_columns(engine) -> List[str]:
    with engine.begin() as con:
        info = con.execute(sql_text("PRAGMA table_info(vendors)")).fetchall()
    return [r[1] for r in info]

def load_df(engine) -> pd.DataFrame:
    cols = get_columns(engine)
    ordered = [
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
    select_cols = [c for c in ordered if c in cols]
    sql = f"SELECT {', '.join(select_cols)} FROM vendors ORDER BY business_name COLLATE NOCASE"
    with engine.begin() as con:
        df = pd.read_sql_query(sql, con)
    return df

def get_distinct(engine, col: str, where: Tuple[str, dict] | None = None) -> List[str]:
    if where:
        sql = f"SELECT DISTINCT {col} FROM vendors WHERE {where[0]} ORDER BY {col} COLLATE NOCASE"
        params = where[1]
    else:
        sql = f"SELECT DISTINCT {col} FROM vendors ORDER BY {col} COLLATE NOCASE"
        params = {}
    with engine.begin() as con:
        rows = con.execute(sql_text(sql), params).fetchall()
    return [r[0] for r in rows if (r[0] is not None and str(r[0]).strip() != "")]

# ---- Normalizers & validators ----
PHONE_ERR = "Phone must be 10 digits (US) or left blank"
URL_ERR = "Website must be a valid URL (with or without https://) or left blank"

def normalize_phone(s: str) -> str:
    s = re.sub(r"\D", "", (s or ""))
    if not s:
        return ""
    if len(s) == 11 and s.startswith("1"):
        s = s[1:]
    if len(s) != 10:
        raise ValueError(PHONE_ERR)
    return f"({s[0:3]}) {s[3:6]}-{s[6:10]}"

def normalize_url(u: str) -> str:
    u = (u or "").strip()
    if not u:
        return ""
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", u):
        u = "https://" + u
    parsed = urlparse(u)
    if not parsed.netloc:
        raise ValueError(URL_ERR)
    return parsed.geturl()

def normalize_keywords(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    parts = re.split(r"[\s,]+", s)
    parts = sorted(set([p.strip() for p in parts if p.strip()]))
    return ", ".join(parts)

# ---- CRUD ----
def insert_vendor(engine, row: Dict[str, str]) -> None:
    cols = get_columns(engine)
    allowed = [
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
    values = {k: row.get(k, "") for k in allowed if k in cols}
    placeholders = ", ".join([f":{k}" for k in values.keys()])
    columns = ", ".join(values.keys())
    sql = f"INSERT INTO vendors ({columns}) VALUES ({placeholders})"
    with engine.begin() as con:
        con.execute(sql_text(sql), values)

def update_vendor(engine, vid: int, updates: Dict[str, str]) -> None:
    if not updates:
        return
    cols = get_columns(engine)
    allowed = [c for c in updates.keys() if c in cols and c != "id"]
    if not allowed:
        return
    set_clause = ", ".join([f"{c} = :{c}" for c in allowed])
    params = {c: updates[c] for c in allowed}
    params["id"] = vid
    with engine.begin() as con:
        con.execute(sql_text(f"UPDATE vendors SET {set_clause} WHERE id = :id"), params)

def delete_vendor(engine, vid: int) -> None:
    with engine.begin() as con:
        con.execute(sql_text("DELETE FROM vendors WHERE id = :id"), {"id": vid})

# ---- Search ----
def filter_df(df: pd.DataFrame, query: str) -> pd.DataFrame:
    query = (query or "").strip()
    if not query:
        return df
    tokens = [t for t in re.split(r"\s+", query) if t]
    if not tokens:
        return df
    search_cols = [c for c in df.columns if c != "id"]
    lower = df[search_cols].astype(str).apply(lambda s: s.str.lower())
    mask = pd.Series(True, index=df.index)
    for tok in tokens:
        mask &= lower.apply(lambda s: s.str.contains(tok.lower(), na=False)).any(axis=1)
    return df[mask]

# ---- UI Sections ----
def section_browse(engine):
    st.subheader("Browse")
    df = load_df(engine)
    q = st.text_input(
        "Search (AND across all fields)",
        value=st.session_state.get("browse_q", ""),
        placeholder="e.g., plumber alamo heights",
        key="browse_q",
        help="Type one or more words. We use AND logic across all fields (case-insensitive).",
    )
    filtered = filter_df(df, q)
    col_config = {}
    try:
        if hasattr(st, "column_config") and "website" in filtered.columns:
            col_config["website"] = st.column_config.LinkColumn("Website", display_text="Open")
    except Exception:
        col_config = {}
    st.dataframe(filtered, use_container_width=True, hide_index=True, column_config=col_config)

def section_add(engine):
    st.subheader("Add Vendor")
    cols = get_columns(engine)

    # Category selection or new
    categories = get_distinct(engine, "category")
    c1, c2 = st.columns([1, 2])
    with c1:
        cat_mode = st.radio("Category", ["Choose existing", "Add new"], horizontal=True, key="cat_mode")
    with c2:
        if cat_mode == "Choose existing":
            category = st.selectbox("", [""] + categories, index=0, key="cat_select", label_visibility="collapsed")
        else:
            category = st.text_input("New Category", "", key="cat_new")

    # Service depends on category (if present)
    services = get_distinct(engine, "service", where=("category = :c", {"c": category})) if category else []
    s1, s2 = st.columns([1, 2])
    with s1:
        svc_mode = st.radio("Service", ["Choose existing", "Add new"], horizontal=True, key="svc_mode")
    with s2:
        if svc_mode == "Choose existing":
            service = st.selectbox("", [""] + services, index=0, key="svc_select", label_visibility="collapsed")
        else:
            service = st.text_input("New Service", "", key="svc_new")

    business_name = st.text_input("Business Name *", "", key="biz")
    contact_name  = st.text_input("Contact Name", "", key="contact")
    phone_in      = st.text_input("Phone (digits only)", "", key="phone")
    address       = st.text_input("Address", "", key="addr")
    website_in    = st.text_input("Website (optional)", "", key="web")
    notes         = st.text_area("Notes", "", key="notes", height=80)
    keywords      = st.text_input("Keywords (comma or space separated)", "", key="kw") if "keywords" in cols else ""

    if st.button("Add Vendor", type="primary", use_container_width=True):
        try:
            if not category.strip():
                st.error("Category is required.")
                st.stop()
            if not service.strip():
                st.error("Service is required.")
                st.stop()
            if not business_name.strip():
                st.error("Business Name is required.")
                st.stop()

            phone   = normalize_phone(phone_in)
            website = normalize_url(website_in)
            kw_norm = normalize_keywords(keywords) if isinstance(keywords, str) else ""

            row = {
                "category": category.strip(),
                "service": service.strip(),
                "business_name": business_name.strip(),
                "contact_name": contact_name.strip(),
                "phone": phone,
                "address": address.strip(),
                "website": website,
                "notes": notes.strip(),
                "keywords": kw_norm,
            }
            insert_vendor(engine, row)
            st.success("Vendor added.")
            st.rerun()
        except Exception as e:
            st.exception(e)

def section_edit(engine):
    st.subheader("Edit Vendor")
    df = load_df(engine)
    if df.empty:
        st.info("No vendors to edit.")
        return

    # Build sorted selection by Business Name
    df_sorted = df.sort_values("business_name", key=lambda s: s.str.lower())
    def label_for_row(r):
        cat = str(r.get("category", "") or "")
        svc = str(r.get("service", "") or "")
        return f"{r['business_name']} — {cat} / {svc}  [#{r['id']}]"

    id_to_label = {int(r["id"]): label_for_row(r) for _, r in df_sorted.iterrows()}
    ids = list(id_to_label.keys())

    chosen_id = st.selectbox(
        "Select Vendor",
        options=ids,
        format_func=lambda i: id_to_label[i],
        key="edit_vendor_select",
    )
    row = df[df["id"] == chosen_id].iloc[0].to_dict()
    cols = get_columns(engine)

    # Fields (allow changing cat/service via existing/new like Add)
    categories = get_distinct(engine, "category")
    c1, c2 = st.columns([1, 2])
    with c1:
        cat_mode = st.radio("Category", ["Choose existing", "Add new"], horizontal=True, key="edit_cat_mode")
    with c2:
        if cat_mode == "Choose existing":
            category = st.selectbox(
                "",
                [""] + categories,
                index=([""] + categories).index(row.get("category") or "") if (row.get("category") or "") in ([""] + categories) else 0,
                key="edit_cat_select",
                label_visibility="collapsed",
            )
        else:
            category = st.text_input("New Category", row.get("category", ""), key="edit_cat_new")

    # Services under selected category
    services = get_distinct(engine, "service", where=("category = :c", {"c": category})) if category else []
    s1, s2 = st.columns([1, 2])
    with s1:
        svc_mode = st.radio("Service", ["Choose existing", "Add new"], horizontal=True, key="edit_svc_mode")
    with s2:
        if svc_mode == "Choose existing":
            default_service = row.get("service") or ""
            svc_options = [""] + (services if services else [default_service])
            service = st.selectbox(
                "",
                svc_options,
                index=svc_options.index(default_service) if default_service in svc_options else 0,
                key="edit_svc_select",
                label_visibility="collapsed",
            )
        else:
            service = st.text_input("New Service", row.get("service", ""), key="edit_svc_new")

    business_name = st.text_input("Business Name *", row.get("business_name", ""), key="edit_biz")
    contact_name  = st.text_input("Contact Name", row.get("contact_name", ""), key="edit_contact")
    phone_in      = st.text_input("Phone (digits only)", row.get("phone", ""), key="edit_phone")
    address       = st.text_input("Address", row.get("address", ""), key="edit_addr")
    website_in    = st.text_input("Website (optional)", row.get("website", ""), key="edit_web")
    notes         = st.text_area("Notes", row.get("notes", ""), key="edit_notes", height=80)
    kw_enabled    = "keywords" in cols
    keywords_in   = st.text_input("Keywords (comma or space separated)", row.get("keywords", ""), key="edit_kw") if kw_enabled else ""

    if st.button("Save Changes", type="primary", use_container_width=True, key="edit_save"):
        try:
            if not category.strip():
                st.error("Category is required.")
                st.stop()
            if not service.strip():
                st.error("Service is required.")
                st.stop()
            if not business_name.strip():
                st.error("Business Name is required.")
                st.stop()

            phone   = normalize_phone(phone_in)
            website = normalize_url(website_in)
            updates = {
                "category": category.strip(),
                "service": service.strip(),
                "business_name": business_name.strip(),
                "contact_name": contact_name.strip(),
                "phone": phone,
                "address": address.strip(),
                "website": website,
                "notes": notes.strip(),
            }
            if kw_enabled:
                updates["keywords"] = normalize_keywords(keywords_in)

            update_vendor(engine, int(chosen_id), updates)
            st.success("Vendor updated.")
            st.rerun()
        except Exception as e:
            st.exception(e)

def section_delete(engine):
    st.subheader("Delete Vendor")
    df = load_df(engine)
    if df.empty:
        st.info("No vendors to delete.")
        return

    df_sorted = df.sort_values("business_name", key=lambda s: s.str.lower())
    def label_for_row(r):
        cat = str(r.get("category", "") or "")
        svc = str(r.get("service", "") or "")
        return f"{r['business_name']} — {cat} / {svc}  [#{r['id']}]"

    id_to_label = {int(r["id"]): label_for_row(r) for _, r in df_sorted.iterrows()}
    ids = list(id_to_label.keys())

    chosen_id = st.selectbox(
        "Select Vendor to Delete",
        options=ids,
        format_func=lambda i: id_to_label[i],
        key="del_vendor_select",
    )

    st.warning("This action is irreversible.", icon="⚠️")
    cols_btn = st.columns([1, 4])
    with cols_btn[0]:
        do_delete = st.button("Delete Vendor", type="secondary", use_container_width=True)
    if do_delete:
        try:
            delete_vendor(engine, int(chosen_id))
            st.success("Vendor deleted.")
            st.rerun()
        except Exception as e:
            st.exception(e)

def section_debug(engine):
    st.subheader("Database Status & Schema (debug)")
    cols = get_columns(engine)
    try:
        vendor_count = int(load_df(engine).shape[0])
    except Exception:
        vendor_count = -1
    st.json(
        {
            "db_source": DB_URL,
            "vendors_columns": {i: c for i, c in enumerate(cols)},
            "has_keywords": "keywords" in cols,
            "counts": {"vendors": vendor_count},
        }
    )

# ---- Main ----
def main():
    st.title(APP_TITLE)

    engine = get_engine()

    # Lay out the page top-to-bottom. No sidebar.
    with st.expander("Browse (word search across all fields)", expanded=True):
        section_browse(engine)

    st.divider()
    section_add(engine)

    st.divider()
    section_edit(engine)

    st.divider()
    section_delete(engine)

    st.divider()
    section_debug(engine)  # Debug at bottom

if __name__ == "__main__":
    main()
