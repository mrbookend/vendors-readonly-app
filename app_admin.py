# app_admin.py — Vendors Admin (Single page: Browse, Add, Edit, Delete; Taxonomy; Debug)
# No sidebar. Immediate refresh on add/edit/delete. UI clears after actions.
# Non-FTS AND search across all fields. Optional "keywords" column supported.
# Capitalization enforced (Category, Service, Business Name, Contact Name) on Add/Edit.
# Business Name optional. Dedicated Category/Service Library to add taxonomy WITHOUT creating a vendor row.

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
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

def has_table(engine, name: str) -> bool:
    with engine.begin() as con:
        row = con.execute(
            sql_text("SELECT name FROM sqlite_master WHERE type='table' AND name = :n"),
            {"n": name},
        ).fetchone()
    return row is not None

def ensure_vocab_tables(engine) -> None:
    """Create categories/services library tables if they don't exist."""
    with engine.begin() as con:
        con.execute(sql_text(
            "CREATE TABLE IF NOT EXISTS categories ("
            "  name TEXT PRIMARY KEY"
            ");"
        ))
        con.execute(sql_text(
            "CREATE TABLE IF NOT EXISTS services ("
            "  category TEXT NOT NULL,"
            "  service  TEXT NOT NULL,"
            "  PRIMARY KEY(category, service)"
            ");"
        ))

def seed_vocab_from_vendors_if_empty(engine) -> None:
    """If the categories/services tables are empty, seed from vendors distincts."""
    if not has_table(engine, "categories") or not has_table(engine, "services"):
        ensure_vocab_tables(engine)

    with engine.begin() as con:
        # categories empty?
        ncat = con.execute(sql_text("SELECT COUNT(*) FROM categories")).scalar_one()
        if int(ncat) == 0:
            con.execute(sql_text(
                "INSERT OR IGNORE INTO categories(name) "
                "SELECT DISTINCT TRIM(category) FROM vendors WHERE TRIM(IFNULL(category,'')) <> ''"
            ))
        # services empty?
        nsvc = con.execute(sql_text("SELECT COUNT(*) FROM services")).scalar_one()
        if int(nsvc) == 0:
            con.execute(sql_text(
                "INSERT OR IGNORE INTO services(category, service) "
                "SELECT DISTINCT TRIM(category), TRIM(service) "
                "FROM vendors "
                "WHERE TRIM(IFNULL(category,'')) <> '' AND TRIM(IFNULL(service,'')) <> ''"
            ))

def get_columns(engine) -> List[str]:
    with engine.begin() as con:
        info = con.execute(sql_text("PRAGMA table_info(vendors)")).fetchall()
    return [r[1] for r in info]

def load_df(engine) -> pd.DataFrame:
    cols = get_columns(engine)
    ordered = [
        "id","category","service","business_name","contact_name",
        "phone","address","website","notes","keywords",
    ]
    select_cols = [c for c in ordered if c in cols]
    sql = f"SELECT {', '.join(select_cols)} FROM vendors ORDER BY business_name COLLATE NOCASE"
    with engine.begin() as con:
        df = pd.read_sql_query(sql, con)
    return df

# ---- Safe helpers & casing ----
def s(x) -> str:
    """Safe string: None -> '', strings trimmed; other types -> str trimmed."""
    if x is None:
        return ""
    if isinstance(x, str):
        return x.strip()
    return str(x).strip()

_ACRONYM = re.compile(r"^[A-Z0-9]{2,}$")
def cap_words_reasonable(x: str) -> str:
    """
    Title-case words separated by space, '/', '&', ',', or '-'.
    Keep acronyms (ALLCAPS/digits) as-is. Meant for names/categories/services.
    """
    t = s(x)
    if not t:
        return ""
    parts = re.split(r'(\s+|/|&|,|-)', t)  # keep delimiters
    out = []
    for p in parts:
        if not p or re.fullmatch(r'\s+|/|&|,|-', p):
            out.append(p); continue
        if _ACRONYM.match(p):
            out.append(p)
        else:
            out.append(p[:1].upper() + p[1:].lower())
    return "".join(out)

def apply_casing(d: Dict[str, str]) -> Dict[str, str]:
    """Capitalize where it makes sense: category, service, business_name, contact_name."""
    for k in ("category", "service", "business_name", "contact_name"):
        if k in d:
            d[k] = cap_words_reasonable(d[k])
    return d

def clear_keys(*keys: str) -> None:
    for k in keys:
        st.session_state.pop(k, None)

# ---- Normalizers & validators ----
PHONE_ERR = "Phone must be 10 digits (US) or left blank"
URL_ERR = "Website must be a valid URL (with or without https://) or left blank"

def normalize_phone(x) -> str:
    raw = s(x)
    digits = re.sub(r"\D", "", raw)
    if not digits:
        return ""
    if len(digits) == 11 and digits.startswith("1"):
        digits = digits[1:]
    if len(digits) != 10:
        raise ValueError(PHONE_ERR)
    return f"({digits[0:3]}) {digits[3:6]}-{digits[6:10]}"

def normalize_url(u) -> str:
    u = s(u)
    if not u:
        return ""
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", u):
        u = "https://" + u
    parsed = urlparse(u)
    if not parsed.netloc:
        raise ValueError(URL_ERR)
    return parsed.geturl()

def normalize_keywords(sval) -> str:
    sval = s(sval)
    if not sval:
        return ""
    parts = re.split(r"[\s,]+", sval)
    parts = sorted(set([p.strip() for p in parts if p.strip()]))
    return ", ".join(parts)

# ---- Taxonomy fetch/upsert ----
def fetch_categories(engine) -> List[str]:
    if has_table(engine, "categories"):
        with engine.begin() as con:
            rows = con.execute(sql_text(
                "SELECT name FROM categories WHERE TRIM(IFNULL(name,''))<>'' ORDER BY name COLLATE NOCASE"
            )).fetchall()
        return [r[0] for r in rows]
    # fallback to vendors distinct
    with engine.begin() as con:
        rows = con.execute(sql_text(
            "SELECT DISTINCT category FROM vendors WHERE TRIM(IFNULL(category,''))<>'' ORDER BY category COLLATE NOCASE"
        )).fetchall()
    return [r[0] for r in rows]

def fetch_services(engine, category: str) -> List[str]:
    c = s(category)
    if not c:
        return []
    if has_table(engine, "services"):
        with engine.begin() as con:
            rows = con.execute(sql_text(
                "SELECT service FROM services "
                "WHERE category = :c AND TRIM(IFNULL(service,''))<>'' "
                "ORDER BY service COLLATE NOCASE"
            ), {"c": c}).fetchall()
        return [r[0] for r in rows]
    # fallback to vendors distinct
    with engine.begin() as con:
        rows = con.execute(sql_text(
            "SELECT DISTINCT service FROM vendors "
            "WHERE TRIM(IFNULL(category,''))<>'' AND category=:c "
            "  AND TRIM(IFNULL(service,''))<>'' "
            "ORDER BY service COLLATE NOCASE"
        ), {"c": c}).fetchall()
    return [r[0] for r in rows]

def upsert_category(engine, name: str) -> None:
    ensure_vocab_tables(engine)
    with engine.begin() as con:
        con.execute(sql_text(
            "INSERT OR IGNORE INTO categories(name) VALUES(:n)"
        ), {"n": cap_words_reasonable(name)})

def upsert_service(engine, category: str, service: str) -> None:
    ensure_vocab_tables(engine)
    cat = cap_words_reasonable(category)
    svc = cap_words_reasonable(service)
    with engine.begin() as con:
        con.execute(sql_text("INSERT OR IGNORE INTO categories(name) VALUES(:n)"), {"n": cat})
        con.execute(sql_text(
            "INSERT OR IGNORE INTO services(category, service) VALUES(:c, :s)"
        ), {"c": cat, "s": svc})

# ---- CRUD on vendors ----
def insert_vendor(engine, row: Dict[str, str]) -> None:
    cols = get_columns(engine)
    allowed = [
        "category","service","business_name","contact_name",
        "phone","address","website","notes","keywords",
    ]
    values = {k: s(row.get(k, "")) for k in allowed if k in cols}
    values = apply_casing(values)
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
    upd = {c: s(updates[c]) for c in allowed}
    upd = apply_casing(upd)
    set_clause = ", ".join([f"{c} = :{c}" for c in upd.keys()])
    upd["id"] = vid
    with engine.begin() as con:
        con.execute(sql_text(f"UPDATE vendors SET {set_clause} WHERE id = :id"), upd)

def delete_vendor(engine, vid: int) -> None:
    with engine.begin() as con:
        con.execute(sql_text("DELETE FROM vendors WHERE id = :id"), {"id": vid})

def normalize_all_caps_now(engine) -> int:
    """One-click normalize capitalization across all rows (where it makes sense)."""
    df = load_df(engine)
    count = 0
    for _, r in df.iterrows():
        vid = int(r["id"])
        updates = {
            "category": r.get("category"),
            "service": r.get("service"),
            "business_name": r.get("business_name"),
            "contact_name": r.get("contact_name"),
        }
        cased = apply_casing({k: s(v) for k, v in updates.items()})
        if any(s(updates[k]) != cased[k] for k in cased):
            update_vendor(engine, vid, cased)
            count += 1
    return count

# ---- Search ----
def filter_df(df: pd.DataFrame, query: str) -> pd.DataFrame:
    query = s(query)
    if not query:
        return df
    tokens = [t for t in re.split(r"\s+", query) if t]
    if not tokens:
        return df
    search_cols = [c for c in df.columns if c != "id"]
    lower = df[search_cols].astype(str).apply(lambda ser: ser.str.lower())
    mask = pd.Series(True, index=df.index)
    for tok in tokens:
        tok = tok.lower()
        mask &= lower.apply(lambda ser: ser.str.contains(tok, na=False)).any(axis=1)
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
    seed_vocab_from_vendors_if_empty(engine)

    # Category selection or new (pulled from library if present)
    categories = fetch_categories(engine)
    c1, c2 = st.columns([1, 2])
    with c1:
        cat_mode = st.radio("Category", ["Choose existing", "Add new"], horizontal=True, key="cat_mode")
    with c2:
        if cat_mode == "Choose existing":
            category = st.selectbox("", [""] + categories, index=0, key="cat_select", label_visibility="collapsed")
        else:
            category = st.text_input("New Category", "", key="cat_new")

    # Services depend on category, pulled from library if present
    cat_val = s(category)
    services = fetch_services(engine, cat_val) if cat_val else []
    s1, s2 = st.columns([1, 2])
    with s1:
        svc_mode = st.radio("Service", ["Choose existing", "Add new"], horizontal=True, key="svc_mode")
    with s2:
        if svc_mode == "Choose existing":
            service = st.selectbox("", [""] + services, index=0, key="svc_select", label_visibility="collapsed")
        else:
            service = st.text_input("New Service", "", key="svc_new")

    business_name = st.text_input("Business Name (optional)", "", key="biz")
    contact_name  = st.text_input("Contact Name", "", key="contact")
    phone_in      = st.text_input("Phone (digits only)", "", key="phone")
    address       = st.text_input("Address", "", key="addr")
    website_in    = st.text_input("Website (optional)", "", key="web")
    notes         = st.text_area("Notes", "", key="notes", height=80)
    keywords      = st.text_input("Keywords (comma or space separated)", "", key="kw") if "keywords" in cols else ""

    if st.button("Add Vendor", type="primary", use_container_width=True, key="add_btn"):
        try:
            cat = cap_words_reasonable(category)
            svc = cap_words_reasonable(service)
            if not s(cat):
                st.error("Category is required."); st.stop()
            if not s(svc):
                st.error("Service is required."); st.stop()

            # Keep library in sync, but this is NOT how you add taxonomy alone anymore.
            upsert_service(engine, cat, svc)

            phone   = normalize_phone(phone_in)
            website = normalize_url(website_in)
            kw_norm = normalize_keywords(keywords) if "keywords" in cols else ""

            row = {
                "category": cat,
                "service": svc,
                "business_name": s(business_name),
                "contact_name": s(contact_name),
                "phone": phone,
                "address": s(address),
                "website": website,
                "notes": s(notes),
                "keywords": kw_norm,
            }
            insert_vendor(engine, row)
            st.success("Vendor added.")

            # Clear inputs to avoid accidental double-submits
            clear_keys("cat_mode","cat_select","cat_new",
                       "svc_mode","svc_select","svc_new",
                       "biz","contact","phone","addr","web","notes","kw")
            st.rerun()
        except Exception as e:
            st.exception(e)

def section_edit(engine):
    st.subheader("Edit Vendor")
    df = load_df(engine)
    if df.empty:
        st.info("No vendors to edit.")
        return

    df_sorted = df.sort_values("business_name", key=lambda srs: srs.astype(str).str.lower())
    def label_for_row(r):
        cat = cap_words_reasonable(r.get("category"))
        svc = cap_words_reasonable(r.get("service"))
        bn  = cap_words_reasonable(r.get("business_name"))
        name = bn or f"{cat} / {svc}" or "[no name]"
        return f"{name}  [#{r['id']}]"

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

    # Use library lists
    seed_vocab_from_vendors_if_empty(engine)
    categories = fetch_categories(engine)
    c1, c2 = st.columns([1, 2])
    with c1:
        cat_mode = st.radio("Category", ["Choose existing", "Add new"], horizontal=True, key="edit_cat_mode")
    with c2:
        current_cat = s(row.get("category"))
        if cat_mode == "Choose existing":
            cat_options = [""] + categories
            cat_index = cat_options.index(current_cat) if current_cat in cat_options else 0
            category = st.selectbox("", cat_options, index=cat_index, key="edit_cat_select", label_visibility="collapsed")
        else:
            category = st.text_input("New Category", current_cat, key="edit_cat_new")

    cat_val = s(category)
    services = fetch_services(engine, cat_val) if cat_val else []
    s1, s2 = st.columns([1, 2])
    with s1:
        svc_mode = st.radio("Service", ["Choose existing", "Add new"], horizontal=True, key="edit_svc_mode")
    with s2:
        current_svc = s(row.get("service"))
        if svc_mode == "Choose existing":
            svc_options = [""] + (services if services else [current_svc] if current_svc else [])
            svc_index = svc_options.index(current_svc) if current_svc in svc_options else 0
            service = st.selectbox("", svc_options, index=svc_index, key="edit_svc_select", label_visibility="collapsed")
        else:
            service = st.text_input("New Service", current_svc, key="edit_svc_new")

    business_name = st.text_input("Business Name (optional)", cap_words_reasonable(row.get("business_name")), key="edit_biz")
    contact_name  = st.text_input("Contact Name", cap_words_reasonable(row.get("contact_name")), key="edit_contact")
    phone_in      = st.text_input("Phone (digits only)", s(row.get("phone")), key="edit_phone")
    address       = st.text_input("Address", s(row.get("address")), key="edit_addr")
    website_in    = st.text_input("Website (optional)", s(row.get("website")), key="edit_web")
    notes         = st.text_area("Notes", s(row.get("notes")), key="edit_notes", height=80)
    kw_enabled    = "keywords" in cols
    keywords_in   = st.text_input("Keywords (comma or space separated)", s(row.get("keywords")), key="edit_kw") if kw_enabled else ""

    if st.button("Save Changes", type="primary", use_container_width=True, key="edit_save"):
        try:
            cat = cap_words_reasonable(category)
            svc = cap_words_reasonable(service)
            if not s(cat):
                st.error("Category is required."); st.stop()
            if not s(svc):
                st.error("Service is required."); st.stop()

            # Keep library in sync on edits
            upsert_service(engine, cat, svc)

            phone   = normalize_phone(phone_in)
            website = normalize_url(website_in)
            updates = {
                "category": cat,
                "service": svc,
                "business_name": s(business_name),
                "contact_name": s(contact_name),
                "phone": phone,
                "address": s(address),
                "website": website,
                "notes": s(notes),
            }
            if kw_enabled:
                updates["keywords"] = normalize_keywords(keywords_in)

            update_vendor(engine, int(chosen_id), updates)
            st.success("Vendor updated.")

            # Clear selection & inputs
            clear_keys("edit_vendor_select","edit_cat_mode","edit_cat_select","edit_cat_new",
                       "edit_svc_mode","edit_svc_select","edit_svc_new",
                       "edit_biz","edit_contact","edit_phone","edit_addr","edit_web","edit_notes","edit_kw")
            st.rerun()
        except Exception as e:
            st.exception(e)

def section_delete(engine):
    st.subheader("Delete Vendor")
    df = load_df(engine)
    if df.empty:
        st.info("No vendors to delete.")
        return

    df_sorted = df.sort_values("business_name", key=lambda srs: srs.astype(str).str.lower())
    def label_for_row(r):
        cat = cap_words_reasonable(r.get("category"))
        svc = cap_words_reasonable(r.get("service"))
        bn  = cap_words_reasonable(r.get("business_name"))
        name = bn or f"{cat} / {svc}" or "[no name]"
        return f"{name}  [#{r['id']}]"

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
        do_delete = st.button("Delete Vendor", type="secondary", use_container_width=True, key="delete_btn")
    if do_delete:
        try:
            delete_vendor(engine, int(chosen_id))
            st.success("Vendor deleted.")
            clear_keys("del_vendor_select")
            st.rerun()
        except Exception as e:
            st.exception(e)

def section_taxonomy(engine):
    st.subheader("Category/Service Library (no vendor record)")
    seed_vocab_from_vendors_if_empty(engine)

    # --- Add Category Only ---
    st.markdown("**Add Category Only**")
    new_cat = st.text_input("New Category Name", "", key="lib_cat_new", placeholder="e.g., Plumbing")
    if st.button("Add Category Only", key="lib_cat_btn"):
        try:
            if not s(new_cat):
                st.error("Category name is required."); st.stop()
            upsert_category(engine, new_cat)
            st.success("Category added to library.")
            clear_keys("lib_cat_new")
            st.rerun()
        except Exception as e:
            st.exception(e)

    st.write("---")

    # --- Add Service Only ---
    st.markdown("**Add Service Only**")
    categories = fetch_categories(engine)
    c1, c2 = st.columns([1, 2])
    with c1:
        cat_mode = st.radio("Choose Category", ["Existing", "Add new"], horizontal=True, key="lib_svc_cat_mode")
    with c2:
        if cat_mode == "Existing":
            svc_cat = st.selectbox("", [""] + categories, index=0, key="lib_svc_cat", label_visibility="collapsed")
        else:
            svc_cat = st.text_input("New Category for this Service", "", key="lib_svc_cat_new")

    svc_name = st.text_input("Service Name", "", key="lib_svc_name", placeholder="e.g., Water Heaters")
    if st.button("Add Service Only", key="lib_svc_btn"):
        try:
            if not s(svc_cat):
                st.error("Category is required."); st.stop()
            if not s(svc_name):
                st.error("Service name is required."); st.stop()
            upsert_service(engine, svc_cat, svc_name)
            st.success("Service added to library.")
            clear_keys("lib_svc_cat_mode","lib_svc_cat","lib_svc_cat_new","lib_svc_name")
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

    info = {
        "db_source": DB_URL,
        "vendors_columns": {i: c for i, c in enumerate(cols)},
        "has_keywords": "keywords" in cols,
        "has_categories_table": has_table(engine, "categories"),
        "has_services_table": has_table(engine, "services"),
        "counts": {"vendors": vendor_count},
    }
    st.json(info)

    st.write("---")
    if st.button("Normalize capitalization for all vendor rows now", type="secondary"):
        try:
            n = normalize_all_caps_now(engine)
            st.success(f"Capitalization normalized on {n} vendor row(s).")
            st.rerun()
        except Exception as e:
            st.exception(e)

# ---- Main ----
def main():
    st.title(APP_TITLE)

    engine = get_engine()

    # Top-to-bottom layout; no sidebar.
    with st.expander("Browse (word search across all fields)", expanded=True):
        section_browse(engine)

    st.divider()
    section_add(engine)

    st.divider()
    section_edit(engine)

    st.divider()
    section_delete(engine)

    st.divider()
    section_taxonomy(engine)  # <-- Add taxonomy without creating vendor records

    st.divider()
    section_debug(engine)  # Debug at bottom

if __name__ == "__main__":
    main()
