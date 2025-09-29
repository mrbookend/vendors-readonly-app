# app_admin.py
# Vendors Admin for Streamlit: SQLite (default) or libSQL via DATABASE_URL).
# - Add Vendor: Category/Service dropdowns from existing data + “+ Add new…” option.
# - Edit Vendor: Select Vendor sorted A→Z by Business Name.
# - Phone validation robust (10 digits or 11 with leading '1'); auto-format to (xxx) xxx-xxxx.
# - Save button enables correctly once changes are valid & non-noop.
# - Safe cache clearing; category/service rename/delete with reassignment guardrails.

from __future__ import annotations

import os
import re
from typing import Optional, Tuple, Dict, Any, List
from urllib.parse import urlparse

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text

# ---------------------------
# DB setup
# ---------------------------
def _default_sqlite_url() -> str:
    return "sqlite:///vendors.db"

DATABASE_URL = os.environ.get("DATABASE_URL", "").strip() or _default_sqlite_url()
engine = create_engine(DATABASE_URL, future=True)

def run_sql(sql: str, params: Dict[str, Any] | None = None):
    with engine.begin() as conn:
        return conn.execute(text(sql), params or {})

def fetchall(sql: str, params: Dict[str, Any] | None = None) -> List[tuple]:
    with engine.begin() as conn:
        return list(conn.execute(text(sql), params or {}).fetchall())

def fetchone(sql: str, params: Dict[str, Any] | None = None) -> Optional[tuple]:
    with engine.begin() as conn:
        return conn.execute(text(sql), params or {}).fetchone()

@st.cache_data(show_spinner=False, ttl=10)
def has_column(tbl: str, col: str) -> bool:
    with engine.begin() as conn:
        rows = conn.execute(text(f"PRAGMA table_info({tbl})")).fetchall()
    return any(r[1] == col for r in rows)

# ---------------------------
# Helpers: normalize fields
# ---------------------------
DIGITS_RE = re.compile(r"\d")

def phone_digits(raw: str) -> str:
    return "".join(DIGITS_RE.findall(raw or ""))

def normalize_phone(raw: str) -> str:
    """Accept 10 digits or 11 with leading '1'; format to (xxx) xxx-xxxx.
       If >=10 digits, use the LAST 10 (common copy/paste cases)."""
    if not raw:
        return ""
    d = phone_digits(raw)
    if len(d) >= 10:
        # Use last 10 digits; handles leading '1' or pasted junk.
        d10 = d[-10:]
        return f"({d10[0:3]}) {d10[3:6]}-{d10[6:10]}"
    return raw.strip()

def normalize_url(raw: str) -> str:
    if not raw:
        return ""
    raw = raw.strip()
    if not raw:
        return ""
    parsed = urlparse(raw)
    if not parsed.scheme:
        raw = "https://" + raw
    return raw

def parse_keywords(raw: str) -> str:
    if not raw:
        return ""
    parts = re.split(r"[,\s]+", raw.strip())
    parts = [p for p in (x.strip() for x in parts) if p]
    seen = set()
    out = []
    for p in parts:
        pl = p.lower()
        if pl not in seen:
            out.append(pl)
            seen.add(pl)
    return ", ".join(out)

# ---------------------------
# Schema expectations
# ---------------------------
VENDORS_TABLE = "vendors"
REQUIRED_COLS = [
    "id","category","service","business_name","contact_name","phone","address","website","notes"
]
HAS_KEYWORDS = has_column(VENDORS_TABLE, "keywords")

# ---------------------------
# Cached readers
# ---------------------------
@st.cache_data(show_spinner=False, ttl=10)
def list_categories() -> List[str]:
    rows = fetchall(f"""
        SELECT DISTINCT COALESCE(NULLIF(TRIM(category), ''), '') AS c
        FROM {VENDORS_TABLE}
        ORDER BY LOWER(c)
    """)
    return [r[0] for r in rows if r and r[0] is not None]

@st.cache_data(show_spinner=False, ttl=10)
def list_services_for(category: str) -> List[str]:
    rows = fetchall(f"""
        SELECT DISTINCT COALESCE(NULLIF(TRIM(service), ''), '') AS s
        FROM {VENDORS_TABLE}
        WHERE COALESCE(category,'') = :c
        ORDER BY LOWER(s)
    """, {"c": category or ""})
    return [r[0] for r in rows if r and r[0] is not None]

@st.cache_data(show_spinner=False, ttl=10)
def load_vendors_df() -> pd.DataFrame:
    cols = REQUIRED_COLS + (["keywords"] if HAS_KEYWORDS else [])
    q = f"""
        SELECT {', '.join(cols)}
        FROM {VENDORS_TABLE}
        ORDER BY LOWER(COALESCE(category,'')),
                 LOWER(COALESCE(service,'')),
                 LOWER(business_name)
    """
    rows = fetchall(q)
    return pd.DataFrame(rows, columns=cols)

def vendor_by_id(vid: int) -> Dict[str, Any] | None:
    cols = REQUIRED_COLS + (["keywords"] if HAS_KEYWORDS else [])
    row = fetchone(f"SELECT {', '.join(cols)} FROM {VENDORS_TABLE} WHERE id = :i", {"i": vid})
    if not row:
        return None
    return dict(zip(cols, row))

# ---------------------------
# Safe cache clearing
# ---------------------------
def _safe_clear(fn):
    try:
        clear = getattr(fn, "clear", None)
        if callable(clear):
            clear()
    except Exception:
        pass

def clear_caches():
    _safe_clear(list_categories)
    _safe_clear(list_services_for)
    _safe_clear(load_vendors_df)

# ---------------------------
# Stats
# ---------------------------
def count_vendors_in_category(category: str) -> int:
    r = fetchone(f"""
        SELECT COUNT(*) FROM {VENDORS_TABLE}
        WHERE COALESCE(category,'') = :c
    """, {"c": category or ""})
    return int(r[0]) if r else 0

def count_vendors_in_service(category: str, service: str) -> int:
    r = fetchone(f"""
        SELECT COUNT(*) FROM {VENDORS_TABLE}
        WHERE COALESCE(category,'') = :c
          AND COALESCE(service,'')  = :s
    """, {"c": category or "", "s": service or ""})
    return int(r[0]) if r else 0

# ---------------------------
# Category/Service ops
# ---------------------------
def rename_category(old_name: str, new_name: str) -> Tuple[bool, str]:
    if not new_name or new_name.strip() == "":
        return False, "New category name cannot be empty."
    new_name = new_name.strip()
    old_name = (old_name or "").strip()
    if new_name == old_name:
        return True, "No change."
    run_sql(f"""
        UPDATE {VENDORS_TABLE}
           SET category = :new
         WHERE COALESCE(category,'') = :old
    """, {"new": new_name, "old": old_name})
    clear_caches()
    return True, f"Category renamed '{old_name}' → '{new_name}'."

def delete_category_val(name: str, reassign_to: Optional[str] = None) -> Tuple[bool, str]:
    name = (name or "").strip()
    victim_count = count_vendors_in_category(name if name != "" else "")
    if victim_count > 0 and not reassign_to:
        return False, f"Cannot delete '{name}': {victim_count} vendor(s) still use it. Provide a reassignment target."
    if victim_count > 0 and reassign_to is not None:
        run_sql(f"""
            UPDATE {VENDORS_TABLE}
               SET category = :newcat
             WHERE COALESCE(category,'') = :oldcat
        """, {"newcat": reassign_to.strip(), "oldcat": name})
    else:
        run_sql(f"""
            UPDATE {VENDORS_TABLE}
               SET category = NULL
             WHERE COALESCE(category,'') = :oldcat
        """, {"oldcat": name})
    clear_caches()
    return True, f"Category '{name}' deleted" + (f" and reassigned to '{reassign_to}'." if reassign_to else ".")

def rename_service(category: str, old_service: str, new_service: str) -> Tuple[bool, str]:
    if not new_service or new_service.strip() == "":
        return False, "New service name cannot be empty."
    category = (category or "").strip()
    old_service = (old_service or "").strip()
    new_service = new_service.strip()
    if new_service == old_service:
        return True, "No change."
    run_sql(f"""
        UPDATE {VENDORS_TABLE}
           SET service = :news
         WHERE COALESCE(category,'') = :cat
           AND COALESCE(service,'')  = :olds
    """, {"news": new_service, "cat": category, "olds": old_service})
    clear_caches()
    return True, f"Service renamed '{old_service}' → '{new_service}' in '{category}'."

def delete_service_val(category: str, service: str, reassign_to: Optional[str] = None) -> Tuple[bool, str]:
    victim_count = count_vendors_in_service(category, service)
    if victim_count > 0 and not reassign_to:
        return False, f"Cannot delete '{service}' in '{category}': {victim_count} vendor(s) still use it. Provide a reassignment target."
    if victim_count > 0 and reassign_to is not None:
        run_sql(f"""
            UPDATE {VENDORS_TABLE}
               SET service = :news
             WHERE COALESCE(category,'') = :cat
               AND COALESCE(service,'')  = :olds
        """, {"news": reassign_to.strip(), "cat": (category or ""), "olds": (service or "")})
    else:
        run_sql(f"""
            UPDATE {VENDORS_TABLE}
               SET service = NULL
             WHERE COALESCE(category,'') = :cat
               AND COALESCE(service,'')  = :olds
        """, {"cat": (category or ""), "olds": (service or "")})
    clear_caches()
    return True, f"Service '{service}' deleted" + (f" and reassigned to '{reassign_to}'." if reassign_to else ".")

# ---------------------------
# Vendor CRUD
# ---------------------------
def update_vendor(vid: int, data: Dict[str, Any]) -> None:
    cols = ["category","service","business_name","contact_name","phone","address","website","notes"]
    if HAS_KEYWORDS:
        cols.append("keywords")
    sets = ", ".join([f"{c} = :{c}" for c in cols])
    params = {c: data.get(c, None) for c in cols}
    params["id"] = vid
    run_sql(f"UPDATE {VENDORS_TABLE} SET {sets} WHERE id = :id", params)

def insert_vendor(data: Dict[str, Any]) -> int:
    cols = ["category","service","business_name","contact_name","phone","address","website","notes"]
    if HAS_KEYWORDS:
        cols.append("keywords")
    named = ", ".join(cols)
    binds = ", ".join([f":{c}" for c in cols])
    r = run_sql(f"INSERT INTO {VENDORS_TABLE} ({named}) VALUES ({binds}) RETURNING id", {c: data.get(c, None) for c in cols})
    try:
        new_id = r.fetchone()[0]
    except Exception:
        new_id = fetchone("SELECT last_insert_rowid()")[0]
    return int(new_id)

def delete_vendor(vid: int):
    run_sql(f"DELETE FROM {VENDORS_TABLE} WHERE id = :i", {"i": vid})

def required_ok(data: Dict[str, Any]) -> Tuple[bool, str]:
    missing = []
    for k in ("business_name","category","service"):
        if not (data.get(k) or "").strip():
            missing.append(k)
    if missing:
        return False, "Missing: " + ", ".join(missing)

    ph_raw = (data.get("phone") or "").strip()
    if ph_raw:
        d = phone_digits(ph_raw)
        # Accept 10 digits OR 11 with leading 1; normalize formats in normalize_phone
        if not (len(d) == 10 or (len(d) == 11 and d[0] == "1") or len(d) > 11):
            return False, "Phone must be 10 digits (or 11 with leading 1) or left blank."
    return True, ""

def compute_dirty(original: Dict[str, Any], edited: Dict[str, Any]) -> bool:
    o = dict(original)
    e = dict(edited)
    o["phone"] = normalize_phone(o.get("phone",""))
    e["phone"] = normalize_phone(e.get("phone",""))
    o["website"] = normalize_url(o.get("website",""))
    e["website"] = normalize_url(e.get("website",""))
    if HAS_KEYWORDS:
        o["keywords"] = parse_keywords(o.get("keywords",""))
        e["keywords"] = parse_keywords(e.get("keywords",""))
    for k in o.keys():
        if isinstance(o[k], str): o[k] = o[k].strip()
    for k in e.keys():
        if isinstance(e[k], str): e[k] = e[k].strip()
    return any((o.get(k) or "") != (e.get(k) or "") for k in o.keys() if k != "id")

# ---------------------------
# UI
# ---------------------------
st.set_page_config(page_title="Vendors Admin", layout="wide")
st.title("Vendors Admin")

tabs = st.tabs(["Vendors", "Categories & Services Admin", "Database Status"])

# ---- Vendors Tab
with tabs[0]:
    st.subheader("Browse / Edit / Add / Delete")

    df = load_vendors_df()
    show_cols = [c for c in df.columns if c != "id"]
    st.dataframe(df[show_cols], use_container_width=True, hide_index=True)

    st.markdown("---")
    mode = st.radio("Action", ["Add", "Edit", "Delete"], horizontal=True)

    if mode == "Add":
        with st.form("add_vendor_form", clear_on_submit=False):
            # Populate dropdowns
            cats = list_categories()
            cats_display = ["(blank)"] + cats + ["+ Add new…"]

            # Category select
            cat_choice = st.selectbox("Category", options=cats_display, index=0, key="add_cat")
            new_cat = ""
            resolved_cat = ""
            if cat_choice == "(blank)":
                resolved_cat = ""
            elif cat_choice == "+ Add new…":
                new_cat = st.text_input("New Category", placeholder="e.g., Plumbing", key="add_cat_new")
                resolved_cat = new_cat.strip()
            else:
                resolved_cat = cat_choice

            # Service select depends on resolved_cat
            svc_list = list_services_for(resolved_cat) if resolved_cat != "" else []
            svcs_display = ["(blank)"] + svc_list + ["+ Add new…"]
            svc_choice = st.selectbox("Service", options=svcs_display, index=0, key="add_svc")
            new_svc = ""
            resolved_svc = ""
            if svc_choice == "(blank)":
                resolved_svc = ""
            elif svc_choice == "+ Add new…":
                new_svc = st.text_input("New Service", placeholder="e.g., Water Heaters", key="add_svc_new")
                resolved_svc = new_svc.strip()
            else:
                resolved_svc = svc_choice

            col2a, col2b = st.columns(2)
            with col2a:
                business = st.text_input("Business Name", placeholder="Company LLC", key="add_business")
                contact  = st.text_input("Contact Name", placeholder="Jane Smith", key="add_contact")
                phone    = st.text_input("Phone (auto-formats)", placeholder="(210) 555-1212", key="add_phone")
            with col2b:
                address  = st.text_input("Address", placeholder="123 Main St, San Antonio, TX", key="add_addr")
                website  = st.text_input("Website", placeholder="example.com", key="add_site")
                notes    = st.text_area("Notes", placeholder="Free estimates, etc.", height=88, key="add_notes")
            keywords = st.text_input("Keywords (space or comma)", placeholder="drain leak emergency", key="add_kw") if HAS_KEYWORDS else ""

            data = {
                "category": resolved_cat,
                "service": resolved_svc,
                "business_name": (business or "").strip(),
                "contact_name": (contact or "").strip(),
                "phone": normalize_phone(phone or ""),
                "address": (address or "").strip(),
                "website": normalize_url(website or ""),
                "notes": (notes or "").strip(),
            }
            if HAS_KEYWORDS:
                data["keywords"] = parse_keywords(keywords or "")

            ok_req, msg_req = required_ok(data)
            add_btn = st.form_submit_button("Add Vendor", disabled=not ok_req)
            if add_btn:
                new_id = insert_vendor(data)
                st.success(f"Vendor added (id={new_id}).")
                clear_caches()
                st.rerun()

            if not ok_req:
                st.warning(msg_req)

    elif mode == "Edit":
        if df.empty:
            st.info("No vendors to edit.")
        else:
            # Sort vendor list A→Z by business_name (case-insensitive)
            df_names = df.sort_values("business_name", key=lambda s: s.str.lower())
            id_to_label = {int(r.id): f"{r.business_name}" for r in df_names.itertuples()}
            chosen_id = st.selectbox("Select Vendor", options=list(id_to_label.keys()),
                                     format_func=lambda i: id_to_label[i], key="edit_select_vendor")

            original = vendor_by_id(chosen_id)
            if not original:
                st.error("Could not load vendor record.")
            else:
                with st.form("edit_vendor_form"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        category = st.text_input("Category", value=original.get("category") or "", key="edit_cat")
                        service  = st.text_input("Service", value=original.get("service") or "", key="edit_svc")
                        business = st.text_input("Business Name", value=original.get("business_name") or "", key="edit_biz")
                    with col2:
                        contact  = st.text_input("Contact Name", value=original.get("contact_name") or "", key="edit_contact")
                        phone    = st.text_input("Phone (auto-formats)", value=original.get("phone") or "", key="edit_phone")
                        address  = st.text_input("Address", value=original.get("address") or "", key="edit_addr")
                    with col3:
                        website  = st.text_input("Website", value=original.get("website") or "", key="edit_site")
                        notes    = st.text_area("Notes", value=original.get("notes") or "", height=88, key="edit_notes")
                        keywords = st.text_input("Keywords (space or comma)",
                                                 value=original.get("keywords") or "", key="edit_kw") if HAS_KEYWORDS else ""

                    edited = {
                        "id": original["id"],
                        "category": (category or "").strip(),
                        "service": (service or "").strip(),
                        "business_name": (business or "").strip(),
                        "contact_name": (contact or "").strip(),
                        "phone": normalize_phone(phone or ""),
                        "address": (address or "").strip(),
                        "website": normalize_url(website or ""),
                        "notes": (notes or "").strip(),
                    }
                    if HAS_KEYWORDS:
                        edited["keywords"] = parse_keywords(keywords or "")

                    is_dirty = compute_dirty(original, edited)
                    ok_req, msg_req = required_ok(edited)
                    save_btn = st.form_submit_button("Save changes", disabled=(not is_dirty) or (not ok_req))
                    if save_btn:
                        update_vendor(chosen_id, edited)
                        st.success("Changes saved.")
                        clear_caches()
                        st.rerun()

                    if not ok_req:
                        st.warning(msg_req)
                    elif not is_dirty:
                        st.info("No changes detected.")

    else:  # Delete
        if df.empty:
            st.info("No vendors to delete.")
        else:
            # Labels: ONLY Business Name, sorted A→Z
            df_names = df.sort_values("business_name", key=lambda s: s.str.lower())
            id_to_label = {int(r.id): f"{r.business_name}" for r in df_names.itertuples()}
            chosen_id = st.selectbox("Select Vendor to Delete", options=list(id_to_label.keys()),
                                     format_func=lambda i: id_to_label[i], key="del_select_vendor")
            confirm = st.checkbox("Yes, delete this vendor")
            if st.button("Delete", disabled=not confirm):
                delete_vendor(chosen_id)
                st.success(f"Vendor {chosen_id} deleted.")
                clear_caches()
                st.rerun()

# ---- Categories & Services Admin
with tabs[1]:
    st.subheader("Manage Categories & Services")

    st.markdown("##### Categories")
    cats = list_categories()
    if not cats:
        st.info("No categories found (table may be empty).")
    else:
        c1, c2, c3 = st.columns([2,2,2])
        with c1:
            picked_cat = st.selectbox("Select Category", options=cats)
            st.caption(f"{count_vendors_in_category(picked_cat)} vendor(s) use this category.")
        with c2:
            new_cat_name = st.text_input("New name for selected Category", value="")
            if st.button("Rename Category", disabled=not bool(new_cat_name.strip())):
                ok, msg = rename_category(picked_cat, new_cat_name.strip())
                st.success(msg) if ok else st.error(msg)
                if ok: st.rerun()
        with c3:
            reassignment = st.text_input(
                "If deleting, reassign vendors to Category (optional)",
                placeholder="Type target category or leave blank to block"
            )
            if st.button("Delete Category"):
                ok, msg = delete_category_val(picked_cat, reassign_to=(reassignment.strip() or None))
                st.success(msg) if ok else st.error(msg)
                if ok: st.rerun()

    st.markdown("---")
    st.markdown("##### Services (within a Category)")
    cats2 = list_categories()
    if not cats2:
        st.info("No categories for services.")
    else:
        sc1, sc2, sc3 = st.columns([2,2,2])
        with sc1:
            cat_for_services = st.selectbox("Category for Services", options=cats2, key="svc_cat")
            services = list_services_for(cat_for_services)
            picked_svc = st.selectbox("Select Service", options=services or [""], key="svc_pick")
            st.caption(f"{count_vendors_in_service(cat_for_services, picked_svc)} vendor(s) use this service.")
        with sc2:
            new_svc_name = st.text_input("New name for selected Service", value="", key="svc_new")
            if st.button("Rename Service", disabled=not bool(new_svc_name.strip())):
                ok, msg = rename_service(cat_for_services, picked_svc, new_svc_name.strip())
                st.success(msg) if ok else st.error(msg)
                if ok: st.rerun()
        with sc3:
            reassign_svc = st.text_input(
                "If deleting, reassign vendors to Service (optional)",
                placeholder="Type target service or leave blank to block",
                key="svc_reassign"
            )
            if st.button("Delete Service"):
                ok, msg = delete_service_val(cat_for_services, picked_svc, reassign_to=(reassign_svc.strip() or None))
                st.success(msg) if ok else st.error(msg)
                if ok: st.rerun()

# ---- Database Status
with tabs[2]:
    st.subheader("Database Status & Schema (debug)")
    try:
        cols = fetchall("PRAGMA table_info(vendors)")
        schema_df = pd.DataFrame(cols, columns=["cid","name","type","notnull","dflt_value","pk"])
        st.write({
            "db_source": DATABASE_URL,
            "vendors_columns": list(schema_df["name"].values),
            "has_keywords": HAS_KEYWORDS,
        })
        st.dataframe(schema_df, use_container_width=True, hide_index=True)
    except Exception as e:
        st.error(f"Schema read failed: {e}")
