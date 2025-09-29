# app_admin.py
# Vendors Admin for Streamlit: SQLite (default) or libSQL via DATABASE_URL.
# - Fixes NameError for rename/delete category helpers.
# - Safe Category/Service rename & delete operations.
# - Edit Vendor form with proper "dirty" detection and button enablement.
# - Phone/URL normalization; flexible Keywords parsing (spaces or commas).

from __future__ import annotations

import os
import re
from typing import Optional, Tuple, Dict, Any, List
from urllib.parse import urlparse

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text

# ---------------------------
# Database setup
# ---------------------------
def _default_sqlite_url() -> str:
    # Fallback to local vendors.db in repo root.
    return f"sqlite:///vendors.db"

DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()
if not DATABASE_URL:
    DATABASE_URL = _default_sqlite_url()

engine = create_engine(DATABASE_URL, future=True)

@st.cache_data(show_spinner=False, ttl=10)
def has_column(tbl: str, col: str) -> bool:
    with engine.begin() as conn:
        rows = conn.execute(text(f"PRAGMA table_info({tbl})")).fetchall()
    return any(r[1] == col for r in rows)

def run_sql(sql: str, params: Dict[str, Any] | None = None):
    with engine.begin() as conn:
        return conn.execute(text(sql), params or {})

def fetchall(sql: str, params: Dict[str, Any] | None = None) -> List[Tuple]:
    with engine.begin() as conn:
        return list(conn.execute(text(sql), params or {}).fetchall())

def fetchone(sql: str, params: Dict[str, Any] | None = None) -> Optional[Tuple]:
    with engine.begin() as conn:
        return conn.execute(text(sql), params or {}).fetchone()

# ---------------------------
# Helpers: normalize fields
# ---------------------------
PHONE_DIGITS = re.compile(r"\D+")

def normalize_phone(raw: str) -> str:
    if not raw:
        return ""
    digits = PHONE_DIGITS.sub("", raw)
    if len(digits) == 10:
        return f"({digits[0:3]}) {digits[3:6]}-{digits[6:10]}"
    # Leave non-10-digit inputs as-is (but trimmed) so user can correct.
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
    # Split on commas or whitespace; collapse empties; lowercase for consistency.
    parts = re.split(r"[,\s]+", raw.strip())
    parts = [p for p in (x.strip() for x in parts) if p]
    # De-dup while preserving order
    seen = set()
    uniq = []
    for p in parts:
        pl = p.lower()
        if pl not in seen:
            uniq.append(pl)
            seen.add(pl)
    return ", ".join(uniq)

# ---------------------------
# Schema expectations
# ---------------------------
VENDORS_TABLE = "vendors"
REQUIRED_COLS = [
    "id","category","service","business_name","contact_name","phone","address","website","notes"
]
# keywords column optional
HAS_KEYWORDS = has_column(VENDORS_TABLE, "keywords")

# ---------------------------
# Categories / Services — discovery (text-based schema)
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
# Category ops (FIX: functions now exist)
# ---------------------------
def rename_category(old_name: str, new_name: str) -> Tuple[bool, str]:
    if not new_name or new_name.strip() == "":
        return False, "New category name cannot be empty."
    new_name = new_name.strip()
    old_name = (old_name or "").strip()

    # If already equals, noop
    if new_name == old_name:
        return True, "No change."

    run_sql(f"""
        UPDATE {VENDORS_TABLE}
           SET category = :new
         WHERE COALESCE(category,'') = :old
    """, {"new": new_name, "old": old_name})
    # Bust caches
    list_categories.clear()
    list_services_for.clear()
    return True, f"Category renamed '{old_name}' → '{new_name}'."

def delete_category_val(name: str, reassign_to: Optional[str] = None) -> Tuple[bool, str]:
    """Delete a category value. If vendors exist:
       - If reassign_to provided: move them.
       - Else: block delete and tell user how many exist.
    """
    name = (name or "").strip()
    if name == "":
        # deleting the blank category
        victim_count = count_vendors_in_category("")
    else:
        victim_count = count_vendors_in_category(name)

    if victim_count > 0 and not reassign_to:
        return (False,
                f"Cannot delete '{name}': {victim_count} vendor(s) still use it. "
                f"Provide a 'reassign_to' value to move them first.")
    if victim_count > 0 and reassign_to is not None:
        run_sql(f"""
            UPDATE {VENDORS_TABLE}
               SET category = :newcat
             WHERE COALESCE(category,'') = :oldcat
        """, {"newcat": reassign_to.strip(), "oldcat": name})
    else:
        # Optional: normalize exact deletes by setting NULL where matches victim?
        run_sql(f"""
            UPDATE {VENDORS_TABLE}
               SET category = NULL
             WHERE COALESCE(category,'') = :oldcat
        """, {"oldcat": name})

    list_categories.clear()
    list_services_for.clear()
    return True, f"Category '{name}' deleted" + (f" and reassigned to '{reassign_to}'." if reassign_to else ".")

# ---------------------------
# Service ops (parallel to category)
# ---------------------------
def rename_service(category: str, old_service: str, new_service: str) -> Tuple[bool, str]:
    if not new_service or new_service.strip() == "":
        return False, "New service name cannot be empty."
    new_service = new_service.strip()
    old_service = (old_service or "").strip()
    category = (category or "").strip()

    if new_service == old_service:
        return True, "No change."

    run_sql(f"""
        UPDATE {VENDORS_TABLE}
           SET service = :news
         WHERE COALESCE(category,'') = :cat
           AND COALESCE(service,'')  = :olds
    """, {"news": new_service, "cat": category, "olds": old_service})
    list_services_for.clear()
    return True, f"Service renamed '{old_service}' → '{new_service}' in category '{category}'."

def delete_service_val(category: str, service: str, reassign_to: Optional[str] = None) -> Tuple[bool, str]:
    victim_count = count_vendors_in_service(category, service)
    if victim_count > 0 and not reassign_to:
        return (False,
                f"Cannot delete '{service}' in '{category}': {victim_count} vendor(s) still use it. "
                f"Provide 'reassign_to' to move them.")
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
    list_services_for.clear()
    return True, f"Service '{service}' deleted" + (f" and reassigned to '{reassign_to}'." if reassign_to else ".")

# ---------------------------
# UI Utilities
# ---------------------------
st.set_page_config(page_title="Vendors Admin", layout="wide")

def load_vendors_df() -> pd.DataFrame:
    cols = REQUIRED_COLS + (["keywords"] if HAS_KEYWORDS else [])
    q = f"SELECT {', '.join(cols)} FROM {VENDORS_TABLE} ORDER BY LOWER(COALESCE(category,'')), LOWER(COALESCE(service,'')), LOWER(business_name)"
    rows = fetchall(q)
    df = pd.DataFrame(rows, columns=cols)
    return df

def vendor_by_id(vid: int) -> Dict[str, Any] | None:
    cols = REQUIRED_COLS + (["keywords"] if HAS_KEYWORDS else [])
    row = fetchone(f"SELECT {', '.join(cols)} FROM {VENDORS_TABLE} WHERE id = :i", {"i": vid})
    if not row:
        return None
    return dict(zip(cols, row))

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
        # libSQL / SQLite with RETURNING
        new_id = r.fetchone()[0]
    except Exception:
        # SQLite w/o RETURNING
        new_id = fetchone("SELECT last_insert_rowid()")[0]
    return int(new_id)

def delete_vendor(vid: int):
    run_sql(f"DELETE FROM {VENDORS_TABLE} WHERE id = :i", {"i": vid})

def required_ok(data: Dict[str, Any]) -> Tuple[bool, str]:
    missing = []
    # Define required: business_name, category, service
    for k in ("business_name","category","service"):
        if not (data.get(k) or "").strip():
            missing.append(k)
    if missing:
        return False, "Missing: " + ", ".join(missing)
    # Phone format advisory: not hard-fail if not 10 digits
    ph = data.get("phone","").strip()
    d = PHONE_DIGITS.sub("", ph)
    if ph and len(d) not in (0,10):
        return False, "Phone must be 10 digits or left blank."
    return True, ""

def compute_dirty(original: Dict[str, Any], edited: Dict[str, Any]) -> bool:
    # Compare normalized representations
    norm_o = dict(original)
    norm_e = dict(edited)

    norm_o["phone"] = normalize_phone(norm_o.get("phone",""))
    norm_e["phone"] = normalize_phone(norm_e.get("phone",""))

    norm_o["website"] = normalize_url(norm_o.get("website",""))
    norm_e["website"] = normalize_url(norm_e.get("website",""))

    if HAS_KEYWORDS:
        norm_o["keywords"] = parse_keywords(norm_o.get("keywords",""))
        norm_e["keywords"] = parse_keywords(norm_e.get("keywords",""))

    # Normalize whitespace
    for k in norm_o.keys():
        if isinstance(norm_o[k], str): norm_o[k] = norm_o[k].strip()
    for k in norm_e.keys():
        if isinstance(norm_e[k], str): norm_e[k] = norm_e[k].strip()

    return any((norm_o.get(k) or "") != (norm_e.get(k) or "") for k in norm_o.keys() if k != "id")

# ---------------------------
# Pages
# ---------------------------
st.title("Vendors Admin")

tabs = st.tabs(["Vendors", "Categories & Services Admin", "Database Status"])

# ---- Vendors Tab
with tabs[0]:
    st.subheader("Browse / Edit / Add / Delete")

    df = load_vendors_df()
    st.dataframe(df.drop(columns=["id"]), use_container_width=True, hide_index=True)

    st.markdown("---")
    mode = st.radio("Action", ["Add", "Edit", "Delete"], horizontal=True)

    if mode == "Add":
        with st.form("add_vendor_form", clear_on_submit=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                category = st.text_input("Category", placeholder="e.g., Plumbing")
                service  = st.text_input("Service", placeholder="e.g., Water Heaters")
                business = st.text_input("Business Name", placeholder="Company LLC")
            with col2:
                contact  = st.text_input("Contact Name", placeholder="Jane Smith")
                phone    = st.text_input("Phone (auto-formats)", placeholder="(210) 555-1212")
                address  = st.text_input("Address", placeholder="123 Main St, San Antonio, TX")
            with col3:
                website  = st.text_input("Website", placeholder="example.com")
                notes    = st.text_area("Notes", placeholder="Free estimates, etc.", height=88)
                keywords = st.text_input("Keywords (space or comma)", placeholder="drain leak emergency") if HAS_KEYWORDS else ""

            data = {
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
                data["keywords"] = parse_keywords(keywords or "")

            ok_req, msg_req = required_ok(data)
            add_btn = st.form_submit_button("Add Vendor", disabled=not ok_req)
            if add_btn:
                new_id = insert_vendor(data)
                st.success(f"Vendor added (id={new_id}).")
                load_vendors_df.clear()
                list_categories.clear()
                list_services_for.clear()

    elif mode == "Edit":
        # Select Vendor
        id_to_label = {int(r.id): f"[{r.category or ''} / {r.service or ''}] {r.business_name}" for r in df.itertuples()}
        if not id_to_label:
            st.info("No vendors to edit.")
        else:
            chosen_id = st.selectbox("Select Vendor", options=list(id_to_label.keys()), format_func=lambda i: id_to_label[i])
            original = vendor_by_id(chosen_id)
            if not original:
                st.error("Could not load vendor record.")
            else:
                with st.form("edit_vendor_form"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        category = st.text_input("Category", value=original.get("category") or "")
                        service  = st.text_input("Service", value=original.get("service") or "")
                        business = st.text_input("Business Name", value=original.get("business_name") or "")
                    with col2:
                        contact  = st.text_input("Contact Name", value=original.get("contact_name") or "")
                        phone    = st.text_input("Phone (auto-formats)", value=original.get("phone") or "")
                        address  = st.text_input("Address", value=original.get("address") or "")
                    with col3:
                        website  = st.text_input("Website", value=original.get("website") or "")
                        notes    = st.text_area("Notes", value=original.get("notes") or "", height=88)
                        keywords = st.text_input("Keywords (space or comma)",
                                                 value=original.get("keywords") or "") if HAS_KEYWORDS else ""

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
                        load_vendors_df.clear()
                        list_categories.clear()
                        list_services_for.clear()

                    if not ok_req:
                        st.warning(msg_req)
                    elif not is_dirty:
                        st.info("No changes detected.")

    else:  # Delete
        id_to_label = {int(r.id): f"[{r.category or ''} / {r.service or ''}] {r.business_name}" for r in df.itertuples()}
        if not id_to_label:
            st.info("No vendors to delete.")
        else:
            chosen_id = st.selectbox("Select Vendor to Delete", options=list(id_to_label.keys()), format_func=lambda i: id_to_label[i])
            confirm = st.checkbox("Yes, delete this vendor")
            if st.button("Delete", disabled=not confirm):
                delete_vendor(chosen_id)
                st.success(f"Vendor {chosen_id} deleted.")
                load_vendors_df.clear()
                list_categories.clear()
                list_services_for.clear()

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
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)

        with c3:
            reassignment = st.text_input("If deleting, reassign vendors to Category (optional)", placeholder="Type target category or leave blank to block")
            if st.button("Delete Category"):
                ok, msg = delete_category_val(picked_cat, reassign_to=(reassignment.strip() or None))
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)

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
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)

        with sc3:
            reassign_svc = st.text_input("If deleting, reassign vendors to Service (optional)", placeholder="Type target service or leave blank to block", key="svc_reassign")
            if st.button("Delete Service"):
                ok, msg = delete_service_val(cat_for_services, picked_svc, reassign_to=(reassign_svc.strip() or None))
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)

# ---- Database Status
with tabs[2]:
    st.subheader("Database Status & Schema (debug)")
    cols = fetchall("PRAGMA table_info(vendors)")
    schema_df = pd.DataFrame(cols, columns=["cid","name","type","notnull","dflt_value","pk"])
    st.write({
        "db_source": DATABASE_URL,
        "vendors_columns": list(schema_df["name"].values),
        "has_keywords": HAS_KEYWORDS,
    })
    st.dataframe(schema_df, use_container_width=True, hide_index=True)
