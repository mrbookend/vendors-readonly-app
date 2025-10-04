# app_admin.py
# Vendors Admin — Category REQUIRED, Service optional (blank display if missing)
# - Categories/Services Admin surfaces **orphans**
# - All comparisons use lower(trim(...))
# - Clear success notices on Add/Edit/Delete
# - Capitalization & trimming on save; Maintenance tab to normalize
# - Turso/libSQL via sqlite+libsql://…?secure=true with connect_args={"auth_token": ...}
# - VIEW TAB: Sortable HTML grid (true px widths + variable row height via wrapping)

from __future__ import annotations

import os
import re
import time
import html
from typing import List, Dict, Optional, Tuple

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text as sql_text
from sqlalchemy.engine import Engine
import streamlit.components.v1 as components

# Optional layout helper
try:
    from layout_header import apply_layout
    apply_layout()
except Exception:
    pass

# -----------------------------
# Secrets helpers
# -----------------------------
def _get_secret(name: str, default: Optional[str | int] = None):
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.environ.get(name, default if default is None else str(default))

def _get_int_secret(name: str, default: int) -> int:
    val = _get_secret(name, None)
    try:
        return int(val) if val is not None else default
    except Exception:
        return default

def _get_libsql_creds() -> Dict[str, Optional[str]]:
    url = (_get_secret("LIBSQL_URL") or _get_secret("TURSO_DATABASE_URL"))
    token = (_get_secret("LIBSQL_AUTH_TOKEN") or _get_secret("TURSO_AUTH_TOKEN"))
    return {"url": url, "token": token}

def _default_sqlite_path() -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, "vendors.db")

# -----------------------------
# DB selection & engine
# -----------------------------
_LIBSQL_SCHEME_RE = re.compile(r"^libsql://", re.IGNORECASE)

def _normalize_libsql_url(url: str) -> str:
    if _LIBSQL_SCHEME_RE.match(url):
        return _LIBSQL_SCHEME_RE.sub("sqlite+libsql://", url, count=1)
    return url

def _append_secure_param(dsn: str) -> str:
    if "secure=" in dsn:
        return dsn
    return f"{dsn}&secure=true" if "?" in dsn else f"{dsn}?secure=true"

def current_db_info() -> Dict[str, Optional[str]]:
    creds = _get_libsql_creds()
    sqlite_path = _get_secret("SQLITE_PATH") or _default_sqlite_path()
    if creds["url"]:
        dsn = _normalize_libsql_url(str(creds["url"]))
        dsn = _append_secure_param(dsn)
        return {
            "backend": "libsql",
            "dsn": dsn,
            "auth": "token_set" if creds["token"] else "no_token",
            "sqlite_path": None,
            "note": "Remote DB (persistent).",
        }
    return {
        "backend": "sqlite",
        "dsn": f"sqlite:///{sqlite_path}",
        "auth": None,
        "sqlite_path": sqlite_path,
        "note": "Local file; may be ephemeral on cloud. Prefer LIBSQL_URL/TURSO_DATABASE_URL.",
    }

def get_engine() -> Engine:
    info = current_db_info()
    if info["backend"] == "libsql":
        connect_args = {}
        token = (_get_secret("LIBSQL_AUTH_TOKEN") or _get_secret("TURSO_AUTH_TOKEN"))
        if token:
            connect_args["auth_token"] = token
        return create_engine(info["dsn"], connect_args=connect_args, pool_pre_ping=True, future=True)
    return create_engine(info["dsn"], future=True)

engine = get_engine()

def _db_write_probe() -> Dict[str, str]:
    result = {"status": "unknown", "detail": ""}
    try:
        with engine.begin() as conn:
            conn.execute(sql_text("CREATE TABLE IF NOT EXISTS app_admin_probe (ts TEXT)"))
            conn.execute(sql_text("INSERT INTO app_admin_probe(ts) VALUES(:ts)"), {"ts": str(time.time())})
        result["status"] = "ok"
        result["detail"] = "Write succeeded."
    except Exception as e:
        result["status"] = "error"
        result["detail"] = f"{e}"
    return result

# -----------------------------
# Introspection & helpers
# -----------------------------
@st.cache_data(show_spinner=False, ttl=60)
def table_exists(name: str) -> bool:
    q = sql_text("""
        SELECT 1
        FROM sqlite_master
        WHERE type IN ('table','view') AND lower(name) = lower(:n)
        UNION ALL
        SELECT 1 FROM pragma_table_info(:n) LIMIT 1
    """)
    try:
        with engine.begin() as conn:
            r = conn.execute(q, {"n": name}).first()
        return bool(r)
    except Exception:
        try:
            with engine.begin() as conn:
                conn.execute(sql_text(f"SELECT 1 FROM {name} WHERE 1=0"))
            return True
        except Exception:
            return False

@st.cache_data(show_spinner=False, ttl=60)
def get_columns(table: str) -> List[str]:
    try:
        with engine.begin() as conn:
            rows = conn.execute(sql_text(f"PRAGMA table_info({table})")).fetchall()
            return [r[1] for r in rows]
    except Exception:
        try:
            with engine.begin() as conn:
                df = pd.read_sql_query(sql_text(f"SELECT * FROM {table} LIMIT 1"), conn)
            return list(df.columns)
        except Exception:
            return []

@st.cache_data(show_spinner=False, ttl=60)
def vendors_has_keywords() -> bool:
    return "keywords" in get_columns("vendors")

def normalize_phone(raw: str | None) -> str | None:
    if not raw:
        return None
    digits = re.sub(r"\D+", "", raw)
    if len(digits) != 10:
        return raw.strip()
    return f"({digits[0:3]}) {digits[3:6]}-{digits[6:10]}"

def normalize_url(url: str | None) -> str | None:
    if not url:
        return None
    u = url.strip()
    if not u:
        return None
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+\-.]*://", u):
        u = "https://" + u
    return u

# -----------------------------
# Capitalization helpers
# -----------------------------
ACRONYMS = {"LLC", "INC", "LLP", "DBA", "HVAC", "USA", "CPA", "PC", "PA", "MD", "DDS", "P.C.", "P.A."}
_word_splitter = re.compile(r"([\-'/])")

def _cap_token(tok: str) -> str:
    t = tok.strip()
    if not t:
        return t
    if t.upper() in ACRONYMS:
        return t.upper()
    parts = _word_splitter.split(t)
    out: List[str] = []
    for p in parts:
        if p in "-'/":
            out.append(p)
        else:
            out.append(p[:1].upper() + p[1:].lower() if p else p)
    return "".join(out)

def smart_title(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    t = re.sub(r"\s+", " ", s.strip())
    if not t:
        return ""
    return " ".join(_cap_token(w) for w in t.split(" "))

def title_address(s: Optional[str]) -> Optional[str]:
    t = smart_title(s)
    if not t:
        return t
    t = re.sub(r"\b(N|S|E|W|NE|NW|SE|SW)\b", lambda m: m.group(0).upper(), t, flags=re.IGNORECASE)
    t = re.sub(r"\bP[.\s]*O[.\s]*\s*Box\b", "PO Box", t, flags=re.IGNORECASE)
    return t

# -----------------------------
# Column width configuration (robust loader)
# -----------------------------
def _get_column_widths_px() -> Dict[str, int]:
    """
    Load COLUMN_WIDTHS_PX from secrets. Accepts:
      - Mapping (Streamlit secrets AttrDict)
      - JSON string
      - simple 'key = value' string
    Maps 'small'/'medium'/'large' to sentinels (-1/-2/-3); converter turns those into px.
    """
    import json
    from collections.abc import Mapping

    def _coerce_map(obj) -> Dict[str, int]:
        out: Dict[str, int] = {}
        if not isinstance(obj, Mapping):
            return out
        for k, v in obj.items():
            key = str(k).strip().lower()
            if isinstance(v, str) and v.strip().lower() in {"small", "medium", "large"}:
                out[key] = {"small": -1, "medium": -2, "large": -3}[v.strip().lower()]
                continue
            try:
                out[key] = int(str(v).strip())
            except Exception:
                pass
        return out

    try:
        raw = st.secrets.get("COLUMN_WIDTHS_PX", None)
    except Exception:
        raw = None

    if isinstance(raw, Mapping):
        mapping = _coerce_map(raw)
    elif isinstance(raw, str):
        s = raw.strip()
        mapping = {}
        try:
            obj = json.loads(s)
            if isinstance(obj, Mapping):
                mapping = _coerce_map(obj)
        except Exception:
            pass
        if not mapping:
            for line in s.splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                key = k.strip().strip('"').strip("'").lower()
                val = v.strip().strip('"').strip("'")
                if val.lower() in {"small", "medium", "large"}:
                    mapping[key] = {"small": -1, "medium": -2, "large": -3}[val.lower()]
                else:
                    try:
                        mapping[key] = int(val)
                    except Exception:
                        pass
    else:
        mapping = {}

    website_w = _get_int_secret("WEBSITE_COL_WIDTH_PX", 0)
    if website_w and "website" not in mapping:
        mapping["website"] = website_w

    return mapping

def _width_value_to_px(val) -> int:
    """Convert secrets values/sentinels to pixel widths; supports ints and size words."""
    if isinstance(val, int):
        if val in (-1, -2, -3):   # small/medium/large
            return { -1: 100, -2: 160, -3: 240 }[val]
        return max(20, val)
    try:
        s = str(val).strip().lower()
        return {"small": 100, "s": 100, "medium": 160, "m": 160, "large": 240, "l": 240}.get(s, 140)
    except Exception:
        return 140

# -----------------------------
# HTML GRID (sortable + wrap + true px widths)
# -----------------------------
def _render_sortable_wrapped_table(df: pd.DataFrame, px_map: dict[str, int], height_px: int = 700) -> None:
    """
    Render a client-side sortable table with:
      - true pixel widths via <colgroup>
      - row auto-height via wrapping
      - sticky header, hover highlight
      - quick filter box (client-side)
    Sorting by clicking any header (toggles asc/desc).
    """
    if df.empty:
        st.info("No vendors found.")
        return

    cols = list(df.columns)

    def _px(col: str) -> int:
        try:
            v = px_map.get(col, 140)
            return _width_value_to_px(v)
        except Exception:
            return 140

    # Build HTML
    head_css = f"""
    <style>
    .tbl-container {{
        border: 1px solid #e6e6e6;
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }}
    .tbl-toolbar {{
        padding: 8px 10px;
        border-bottom: 1px solid #eee;
        background: #fafafa;
        display: flex;
        gap: 8px;
        align-items: center;
        font-size: 14px;
    }}
    .tbl-filter {{
        flex: 1;
        padding: 6px 8px;
        border: 1px solid #ddd;
        border-radius: 6px;
        outline: none;
    }}
    .tbl-viewport {{
        max-height: {height_px - 46}px;
        overflow: auto;
    }}
    table.tbl {{
        table-layout: fixed;
        width: 100%;
        border-collapse: collapse;
        font-size: 14px;
    }}
    .tbl th, .tbl td {{
        border-bottom: 1px solid #f0f0f0;
        padding: 6px 8px;
        vertical-align: top;
    }}
    .tbl thead th {{
        position: sticky;
        top: 0;
        background: #ffffff;
        z-index: 2;
        cursor: pointer;
        user-select: none;
    }}
    .tbl tbody tr:hover {{
        background: #fcfcfc;
    }}
    .tbl td {{
        white-space: normal;        /* allow wrapping */
        word-break: break-word;
        text-overflow: clip;
        overflow: visible;
    }}
    .th-inner {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }}
    .sort-arrow {{
        font-size: 12px;
        color: #888;
        visibility: hidden;
    }}
    th.sorted .sort-arrow {{
        visibility: visible;
    }}
    .text-right {{ text-align: right; }}
    .text-center {{ text-align: center; }}
    </style>
    """

    # colgroup with explicit widths
    colgroup = ["<colgroup>"] + [f'<col style="width:{_px(c)}px">' for c in cols] + ["</colgroup>"]

    # header (clickable)
    thead = ["<thead><tr>"]
    for c in cols:
        label = html.escape(c.replace("_", " ").title())
        thead.append(f'<th data-col="{html.escape(c)}"><span class="th-inner">{label}<span class="sort-arrow">▲</span></span></th>')
    thead.append("</tr></thead>")

    # body
    tbody = ["<tbody>"]
    for _, row in df.iterrows():
        tbody.append("<tr>")
        for c in cols:
            val = row[c]
            if c == "website" and isinstance(val, str) and val.strip():
                safe_url = html.escape(val)
                safe_text = html.escape(val)
                tbody.append(f'<td><a href="{safe_url}" target="_blank" rel="noopener noreferrer">{safe_text}</a></td>')
            else:
                safe = "" if pd.isna(val) else str(val)
                tbody.append(f"<td>{html.escape(safe)}</td>")
        tbody.append("</tr>")
    tbody.append("</tbody>")

    # quick filter + sorting JS
    script = f"""
    <script>
    (function() {{
        const container = document.currentScript.closest('.tbl-container');
        const input = container.querySelector('.tbl-filter');
        const table = container.querySelector('table.tbl');
        const thead = table.tHead;
        const tbody = table.tBodies[0];
        const headers = Array.from(thead.rows[0].cells);
        let sortState = {{}}; // colIndex -> 'asc'|'desc'

        function getCellText(td) {{
            return (td.textContent || td.innerText || '').trim();
        }}

        function compare(a, b, idx, asc) {{
            const ta = getCellText(a.cells[idx]);
            const tb = getCellText(b.cells[idx]);
            // try numeric compare
            const na = parseFloat(ta.replace(/[^0-9.-]/g, ''));
            const nb = parseFloat(tb.replace(/[^0-9.-]/g, ''));
            const bothNumeric = !isNaN(na) && !isNaN(nb) && ta !== '' && tb !== '';
            let cmp = 0;
            if (bothNumeric) {{
                cmp = na - nb;
            }} else {{
                cmp = ta.localeCompare(tb, undefined, {{ sensitivity: 'base' }});
            }}
            return asc ? cmp : -cmp;
        }}

        headers.forEach((th, idx) => {{
            th.addEventListener('click', () => {{
                // toggle sort dir
                const dir = sortState[idx] === 'asc' ? 'desc' : 'asc';
                sortState = {{}}; sortState[idx] = dir;

                // visual arrow
                headers.forEach(h => h.classList.remove('sorted'));
                th.classList.add('sorted');
                const arrow = th.querySelector('.sort-arrow');
                if (arrow) arrow.textContent = dir === 'asc' ? '▲' : '▼';

                // sort rows
                const rows = Array.from(tbody.rows);
                rows.sort((ra, rb) => compare(ra, rb, idx, dir === 'asc'));
                rows.forEach(r => tbody.appendChild(r));
            }});
        }});

        // quick filter: match any cell
        input.addEventListener('input', () => {{
            const q = input.value.toLowerCase();
            Array.from(tbody.rows).forEach(tr => {{
                const text = tr.innerText.toLowerCase();
                tr.style.display = text.includes(q) ? '' : 'none';
            }});
        }});
    }})();
    </script>
    """

    toolbar = """
    <div class="tbl-toolbar">
      <input class="tbl-filter" placeholder="Quick filter (matches any column)…" />
    </div>
    """

    html_doc = (
        '<div class="tbl-container">'
        + head_css
        + toolbar
        + '<div class="tbl-viewport"><table class="tbl">'
        + "".join(colgroup)
        + "".join(thead)
        + "".join(tbody)
        + "</table></div>"
        + script
        + "</div>"
    )

    # Render inside an iframe so JS works (Streamlit sanitizes <script> in markdown)
    components.html(html_doc, height=height_px, scrolling=True)
    # Note: 'height_px' controls the viewport; table itself expands row height as needed.

# -----------------------------
# Categories/Services: lists & orphans
# -----------------------------
@st.cache_data(show_spinner=False, ttl=30)
def list_categories_table() -> List[str]:
    if table_exists("categories") and "name" in get_columns("categories"):
        with engine.begin() as conn:
            rows = conn.execute(sql_text("SELECT name FROM categories ORDER BY lower(name) ASC")).fetchall()
        return [r[0] for r in rows if r[0]]
    return []

@st.cache_data(show_spinner=False, ttl=30)
def list_categories_from_vendors() -> List[str]:
    with engine.begin() as conn:
        rows = conn.execute(sql_text(
            "SELECT DISTINCT trim(category) AS c "
            "FROM vendors WHERE IFNULL(trim(category),'')<>'' "
            "ORDER BY lower(c) ASC"
        )).fetchall()
    return [r[0] for r in rows if r[0]]

@st.cache_data(show_spinner=False, ttl=30)
def list_services_table() -> List[str]:
    if table_exists("services") and "name" in get_columns("services"):
        with engine.begin() as conn:
            rows = conn.execute(sql_text("SELECT name FROM services ORDER BY lower(name) ASC")).fetchall()
        return [r[0] for r in rows if r[0]]
    return []

@st.cache_data(show_spinner=False, ttl=30)
def list_services_from_vendors() -> List[str]:
    with engine.begin() as conn:
        rows = conn.execute(sql_text(
            "SELECT DISTINCT trim(service) AS s "
            "FROM vendors WHERE IFNULL(trim(service),'')<>'' "
            "ORDER BY lower(s) ASC"
        )).fetchall()
    return [r[0] for r in rows if r[0]]

def list_categories() -> List[str]:
    cats_tbl = list_categories_table()
    if cats_tbl:
        return cats_tbl
    return list_categories_from_vendors()

def list_services() -> List[str]:
    svcs_tbl = list_services_table()
    if svcs_tbl:
        return svcs_tbl
    return list_services_from_vendors()

def invalidate_caches():
    for f in (
        list_categories_table, list_categories_from_vendors, list_categories,
        list_services_table, list_services_from_vendors, list_services,
        get_columns, table_exists, vendors_has_keywords
    ):
        try:
            f.clear()  # type: ignore[attr-defined]
        except Exception:
            pass
    try:
        st.cache_data.clear()
    except Exception:
        pass

# ---- Usage counters + previews (TRIM-aware) ----
def count_vendors_with_category(name: str) -> int:
    with engine.begin() as conn:
        return int(conn.execute(sql_text(
            "SELECT COUNT(1) FROM vendors WHERE lower(trim(category)) = lower(trim(:n))"
        ), {"n": name}).scalar_one())

def count_vendors_with_service(name: str) -> int:
    with engine.begin() as conn:
        return int(conn.execute(sql_text(
            "SELECT COUNT(1) FROM vendors WHERE lower(trim(service)) = lower(trim(:n))"
        ), {"n": name}).scalar_one())

def list_vendors_by_category(name: str, limit: int = 15) -> List[Tuple[int, str]]:
    with engine.begin() as conn:
        rows = conn.execute(sql_text(
            "SELECT id, business_name FROM vendors "
            "WHERE lower(trim(category)) = lower(trim(:n)) "
            "ORDER BY lower(business_name) ASC LIMIT :lim"
        ), {"n": name, "lim": limit}).fetchall()
    return [(int(r[0]), r[1] or "") for r in rows]

def list_vendors_by_service(name: str, limit: int = 15) -> List[Tuple[int, str]]:
    with engine.begin() as conn:
        rows = conn.execute(sql_text(
            "SELECT id, business_name FROM vendors "
            "WHERE lower(trim(service)) = lower(trim(:n)) "
            "ORDER BY lower(business_name) ASC LIMIT :lim"
        ), {"n": name, "lim": limit}).fetchall()
    return [(int(r[0]), r[1] or "") for r in rows]

# -----------------------------
# Load vendors
# -----------------------------
@st.cache_data(show_spinner=False, ttl=30)
def load_vendors_df_cached(sel_cols: List[str]) -> pd.DataFrame:
    with engine.begin() as conn:
        return pd.read_sql_query(sql_text(f"SELECT {', '.join(sel_cols)} FROM vendors"), conn)

def load_vendors_df() -> pd.DataFrame:
    cols = get_columns("vendors")
    base_cols = ["id", "category", "service", "business_name", "contact_name",
                 "phone", "address", "website", "notes"]
    optional = []
    if "keywords" in cols:
        optional.append("keywords")
    sel_cols = [c for c in base_cols + optional if c in cols]
    if not sel_cols:
        return pd.DataFrame()
    df = load_vendors_df_cached(sel_cols)
    if "service" in df.columns:
        df["service"] = df["service"].apply(lambda v: v if (isinstance(v, str) and v.strip()) else "")
    if "category" in df.columns:
        df["category"] = df["category"].apply(lambda v: v.strip() if isinstance(v, str) else v)
    if "business_name" in df.columns:
        df = df.sort_values("business_name", key=lambda s: s.str.lower()).reset_index(drop=True)
    return df

# -----------------------------
# Upserts & deletes (forced capitalization/trim)
# -----------------------------
def insert_vendor(
    category: str,
    service: Optional[str],
    business_name: str,
    contact_name: Optional[str],
    phone: Optional[str],
    address: Optional[str],
    website: Optional[str],
    notes: Optional[str],
    keywords: Optional[str],
):
    if not category or not str(category).strip():
        raise ValueError("Category is required and cannot be blank.")
    cols = get_columns("vendors")
    has_kw = "keywords" in cols
    sql = """
        INSERT INTO vendors (category, service, business_name, contact_name, phone, address, website, notes{kwc})
        VALUES (:category, :service, :business_name, :contact_name, :phone, :address, :website, :notes{kwv})
    """.format(kwc=", keywords" if has_kw else "", kwv=", :keywords" if has_kw else "")
    params = {
        "category": smart_title(category).strip(),
        "service": smart_title(service) if service else None,
        "business_name": smart_title(business_name),
        "contact_name": smart_title(contact_name) if contact_name else None,
        "phone": normalize_phone(phone),
        "address": title_address(address) if address else None,
        "website": normalize_url(website),
        "notes": (notes.strip() if notes else None),
    }
    if has_kw:
        params["keywords"] = (keywords or None)
    with engine.begin() as conn:
        conn.execute(sql_text(sql), params)

def update_vendor(
    vid: int,
    category: str,
    service: Optional[str],
    business_name: str,
    contact_name: Optional[str],
    phone: Optional[str],
    address: Optional[str],
    website: Optional[str],
    notes: Optional[str],
    keywords: Optional[str],
):
    if not category or not str(category).strip():
        raise ValueError("Category is required and cannot be blank.")
    cols = get_columns("vendors")
    has_kw = "keywords" in cols
    sql = """
        UPDATE vendors
        SET category = :category,
            service = :service,
            business_name = :business_name,
            contact_name = :contact_name,
            phone = :phone,
            address = :address,
            website = :website,
            notes = :notes{kwset}
        WHERE id = :id
    """.format(kwset=", keywords = :keywords" if has_kw else "")
    params = {
        "id": int(vid),
        "category": smart_title(category).strip(),
        "service": smart_title(service) if service else None,
        "business_name": smart_title(business_name),
        "contact_name": smart_title(contact_name) if contact_name else None,
        "phone": normalize_phone(phone),
        "address": title_address(address) if address else None,
        "website": normalize_url(website),
        "notes": (notes.strip() if notes else None),
    }
    if has_kw:
        params["keywords"] = (keywords or None)
    with engine.begin() as conn:
        conn.execute(sql_text(sql), params)

def delete_vendor(vid: int):
    with engine.begin() as conn:
        conn.execute(sql_text("DELETE FROM vendors WHERE id = :id"), {"id": int(vid)})

# -----------------------------
# UI helpers
# -----------------------------
def rerun():
    st.rerun()

def text_input_w(label: str, value: Optional[str], key: str) -> str:
    return st.text_input(label, value=value or "", key=key)

def select_category_required(label: str, value: Optional[str], key: str) -> str:
    cats = list_categories()
    if cats:
        idx = 0
        if value and value in cats:
            idx = cats.index(value)
        selected = st.selectbox(label + " *", options=cats, index=idx, key=key, help="Category is required.")
        return selected.strip() if selected else ""
    else:
        typed = st.text_input(label + " * (type a category name)", value=value or "", key=key, help="Enter a new category.")
        return typed.strip()

def select_service_optional(label: str, value: Optional[str], key: str) -> Optional[str]:
    svcs = list_services()
    options = [""] + svcs
    idx = 0
    if value and value in svcs:
        idx = options.index(value)
    chosen = st.selectbox(label, options=options, index=idx, key=key, help="Blank is allowed.")
    return None if (chosen is None or str(chosen).strip() == "") else chosen

def load_vendor_by_id(vid: int) -> Optional[Dict]:
    with engine.begin() as conn:
        df = pd.read_sql_query(sql_text("SELECT * FROM vendors WHERE id = :id"), conn, params={"id": int(vid)})
    if df.empty:
        return None
    row = df.iloc[0].to_dict()
    if not row.get("service"):
        row["service"] = ""
    return row

# -----------------------------
# App layout
# -----------------------------
st.title("Vendors Admin")

# DB diagnostics + Secrets debug
with st.expander("Database Status & Schema (debug)", expanded=False):
    info = current_db_info()
    st.write("**DB Target**")
    st.json(info)

    probe = _db_write_probe()
    if probe["status"] != "ok":
        st.error(f"DB write probe failed: {probe['detail']}")
    else:
        st.success("DB write probe: OK (this usually means persistence is configured correctly).")

    status = {}
    try:
        cols = get_columns("vendors")
        status["vendors_columns"] = cols
        status["has_categories_table"] = table_exists("categories")
        status["has_services_table"] = table_exists("services")
        status["has_keywords"] = vendors_has_keywords()
        with engine.begin() as conn:
            cnt = conn.execute(sql_text("SELECT COUNT(1) FROM vendors")).scalar_one()
        status["counts"] = {"vendors": int(cnt)}
    except Exception as e:
        status["error"] = str(e)

    st.write("**Schema**")
    st.json(status)

    try:
        st.write("**Secrets keys (debug)**", sorted(list(st.secrets.keys())))
        if "COLUMN_WIDTHS_PX" in st.secrets:
            st.write("**Raw COLUMN_WIDTHS_PX (debug)**")
            raw = st.secrets.get("COLUMN_WIDTHS_PX", None)
            st.write("type:", type(raw).__name__)
            try:
                st.json(raw)
            except Exception:
                st.code(repr(raw))
        else:
            st.warning("COLUMN_WIDTHS_PX not found in st.secrets")
        st.write("**Loaded column widths (debug)**")
        st.json(_get_column_widths_px())
    except Exception as _e:
        st.write("Secrets debug error:", str(_e))

# Tabs
tab_view, tab_add, tab_edit, tab_delete, tab_cat, tab_svc, tab_maint = st.tabs(
    ["View", "Add", "Edit", "Delete", "Categories Admin", "Services Admin", "Maintenance"]
)

# -----------------------------
# View tab (HTML grid)
# -----------------------------
with tab_view:
    st.subheader("Browse Vendors")
    df = load_vendors_df()

    desired = ["business_name","category","service","contact_name",
               "phone","address","website","notes","keywords"]
    show_cols = [c for c in df.columns if c in desired]
    columns_order = (["id"] + show_cols) if "id" in df.columns else show_cols

    # Reorder the frame to match what we'll show
    df_view = df[columns_order].copy()

    # Load exact px widths from secrets
    widths_px = _get_column_widths_px()

    # Render sortable, wrapping HTML grid
    # Adjust height_px as you like (viewport height; table rows expand freely)
    _render_sortable_wrapped_table(df_view, widths_px, height_px=720)

# -----------------------------
# Add tab — persistent success
# -----------------------------
with tab_add:
    st.subheader("Add Vendor")
    col1, col2, col3 = st.columns(3)
    with col1:
        business_name = st.text_input("Business Name *", key="add_bn")
        contact_name  = st.text_input("Contact Name", key="add_cn")
        phone         = st.text_input("Phone", key="add_ph")
    with col2:
        address       = st.text_input("Address", key="add_addr")
        website       = st.text_input("Website (URL)", key="add_url")
        category      = select_category_required("Category", value=None, key="add_cat")
    with col3:
        service       = select_service_optional("Service (optional)", value=None, key="add_svc")
        notes         = st.text_area("Notes", key="add_notes", height=100)
        keywords_val  = st.text_input("Keywords (comma/space OK)" if vendors_has_keywords() else "Keywords (not in DB)", key="add_kw")

    colA, _ = st.columns([1,1])
    with colA:
        add_feedback = st.empty()
        add_clicked = st.button("Add Vendor", type="primary", key="btn_add_vendor")
        if add_clicked:
            errs = []
            if not business_name.strip():
                errs.append("Business Name is required.")
            if not category.strip():
                errs.append("Category is required.")
            if errs:
                for e in errs:
                    add_feedback.error(e)
            else:
                try:
                    insert_vendor(
                        category=category,
                        service=service,
                        business_name=business_name.strip(),
                        contact_name=contact_name.strip() if contact_name else None,
                        phone=phone.strip() if phone else None,
                        address=address.strip() if address else None,
                        website=website.strip() if website else None,
                        notes=notes.strip() if notes else None,
                        keywords=keywords_val.strip() if vendors_has_keywords() and keywords_val else None,
                    )
                    st.session_state["add_success_msg"] = f"Vendor successfully added: {smart_title(business_name.strip())}"
                    invalidate_caches()
                    rerun()
                except Exception as ex:
                    add_feedback.error(f"Failed to add vendor: {ex}")
        if "add_success_msg" in st.session_state:
            add_feedback.success(st.session_state.pop("add_success_msg"))

# -----------------------------
# Edit tab — persistent success
# -----------------------------
with tab_edit:
    st.subheader("Edit Vendor")
    df_all = load_vendors_df()
    if df_all.empty:
        st.info("No vendors found.")
    else:
        id_to_label = {int(r["id"]): f'{r.get("business_name","(no name)")} — #{int(r["id"])}' for _, r in df_all.iterrows()}
        sort_ids = sorted(id_to_label.keys(), key=lambda i: id_to_label[i].lower())
        chosen_id = st.selectbox("Select Vendor", options=sort_ids, format_func=lambda i: id_to_label[i], key="edit_sel")

        if chosen_id:
            row = load_vendor_by_id(int(chosen_id))
            if not row:
                st.error("Selected vendor not found.")
            else:
                col1, col2, col3 = st.columns(3)
                with col1:
                    business_name_e = text_input_w("Business Name *", row.get("business_name"), key="e_bn")
                    contact_name_e  = text_input_w("Contact Name", row.get("contact_name"), key="e_cn")
                    phone_e         = text_input_w("Phone", row.get("phone"), key="e_ph")
                with col2:
                    address_e       = text_input_w("Address", row.get("address"), key="e_addr")
                    website_e       = text_input_w("Website (URL)", row.get("website"), key="e_url")
                    category_e      = select_category_required("Category", row.get("category"), key="e_cat")
                with col3:
                    service_val = row.get("service") or ""
                    service_e   = select_service_optional("Service (optional)", value=service_val if service_val else None, key="e_svc")
                    notes_e     = st.text_area("Notes", value=row.get("notes") or "", key="e_notes", height=100)
                    if vendors_has_keywords():
                        keywords_e = text_input_w("Keywords (comma/space OK)", row.get("keywords"), key="e_kw")
                    else:
                        keywords_e = None

                save_feedback = st.empty()
                if st.button("Save Changes", type="primary", key="btn_save_vendor"):
                    errs = []
                    if not business_name_e.strip():
                        errs.append("Business Name is required.")
                    if not category_e.strip():
                        errs.append("Category is required.")
                    if errs:
                        for e in errs:
                            save_feedback.error(e)
                    else:
                        try:
                            update_vendor(
                                vid=int(chosen_id),
                                category=category_e,
                                service=service_e,
                                business_name=business_name_e.strip(),
                                contact_name=contact_name_e.strip() if contact_name_e else None,
                                phone=phone_e.strip() if phone_e else None,
                                address=address_e.strip() if address_e else None,
                                website=website_e.strip() if website_e else None,
                                notes=notes_e.strip() if notes_e else None,
                                keywords=keywords_e.strip() if (vendors_has_keywords() and keywords_e) else None,
                            )
                            st.session_state["edit_success_msg"] = "Changes successfully applied."
                            invalidate_caches()
                            rerun()
                        except Exception as ex:
                            save_feedback.error(f"Failed to save changes: {ex}")
                if "edit_success_msg" in st.session_state:
                    save_feedback.success(st.session_state.pop("edit_success_msg"))

# -----------------------------
# Delete tab — persistent success
# -----------------------------
with tab_delete:
    st.subheader("Delete Vendor")
    df_all = load_vendors_df()
    if df_all.empty:
        st.info("No vendors to delete.")
    else:
        id_to_label = {int(r["id"]): f'{r.get("business_name","(no name)")} — #{int(r["id"])}' for _, r in df_all.iterrows()}
        sort_ids = sorted(id_to_label.keys(), key=lambda i: id_to_label[i].lower())
        del_id = st.selectbox("Select Vendor to Delete", options=sort_ids, format_func=lambda i: id_to_label[i], key="del_sel")
        del_feedback = st.empty()
        if del_id and st.button("Delete", type="secondary", key="btn_delete_vendor"):
            try:
                delete_vendor(int(del_id))
                st.session_state["delete_success_msg"] = f"Vendor successfully deleted: {id_to_label[int(del_id)]}"
                invalidate_caches()
                rerun()
            except Exception as ex:
                del_feedback.error(f"Failed to delete vendor: {ex}")
        if "delete_success_msg" in st.session_state:
            del_feedback.success(st.session_state.pop("delete_success_msg"))

# -----------------------------
# Categories Admin (add + orphans)
# -----------------------------
with tab_cat:
    st.subheader("Categories Admin")
    if not table_exists("categories") or "name" not in get_columns("categories"):
        st.info("Table 'categories(name)' not found. You can still assign Category text directly in vendors.")

    new_cat = st.text_input("Add a new Category", help="Creates it in the categories table (smart Title Case applied).")
    if st.button("Add Category", key="btn_add_cat"):
        if not new_cat.strip():
            st.error("Category name required.")
        else:
            with engine.begin() as conn:
                conn.execute(sql_text("CREATE TABLE IF NOT EXISTS categories(name TEXT PRIMARY KEY)"))
                conn.execute(sql_text("INSERT OR IGNORE INTO categories(name) VALUES(:n)"),
                             {"n": smart_title(new_cat.strip())})
            st.success(f"Category added/kept: {smart_title(new_cat.strip())}")
            invalidate_caches()
            rerun()

    st.info(
        "How to edit Categories:\n"
        "1) Pick a category (regular or **[orphan]**).\n"
        "2) See how many vendors use it and preview some names.\n"
        "3) Choose what to do:\n"
        "   • **Reassign Vendors** to another category (existing or new), keeping the old category.\n"
        "   • **Reassign Vendors then Delete Category**.\n"
        "   • **Delete Category (no vendors use it)** — only shown when usage is 0.\n"
        "Notes: Categories marked **[orphan]** exist only on vendor rows (no categories-table entry)."
    )

    st.divider()
    cats_tbl = list_categories_table()
    cats_vendors = list_categories_from_vendors()
    tbl_set = {c for c in cats_tbl}
    orphans = [c for c in cats_vendors if c not in tbl_set]

    options_labels: List[str] = []
    label_to_value: Dict[str, Tuple[str, bool]] = {}
    for c in cats_tbl:
        lbl = c
        options_labels.append(lbl)
        label_to_value[lbl] = (c, False)
    for c in orphans:
        lbl = f"{c} [orphan]"
        options_labels.append(lbl)
        label_to_value[lbl] = (c, True)

    if not options_labels:
        st.write("(no categories yet)")
    else:
        sel_label = st.selectbox(
            "Step 1 — Select category to manage",
            options=sorted(options_labels, key=lambda s: s.lower()),
            key="cat_manage",
            help="Regular categories come from the categories table; [orphan] values come from vendor rows."
        )
        sel_cat, is_orphan = label_to_value[sel_label]

        use_count = count_vendors_with_category(sel_cat)
        preview = list_vendors_by_category(sel_cat, limit=15)
        st.write(f"Step 2 — Impact preview: **{use_count}** vendors currently use **{sel_cat}**{' (orphan)' if is_orphan else ''}.")
        if preview:
            st.caption("First 15 impacted vendors:")
            st.write(", ".join([f"{name} (#{vid})" for vid, name in preview]))

        target_options = ["(choose)"] + cats_tbl
        cols = st.columns([1,1])
        with cols[0]:
            target_sel = st.selectbox(
                "Step 3 — Reassign vendors to (existing category from table)",
                options=target_options,
                key="cat_reassign_sel",
                help="Pick a table-backed category to move these vendors into."
            )
        with cols[1]:
            target_new = st.text_input(
                "...or type a NEW category to create and reassign to",
                key="cat_reassign_new",
                help="If provided, a new category will be created (in the table) and all vendors moved there."
            )

        act = st.empty()
        confirm_reassign = st.checkbox("Confirm: I want to reassign these vendors.", key="cat_confirm_reassign")

        if st.button("Reassign Vendors (keep old category entry if it exists)", key="btn_cat_reassign"):
            if not confirm_reassign:
                act.error("Tick the confirm checkbox above.")
            else:
                target = None
                if target_sel and target_sel != "(choose)":
                    target = target_sel
                elif target_new.strip():
                    target = smart_title(target_new.strip())
                if not target:
                    act.error("Pick an existing target or type a new one.")
                else:
                    try:
                        with engine.begin() as conn:
                            conn.execute(sql_text("CREATE TABLE IF NOT EXISTS categories(name TEXT PRIMARY KEY)"))
                            conn.execute(sql_text("INSERT OR IGNORE INTO categories(name) VALUES(:n)"), {"n": target})
                            conn.execute(sql_text(
                                "UPDATE vendors SET category=:tgt WHERE lower(trim(category)) = lower(trim(:old))"
                            ), {"tgt": target, "old": sel_cat})
                        st.success(f"Reassigned vendors from '{sel_cat}' to '{target}'.")
                        invalidate_caches()
                        rerun()
                    except Exception as ex:
                        act.error(f"Reassign failed: {ex}")

        confirm_reassign_delete = st.checkbox(
            "Confirm: Reassign vendors and then delete the old category entry (if present).",
            key="cat_confirm_reassign_delete"
        )
        if st.button("Reassign Vendors then Delete Category", key="btn_cat_reassign_delete"):
            if not confirm_reassign_delete:
                act.error("Tick the confirm checkbox above.")
            else:
                target = None
                if target_sel and target_sel != "(choose)":
                    target = target_sel
                elif target_new.strip():
                    target = smart_title(target_new.strip())
                if not target:
                    act.error("Pick an existing target or type a new one.")
                else:
                    try:
                        with engine.begin() as conn:
                            conn.execute(sql_text("CREATE TABLE IF NOT EXISTS categories(name TEXT PRIMARY KEY)"))
                            conn.execute(sql_text("INSERT OR IGNORE INTO categories(name) VALUES(:n)"), {"n": target})
                            conn.execute(sql_text(
                                "UPDATE vendors SET category=:tgt WHERE lower(trim(category)) = lower(trim(:old))"
                            ), {"tgt": target, "old": sel_cat})
                            conn.execute(sql_text("DELETE FROM categories WHERE lower(name)=lower(:n)"), {"n": sel_cat})
                        st.success(f"Reassigned to '{target}' and deleted category entry for '{sel_cat}' (if it existed).")
                        invalidate_caches()
                        rerun()
                    except Exception as ex:
                        act.error(f"Reassign+Delete failed: {ex}")

        if use_count == 0:
            confirm_delete_only = st.checkbox(
                "Confirm: Delete this unused category entry (if present).",
                key="cat_confirm_delete_only"
            )
            if st.button("Delete Category (no vendors use it)", key="cat_delete_only"):
                if not confirm_delete_only:
                    act.error("Tick the confirm checkbox above.")
                else:
                    try:
                        with engine.begin() as conn:
                            conn.execute(sql_text("DELETE FROM categories WHERE lower(name)=lower(:n)"), {"n": sel_cat})
                        st.success(f"Deleted category entry for '{sel_cat}' (if it existed).")
                        invalidate_caches()
                        rerun()
                    except Exception as ex:
                        act.error(f"Delete failed: {ex}")
        else:
            st.caption("Delete button appears only when usage is 0.")

# -----------------------------
# Services Admin (add + orphans)
# -----------------------------
with tab_svc:
    st.subheader("Services Admin")
    if not table_exists("services") or "name" not in get_columns("services"):
        st.info("Table 'services(name)' not found. You can still leave Service blank for vendors, or type free-form in vendors.service if you later add the column.")

    new_svc = st.text_input("Add a new Service", help="Creates it in the services table (smart Title Case applied).")
    if st.button("Add Service", key="btn_add_svc"):
        if not new_svc.strip():
            st.error("Service name required.")
        else:
            with engine.begin() as conn:
                conn.execute(sql_text("CREATE TABLE IF NOT EXISTS services(name TEXT PRIMARY KEY)"))
                conn.execute(sql_text("INSERT OR IGNORE INTO services(name) VALUES(:n)"),
                             {"n": smart_title(new_svc.strip())})
            st.success(f"Service added/kept: {smart_title(new_svc.strip())}")
            invalidate_caches()
            rerun()

    st.info(
        "How to edit Services:\n"
        "1) Pick a service (regular or **[orphan]**).\n"
        "2) See how many vendors use it and preview some names.\n"
        "3) Choose what to do:\n"
        "   • **Reassign Vendors (Service)** to another service (existing or new).\n"
        "   • **Clear Service to blank** (Service is optional).\n"
        "   • **Apply (Reassign or Clear) then Delete Service**.\n"
        "   • **Delete Service (no vendors use it)** — only shown when usage is 0.\n"
        "Notes: Services marked **[orphan]** exist only on vendor rows (no services-table entry)."
    )

    st.divider()
    svcs_tbl = list_services_table()
    svcs_vendors = list_services_from_vendors()
    tbl_set_s = {s for s in svcs_tbl}
    orphans_s = [s for s in svcs_vendors if s not in tbl_set_s]

    options_labels_s: List[str] = []
    label_to_value_s: Dict[str, Tuple[str, bool]] = {}
    for s in svcs_tbl:
        lbl = s
        options_labels_s.append(lbl)
        label_to_value_s[lbl] = (s, False)
    for s in orphans_s:
        lbl = f"{s} [orphan]"
        options_labels_s.append(lbl)
        label_to_value_s[lbl] = (s, True)

    if not options_labels_s:
        st.write("(no services yet)")
    else:
        sel_label_s = st.selectbox(
            "Step 1 — Select service to manage",
            options=sorted(options_labels_s, key=lambda s: s.lower()),
            key="svc_manage",
            help="Regular services come from the services table; [orphan] values come from vendor rows."
        )
        sel_svc, svc_is_orphan = label_to_value_s[sel_label_s]

        use_count = count_vendors_with_service(sel_svc)
        preview = list_vendors_by_service(sel_svc, limit=15)
        st.write(f"Step 2 — Impact preview: **{use_count}** vendors currently use **{sel_svc}**{' (orphan)' if svc_is_orphan else ''}.")
        if preview:
            st.caption("First 15 impacted vendors:")
            st.write(", ".join([f"{name} (#{vid})" for vid, name in preview]))

        cols = st.columns([1,1,1])
        with cols[0]:
            target_options = ["(choose)"] + svcs_tbl
            target_sel = st.selectbox(
                "Step 3a — Reassign vendors to (existing service from table)",
                options=target_options,
                key="svc_reassign_sel",
                help="Pick an existing service for reassignment (optional if you choose 'clear to blank')."
            )
        with cols[1]:
            target_new = st.text_input(
                "Step 3b — ...or type a NEW service",
                key="svc_reassign_new",
                help="If provided, a new service will be created and vendors reassigned to it."
            )
        with cols[2]:
            clear_to_blank = st.checkbox(
                "Or clear to blank",
                value=False,
                key="svc_clear_blank",
                help="Sets service to blank (NULL) for all vendors using the selected service."
            )

        act = st.empty()
        confirm_apply = st.checkbox("Confirm: Apply the change (reassign or clear).", key="svc_confirm_apply")

        if st.button("Reassign Vendors (Service) / Clear Only", key="btn_svc_reassign"):
            if not confirm_apply:
                act.error("Tick the confirm checkbox above.")
            else:
                try:
                    with engine.begin() as conn:
                        if clear_to_blank:
                            conn.execute(sql_text(
                                "UPDATE vendors SET service=NULL WHERE lower(trim(service)) = lower(trim(:old))"
                            ), {"old": sel_svc})
                        else:
                            target = None
                            if target_sel and target_sel != "(choose)":
                                target = target_sel
                            elif target_new.strip():
                                target = smart_title(target_new.strip())
                            if not target:
                                raise ValueError("Pick a target service or check 'clear to blank'.")
                            conn.execute(sql_text("CREATE TABLE IF NOT EXISTS services(name TEXT PRIMARY KEY)"))
                            conn.execute(sql_text("INSERT OR IGNORE INTO services(name) VALUES(:n)"), {"n": target})
                            conn.execute(sql_text(
                                "UPDATE vendors SET service=:tgt WHERE lower(trim(service)) = lower(trim(:old))"
                            ), {"tgt": target, "old": sel_svc})
                    st.success("Applied change.")
                    invalidate_caches()
                    rerun()
                except Exception as ex:
                    act.error(f"Change failed: {ex}")

        confirm_apply_delete = st.checkbox("Confirm: Apply change and then delete the old service entry (if present).", key="svc_confirm_apply_delete")
        if st.button("Apply (Reassign or Clear) then Delete Service", key="btn_svc_apply_delete"):
            if not confirm_apply_delete:
                act.error("Tick the confirm checkbox above.")
            else:
                try:
                    with engine.begin() as conn:
                        if clear_to_blank:
                            conn.execute(sql_text(
                                "UPDATE vendors SET service=NULL WHERE lower(trim(service)) = lower(trim(:old))"
                            ), {"old": sel_svc})
                        else:
                            target = None
                            if target_sel and target_sel != "(choose)":
                                target = target_sel
                            elif target_new.strip():
                                target = smart_title(target_new.strip())
                            if not target:
                                raise ValueError("Pick a target service or check 'clear to blank'.")
                            conn.execute(sql_text("CREATE TABLE IF NOT EXISTS services(name TEXT PRIMARY KEY)"))
                            conn.execute(sql_text("INSERT OR IGNORE INTO services(name) VALUES(:n)"), {"n": target})
                            conn.execute(sql_text(
                                "UPDATE vendors SET service=:tgt WHERE lower(trim(service)) = lower(trim(:old))"
                            ), {"tgt": target, "old": sel_svc})
                        conn.execute(sql_text("DELETE FROM services WHERE lower(name)=lower(:n)"), {"n": sel_svc})
                    st.success(f"Applied change and deleted service entry for '{sel_svc}' (if it existed).")
                    invalidate_caches()
                    rerun()
                except Exception as ex:
                    act.error(f"Apply+Delete failed: {ex}")

        if use_count == 0:
            confirm_delete_only = st.checkbox("Confirm: Delete this unused service entry (if present).", key="svc_confirm_delete_only")
            if st.button("Delete Service (no vendors use it)", key="svc_delete_only"):
                if not confirm_delete_only:
                    act.error("Tick the confirm checkbox above.")
                else:
                    try:
                        with engine.begin() as conn:
                            conn.execute(sql_text("DELETE FROM services WHERE lower(name)=lower(:n)"), {"n": sel_svc})
                        st.success(f"Deleted service entry for '{sel_svc}' (if it existed).")
                        invalidate_caches()
                        rerun()
                    except Exception as ex:
                        act.error(f"Delete failed: {ex}")
        else:
            st.caption("Delete button appears only when usage is 0.")

# -----------------------------
# Maintenance — normalize existing rows
# -----------------------------
with tab_maint:
    st.subheader("Maintenance")
    st.write("Normalize ALL vendors to Title Case **and trim whitespace** (business_name, contact_name, category, service, address).")
    if st.button("Normalize existing records now", type="primary", key="btn_norm_all"):
        try:
            df_all = load_vendors_df()
            if df_all.empty:
                st.info("No vendors to normalize.")
            else:
                updated = 0
                for _, r in df_all.iterrows():
                    vid = int(r["id"])
                    bn  = smart_title(r.get("business_name") or "")
                    cn  = smart_title(r.get("contact_name") or None) if r.get("contact_name") else None
                    cat = smart_title((r.get("category") or "").strip())
                    svc_raw = r.get("service")
                    svc = smart_title((svc_raw or "").strip()) if (svc_raw and str(svc_raw).strip()) else None
                    addr= title_address((r.get("address") or "").strip()) if r.get("address") else None
                    web = normalize_url((r.get("website") or "").strip()) if r.get("website") else None
                    ph  = normalize_phone((r.get("phone") or "").strip()) if r.get("phone") else None
                    notes = (r.get("notes") or None)
                    kw    = r.get("keywords") if "keywords" in r and pd.notna(r.get("keywords")) else None

                    if any([
                        bn != r.get("business_name"),
                        (cn or "") != (r.get("contact_name") or ""),
                        cat != (r.get("category") or ""),
                        (svc or "") != (r.get("service") or ""),
                        (addr or "") != (r.get("address") or ""),
                        (web or "") != (r.get("website") or ""),
                        (ph or "") != (r.get("phone") or ""),
                    ]):
                        update_vendor(
                            vid=vid,
                            category=cat,
                            service=svc,
                            business_name=bn,
                            contact_name=cn,
                            phone=ph,
                            address=addr,
                            website=web,
                            notes=notes,
                            keywords=kw,
                        )
                        updated += 1
                st.success(f"Normalization complete. Rows updated: {updated}.")
                invalidate_caches()
                rerun()
        except Exception as ex:
            st.error(f"Normalization failed: {ex}")
