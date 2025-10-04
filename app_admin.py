# app_admin.py — Vendors Admin (v3.6.9)
# View | Add | Edit | Delete | Categories Admin | Services Admin | Maintenance | Changelog
# - AgGrid optional: wrap/auto-height for ALL columns EXCEPT notes & keywords (fixed height)
# - Website column: shows literal "Website", underlined/pointer; click opens in new tab
#   (guarded: does not open while editing; ignores non-click events)
# - Copy UX: keyboard selection + context menu (“Copy”, “Copy with headers”, “Copy row (TSV)”)
# - Disabled full-row hover highlight and row selection on click (so only cells/ranges highlight)
# - Fallback uses st.table; CSS: wrap everywhere except notes/keywords; fixed widths
# - Validators, audit trail, schema guardrails; CSV + Debug under the table in View

from __future__ import annotations

import os
import re
from typing import Dict, List, Tuple, Optional

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine
from sqlalchemy.sql import text as sql_text
from urllib.parse import urlparse
from datetime import datetime, timezone

# ---- AgGrid (optional; app falls back if not installed) ----
try:
    from st_aggrid import AgGrid, GridOptionsBuilder, JsCode, GridUpdateMode, ColumnsAutoSizeMode
    _AGGRID_AVAILABLE = True
except Exception:
    _AGGRID_AVAILABLE = False


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
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


# -----------------------------
# Page config / layout
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
        f"<style>.block-container {{ max-width: {maxw}px; }}</style>",
        unsafe_allow_html=True,
    )

_apply_layout()


# -----------------------------
# Config from secrets
# -----------------------------
LABEL_OVERRIDES: Dict[str, str] = _read_secret("READONLY_COLUMN_LABELS", {}) or {}
COLUMN_WIDTHS: Dict[str, int]   = _read_secret("COLUMN_WIDTHS_PX_READONLY", {}) or {}
STICKY_FIRST: bool              = _get_bool(_read_secret("READONLY_STICKY_FIRST_COL", False), False)

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
    "id","category","service","business_name","contact_name",
    "phone","address","website","notes","keywords",
]
AUDIT_COLS = ["created_at","updated_at","updated_by"]


# -----------------------------
# Engine (Turso/libSQL with fallback)
# -----------------------------
def _engine():
    url = _read_secret("TURSO_DATABASE_URL", "sqlite+libsql://vendors-prod-mrbookend.aws-us-west-2.turso.io?secure=true")
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
# Schema migration / guardrails
# -----------------------------
def _column_exists(conn, table: str, col: str) -> bool:
    try:
        res = conn.execute(sql_text(f"PRAGMA table_info({table})")).fetchall()
        cols = [r[1] for r in res]
        return col in cols
    except Exception:
        return False

def _table_exists(conn, table: str) -> bool:
    try:
        res = conn.execute(sql_text(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=:t"
        ), {"t": table}).fetchone()
        return bool(res)
    except Exception:
        return False

def _ensure_ref_tables(conn):
    conn.execute(sql_text("""
        CREATE TABLE IF NOT EXISTS categories (
            id   INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL
        )
    """))
    conn.execute(sql_text("""
        CREATE TABLE IF NOT EXISTS services (
            id   INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL
        )
    """))

def _ensure_schema():
    with eng.begin() as c:
        _ensure_ref_tables(c)
        if _table_exists(c, "vendors"):
            if not _column_exists(c, "vendors", "created_at"):
                c.execute(sql_text("ALTER TABLE vendors ADD COLUMN created_at TEXT"))
            if not _column_exists(c, "vendors", "updated_at"):
                c.execute(sql_text("ALTER TABLE vendors ADD COLUMN updated_at TEXT"))
            if not _column_exists(c, "vendors", "updated_by"):
                c.execute(sql_text("ALTER TABLE vendors ADD COLUMN updated_by TEXT"))
        c.execute(sql_text("""
            CREATE TABLE IF NOT EXISTS vendor_changes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                vendor_id INTEGER NOT NULL,
                changed_at TEXT NOT NULL,
                changed_by TEXT NOT NULL,
                action TEXT NOT NULL,   -- 'insert'|'update'|'delete'
                field TEXT,
                old_value TEXT,
                new_value TEXT
            )
        """))

_ensure_schema()


# -----------------------------
# CSS for table-based fallback (and some shared rules)
# -----------------------------
def _apply_css_for_table(field_order: List[str]):
    """CSS that affects st.table AND legacy st.dataframe; wrap everywhere EXCEPT notes/keywords."""
    tbl_sel = "div[data-testid='stTable'] table"
    df_sel  = "div[data-testid='stDataFrame'] table"
    base = f"""
    {tbl_sel}, {df_sel} {{ table-layout: fixed !important; }}
    {tbl_sel} td, {tbl_sel} th, {df_sel} td, {df_sel} th {{
        white-space: normal !important;        /* allow wrapping */
        overflow-wrap: normal !important;      /* wrap at word boundaries */
        word-break: normal !important;         /* no mid-word breaks */
        hyphens: auto !important;              /* optional hyphenation for long words */
    }}
    """

    rules = [base]

    # Fixed per-column widths (1-based nth-child)
    for idx, col in enumerate(field_order, start=1):
        px = COLUMN_WIDTHS.get(col)
        if not px:
            continue
        rules.append(f"""
        {tbl_sel} thead tr th:nth-child({idx}), {tbl_sel} tbody tr td:nth-child({idx}),
        {df_sel}  thead tr th:nth-child({idx}), {df_sel}  tbody tr td:nth-child({idx}) {{
            min-width:{px}px !important; max-width:{px}px !important; width:{px}px !important;
        }}
        """)

    # Turn OFF wrapping for notes & keywords
    def _nth_for(colname: str) -> Optional[int]:
        try:
            return field_order.index(colname) + 1
        except ValueError:
            return None

    for colname in ["notes", "keywords"]:
        nth = _nth_for(LABEL_OVERRIDES.get(colname, colname)) or _nth_for(colname)
        if nth:
            rules.append(f"""
            {tbl_sel} thead tr th:nth-child({nth}), {tbl_sel} tbody tr td:nth-child({nth}),
            {df_sel}  thead tr th:nth-child({nth}), {df_sel}  tbody tr td:nth-child({nth}) {{
                white-space: nowrap !important;     /* no wrap */
                overflow-wrap: normal !important;
                word-break: normal !important;
            }}
            """)

    sticky = ""
    if STICKY_FIRST and field_order:
        sticky = f"""
        {tbl_sel} thead tr th:nth-child(1), {tbl_sel} tbody tr td:nth-child(1),
        {df_sel}  thead tr th:nth-child(1), {df_sel}  tbody tr td:nth-child(1) {{
            position: sticky; left: 0; z-index: 2;
            background: var(--background-color, white);
            box-shadow: 1px 0 0 rgba(0,0,0,0.06);
        }}
        {tbl_sel} thead tr th:nth-child(1), {df_sel} thead tr th:nth-child(1) {{ z-index: 3; }}
        """

    st.markdown("<style>" + "\n".join(rules) + sticky + "</style>", unsafe_allow_html=True)


# -----------------------------
# URL / phone validators
# -----------------------------
def _normalize_url(u: str) -> str:
    s = (u or "").strip()
    if not s:
        return ""
    if not s.lower().startswith(("http://","https://")):
        s = "http://" + s
    try:
        p = urlparse(s)
        if p.scheme not in ("http","https"): return ""
        if not p.netloc or "." not in p.netloc: return ""
        return s
    except Exception:
        return ""

def _digits_only(s: str) -> str:
    return re.sub(r"[^0-9]", "", s or "")

def _format_phone_10(digits: str) -> str:
    d = _digits_only(digits)
    if len(d) != 10: return ""
    return f"({d[0:3]}) {d[3:6]}-{d[6:10]}"

def _validate_vendor_fields(business_name: str, category: str, service: str, phone: str, website: str) -> Tuple[bool, Dict[str,str], Dict[str,str]]:
    errors: Dict[str,str] = {}
    cleaned: Dict[str,str] = {}

    if not (business_name or "").strip():
        errors["business_name"] = "Business name is required."
    else:
        cleaned["business_name"] = business_name.strip()

    if not (category or "").strip():
        errors["category"] = "Category is required."
    else:
        cleaned["category"] = category.strip()

    if (category or "").strip().casefold() != "home repair":
        if not (service or "").strip():
            errors["service"] = "Service is required unless Category is 'Home Repair'."
        else:
            cleaned["service"] = service.strip()
    else:
        cleaned["service"] = (service or "").strip()

    fmt_phone = _format_phone_10(phone)
    if not fmt_phone:
        errors["phone"] = "Phone must have exactly 10 digits."
    else:
        cleaned["phone"] = fmt_phone

    web_norm = _normalize_url(website)
    if (website or "").strip() and not web_norm:
        errors["website"] = "Website must start with http/https and include a valid host (e.g., example.com)."
    else:
        cleaned["website"] = web_norm  # may be ""

    return (len(errors) == 0), cleaned, errors


# -----------------------------
# Help + Quick filter
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

def _quick_filter_ui():
    return st.text_input("Quick filter (type words or parts of words):", "", placeholder="e.g., plumber roof 78240")

def _apply_quick_filter(df: pd.DataFrame, query: str) -> pd.DataFrame:
    q = (query or "").strip().lower()
    if not q:
        return df
    terms = [t for t in q.split() if t]
    if not terms:
        return df
    cols = [c for c in df.columns if c.lower() in {
        "business_name","provider","category","service","contact_name",
        "phone","address","website","notes","keywords","website url"
    }]
    if not cols:
        return df
    blob = df[cols].fillna("").astype(str).agg(" ".join, axis=1).str.lower()
    mask = pd.Series(True, index=df.index)
    for t in terms:
        mask &= blob.str.contains(t, na=False)
    return df[mask]


# -----------------------------
# Data access + audit helpers
# -----------------------------
def _fetch_df() -> pd.DataFrame:
    with eng.connect() as c:
        df = pd.read_sql_query(sql_text("SELECT * FROM vendors"), c)
    for col in RAW_COLS + AUDIT_COLS:
        if col not in df.columns:
            df[col] = ""
    return df[RAW_COLS + AUDIT_COLS]

def _cats() -> pd.DataFrame:
    with eng.connect() as c:
        try:
            return pd.read_sql_query(sql_text("SELECT id, name FROM categories ORDER BY name COLLATE NOCASE"), c)
        except Exception:
            return pd.DataFrame(columns=["id","name"]).astype({"id":int,"name":str})

def _svcs() -> pd.DataFrame:
    with eng.connect() as c:
        try:
            return pd.read_sql_query(sql_text("SELECT id, name FROM services ORDER BY name COLLATE NOCASE"), c)
        except Exception:
            return pd.DataFrame(columns=["id","name"]).astype({"id":int,"name":str})

def _insert_vendor(row: Dict[str,str]):
    now = _now_iso()
    row2 = dict(row, created_at=now, updated_at=now, updated_by=ADMIN_DISPLAY_NAME)
    q = sql_text("""
        INSERT INTO vendors(category, service, business_name, contact_name, phone, address, website, notes, keywords, created_at, updated_at, updated_by)
        VALUES(:category,:service,:business_name,:contact_name,:phone,:address,:website,:notes,:keywords,:created_at,:updated_at,:updated_by)
    """)
    with eng.begin() as c:
        c.execute(q, row2)
        vid = c.execute(sql_text("SELECT last_insert_rowid()")).scalar()
        c.execute(sql_text("""
            INSERT INTO vendor_changes(vendor_id, changed_at, changed_by, action, field, old_value, new_value)
            VALUES(:vid, :ts, :by, 'insert', NULL, NULL, NULL)
        """), {"vid": vid, "ts": now, "by": ADMIN_DISPLAY_NAME})

def _update_vendor(vid: int, new_row: Dict[str,str]):
    with eng.begin() as c:
        cur = c.execute(sql_text("SELECT * FROM vendors WHERE id = :id"), {"id": vid}).mappings().first()
        if not cur:
            return
        now = _now_iso()
        upd = dict(new_row, updated_at=now, updated_by=ADMIN_DISPLAY_NAME)
        sets = ", ".join([f"{k} = :{k}" for k in upd.keys()])
        c.execute(sql_text(f"UPDATE vendors SET {sets} WHERE id = :id"), {**upd, "id": vid})
        for k, newv in new_row.items():
            oldv = cur.get(k, "")
            if (oldv or "") != (newv or ""):
                c.execute(sql_text("""
                    INSERT INTO vendor_changes(vendor_id, changed_at, changed_by, action, field, old_value, new_value)
                    VALUES(:vid, :ts, :by, 'update', :field, :oldv, :newv)
                """), {"vid": vid, "ts": now, "by": ADMIN_DISPLAY_NAME, "field": k,
                       "oldv": str(oldv) if oldv is not None else "", "newv": str(newv) if newv is not None else ""})

def _delete_vendor(vid: int):
    now = _now_iso()
    with eng.begin() as c:
        c.execute(sql_text("""
            INSERT INTO vendor_changes(vendor_id, changed_at, changed_by, action, field, old_value, new_value)
            VALUES(:vid, :ts, :by, 'delete', NULL, NULL, NULL)
        """), {"vid": vid, "ts": now, "by": ADMIN_DISPLAY_NAME})
        c.execute(sql_text("DELETE FROM vendors WHERE id = :id"), {"id": vid})


# -----------------------------
# Label mapping (display only)
# -----------------------------
def _apply_labels(df: pd.DataFrame) -> pd.DataFrame:
    if not LABEL_OVERRIDES:
        return df
    mapping = {k:v for k,v in LABEL_OVERRIDES.items() if k in df.columns and isinstance(v,str) and v}
    return df.rename(columns=mapping)


# -----------------------------
# AgGrid renderer (wrap all but notes/keywords; clickable website; copy)
# -----------------------------
def _aggrid_view(df_show: pd.DataFrame, website_label: str = "website"):
    if df_show.empty:
        st.info("No rows to display.")
        return

    website_key = website_label if website_label in df_show.columns else next((c for c in df_show.columns if c.lower() == "website"), None)

    _df = df_show.copy()

    # Insert normalized URL helper column immediately after the website column
    url_col = None
    if website_key:
        norm = _df[website_key].map(_normalize_url) if website_key in _df.columns else ""
        url_col = f"{website_key} URL" if website_key.lower() != "website" else "Website URL"
        widx = _df.columns.get_loc(website_key)
        _df.insert(widx + 1, url_col, norm)

    gob = GridOptionsBuilder.from_dataframe(_df)

    # Defaults: enable wrapping/autoHeight everywhere (we'll turn OFF for notes/keywords)
    gob.configure_default_column(
        resizable=True,
        sortable=True,
        filter=False,         # no per-column filters
        wrapText=True,
        autoHeight=True,
    )

    # Fixed widths via secrets; map displayed->raw if renamed
    display_to_raw = {disp: raw for raw, disp in LABEL_OVERRIDES.items() if isinstance(disp, str) and disp}
    for col in _df.columns:
        raw_key = display_to_raw.get(col, col)
        px = COLUMN_WIDTHS.get(raw_key)
        if px:
            gob.configure_column(col, width=px)

    # Turn OFF wrap/autoHeight only for notes & keywords (single-line, ellipsis)
    for col in _df.columns:
        low = col.lower()
        raw_match = low in {"notes", "keywords"}
        renamed_match = any((k in {"notes", "keywords"}) and (LABEL_OVERRIDES.get(k, k) == col) for k in ["notes", "keywords"])
        if raw_match or renamed_match:
            gob.configure_column(
                col,
                wrapText=False,
                autoHeight=False,
                cellStyle={"whiteSpace": "nowrap", "textOverflow": "ellipsis", "overflow": "hidden"},
            )

    # Make Website column clickable (return HTML STRING, not DOM node — prevents React invariant #31)
    if website_key and url_col:
        link_renderer = JsCode(f"""
            function(params) {{
                const url = params.data && params.data["{url_col}"] ? params.data["{url_col}"] : "";
                if (!url) return "";
                // HTML string only; inline stopPropagation so selection/copy still works
                return '<a href="' + url + '" target="_blank" rel="noopener noreferrer" onclick="event.stopPropagation();" '
                       + 'style="text-decoration:underline; cursor:pointer;">Website</a>';
            }}
        """)
        # Provide a formatted label so clipboard picks "Website" when copying formatted value
        label_formatter = JsCode("""
            function(params) {
                if (params && params.data) {
                    // show the literal label if a valid URL exists
                    const url = params.data[params.colDef.urlField];
                    return (url && typeof url === 'string' && url.length > 0) ? 'Website' : '';
                }
                return '';
            }
        """)
        gob.configure_column(
            website_key,
            cellRenderer=link_renderer,
            valueFormatter=label_formatter,
        )

    # Remove filter/menu on 'id' column header (sorting stays)
    if "id" in _df.columns:
        gob.configure_column("id", filter=False, suppressMenu=True)

    grid_options = gob.build()

    # Whole-word wrapping everywhere by default (notes/keywords already overridden above)
    grid_options.setdefault("defaultColDef", {})
    grid_options["defaultColDef"]["cellStyle"] = {
        "whiteSpace": "normal",     # allow wrapping
        "overflowWrap": "normal",   # wrap at word boundaries
        "wordBreak": "normal",      # no mid-word breaks
        "hyphens": "auto",          # optional hyphenation for very long tokens
    }

    # Tell Website formatter which hidden field holds the URL
    if website_key and url_col:
        # attach a custom key on the columnDef so JS formatter can find it
        for coldef in grid_options.get("columnDefs", []):
            if coldef.get("field") == website_key:
                coldef["urlField"] = url_col
                break

    grid_options["floatingFilter"] = False
    grid_options["suppressMenuHide"] = True
    grid_options["domLayout"] = "normal"

    # --- Live column width HUD (pinned top row while resizing) ---
    grid_options["pinnedTopRowData"] = []
    grid_options["pinnedTopRowHeight"] = 22  # slim HUD row

    st.markdown("""
    <style>
      .ag-theme-streamlit .ag-pinned-top .ag-row {
        background: rgba(0,0,0,0.04) !important;
        font-size: 11px !important;
        line-height: 18px !important;
      }
      .ag-theme-streamlit .ag-pinned-top .ag-cell {
        padding-top: 0 !important;
        padding-bottom: 0 !important;
        text-align: center;
        color: rgba(0,0,0,0.75);
        user-select: none;
      }
    </style>
    """, unsafe_allow_html=True)

    grid_options.setdefault("context", {})
    grid_options["context"]["_widthHudMs"] = 700  # keep HUD this long after release

    grid_options["onGridReady"] = JsCode("""
        function(params){
          const api = params.api, colApi = params.columnApi;
          const go = api.gridOptionsWrapper.gridOptions;
          go._buildWidthHudRow = function(){
            const row = {};
            const cols = colApi.getAllDisplayedColumns();
            if (cols && cols.forEach){
              cols.forEach(c => {
                const id = c.getColId();
                const w  = Math.round(c.getActualWidth());
                row[id]  = w ? (w + 'px') : '';
              });
            }
            return row;
          };
        }
    """)

    grid_options["onColumnResized"] = JsCode("""
        function(event){
            if (!event || !event.api || !event.column) return;
            const api = event.api;
            const go  = api.gridOptionsWrapper.gridOptions;
            const build = (go && go._buildWidthHudRow) ? go._buildWidthHudRow : function(){ return {}; };
            try {
                const row = build();
                api.setPinnedTopRowData([row]);
            } catch(e) {}
            if (event.finished) {
                const delay = (go && go.context && go.context._widthHudMs) ? go.context._widthHudMs : 700;
                setTimeout(function(){ try { api.setPinnedTopRowData([]); } catch(e) {} }, delay);
            }
        }
    """)

    # Improve copy behavior: copy exactly the selected cell(s)/range; prefer formatted value
    grid_options["ensureDomOrder"] = True
    grid_options["enableRangeSelection"] = True           # drag to select ranges
    grid_options["enableCellTextSelection"] = True        # single-cell text selection
    grid_options["suppressCopySingleCellRanges"] = False  # allow copying a single focused cell

    # Avoid whole-row selection/hover highlight
    grid_options["rowSelection"] = "single"
    grid_options["rowMultiSelectWithClick"] = False
    grid_options["suppressRowClickSelection"] = True
    grid_options["suppressRowHoverHighlight"] = True

    # Clipboard defaults: TSV; no headers; formatted value if available
    grid_options["suppressCopyRowsToClipboard"] = False
    grid_options["clipboardDelimiter"] = "\t"
    grid_options["copyHeadersToClipboard"] = False
    grid_options["processCellForClipboard"] = JsCode("""
        function(params) {
          return (params && params.valueFormatted != null) ? String(params.valueFormatted) :
                 (params && params.value != null) ? String(params.value) : "";
        }
    """)

    # Context menu: default copy + copy row (TSV) + copy Website URL + copy selection as CSV
    grid_options["getContextMenuItems"] = JsCode(f"""
        function(params) {{
          const res = ['copy', 'copyWithHeaders', 'paste'];
          const node = params.node;
          const cols = params.columnApi.getAllDisplayedColumns();

          // Copy row (TSV)
          if (node && node.data) {{
            res.push({{
              name: 'Copy row (TSV)',
              action: () => {{
                const vals = [];
                cols.forEach(c => {{
                  const id = c.getColId();
                  if (id === '__copy__') return;
                  const v = node.data[id] != null ? String(node.data[id]) : '';
                  vals.push(v);
                }});
                const txt = vals.join('\\t');
                if (navigator.clipboard && navigator.clipboard.writeText) {{
                  navigator.clipboard.writeText(txt);
                }} else {{
                  const ta = document.createElement('textarea');
                  ta.value = txt; document.body.appendChild(ta);
                  ta.select(); document.execCommand('copy'); document.body.removeChild(ta);
                }}
              }}
            }});
          }}

          // Copy Website URL (only if right-clicking Website column and URL exists)
          try {{
            const websiteField = {json.dumps(website_key) if website_key else 'null'};
            const urlField = {json.dumps(url_col) if url_col else 'null'};
            if (websiteField && urlField && params && params.column && params.node && params.node.data) {{
              const colId = params.column.getColId();
              if (colId === websiteField) {{
                const url = params.node.data[urlField] || "";
                if (url) {{
                  res.push({{
                    name: 'Copy Website URL',
                    action: () => {{
                      if (navigator.clipboard && navigator.clipboard.writeText) {{
                        navigator.clipboard.writeText(url);
                      }} else {{
                        const ta = document.createElement('textarea');
                        ta.value = url; document.body.appendChild(ta);
                        ta.select(); document.execCommand('copy'); document.body.removeChild(ta);
                      }}
                    }}
                  }});
                }}
              }}
            }}
          }} catch(e) {{}}

          // Copy selection as CSV
          res.push({{
            name: 'Copy selection as CSV',
            action: () => {{
              const csv = params.api.getDataAsCsv({{ suppressQuotes: true, onlySelected: true }});
              if (csv) {{
                if (navigator.clipboard && navigator.clipboard.writeText) {{
                  navigator.clipboard.writeText(csv);
                }} else {{
                  const ta = document.createElement('textarea');
                  ta.value = csv; document.body.appendChild(ta);
                  ta.select(); document.execCommand('copy'); document.body.removeChild(ta);
                }}
              }}
            }}
          }});

          return res;
        }}
    """)

    AgGrid(
        _df,
        gridOptions=grid_options,
        update_mode=GridUpdateMode.NO_UPDATE,
        fit_columns_on_grid_load=False,
        allow_unsafe_jscode=True,
        enable_enterprise_modules=False,
        columns_auto_size_mode=ColumnsAutoSizeMode.NO_AUTOSIZE,
        height=600,
        theme="streamlit",
    )


# -----------------------------
# VIEW tab (quick filter; CSV; DEBUG right under CSV)
# -----------------------------
def tab_view(query: str):
    df = _fetch_df()
    df_show = _apply_labels(df)
    df_show = _apply_quick_filter(df_show, query)

    if _AGGRID_AVAILABLE:
        _aggrid_view(df_show, website_label=LABEL_OVERRIDES.get("website", "website"))
    else:
        st.warning("`streamlit-aggrid` not installed — basic table fallback. Links will not be clickable.")
        _apply_css_for_table(df_show.columns.tolist())
        df_sorted = df_show.sort_values(
            by=["business_name","category","id"],
            key=lambda s: s.astype(str).str.casefold(),
            ignore_index=True
        )
        st.table(df_sorted)

    st.download_button(
        label="Download Providers (CSV)",
        data=df_show.to_csv(index=False).encode("utf-8"),
        file_name="providers_admin_view.csv",
        mime="text/csv",
    )

    render_status_debug(expanded=False)


# -----------------------------
# ADD / EDIT / DELETE / Admin tabs
# -----------------------------
def tab_add():
    cats = sorted(_cats()["name"].dropna().astype(str).tolist(), key=lambda s: s.casefold())
    svcs = sorted(_svcs()["name"].dropna().astype(str).tolist(), key=lambda s: s.casefold())

    st.info("Fields in **bold** are required. Service is optional only when Category is 'Home Repair'.")
    with st.form("add_vendor_form", clear_on_submit=True):
        col = st.columns(2)
        with col[0]:
            business_name = st.text_input(f"**{LABEL_OVERRIDES.get('business_name','business_name')}** *", placeholder="Acme Plumbing")
            category = st.selectbox(LABEL_OVERRIDES.get("category", "category"), options=cats)
            service = st.selectbox(LABEL_OVERRIDES.get("service", "service"), options=[""] + svcs)
            contact_name = st.text_input(LABEL_OVERRIDES.get("contact_name", "contact_name"))
            phone = st.text_input(LABEL_OVERRIDES.get("phone", "phone"), placeholder="(210) 555-0123 or 210-555-0123")
        with col[1]:
            address = st.text_input(LABEL_OVERRIDES.get("address", "address"))
            website = st.text_input(LABEL_OVERRIDES.get("website", "website"), placeholder="https://example.com")
            notes = st.text_area(LABEL_OVERRIDES.get("notes", "notes"))
            keywords = st.text_input(LABEL_OVERRIDES.get("keywords", "keywords"))
        submitted = st.form_submit_button("Add")
        if submitted:
            ok, cleaned, errors = _validate_vendor_fields(business_name, category, service, phone, website)
            if not ok:
                for field, msg in errors.items():
                    st.error(f"{field}: {msg}")
                return
            row = {
                "business_name": cleaned["business_name"], "category": cleaned["category"], "service": cleaned["service"],
                "contact_name": (contact_name or "").strip(), "phone": cleaned["phone"], "address": (address or "").strip(),
                "website": cleaned["website"], "notes": (notes or "").strip(), "keywords": (keywords or "").strip(),
            }
            _insert_vendor(row)
            st.success("Vendor added.")
            st.rerun()

def _parse_vid_from_label(label: str) -> Optional[int]:
    if "id#" not in label:
        return None
    try:
        return int(label.split("id#")[-1].strip())
    except Exception:
        return None

def tab_edit():
    df = _fetch_df()
    if df.empty:
        st.info("No vendors available.")
        return

    df_sel = df.copy()
    df_sel = df_sel.sort_values(
        by=["business_name","category","id"],
        key=lambda s: s.astype(str).str.casefold()
    )
    df_sel["label"] = (
        df_sel["business_name"].fillna("").astype(str) + " (" +
        df_sel["category"].fillna("").astype(str) + ") — id#" +
        df_sel["id"].astype(str)
    )

    choice = st.selectbox("Select Vendor", options=df_sel["label"].tolist())
    vid = _parse_vid_from_label(choice)
    if vid is None:
        st.error("Could not parse selected vendor id.")
        return

    row = df[df["id"] == vid].iloc[0].to_dict()

    cats = sorted(_cats()["name"].dropna().astype(str).tolist(), key=lambda s: s.casefold())
    svcs = sorted(_svcs()["name"].dropna().astype(str).tolist(), key=lambda s: s.casefold())

    with st.form("edit_vendor_form"):
        col = st.columns(2)
        with col[0]:
            row["business_name"] = st.text_input(LABEL_OVERRIDES.get("business_name", "business_name"), value=row["business_name"])
            cat_idx = 0
            if row.get("category") in cats:
                cat_idx = cats.index(row["category"])
            row["category"] = st.selectbox(LABEL_OVERRIDES.get("category", "category"), options=cats, index=cat_idx)

            svc_options = [""] + svcs
            svc_idx = 0
            if row.get("service") in svcs:
                svc_idx = 1 + svcs.index(row["service"])
            row["service"] = st.selectbox(LABEL_OVERRIDES.get("service", "service"), options=svc_options, index=svc_idx)

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
                    row.get("business_name",""), row.get("category",""), row.get("service",""),
                    row.get("phone",""), row.get("website","")
                )
                if not ok:
                    for field, msg in errors.items():
                        st.error(f"{field}: {msg}")
                    return
                upd = {
                    "business_name": cleaned["business_name"], "category": cleaned["category"], "service": cleaned["service"],
                    "contact_name": (row.get("contact_name","")).strip(), "phone": cleaned["phone"], "address": (row.get("address","")).strip(),
                    "website": cleaned["website"], "notes": (row.get("notes","")).strip(), "keywords": (row.get("keywords","")).strip(),
                }
                _update_vendor(vid, upd)
                st.success("Saved.")
                st.rerun()
        with col2[1]:
            if st.form_submit_button("Cancel"):
                st.stop()

def tab_delete():
    df = _fetch_df()
    if df.empty:
        st.info("No vendors to delete.")
        return

    df_sel = df.copy().sort_values(
        by=["business_name","category","id"],
        key=lambda s: s.astype(str).str.casefold()
    )
    df_sel["label"] = (
        df_sel["business_name"].fillna("").astype(str) + " (" +
        df_sel["category"].fillna("").astype(str) + ") — id#" +
        df_sel["id"].astype(str)
    )
    choice = st.selectbox("Select Vendor to Delete", options=df_sel["label"].tolist())
    vid = _parse_vid_from_label(choice)
    if vid is None:
        st.error("Could not parse selected vendor id.")
        return

    if st.button("Delete", type="primary"):
        _delete_vendor(vid)
        st.success("Deleted.")
        st.rerun()

def tab_categories():
    st.subheader("Categories Admin")
    cats = _cats()
    st.dataframe(cats, use_container_width=True, hide_index=True)

    with st.form("add_cat"):
        new = st.text_input("Add Category")
        if st.form_submit_button("Add") and (new or "").strip():
            _upsert_category(new)
            st.success("Category added (or already existed). Reload to see it.")
            st.rerun()

    with st.form("del_cat"):
        delname = st.text_input("Delete Category (exact name)")
        if st.form_submit_button("Delete") and (delname or "").strip():
            _delete_category(delname)
            st.success("Category deleted if it existed. Reassign vendors first if needed.")
            st.rerun()

def _upsert_category(name: str):
    with eng.begin() as c:
        c.execute(sql_text("INSERT OR IGNORE INTO categories(name) VALUES(:n)"), {"n": name.strip()})

def _delete_category(name: str):
    with eng.begin() as c:
        c.execute(sql_text("DELETE FROM categories WHERE name = :n"), {"n": name.strip()})

def tab_services():
    st.subheader("Services Admin")
    svcs = _svcs()
    st.dataframe(svcs, use_container_width=True, hide_index=True)

    with st.form("add_svc"):
        new = st.text_input("Add Service")
        if st.form_submit_button("Add") and (new or "").strip():
            _upsert_service(new)
            st.success("Service added (or already existed). Reload to see it.")
            st.rerun()

    with st.form("del_svc"):
        delname = st.text_input("Delete Service (exact name)")
        if st.form_submit_button("Delete") and (delname or "").strip():
            _delete_service(delname)
            st.success("Service deleted if it existed. Reassign vendors first if needed.")
            st.rerun()

def _upsert_service(name: str):
    with eng.begin() as c:
        c.execute(sql_text("INSERT OR IGNORE INTO services(name) VALUES(:n)"), {"n": name.strip()})

def _delete_service(name: str):
    with eng.begin() as c:
        c.execute(sql_text("DELETE FROM services WHERE name = :n"), {"n": name.strip()})

def tab_maintenance():
    st.subheader("Maintenance")
    st.caption("Quick utilities for data hygiene.")
    col = st.columns(2)
    with col[0]:
        if st.button("Normalize phone format to (xxx) xxx-xxxx"):
            with eng.begin() as c:
                dfp = pd.read_sql_query("SELECT id, phone FROM vendors", c)
                for _, r in dfp.iterrows():
                    oldp = str(r.get("phone",""))
                    fmt = _format_phone_10(oldp)
                    if fmt and fmt != oldp:
                        now = _now_iso()
                        c.execute(sql_text(
                            "UPDATE vendors SET phone = :p, updated_at = :ua, updated_by = :ub WHERE id = :id"
                        ), {"p": fmt, "ua": now, "ub": ADMIN_DISPLAY_NAME, "id": int(r["id"])})
                        c.execute(sql_text("""
                            INSERT INTO vendor_changes(vendor_id, changed_at, changed_by, action, field, old_value, new_value)
                            VALUES(:vid, :ts, :by, 'update', 'phone', :oldv, :newv)
                        """), {"vid": int(r["id"]), "ts": now, "by": ADMIN_DISPLAY_NAME,
                               "oldv": oldp, "newv": fmt})
            st.success("Phone numbers normalized where possible.")
            st.rerun()
    with col[1]:
        if st.button("Trim whitespace in text fields"):
            with eng.begin() as c:
                for k in ["category","service","business_name","contact_name","address","website","notes","keywords"]:
                    c.execute(sql_text(f"UPDATE vendors SET {k} = TRIM({k})"))
                c.execute(sql_text("UPDATE vendors SET updated_at = :ua, updated_by = :ub"),
                          {"ua": _now_iso(), "ub": ADMIN_DISPLAY_NAME})
            st.success("Whitespace trimmed.")
            st.rerun()

def _fmt_when(s: str) -> str:
    try:
        dt = datetime.fromisoformat(s.replace("Z","+00:00"))
        return dt.astimezone().strftime("%b %d, %Y, %I:%M %p")
    except Exception:
        return s

def tab_changelog():
    st.subheader("Changelog")
    f1, f2 = st.columns(2)
    with f1:
        vendor_filter = st.text_input("Filter by vendor name contains", "")
    with f2:
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
            _ = f"`{oldv}` → `{newv}`"  # keep arrow out of bullet text to reduce Markdown escaping issues
            lines.append(f"{when} — Updated **{field}** for **{name}** (by {by})")
        else:
            lines.append(f"{when} — Change on **{name}** (by {by})")
    if not lines:
        st.info("No changes recorded yet.")
    else:
        st.markdown("\n\n".join(f"- {ln}" for ln in lines))


# -----------------------------
# Status & Secrets (debug) — helper
# -----------------------------
def render_status_debug(expanded=False):
    with st.expander("Status & Secrets (debug)", expanded=expanded):
        backend = "libsql" if str(_read_secret("TURSO_DATABASE_URL","")).startswith("sqlite+libsql://") else "sqlite"
        st.write("DB")
        st.code({"backend": backend, "dsn": _read_secret("TURSO_DATABASE_URL",""),
                 "auth": "token_set" if bool(_read_secret("TURSO_AUTH_TOKEN")) else "none"})
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
    q = _quick_filter_ui()

    tabs = st.tabs(["View","Add","Edit","Delete","Categories Admin","Services Admin","Maintenance","Changelog"])
    with tabs[0]: tab_view(q)
    with tabs[1]: tab_add()
    with tabs[2]: tab_edit()
    with tabs[3]: tab_delete()
    with tabs[4]: tab_categories()
    with tabs[5]: tab_services()
    with tabs[6]: tab_maintenance()
    with tabs[7]: tab_changelog()

if __name__ == "__main__":
    main()
