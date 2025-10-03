# app_readonly.py
# Read-only Vendors view with:
# - True pixel widths from secrets (COLUMN_WIDTHS_PX_READONLY or fallback to COLUMN_WIDTHS_PX)
# - Wrapped cells; rows auto-grow in height
# - Click-to-sort on every column (client-side)
# - Quick filter (client-side)
# - Optional sticky first column (id) — default OFF
# - Wide page layout with configurable max width via secrets
# - Help section via st.expander (no modal/iframe)
# - CSV download button for the entire dataset

from __future__ import annotations

import os
import re
import html as py_html  # avoid name clash with HTML strings
from typing import List, Dict, Optional

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text as sql_text
from sqlalchemy.engine import Engine
import streamlit.components.v1 as components

# -----------------------------
# Early layout setup
# -----------------------------
def _read_secret_early(name: str, default=None):
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.environ.get(name, default)

_title   = _read_secret_early("page_title", "Vendors (Read-only)")
_sidebar = _read_secret_early("sidebar_state", "collapsed")  # "collapsed" | "expanded"
_max_w   = _read_secret_early("page_max_width_px", 3000)
try:
    _max_w = int(_max_w)
except Exception:
    _max_w = 3000

st.set_page_config(page_title=_title, layout="wide", initial_sidebar_state=_sidebar)
st.markdown(
    f"""
    <style>
      .main .block-container {{
        max-width: {_max_w}px;
        padding-left: 16px;
        padding-right: 16px;
      }}
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Config + secrets helpers
# -----------------------------
STICKY_FIRST_COL_DEFAULT = False

def _get_secret(name: str, default: Optional[str | int | bool] = None):
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

def _get_bool_secret(name: str, default: bool) -> bool:
    val = _get_secret(name, None)
    if val is None:
        return default
    s = str(val).strip().lower()
    if s in {"1", "true", "yes", "on"}:
        return True
    if s in {"0", "false", "no", "off"}:
        return False
    return default

# -----------------------------
# DB engine
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

def _get_libsql_creds() -> Dict[str, Optional[str]]:
    url = (_get_secret("LIBSQL_URL") or _get_secret("TURSO_DATABASE_URL"))
    token = (_get_secret("LIBSQL_AUTH_TOKEN") or _get_secret("TURSO_AUTH_TOKEN"))
    return {"url": url, "token": token}

def current_db_info() -> Dict[str, Optional[str]]:
    creds = _get_libsql_creds()
    if creds["url"]:
        dsn = _normalize_libsql_url(str(creds["url"]))
        dsn = _append_secure_param(dsn)
        return {
            "backend": "libsql",
            "dsn": dsn,
            "auth": "token_set" if creds["token"] else "no_token",
        }
    sqlite_path = _get_secret("SQLITE_PATH") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendors.db")
    return {"backend": "sqlite", "dsn": f"sqlite:///{sqlite_path}", "auth": None}

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

# -----------------------------
# Column widths loader
# -----------------------------
def _width_value_to_px(val) -> int:
    """Convert size words or ints/strings to px."""
    if isinstance(val, int):
        return max(20, val)
    s = str(val).strip().lower()
    if s in {"small", "s"}:  return 100
    if s in {"medium", "m"}: return 160
    if s in {"large", "l"}:  return 240
    try:
        return max(20, int(s))
    except Exception:
        return 140

def _coerce_map(obj) -> Dict[str, int]:
    out: Dict[str, int] = {}
    try:
        items = obj.items()
    except Exception:
        return out
    for k, v in items:
        out[str(k).strip().lower()] = _width_value_to_px(v)
    return out

def _get_column_widths_px() -> Dict[str, int]:
    """
    Prefer COLUMN_WIDTHS_PX_READONLY; fallback to COLUMN_WIDTHS_PX.
    Accepts Streamlit AttrDict mapping, JSON mapping string, or key/val mapping text.
    """
    import json
    from collections.abc import Mapping

    def _load_block(name: str):
        try:
            return st.secrets.get(name, None)
        except Exception:
            return None

    raw = _load_block("COLUMN_WIDTHS_PX_READONLY")
    if raw is None:
        raw = _load_block("COLUMN_WIDTHS_PX")

    mapping: Dict[str, int] = {}
    if isinstance(raw, Mapping):
        mapping = _coerce_map(raw)
    elif isinstance(raw, str):
        s = raw.strip()
        try:
            obj = json.loads(s)
            if isinstance(obj, Mapping):
                mapping = _coerce_map(obj)
        except Exception:
            local_map: Dict[str, int] = {}
            for line in s.splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                key = k.strip().strip('"').strip("'").lower()
                val = v.strip().strip('"').strip("'")
                local_map[key] = _width_value_to_px(val)
            mapping = local_map
    else:
        mapping = {}

    website_w = _get_int_secret("WEBSITE_COL_WIDTH_PX", 0)
    if website_w and "website" not in mapping:
        mapping["website"] = website_w

    return mapping

def _sticky_first_col_enabled() -> bool:
    return _get_bool_secret("READONLY_STICKY_FIRST_COL", STICKY_FIRST_COL_DEFAULT)

# -----------------------------
# Data loading
# -----------------------------
@st.cache_data(show_spinner=False, ttl=30)
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

@st.cache_data(show_spinner=False, ttl=30)
def load_vendors_df() -> pd.DataFrame:
    cols = get_columns("vendors")
    base_cols = ["id", "category", "service", "business_name", "contact_name",
                 "phone", "address", "website", "notes", "keywords"]
    sel_cols = [c for c in base_cols if c in cols]
    if not sel_cols:
        return pd.DataFrame()
    with engine.begin() as conn:
        df = pd.read_sql_query(sql_text(f"SELECT {', '.join(sel_cols)} FROM vendors"), conn)
    if "service" in df.columns:
        df["service"] = df["service"].apply(lambda v: v if (isinstance(v, str) and v.strip()) else "")
    if "category" in df.columns:
        df["category"] = df["category"].apply(lambda v: v.strip() if isinstance(v, str) else v)
    if "business_name" in df.columns:
        df = df.sort_values("business_name", key=lambda s: s.str.lower()).reset_index(drop=True)
    return df

# -----------------------------
# HTML grid renderer (sortable + wrap)
# -----------------------------
def _render_sortable_wrapped_table(
    df: pd.DataFrame,
    px_map: Dict[str, int],
    height_px: int = 720,
    sticky_first_col: bool = False,
) -> None:
    if df.empty:
        st.info("No vendors found.")
        return

    cols = list(df.columns)

    def _px(col: str) -> int:
        try:
            return px_map.get(col, 140)
        except Exception:
            return 140

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
      .tbl tbody tr:hover {{ background: #fcfcfc; }}
      .tbl td {{
        white-space: normal;
        word-break: break-word;
        text-overflow: clip;
        overflow: visible;
      }}
      .th-inner {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
      }}
      .sort-arrow {{ font-size: 12px; color: #888; visibility: hidden; }}
      th.sorted .sort-arrow {{ visibility: visible; }}

      .tbl .sticky-col {{
        position: sticky;
        left: 0;
        z-index: 3;
        background: #ffffff;
      }}
      .tbl thead th.sticky-col {{ z-index: 4; }}
      .sticky-shadow {{ box-shadow: 2px 0 0 rgba(0,0,0,0.06); }}
    </style>
    """

    colgroup = ["<colgroup>"] + [f'<col style="width:{_px(c)}px">' for c in cols] + ["</colgroup>"]

    thead = ["<thead><tr>"]
    for idx, c in enumerate(cols):
        label = py_html.escape(c.replace("_", " ").title())
        th_classes = ' class="sticky-col sticky-shadow"' if (sticky_first_col and idx == 0) else ""
        thead.append(
            f'<th{th_classes} data-col="{py_html.escape(c)}">'
            f'<span class="th-inner">{label}<span class="sort-arrow">▲</span></span>'
            f"</th>"
        )
    thead.append("</tr></thead>")

    tbody = ["<tbody>"]
    for _, row in df.iterrows():
        tbody.append("<tr>")
        for idx, c in enumerate(cols):
            val = row[c]
            td_class = ' class="sticky-col sticky-shadow"' if (sticky_first_col and idx == 0) else ""
            if c == "website" and isinstance(val, str) and val.strip():
                href = py_html.escape(val)
                text = py_html.escape(val)
                tbody.append(f'<td{td_class}><a href="{href}" target="_blank" rel="noopener noreferrer">{text}</a></td>')
            else:
                safe = "" if pd.isna(val) else str(val)
                tbody.append(f"<td{td_class}>{py_html.escape(safe)}</td>")
        tbody.append("</tr>")
    tbody.append("</tbody>")

    # (plain triple-quoted string — no f-string, so braces are fine)
    script = """
    <script>
      (function() {
        const container = document.currentScript.closest('.tbl-container');
        const input = container.querySelector('.tbl-filter');
        const table = container.querySelector('table.tbl');
        const thead = table.tHead;
        const tbody = table.tBodies[0];
        const headers = Array.from(thead.rows[0].cells);
        let sortState = {}; // colIndex -> 'asc'|'desc'

        function getCellText(td) {
          return (td.textContent || td.innerText || '').trim();
        }
        function compare(a, b, idx, asc) {
          const ta = getCellText(a.cells[idx]);
          const tb = getCellText(b.cells[idx]);
          const na = parseFloat(ta.replace(/[^0-9.-]/g, ''));
          const nb = parseFloat(tb.replace(/[^0-9.-]/g, ''));
          const bothNumeric = !isNaN(na) && !isNaN(nb) && ta !== '' && tb !== '';
          let cmp = 0;
          if (bothNumeric) cmp = na - nb;
          else cmp = ta.localeCompare(tb, undefined, {sensitivity: 'base'});
          return asc ? cmp : -cmp;
        }

        headers.forEach((th, idx) => {
          th.addEventListener('click', () => {
            const dir = sortState[idx] === 'asc' ? 'desc' : 'asc';
            sortState = {}; sortState[idx] = dir;

            headers.forEach(h => h.classList.remove('sorted'));
            th.classList.add('sorted');
            const arrow = th.querySelector('.sort-arrow');
            if (arrow) arrow.textContent = dir === 'asc' ? '▲' : '▼';

            const rows = Array.from(tbody.rows);
            rows.sort((ra, rb) => compare(ra, rb, idx, dir === 'asc'));
            rows.forEach(r => tbody.appendChild(r));
          });
        });

        input.addEventListener('input', () => {
          const q = input.value.toLowerCase();
          Array.from(tbody.rows).forEach(tr => {
            const text = tr.innerText.toLowerCase();
            tr.style.display = text.includes(q) ? '' : 'none';
          });
        });
      })();
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
        + "".join(colgroup) + "".join(thead) + "".join(tbody)
        + "</table></div>"
        + script
        + "</div>"
    )

    components.html(html_doc, height=height_px, scrolling=True)

# -----------------------------
# Help content (expander, no modal)
# -----------------------------
DEFAULT_HELP_HTML = """
<div style="line-height:1.55;">
  <h2 style="margin:0 0 10px;">Vendors — How to Use</h2>
  <h3 style="margin:14px 0 6px;">Sorting</h3>
  <ul>
    <li>Click a column header to sort ascending; click again for descending (▲ / ▼).</li>
    <li>Numeric columns sort numerically; others sort A→Z.</li>
  </ul>
  <h3 style="margin:14px 0 6px;">Filtering</h3>
  <ul>
    <li>Use the <strong>Quick filter</strong> box above the table.</li>
    <li>It matches text across <em>all</em> columns.</li>
  </ul>
  <h3 style="margin:14px 0 6px;">Viewing Long Text</h3>
  <ul>
    <li>Cells wrap automatically; row height expands to fit content.</li>
    <li>Scroll horizontally if a column is off-screen.</li>
  </ul>
  <h3 style="margin:14px 0 6px;">Copy & Links</h3>
  <ul>
    <li>Select text in any cell and press Ctrl/Cmd+C to copy.</li>
    <li>Website values are clickable and open in a new tab.</li>
  </ul>
  <h3 style="margin:14px 0 6px;">Column Widths</h3>
  <ul>
    <li>Set pixel widths in <code>[COLUMN_WIDTHS_PX_READONLY]</code> in secrets.</li>
    <li>Example: <code>address = 220</code>, <code>notes = 320</code>.</li>
  </ul>
</div>
"""

def _load_help_html() -> str:
    html_from_secrets = _get_secret("READONLY_HELP_HTML", None)
    if html_from_secrets and str(html_from_secrets).strip():
        return str(html_from_secrets)

    md = _get_secret("READONLY_HELP_MD", None)
    if md and str(md).strip():
        # Render markdown directly in Streamlit
        return None  # Signal that we should use st.markdown(md)

    return DEFAULT_HELP_HTML

def render_help_expander():
    title = _get_secret("READONLY_HELP_TITLE", "Help / Tips")
    md = _get_secret("READONLY_HELP_MD", None)
    with st.expander(title, expanded=False):
        if md and str(md).strip():
            st.markdown(str(md))
        else:
            html_block = _load_help_html()
            if html_block:  # HTML fallback
                st.markdown(html_block, unsafe_allow_html=True)
            else:
                st.info("No help text configured.")

# -----------------------------
# App UI
# -----------------------------
st.title("Vendors (Read-only)")
render_help_expander()

with st.expander("Status & Secrets (debug)", expanded=False):
    st.write("**DB**", current_db_info())
    try:
        st.write("**Secrets keys**", sorted(list(st.secrets.keys())))
        raw_ro = st.secrets.get("COLUMN_WIDTHS_PX_READONLY", None)
        raw = st.secrets.get("COLUMN_WIDTHS_PX", None)
        st.write("**Raw COLUMN_WIDTHS_PX_READONLY (type)**", type(raw_ro).__name__)
        if raw_ro is not None:
            try:
                st.json(dict(raw_ro))
            except Exception:
                st.code(repr(raw_ro))
        st.write("**Raw COLUMN_WIDTHS_PX (type)**", type(raw).__name__)
        if raw is not None:
            try:
                st.json(dict(raw))
            except Exception:
                st.code(repr(raw))
        st.write("**Loaded column widths (effective)**")
        st.json(_get_column_widths_px())
        st.write("**Sticky first col enabled**:", _sticky_first_col_enabled())
    except Exception as e:
        st.write("Debug error:", str(e))

# Load and render data
df = load_vendors_df()
desired = ["business_name","category","service","contact_name","phone","address","website","notes","keywords"]
show_cols = [c for c in df.columns if c in desired]
columns_order = (["id"] + show_cols) if "id" in df.columns else show_cols

if df.empty:
    st.info("No vendors found.")
else:
    df_view = df[columns_order].copy()
    widths_px = _get_column_widths_px()

    # --- Download button (entire dataset) ---
    csv_bytes = df_view.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="Download vendors as CSV",
        data=csv_bytes,
        file_name="vendors.csv",
        mime="text/csv",
        use_container_width=True,
    )

    # Render grid
    _render_sortable_wrapped_table(
        df_view,
        widths_px,
        height_px=720,
        sticky_first_col=_sticky_first_col_enabled(),
    )
