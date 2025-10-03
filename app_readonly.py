# app_readonly.py
# Read-only Vendors view with:
# - True pixel widths from secrets (COLUMN_WIDTHS_PX_READONLY or fallback to COLUMN_WIDTHS_PX)
# - Wrapped cells; rows auto-grow in height
# - Click-to-sort on every column (client-side)
# - Quick filter (client-side)
# - Optional sticky first column (id)
# - Wide page layout with configurable max width via secrets

from __future__ import annotations

import os
import re
import html
from typing import List, Dict, Optional

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text as sql_text
from sqlalchemy.engine import Engine
import streamlit.components.v1 as components

# -----------------------------
# Page layout (must happen before any other Streamlit UI output)
# -----------------------------
def _read_secret_early(name: str, default=None):
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.environ.get(name, default)

_title   = _read_secret_early("page_title", "Vendors")
_sidebar = _read_secret_early("sidebar_state", "collapsed")  # "collapsed" | "expanded"
_max_w   = _read_secret_early("page_max_width_px", 3000)
try:
    _max_w = int(_max_w)
except Exception:
    _max_w = 3000

# First UI call
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
# Config
# -----------------------------
STICKY_FIRST_COL_DEFAULT = False  # can be overridden by READONLY_STICKY_FIRST_COL in secrets

# -----------------------------
# Secrets helpers
# -----------------------------
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

def _get_libsql_creds() -> Dict[str, Optional[str]]:
    url = (_get_secret("LIBSQL_URL") or _get_secret("TURSO_DATABASE_URL"))
    token = (_get_secret("LIBSQL_AUTH_TOKEN") or _get_secret("TURSO_AUTH_TOKEN"))
    return {"url": url, "token": token}

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
    return {
        "backend": "sqlite",
        "dsn": f"sqlite:///{sqlite_path}",
        "auth": None,
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

# -----------------------------
# Column widths loader
# -----------------------------
def _width_value_to_px(val) -> int:
    """Convert size words or ints/strings to px."""
    if isinstance(val, int):
        return max(20, val)
    s = str(val).strip().lower()
    if s in {"small", "s"}:
        return 100
    if s in {"medium", "m"}:
        return 160
    if s in {"large", "l"}:
        return 240
    try:
        return max(20, int(s))
    except Exception:
        return 140

def _coerce_map(obj) -> Dict[str, int]:
    out: Dict[str, int] = {}
    try:
        items = obj.items()  # Mapping/AttrDict-like
    except Exception:
        return out
    for k, v in items:
        key = str(k).strip().lower()
        out[key] = _width_value_to_px(v)
    return out

def _get_column_widths_px() -> Dict[str, int]:
    """
    Prefer COLUMN_WIDTHS_PX_READONLY; fallback to COLUMN_WIDTHS_PX.
    Accepts Streamlit AttrDict mapping, JSON-like mapping, or key/val mapping in TOML block.
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

    # Optional website width override (legacy)
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
    # Tidy display
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
    sticky_first_col: bool = True,
) -> None:
    """
    Client-side sortable table with:
      - True pixel widths via <colgroup>
      - Wrapped cells -> rows auto-grow
      - Sticky header
      - Optional sticky first column (id)
      - Quick filter
    """
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
        max-height: {height_px - 46}px; /* toolbar height */
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
        white-space: normal;        /* wrapping on */
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

    /* Sticky first column support */
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

    # Build colgroup with explicit widths
    colgroup = ["<colgroup>"] + [f'<col style="width:{_px(c)}px">' for c in cols] + ["</colgroup>"]

    # Header (click-to-sort)
    thead = ["<thead><tr>"]
    for idx, c in enumerate(cols):
        label = html.escape(c.replace("_", " ").title())
        th_classes = ' class="sticky-col sticky-shadow"' if (sticky_first_col and idx == 0) else ""
        thead.append(
            f'<th{th_classes} data-col="{html.escape(c)}">'
            f'<span class="th-inner">{label}<span class="sort-arrow">▲</span></span>'
            f"</th>"
        )
    thead.append("</tr></thead>")

    # Body
    tbody = ["<tbody>"]
    for _, row in df.iterrows():
        tbody.append("<tr>")
        for idx, c in enumerate(cols):
            val = row[c]
            td_class = ' class="sticky-col sticky-shadow"' if (sticky_first_col and idx == 0) else ""
            if c == "website" and isinstance(val, str) and val.strip():
                href = html.escape(val)
                text = html.escape(val)
                tbody.append(f'<td{td_class}><a href="{href}" target="_blank" rel="noopener noreferrer">{text}</a></td>')
            else:
                safe = "" if pd.isna(val) else str(val)
                tbody.append(f"<td{td_class}>{html.escape(safe)}</td>")
        tbody.append("</tr>")
    tbody.append("</tbody>")

    # Quick filter + sort JS
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
# App UI
# -----------------------------
st.title("Vendors (Read-only)")

# ---- Help modal (large content friendly) ----
HELP_MD = """
**How to use this table**

- **Sort**: Click any column header. Click again to reverse (▲ / ▼).
- **Filter**: Use the “Quick filter” above the table — matches any column.
- **Wrap & row height**: Long text wraps; the row grows so everything is visible.
- **Copy text**: Select text in a cell and press Ctrl/Cmd+C. Website links open in a new tab.
- **Horizontal scroll**: Scroll the grid if columns are off-screen.
- **Column widths**: Controlled in secrets (`[COLUMN_WIDTHS_PX_READONLY]` or `[COLUMN_WIDTHS_PX]`).
"""

# Same content as HTML; you can expand this to your “half-sheet” of tips.
HELP_HTML = f"""
<div style="line-height:1.55;">
  {HELP_MD.replace('**', '<strong>').replace('__','<u>').replace('*','')}
</div>
"""

def _help_modal_dims_from_secrets():
    # Defaults tuned for long content; override via secrets if desired
    import streamlit as st
    def _get(name, default): 
        try:
            if name in st.secrets: return st.secrets[name]
        except Exception:
            pass
        return default
    try:    width_px = int(_get("READONLY_HELP_WIDTH_PX", 900))   # modal width cap
    except: width_px = 900
    try:    max_h_vh = int(_get("READONLY_HELP_MAX_H_VH", 90))    # modal max height in vh
    except: max_h_vh = 90
    try:    font_px  = int(_get("READONLY_HELP_FONT_PX", 14))     # body font size
    except: font_px  = 14
    return width_px, max_h_vh, font_px

def _render_help(style: str = "modal"):
    style = (style or "modal").strip().lower()
    if style != "modal":
        # keep other styles if you ever want to switch
        try:
            if style == "popover" and hasattr(st, "popover"):
                with st.popover("Help"):
                    st.markdown(HELP_MD)
                return
        except Exception:
            pass
        with st.expander("Help / Tips", expanded=False):
            st.markdown(HELP_MD)
        return

    # Modal implementation (HTML/CSS/JS)
    import uuid, streamlit.components.v1 as components
    w, mh, fs = _help_modal_dims_from_secrets()
    uid = f"help_{uuid.uuid4().hex[:8]}"
    components.html(f"""
    <div id="{uid}_root">
      <button id="{uid}_open" class="help-btn">Help</button>
      <div id="{uid}_overlay" class="help-overlay" style="display:none;">
        <div class="help-modal" role="dialog" aria-modal="true" aria-labelledby="{uid}_title" tabindex="-1">
          <div class="help-header">
            <span id="{uid}_title">Help</span>
            <div class="help-actions">
              <button id="{uid}_copy" class="act-btn" title="Copy all">Copy</button>
              <button id="{uid}_print" class="act-btn" title="Print">Print</button>
              <button id="{uid}_close" class="close-btn" aria-label="Close">&times;</button>
            </div>
          </div>
          <div id="{uid}_body" class="help-body">{HELP_HTML}</div>
        </div>
      </div>
    </div>
    <style>
      .help-btn {{
        padding: 6px 10px; border: 1px solid #ddd; border-radius: 6px; background:#fff; cursor:pointer;
      }}
      .help-overlay {{
        position: fixed; inset: 0; background: rgba(0,0,0,0.35);
        display: none; align-items: center; justify-content: center; z-index: 9999;
      }}
      .help-modal {{
        background:#fff; border-radius:10px; width: min({w}px, 96vw);
        max-height: {mh}vh; overflow:auto; box-shadow: 0 10px 30px rgba(0,0,0,0.25);
      }}
      .help-header {{
        padding: 10px 14px; border-bottom: 1px solid #eee; display:flex; justify-content: space-between; align-items:center;
        position: sticky; top:0; background:#fff; z-index: 1;
      }}
      .help-actions {{ display:flex; gap:8px; align-items:center; }}
      .act-btn {{
        padding: 4px 8px; border: 1px solid #ddd; border-radius: 6px; background:#fafafa; cursor:pointer; font-size: 13px;
      }}
      .close-btn {{
        font-size: 22px; line-height: 1; padding: 2px 8px; border:none; background:transparent; cursor:pointer;
      }}
      .help-body {{ padding: 14px; font-size: {fs}px; }}
      .help-body p {{ margin: 0 0 10px 0; }}
      .help-body ul {{ margin: 0 0 10px 22px; }}
      .help-body li {{ margin: 4px 0; }}
    </style>
    <script>
      (function(){{
        const overlay = document.getElementById("{uid}_overlay");
        const openBtn = document.getElementById("{uid}_open");
        const closeBtn = document.getElementById("{uid}_close");
        const modal = overlay.querySelector(".help-modal");
        const body = document.getElementById("{uid}_body");
        const copyBtn = document.getElementById("{uid}_copy");
        const printBtn = document.getElementById("{uid}_print");

        function openModal() {{
          overlay.style.display = "flex";
          modal.focus();
        }}
        function closeModal() {{
          overlay.style.display = "none";
          openBtn.focus();
        }}
        openBtn.addEventListener("click", openModal);
        closeBtn.addEventListener("click", closeModal);
        overlay.addEventListener("click", (e) => {{ if (e.target === overlay) closeModal(); }});
        document.addEventListener("keydown", (e) => {{ if (e.key === "Escape") closeModal(); }});

        copyBtn.addEventListener("click", async () => {{
          try {{
            const txt = body.textContent || "";
            await navigator.clipboard.writeText(txt);
            copyBtn.textContent = "Copied";
            setTimeout(()=> copyBtn.textContent = "Copy", 1200);
          }} catch (err) {{
            alert("Copy failed");
          }}
        }});

        printBtn.addEventListener("click", () => {{
          const w = window.open("", "_blank");
          w.document.write("<html><head><title>Help</title></head><body>" + body.innerHTML + "</body></html>");
          w.document.close();
          w.focus();
          w.print();
          setTimeout(()=> w.close(), 250);
        }});
      }})();
    </script>
    """, height=1, scrolling=False)

# Choose style via secrets (defaults to modal)
_help_style = str(st.secrets.get("READONLY_HELP_STYLE", "modal")).lower()
_render_help(_help_style)
# ---- end Help modal ----


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

# Load data
df = load_vendors_df()

# Column order to display
desired = ["business_name","category","service","contact_name",
           "phone","address","website","notes","keywords"]
show_cols = [c for c in df.columns if c in desired]
columns_order = (["id"] + show_cols) if "id" in df.columns else show_cols

if df.empty:
    st.info("No vendors found.")
else:
    df_view = df[columns_order].copy()
    widths_px = _get_column_widths_px()

    # Prevent accidental double-render
    if not st.session_state.get("_rendered_grid_once"):
        st.session_state["_rendered_grid_once"] = True
        _render_sortable_wrapped_table(
            df_view,
            widths_px,
            height_px=720,  # viewport height
            sticky_first_col=_sticky_first_col_enabled(),  # defaults to False per your config
        )
