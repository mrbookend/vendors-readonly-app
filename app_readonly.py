# app_readonly.py — Vendors Directory (Read-only, Live DB via Turso or local fallback)
return


# Search (AND across all fields)
q = st.text_input(
"Search (AND across all fields)",
value=st.session_state.get("q", ""),
placeholder="e.g., plumber alamo heights",
key="q",
help="Type one or more words. We use AND logic across all fields (case-insensitive).",
)


df = filter_df(df, q)


# Keep only the desired display columns that exist
present = [c for c in DISPLAY_COLUMNS if c in df.columns]
df = df[present]


# Apply character clipping (except website)
disp = df.copy()
for col, width in CHAR_WIDTHS.items():
if col in disp.columns and col != "website":
disp[col] = disp[col].map(lambda v: clip_value(v, width))


# Rename columns for friendly labels
disp = disp.rename(columns={c: LABELS.get(c, c) for c in disp.columns})


# Configure Website as a clickable link labeled "Open"
col_config = {}
if "Website" in disp.columns:
try:
col_config["Website"] = st.column_config.LinkColumn("Website", display_text="Open")
except Exception:
# Older Streamlit versions may not have LinkColumn; leave as plain text
pass


st.dataframe(disp, use_container_width=True, hide_index=True, column_config=col_config)


with st.expander("About this directory", expanded=False):
st.markdown(
"""
**Live data**: This page reads directly from the configured database (Turso if set, otherwise local SQLite).
Use the search box to filter; multiple words are combined with AND.
To adjust column clipping widths, edit the `CHAR_WIDTHS` dict near the top of the file.
"""
)




if __name__ == "__main__":
main()
