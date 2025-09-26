# Vendors Directory — Read-Only (Streamlit)

This repo ships the SQLite database **in the repo** for a public, read-only directory with CSV export.

## Files
- `app_readonly.py` — Streamlit app (read-only, CSV download)
- `vendors.db` — SQLite DB (included)
- `requirements.txt` — Python deps

## Deploy — Streamlit Community Cloud
1. Create a new GitHub repo and upload **all three files** above.
2. Go to https://streamlit.io, sign in → **Deploy an app**.
3. Choose your repo and set **App file** to `app_readonly.py`.
4. Deploy — share the URL.

## Local test (optional)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app_readonly.py
```

Notes:
- App opens SQLite in **read-only** mode; public cannot alter data.
- Schema after your swap: **`website` holds notes text**, **`notes` holds URL**. UI labels reflect this.
- To update data: edit `vendors.db` offline, commit a new copy, and redeploy.