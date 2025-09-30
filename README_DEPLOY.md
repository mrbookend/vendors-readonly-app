# Live Data (Turso/libSQL) — Quick Setup


Goal: Make **both apps** (admin + read-only) use the **same remote DB** so the read-only URL always shows current data.


## 1) Import your current SQLite file into Turso
From your repo folder (Crostini terminal):


```bash
cd ~/vendors-readonly-app
# Optional tidy
sqlite3 vendors.db "PRAGMA wal_checkpoint(TRUNCATE);"
# Import
turso db import ./vendors.db
