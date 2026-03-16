"""database.py — SQLite scan history."""
import datetime, sqlite3
from config import DB_PATH

_SQL = """CREATE TABLE IF NOT EXISTS scans (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL, denomination TEXT NOT NULL,
    verdict TEXT NOT NULL, auth_score REAL DEFAULT 0, confidence REAL DEFAULT 0
);"""

def init_db():
    with sqlite3.connect(DB_PATH) as c: c.execute(_SQL); c.commit()

def save_scan(r: dict):
    init_db()
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with sqlite3.connect(DB_PATH) as c:
        c.execute("INSERT INTO scans(timestamp,denomination,verdict,auth_score,confidence)"
                  " VALUES(?,?,?,?,?)",
                  (ts, r.get("denomination","?"), r.get("verdict","?"),
                   r.get("auth_score",0), r.get("model_conf",0)))
        c.commit()

def get_history(limit=20):
    init_db()
    with sqlite3.connect(DB_PATH) as c:
        rows = c.execute(
            "SELECT timestamp,denomination,verdict,auth_score,confidence"
            " FROM scans ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [{"timestamp":r[0],"denomination":r[1],"verdict":r[2],
             "auth_score":round(r[3],1),"confidence":round(r[4],1)} for r in rows]
