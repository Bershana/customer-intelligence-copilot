"""Repository paths. Scripts can be run from any working directory."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
CLEAN = DATA / "clean"
POLICIES = DATA / "policies"
DB_PATH = DATA / "instamart.db"
SQL_PATH = ROOT / "sql" / "queries.sql"
OUTPUTS = ROOT / "outputs"
STATE = ROOT / "state"
