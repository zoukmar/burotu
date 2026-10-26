"""Stockage SQLite : centres de prélèvement, réservations et suivi des échantillons."""

from __future__ import annotations

import secrets
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS sites (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    city TEXT NOT NULL,
    address TEXT NOT NULL,
    open_from TEXT NOT NULL DEFAULT '07:30',
    open_to TEXT NOT NULL DEFAULT '11:30',
    slot_minutes INTEGER NOT NULL DEFAULT 15,
    capacity INTEGER NOT NULL DEFAULT 2,
    weekdays TEXT NOT NULL DEFAULT '0,1,2,3,4',
    active INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE IF NOT EXISTS closures (
    site_id INTEGER NOT NULL REFERENCES sites(id),
    day TEXT NOT NULL,
    PRIMARY KEY (site_id, day)
);
CREATE TABLE IF NOT EXISTS bookings (
    id INTEGER PRIMARY KEY,
    ref TEXT UNIQUE NOT NULL,
    created_at TEXT NOT NULL,
    lang TEXT NOT NULL,
    site_id INTEGER NOT NULL REFERENCES sites(id),
    day TEXT NOT NULL,
    slot TEXT NOT NULL,
    first_name TEXT NOT NULL,
    last_name TEXT NOT NULL,
    email TEXT NOT NULL,
    phone TEXT NOT NULL,
    birth_year INTEGER NOT NULL,
    doctor TEXT NOT NULL,
    amount_cents INTEGER NOT NULL,
    payment_status TEXT NOT NULL DEFAULT 'test_paid',
    status TEXT NOT NULL DEFAULT 'requested',
    admin_note TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS ix_bookings_slot ON bookings(site_id, day, slot);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY,
    booking_id INTEGER NOT NULL REFERENCES bookings(id),
    at TEXT NOT NULL,
    status TEXT NOT NULL,
    note TEXT NOT NULL DEFAULT '',
    actor TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_events_booking ON events(booking_id);
"""

DEMO_SITES = [
    ("Burotu Point Clerkenwell", "London", "14 St John Street, EC1M (demo)"),
    ("Burotu Point Bastille", "Paris", "22 rue de la Roquette, 75011 (démo)"),
    ("Burotu Point Part-Dieu", "Lyon", "8 rue Servient, 69003 (démo)"),
    ("Burotu Point Chamberí", "Madrid", "Calle de Santa Engracia 40 (demo)"),
    ("Burotu Point Eixample", "Barcelona", "Carrer del Consell de Cent 310 (demo)"),
]


@dataclass(frozen=True)
class Status:
    key: str
    label: str
    active: bool = True  # occupe un créneau


# Parcours d'un échantillon, de la demande au compte rendu.
STATUSES: tuple[Status, ...] = (
    Status("requested", "Demande reçue"),
    Status("confirmed", "Rendez-vous confirmé"),
    Status("drawn", "Sang prélevé"),
    Status("in_lab", "Au laboratoire"),
    Status("analysed", "Analysé"),
    Status("reported", "Compte rendu envoyé"),
    Status("cancelled", "Annulé", active=False),
)
STATUS_KEYS = tuple(s.key for s in STATUSES)
STATUS_LABELS = {s.key: s.label for s in STATUSES}
PIPELINE = tuple(s.key for s in STATUSES if s.active)
PRICE_CENTS = 269000
SITE_FIELDS = {"name", "city", "address", "open_from", "open_to", "slot_minutes", "capacity", "weekdays", "active"}


class Database:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(self.path, isolation_level=None, timeout=10)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys=ON")
        con.execute("PRAGMA journal_mode=WAL")
        try:
            yield con
        finally:
            con.close()

    def init(self) -> None:
        with self.connect() as con:
            con.executescript(SCHEMA)
            if con.execute("SELECT COUNT(*) FROM sites").fetchone()[0] == 0:
                con.executemany("INSERT INTO sites(name, city, address) VALUES (?,?,?)", DEMO_SITES)

    # --- Centres -------------------------------------------------------------

    def list_sites(self, active_only: bool = True) -> list[sqlite3.Row]:
        q = "SELECT * FROM sites" + (" WHERE active=1" if active_only else "") + " ORDER BY city, name"
        with self.connect() as con:
            return con.execute(q).fetchall()

    def get_site(self, site_id: int) -> sqlite3.Row | None:
        with self.connect() as con:
            return con.execute("SELECT * FROM sites WHERE id=?", (site_id,)).fetchone()

    def add_site(self, name: str, city: str, address: str) -> None:
        with self.connect() as con:
            con.execute("INSERT INTO sites(name, city, address) VALUES (?,?,?)", (name, city, address))

    def update_site(self, site_id: int, **fields) -> None:
        fields = {k: v for k, v in fields.items() if k in SITE_FIELDS}
        if not fields:
            return
        cols = ", ".join(f"{k}=?" for k in fields)  # noms de colonnes issus de la liste blanche
        with self.connect() as con:
            con.execute(f"UPDATE sites SET {cols} WHERE id=?", (*fields.values(), site_id))

    def set_closure(self, site_id: int, day: str, closed: bool) -> None:
        with self.connect() as con:
            if closed:
                con.execute("INSERT OR IGNORE INTO closures(site_id, day) VALUES (?,?)", (site_id, day))
            else:
                con.execute("DELETE FROM closures WHERE site_id=? AND day=?", (site_id, day))

    def list_closures(self) -> list[sqlite3.Row]:
        with self.connect() as con:
            return con.execute(
                """SELECT c.*, s.name, s.city FROM closures c JOIN sites s ON s.id=c.site_id
                   WHERE c.day>=? ORDER BY c.day""",
                (date.today().isoformat(),),
            ).fetchall()

    # --- Créneaux ------------------------------------------------------------

    @staticmethod
    def _slot_times(site: sqlite3.Row) -> list[str]:
        t0 = datetime.combine(date.today(), time.fromisoformat(site["open_from"]))
        t1 = datetime.combine(date.today(), time.fromisoformat(site["open_to"]))
        out = []
        while t0 < t1:
            out.append(t0.strftime("%H:%M"))
            t0 += timedelta(minutes=site["slot_minutes"])
        return out

    def available_slots(self, site_id: int, day: date, now: datetime | None = None, con=None) -> list[str]:
        now = now or datetime.now()
        if day < now.date() or day > now.date() + timedelta(days=60):
            return []
        if con is None:
            with self.connect() as c:
                return self.available_slots(site_id, day, now, c)
        site = con.execute("SELECT * FROM sites WHERE id=?", (site_id,)).fetchone()
        if site is None or not site["active"] or str(day.weekday()) not in site["weekdays"].split(","):
            return []
        if con.execute("SELECT 1 FROM closures WHERE site_id=? AND day=?", (site_id, day.isoformat())).fetchone():
            return []
        taken = dict(
            con.execute(
                "SELECT slot, COUNT(*) FROM bookings WHERE site_id=? AND day=? AND status!='cancelled' GROUP BY slot",
                (site_id, day.isoformat()),
            ).fetchall()
        )
        slots = [s for s in self._slot_times(site) if taken.get(s, 0) < site["capacity"]]
        if day == now.date():  # pas de rendez-vous dans moins d'une heure
            limit = (now + timedelta(hours=1)).strftime("%H:%M")
            slots = [s for s in slots if s >= limit]
        return slots

    # --- Réservations ----------------------------------------------------------

    def create_booking(self, data: dict) -> str | None:
        """Crée la réservation si le créneau est libre. Renvoie la référence, ou None si complet."""
        with self.connect() as con:
            con.execute("BEGIN IMMEDIATE")  # verrou d'écriture : pas de double réservation
            try:
                if data["slot"] not in self.available_slots(data["site_id"], date.fromisoformat(data["day"]), con=con):
                    con.execute("ROLLBACK")
                    return None
                ref = "BRT-" + secrets.token_hex(3).upper()
                cur = con.execute(
                    """INSERT INTO bookings(ref, created_at, lang, site_id, day, slot, first_name, last_name,
                       email, phone, birth_year, doctor, amount_cents) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        ref,
                        _now(),
                        data["lang"],
                        data["site_id"],
                        data["day"],
                        data["slot"],
                        data["first_name"],
                        data["last_name"],
                        data["email"],
                        data["phone"],
                        data["birth_year"],
                        data["doctor"],
                        PRICE_CENTS,
                    ),
                )
                con.execute(
                    "INSERT INTO events(booking_id, at, status, note, actor) VALUES (?,?,?,?,?)",
                    (cur.lastrowid, _now(), "requested", "Réservation en ligne", "site"),
                )
                con.execute("COMMIT")
                return ref
            except Exception:
                con.execute("ROLLBACK")
                raise

    def get_booking(self, ref: str) -> sqlite3.Row | None:
        with self.connect() as con:
            return con.execute(
                """SELECT b.*, s.name AS site_name, s.city, s.address
                   FROM bookings b JOIN sites s ON s.id=b.site_id WHERE ref=?""",
                (ref,),
            ).fetchone()

    def list_bookings(self, status: str | None = None, q: str | None = None) -> list[sqlite3.Row]:
        sql = "SELECT b.*, s.name AS site_name, s.city FROM bookings b JOIN sites s ON s.id=b.site_id WHERE 1=1"
        args: list = []
        if status:
            sql += " AND b.status=?"
            args.append(status)
        if q:
            sql += " AND (b.ref LIKE ? OR b.last_name LIKE ? OR b.first_name LIKE ? OR b.email LIKE ?)"
            args += [f"%{q}%"] * 4
        sql += " ORDER BY b.day DESC, b.slot DESC"
        with self.connect() as con:
            return con.execute(sql, args).fetchall()

    def events(self, ref: str) -> list[sqlite3.Row]:
        with self.connect() as con:
            return con.execute(
                "SELECT e.* FROM events e JOIN bookings b ON b.id=e.booking_id WHERE b.ref=? ORDER BY e.id",
                (ref,),
            ).fetchall()

    def set_status(self, ref: str, status: str, note: str = "", actor: str = "admin") -> None:
        if status not in STATUS_KEYS:
            raise ValueError(f"statut inconnu : {status}")
        with self.connect() as con:
            row = con.execute("SELECT id, status FROM bookings WHERE ref=?", (ref,)).fetchone()
            if row is None:
                raise KeyError(ref)
            if row["status"] == status and not note:
                return
            con.execute("UPDATE bookings SET status=? WHERE id=?", (status, row["id"]))
            con.execute(
                "INSERT INTO events(booking_id, at, status, note, actor) VALUES (?,?,?,?,?)",
                (row["id"], _now(), status, note[:1000], actor),
            )

    def set_note(self, ref: str, note: str) -> None:
        with self.connect() as con:
            con.execute("UPDATE bookings SET admin_note=? WHERE ref=?", (note[:2000], ref))

    def stats(self) -> dict[str, int]:
        cases = ", ".join(f"SUM(status='{k}') AS {k}" for k in STATUS_KEYS)
        with self.connect() as con:
            row = con.execute(
                f"""SELECT COUNT(*) AS total, {cases},
                    SUM(CASE WHEN status!='cancelled' THEN amount_cents ELSE 0 END) AS revenue_cents,
                    SUM(day>=date('now') AND status IN ('requested','confirmed')) AS upcoming
                    FROM bookings"""
            ).fetchone()
            return {k: (row[k] or 0) for k in row.keys()}  # noqa: SIM118 (sqlite3.Row)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")
