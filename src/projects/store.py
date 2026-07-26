"""
SQLite store for projects, branches, story parts, and the fact ledger.

Local rather than Delta for two reasons. Facts need UPDATE when they are
superseded, which a Delta Sync vector index handles poorly, and the branch tree
is relational data that would be awkward and slow to walk over a SQL warehouse.
Keeping it local also means the app opens instantly and keeps working when the
warehouse is asleep -- passage search degrades, the story does not.

Embedded prose passages go to Databricks Vector Search instead. See
`canon_store.py`.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .schemas import MAIN_BRANCH_NAME, Branch, Fact, Project, Segment
from .visibility import BranchGraph

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("projects/anubhuti.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    logline     TEXT NOT NULL DEFAULT '',
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS branches (
    id          TEXT PRIMARY KEY,
    project_id  TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    name        TEXT NOT NULL,
    parent_id   TEXT REFERENCES branches(id),
    forked_at   INTEGER,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_branches_project ON branches(project_id);

CREATE TABLE IF NOT EXISTS segments (
    id          TEXT PRIMARY KEY,
    project_id  TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    branch_id   TEXT NOT NULL REFERENCES branches(id),
    position    INTEGER NOT NULL,
    title       TEXT NOT NULL DEFAULT '',
    text        TEXT NOT NULL,
    word_count  INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT NOT NULL,
    UNIQUE (branch_id, position)
);
CREATE INDEX IF NOT EXISTS idx_segments_project ON segments(project_id);

CREATE TABLE IF NOT EXISTS facts (
    id                   TEXT PRIMARY KEY,
    project_id           TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    subject              TEXT NOT NULL,
    claim                TEXT NOT NULL,
    kind                 TEXT NOT NULL DEFAULT 'state',
    established_branch   TEXT NOT NULL,
    established_position INTEGER NOT NULL,
    superseded_branch    TEXT,
    superseded_position  INTEGER,
    superseded_by        TEXT,
    source_segment_id    TEXT NOT NULL DEFAULT '',
    quote                TEXT NOT NULL DEFAULT '',
    embedding            TEXT,
    created_at           TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_facts_project ON facts(project_id);
CREATE INDEX IF NOT EXISTS idx_facts_subject ON facts(project_id, subject);

CREATE TABLE IF NOT EXISTS synopses (
    branch_id   TEXT PRIMARY KEY REFERENCES branches(id) ON DELETE CASCADE,
    project_id  TEXT NOT NULL,
    position    INTEGER NOT NULL,
    text        TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_time(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return datetime.now(timezone.utc)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class ProjectStore:
    """
    Everything on disk for every project.

    One connection guarded by a lock. Streamlit reruns the script on every
    interaction and can service more than one session, so the naive
    "connection per module import" pattern trips SQLite's thread checks.
    """

    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        # Readers do not block the writer, which matters because canon ingest
        # runs on a background thread while the UI keeps reading.
        self._conn.execute("PRAGMA journal_mode = WAL")
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # -- projects ----------------------------------------------------------

    def create_project(self, title: str, logline: str = "") -> tuple[Project, Branch]:
        """Create a project and its main timeline together."""
        project = Project(id=new_id("prj"), title=title, logline=logline.strip())
        branch = Branch(
            id=new_id("brn"),
            project_id=project.id,
            name=MAIN_BRANCH_NAME,
            parent_id=None,
            forked_at=None,
        )

        with self._lock:
            self._conn.execute(
                "INSERT INTO projects (id, title, logline, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (project.id, project.title, project.logline, _now(), _now()),
            )
            self._conn.execute(
                "INSERT INTO branches (id, project_id, name, parent_id, forked_at, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (branch.id, branch.project_id, branch.name, None, None, _now()),
            )
            self._conn.commit()

        logger.info("Created project %s (%s)", project.title, project.id)
        return project, branch

    def list_projects(self) -> list[Project]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM projects ORDER BY updated_at DESC"
            ).fetchall()
        return [self._row_to_project(r) for r in rows]

    def get_project(self, project_id: str) -> Project | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM projects WHERE id = ?", (project_id,)
            ).fetchone()
        return self._row_to_project(row) if row else None

    def rename_project(self, project_id: str, title: str, logline: str = "") -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE projects SET title = ?, logline = ?, updated_at = ? WHERE id = ?",
                (title.strip(), logline.strip(), _now(), project_id),
            )
            self._conn.commit()

    def touch_project(self, project_id: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE projects SET updated_at = ? WHERE id = ?", (_now(), project_id)
            )
            self._conn.commit()

    def delete_project(self, project_id: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM facts WHERE project_id = ?", (project_id,))
            self._conn.execute("DELETE FROM segments WHERE project_id = ?", (project_id,))
            self._conn.execute(
                "DELETE FROM synopses WHERE project_id = ?", (project_id,)
            )
            self._conn.execute("DELETE FROM branches WHERE project_id = ?", (project_id,))
            self._conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))
            self._conn.commit()

    # -- branches ----------------------------------------------------------

    def list_branches(self, project_id: str) -> list[Branch]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM branches WHERE project_id = ? ORDER BY created_at",
                (project_id,),
            ).fetchall()
        return [self._row_to_branch(r) for r in rows]

    def get_branch(self, branch_id: str) -> Branch | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM branches WHERE id = ?", (branch_id,)
            ).fetchone()
        return self._row_to_branch(row) if row else None

    def graph(self, project_id: str) -> BranchGraph:
        return BranchGraph(self.list_branches(project_id))

    def create_branch(self, from_branch_id: str, at_position: int, name: str) -> Branch:
        """
        Fork a new timeline off an existing one.

        `at_position` is exclusive: the new branch inherits parts strictly
        below it and writes its own first part *at* it. Numbering continues
        rather than restarting, so positions stay comparable across the
        lineage and one integer comparison can order the whole tree.
        """
        parent = self.get_branch(from_branch_id)
        if parent is None:
            raise ValueError(f"No such branch: {from_branch_id}")
        if at_position < 0:
            raise ValueError("Cannot fork before the start of the story.")

        branch = Branch(
            id=new_id("brn"),
            project_id=parent.project_id,
            name=name,
            parent_id=parent.id,
            forked_at=at_position,
        )

        with self._lock:
            self._conn.execute(
                "INSERT INTO branches (id, project_id, name, parent_id, forked_at, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    branch.id,
                    branch.project_id,
                    branch.name,
                    branch.parent_id,
                    branch.forked_at,
                    _now(),
                ),
            )
            self._conn.commit()

        logger.info(
            "Forked '%s' from '%s' at position %d", branch.name, parent.name, at_position
        )
        return branch

    def rename_branch(self, branch_id: str, name: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE branches SET name = ? WHERE id = ?", (name.strip(), branch_id)
            )
            self._conn.commit()

    def delete_branch(self, branch_id: str) -> None:
        """Remove a branch and everything written on it. Roots are protected."""
        branch = self.get_branch(branch_id)
        if branch is None:
            return
        if branch.is_root:
            raise ValueError("The main timeline cannot be deleted.")

        graph = self.graph(branch.project_id)
        if graph.children_of(branch_id):
            raise ValueError(
                "This timeline has branches of its own. Delete those first."
            )

        with self._lock:
            self._conn.execute("DELETE FROM segments WHERE branch_id = ?", (branch_id,))
            self._conn.execute(
                "DELETE FROM facts WHERE established_branch = ?", (branch_id,)
            )
            # A fact superseded only on the branch being removed becomes live
            # again, which is the correct outcome: the event that ended it no
            # longer happened anywhere.
            self._conn.execute(
                "UPDATE facts SET superseded_branch = NULL, superseded_position = NULL, "
                "superseded_by = NULL WHERE superseded_branch = ?",
                (branch_id,),
            )
            self._conn.execute("DELETE FROM synopses WHERE branch_id = ?", (branch_id,))
            self._conn.execute("DELETE FROM branches WHERE id = ?", (branch_id,))
            self._conn.commit()

    # -- segments ----------------------------------------------------------

    def list_segments(self, project_id: str) -> list[Segment]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM segments WHERE project_id = ? ORDER BY position",
                (project_id,),
            ).fetchall()
        return [self._row_to_segment(r) for r in rows]

    def add_segment(
        self,
        project_id: str,
        branch_id: str,
        text: str,
        *,
        title: str = "",
        position: int | None = None,
    ) -> Segment:
        """Finalise a part onto a branch at the next free position."""
        if position is None:
            graph = self.graph(project_id)
            position = graph.next_position(branch_id, self.list_segments(project_id))

        segment = Segment(
            id=new_id("seg"),
            project_id=project_id,
            branch_id=branch_id,
            position=position,
            title=title.strip(),
            text=text,
            word_count=len(text.split()),
        )

        with self._lock:
            self._conn.execute(
                "INSERT INTO segments "
                "(id, project_id, branch_id, position, title, text, word_count, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    segment.id,
                    segment.project_id,
                    segment.branch_id,
                    segment.position,
                    segment.title,
                    segment.text,
                    segment.word_count,
                    _now(),
                ),
            )
            self._conn.commit()

        self.touch_project(project_id)
        return segment

    def delete_segment(self, segment_id: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM segments WHERE id = ?", (segment_id,))
            self._conn.commit()

    def story_so_far(
        self, project_id: str, branch_id: str, position: int | None = None
    ) -> list[Segment]:
        """Every part visible from this point on this timeline, in reading order."""
        graph = self.graph(project_id)
        segments = self.list_segments(project_id)
        if position is None:
            from .visibility import UNBOUNDED

            position = UNBOUNDED
        return graph.visible_segments(segments, branch_id, position)

    def next_position(self, project_id: str, branch_id: str) -> int:
        graph = self.graph(project_id)
        return graph.next_position(branch_id, self.list_segments(project_id))

    # -- facts -------------------------------------------------------------

    def list_facts(self, project_id: str) -> list[Fact]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM facts WHERE project_id = ? ORDER BY established_position",
                (project_id,),
            ).fetchall()
        return [self._row_to_fact(r) for r in rows]

    def list_facts_with_embeddings(
        self, project_id: str
    ) -> list[tuple[Fact, list[float] | None]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM facts WHERE project_id = ? ORDER BY established_position",
                (project_id,),
            ).fetchall()

        out: list[tuple[Fact, list[float] | None]] = []
        for row in rows:
            raw = row["embedding"]
            vector = json.loads(raw) if raw else None
            out.append((self._row_to_fact(row), vector))
        return out

    def add_facts(self, facts: list[Fact], embeddings: dict[str, list[float]] | None = None) -> None:
        embeddings = embeddings or {}
        with self._lock:
            self._conn.executemany(
                "INSERT INTO facts (id, project_id, subject, claim, kind, "
                "established_branch, established_position, superseded_branch, "
                "superseded_position, superseded_by, source_segment_id, quote, "
                "embedding, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        f.id,
                        f.project_id,
                        f.subject,
                        f.claim,
                        f.kind,
                        f.established_branch,
                        f.established_position,
                        f.superseded_branch,
                        f.superseded_position,
                        f.superseded_by,
                        f.source_segment_id,
                        f.quote,
                        json.dumps(embeddings[f.id]) if f.id in embeddings else None,
                        _now(),
                    )
                    for f in facts
                ],
            )
            self._conn.commit()

    def supersede_fact(
        self,
        fact_id: str,
        branch_id: str,
        position: int,
        superseded_by: str | None = None,
    ) -> None:
        """
        Record that a fact stopped being true, on a branch, at a position.

        The supersession is itself an event on a timeline, which is why it
        stores a branch. A fact ended on the main line stays live on a branch
        that forked before the ending.
        """
        with self._lock:
            self._conn.execute(
                "UPDATE facts SET superseded_branch = ?, superseded_position = ?, "
                "superseded_by = ? WHERE id = ?",
                (branch_id, position, superseded_by, fact_id),
            )
            self._conn.commit()

    def revive_fact(self, fact_id: str) -> None:
        """Undo a supersession, used when the part that caused it is removed."""
        with self._lock:
            self._conn.execute(
                "UPDATE facts SET superseded_branch = NULL, superseded_position = NULL, "
                "superseded_by = NULL WHERE id = ?",
                (fact_id,),
            )
            self._conn.commit()

    def delete_facts_from_segment(self, segment_id: str) -> None:
        with self._lock:
            self._conn.execute(
                "DELETE FROM facts WHERE source_segment_id = ?", (segment_id,)
            )
            self._conn.commit()

    # -- synopsis ----------------------------------------------------------

    def get_synopsis(self, branch_id: str) -> tuple[str, int]:
        with self._lock:
            row = self._conn.execute(
                "SELECT text, position FROM synopses WHERE branch_id = ?", (branch_id,)
            ).fetchone()
        return (row["text"], row["position"]) if row else ("", -1)

    def set_synopsis(
        self, project_id: str, branch_id: str, text: str, position: int
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO synopses (branch_id, project_id, position, text, updated_at) "
                "VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(branch_id) DO UPDATE SET "
                "position = excluded.position, text = excluded.text, "
                "updated_at = excluded.updated_at",
                (branch_id, project_id, position, text.strip(), _now()),
            )
            self._conn.commit()

    # -- row mapping -------------------------------------------------------

    @staticmethod
    def _row_to_project(row: sqlite3.Row) -> Project:
        return Project(
            id=row["id"],
            title=row["title"],
            logline=row["logline"],
            created_at=_parse_time(row["created_at"]),
            updated_at=_parse_time(row["updated_at"]),
        )

    @staticmethod
    def _row_to_branch(row: sqlite3.Row) -> Branch:
        return Branch(
            id=row["id"],
            project_id=row["project_id"],
            name=row["name"],
            parent_id=row["parent_id"],
            forked_at=row["forked_at"],
            created_at=_parse_time(row["created_at"]),
        )

    @staticmethod
    def _row_to_segment(row: sqlite3.Row) -> Segment:
        return Segment(
            id=row["id"],
            project_id=row["project_id"],
            branch_id=row["branch_id"],
            position=row["position"],
            title=row["title"],
            text=row["text"],
            word_count=row["word_count"],
            created_at=_parse_time(row["created_at"]),
        )

    @staticmethod
    def _row_to_fact(row: sqlite3.Row) -> Fact:
        return Fact(
            id=row["id"],
            project_id=row["project_id"],
            subject=row["subject"],
            claim=row["claim"],
            kind=row["kind"],
            established_branch=row["established_branch"],
            established_position=row["established_position"],
            superseded_branch=row["superseded_branch"],
            superseded_position=row["superseded_position"],
            superseded_by=row["superseded_by"],
            source_segment_id=row["source_segment_id"],
            quote=row["quote"],
            created_at=_parse_time(row["created_at"]),
        )
