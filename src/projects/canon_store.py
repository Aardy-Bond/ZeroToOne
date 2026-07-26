"""
Passage canon in Databricks Vector Search, scoped to one branch's lineage.

This is the texture layer: prose chunks that answer "how does this character
talk", "what did that room feel like", "was there a callback here". State and
plot-hole logic live in the fact ledger instead, because similarity cannot
represent a claim that stopped being true.

Deliberately a *new* table rather than a change to `main.anubhuti.story_lore`.
The existing table has no project or branch column, and adding one would mean
rebuilding the Vector Search index that the Writers Room continuity path
depends on. `src/lore_engine/` is left exactly as it is.

Retrieval mirrors `visibility.py`: a server-side filter narrows to the lineage,
then a client-side trim drops ancestor rows at or past their fork point. The
server cannot express "different upper bound per ancestor" in one filter, so
the cheap part runs remotely and the exact part runs locally.
"""

from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import DatabricksError as SdkDatabricksError
from databricks.sdk.service.vectorsearch import (
    DeltaSyncVectorIndexSpecRequest,
    EmbeddingVectorColumn,
    PipelineType,
    VectorIndexType,
)
from openai import OpenAI

from lore_engine.lore_manager import (
    CATALOG,
    EMBEDDING_DIMENSION,
    ENDPOINT_NAME,
    SCHEMA,
    build_workspace_client,
    execute_sql,
    resolve_sql_warehouse_id,
)

from .facts import embed_texts
from .visibility import UNBOUNDED, BranchGraph, Event

logger = logging.getLogger(__name__)

CANON_TABLE = f"{CATALOG}.{SCHEMA}.project_canon"
CANON_INDEX = f"{CATALOG}.{SCHEMA}.project_canon_index"

# Over-fetch before the client-side fork trim, since the server filter is a
# superset of what is actually visible.
OVERFETCH = 4


class CanonStoreError(Exception):
    """Raised when the passage store cannot be reached or written."""


@dataclass
class Passage:
    """One retrieved chunk of earlier prose."""

    id: str
    project_id: str
    branch_id: str
    position: int
    character_id: str
    text: str
    score: float = 0.0


def _sql_str(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _sql_array(values) -> str:
    return "array(" + ", ".join(repr(float(v)) for v in values) + ")"


class CanonStore:
    """Branch-scoped passage storage and retrieval."""

    def __init__(
        self,
        workspace: WorkspaceClient | None = None,
        openai_client: OpenAI | None = None,
        warehouse_id: str | None = None,
    ) -> None:
        try:
            self.workspace = workspace or build_workspace_client()
            self.warehouse_id = resolve_sql_warehouse_id(self.workspace, warehouse_id)
        except Exception as exc:
            raise CanonStoreError(f"Databricks is unavailable: {exc}") from exc

        self.openai = openai_client
        self._sync_thread: threading.Thread | None = None

    # -- bootstrap ---------------------------------------------------------

    def ensure_table(self) -> None:
        execute_sql(
            self.workspace,
            self.warehouse_id,
            f"""
CREATE TABLE IF NOT EXISTS {CANON_TABLE} (
  id           STRING NOT NULL,
  project_id   STRING NOT NULL,
  branch_id    STRING NOT NULL,
  position     INT    NOT NULL,
  character_id STRING NOT NULL,
  canon_text   STRING NOT NULL,
  embedding    ARRAY<FLOAT> NOT NULL
)
USING DELTA
TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')
COMMENT 'Project Anubhuti per-branch story passages with OpenAI embeddings'
""",
        )

    def ensure_index(self) -> bool:
        """Create the Delta Sync index if missing. True when it is queryable."""
        try:
            index = self.workspace.vector_search_indexes.get_index(
                index_name=CANON_INDEX
            )
            return bool(index.status and index.status.ready)
        except SdkDatabricksError:
            pass

        logger.info("Creating vector index %s", CANON_INDEX)
        try:
            self.workspace.vector_search_indexes.create_index(
                name=CANON_INDEX,
                endpoint_name=ENDPOINT_NAME,
                primary_key="id",
                index_type=VectorIndexType.DELTA_SYNC,
                delta_sync_index_spec=DeltaSyncVectorIndexSpecRequest(
                    source_table=CANON_TABLE,
                    embedding_vector_columns=[
                        EmbeddingVectorColumn(
                            name="embedding",
                            embedding_dimension=EMBEDDING_DIMENSION,
                        )
                    ],
                    pipeline_type=PipelineType.TRIGGERED,
                ),
            )
        except SdkDatabricksError as exc:
            if "already exists" not in str(exc).lower():
                raise CanonStoreError(f"Could not create the canon index: {exc}") from exc

        return False

    def bootstrap(self) -> bool:
        self.ensure_table()
        return self.ensure_index()

    # -- ingest ------------------------------------------------------------

    def ingest_passages(
        self,
        project_id: str,
        branch_id: str,
        position: int,
        passages: list[tuple[str, str]],
        *,
        sync: bool = True,
        block: bool = False,
    ) -> int:
        """
        Write one part's passages as a single statement.

        `passages` is (character_id, text). Everything goes in one multi-row
        INSERT and triggers at most one index sync, because the per-row path in
        `LoreGraph.ingest_lore` blocks for up to ten minutes polling the
        pipeline and would freeze the UI on a long story.
        """
        rows = [(c, t.strip()) for c, t in passages if t and t.strip()]
        if not rows:
            return 0

        if self.openai is None:
            raise CanonStoreError("An OpenAI client is required to embed passages.")

        vectors = embed_texts([t for _, t in rows], client=self.openai)

        values = ",\n".join(
            f"({_sql_str(uuid.uuid4().hex)}, {_sql_str(project_id)}, "
            f"{_sql_str(branch_id)}, {int(position)}, {_sql_str(character)}, "
            f"{_sql_str(text)}, {_sql_array(vector)})"
            for (character, text), vector in zip(rows, vectors)
        )

        execute_sql(
            self.workspace,
            self.warehouse_id,
            f"INSERT INTO {CANON_TABLE} "
            f"(id, project_id, branch_id, position, character_id, canon_text, embedding) "
            f"VALUES\n{values}",
        )

        logger.info(
            "Wrote %d passage(s) for %s/%s at position %d",
            len(rows),
            project_id,
            branch_id,
            position,
        )

        if sync:
            self.trigger_sync(block=block)
        return len(rows)

    def trigger_sync(self, *, block: bool = False) -> None:
        """
        Refresh the index. Off the UI thread unless explicitly told otherwise.

        A triggered Delta Sync pipeline takes minutes. The writer should be
        able to keep working, and the fact ledger -- which is what continuity
        checking actually depends on -- is already queryable from SQLite.
        """

        def run() -> None:
            try:
                self.workspace.vector_search_indexes.sync_index(index_name=CANON_INDEX)
                logger.info("Canon index sync requested.")
            except SdkDatabricksError as exc:
                logger.warning("Canon index sync could not be triggered: %s", exc)

        if block:
            run()
            return

        thread = threading.Thread(target=run, name="canon-sync", daemon=True)
        thread.start()
        self._sync_thread = thread

    def sync_in_progress(self) -> bool:
        return self._sync_thread is not None and self._sync_thread.is_alive()

    # -- retrieval ---------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        project_id: str,
        branch_id: str,
        position: int,
        graph: BranchGraph,
        top_k: int = 6,
    ) -> list[Passage]:
        """
        Passages from this branch's lineage, returned in story order.

        Similarity decides what is included; position decides how it reads. A
        model handed a score-ordered shuffle of a timeline loses the thread,
        so the final sort is always chronological.
        """
        query = query.strip()
        if not query:
            return []
        if self.openai is None:
            raise CanonStoreError("An OpenAI client is required to search passages.")

        cutoff = graph.lineage_cutoff(branch_id)
        if not cutoff:
            return []

        vector = embed_texts([query], client=self.openai)[0]

        import json

        filters = {
            "project_id": project_id,
            "branch_id": list(cutoff.keys()),
        }
        if position < UNBOUNDED:
            filters["position <"] = position

        try:
            response = self.workspace.vector_search_indexes.query_index(
                index_name=CANON_INDEX,
                columns=[
                    "id",
                    "project_id",
                    "branch_id",
                    "position",
                    "character_id",
                    "canon_text",
                ],
                query_vector=vector,
                filters_json=json.dumps(filters),
                num_results=top_k * OVERFETCH,
            )
        except SdkDatabricksError as exc:
            raise CanonStoreError(f"Canon search failed: {exc}") from exc

        if not response.result or not response.result.data_array:
            return []

        found: list[Passage] = []
        for row in response.result.data_array:
            if len(row) < 7:
                continue
            passage = Passage(
                id=row[0],
                project_id=row[1],
                branch_id=row[2],
                position=int(row[3]),
                character_id=row[4],
                text=row[5],
                score=float(row[6]),
            )
            # The server filter is a superset: it cannot express a different
            # upper bound per ancestor, so the fork trim happens here.
            if not graph.visible(
                Event(passage.branch_id, passage.position), branch_id, position
            ):
                continue
            found.append(passage)

        found.sort(key=lambda p: -p.score)
        found = found[:top_k]
        found.sort(key=lambda p: p.position)
        return found

    def delete_branch(self, project_id: str, branch_id: str) -> None:
        execute_sql(
            self.workspace,
            self.warehouse_id,
            f"DELETE FROM {CANON_TABLE} WHERE project_id = {_sql_str(project_id)} "
            f"AND branch_id = {_sql_str(branch_id)}",
        )

    def delete_project(self, project_id: str) -> None:
        execute_sql(
            self.workspace,
            self.warehouse_id,
            f"DELETE FROM {CANON_TABLE} WHERE project_id = {_sql_str(project_id)}",
        )
