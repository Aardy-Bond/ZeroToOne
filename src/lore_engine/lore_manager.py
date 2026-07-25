"""
Lore Graph manager for Project Anubhuti.

Embeds story lore with OpenAI, persists to Delta Lake, and queries
Databricks Vector Search for continuity checks against established canon.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from typing import Any

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import DatabricksError as SdkDatabricksError
from databricks.sdk.service.sql import StatementState
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

logger = logging.getLogger(__name__)

# Unity Catalog + Vector Search identifiers
CATALOG = "main"
SCHEMA = "anubhuti"
TABLE_NAME = "story_lore"
FULL_TABLE_NAME = f"{CATALOG}.{SCHEMA}.{TABLE_NAME}"
INDEX_NAME = f"{CATALOG}.{SCHEMA}.lore_index"
ENDPOINT_NAME = "anubhuti-lore-vs-endpoint"

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536

DEFAULT_PROFILE = "DEFAULT"
WAREHOUSE_ENV = "DATABRICKS_WAREHOUSE_ID"
STATEMENT_WAIT_TIMEOUT = "50s"
INDEX_SYNC_TIMEOUT_SECONDS = 600
INDEX_SYNC_POLL_INTERVAL_SECONDS = 10


class LoreEngineError(Exception):
    """Base exception for Lore Engine operations."""


class ConfigurationError(LoreEngineError):
    """Raised when required configuration is missing or invalid."""


class EmbeddingError(LoreEngineError):
    """Raised when OpenAI embedding generation fails."""


class DatabricksOperationError(LoreEngineError):
    """Raised when a Databricks SDK or SQL operation fails."""


def _require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ConfigurationError(f"Missing required environment variable: {name}")
    return value


def build_workspace_client(
    host: str | None = None,
    token: str | None = None,
    profile: str = DEFAULT_PROFILE,
) -> WorkspaceClient:
    """Create a WorkspaceClient from explicit credentials or CLI profile."""
    resolved_host = (host or os.getenv("DATABRICKS_HOST", "")).strip() or None
    resolved_token = (token or os.getenv("DATABRICKS_TOKEN", "")).strip() or None

    try:
        if resolved_token:
            if not resolved_host:
                raise ConfigurationError(
                    "DATABRICKS_HOST is required when DATABRICKS_TOKEN is set."
                )
            return WorkspaceClient(host=resolved_host, token=resolved_token)

        if resolved_host:
            return WorkspaceClient(host=resolved_host, profile=profile)

        return WorkspaceClient(profile=profile)
    except SdkDatabricksError as exc:
        raise DatabricksOperationError(
            f"Failed to initialize Databricks WorkspaceClient: {exc}"
        ) from exc


def resolve_sql_warehouse_id(
    workspace: WorkspaceClient,
    warehouse_id: str | None = None,
) -> str:
    """Resolve a SQL warehouse ID from env or the first running warehouse."""
    if warehouse_id:
        return warehouse_id

    env_warehouse = os.getenv(WAREHOUSE_ENV, "").strip()
    if env_warehouse:
        return env_warehouse

    try:
        warehouses = list(workspace.warehouses.list())
    except SdkDatabricksError as exc:
        raise DatabricksOperationError(
            f"Unable to list SQL warehouses: {exc}"
        ) from exc

    running = [
        wh
        for wh in warehouses
        if wh.id and wh.state and wh.state.value == "RUNNING"
    ]
    if running:
        logger.info("Using SQL warehouse: %s (%s)", running[0].name, running[0].id)
        return running[0].id

    starting = [
        wh
        for wh in warehouses
        if wh.id and wh.state and wh.state.value in {"STARTING", "STOPPED"}
    ]
    if starting:
        raise DatabricksOperationError(
            "Found SQL warehouse(s) but none are RUNNING. "
            f"Start warehouse '{starting[0].name}' ({starting[0].id}) or set "
            f"{WAREHOUSE_ENV}."
        )

    raise DatabricksOperationError(
        "No SQL warehouse available. Create one in Databricks SQL and set "
        f"{WAREHOUSE_ENV}, or start an existing warehouse."
    )


def execute_sql(
    workspace: WorkspaceClient,
    warehouse_id: str,
    statement: str,
    *,
    wait_timeout: str = STATEMENT_WAIT_TIMEOUT,
) -> None:
    """Execute a SQL statement on a SQL warehouse and raise on failure."""
    try:
        response = workspace.statement_execution.execute_statement(
            warehouse_id=warehouse_id,
            statement=statement,
            wait_timeout=wait_timeout,
        )
    except SdkDatabricksError as exc:
        raise DatabricksOperationError(f"SQL execution failed: {exc}") from exc

    status = response.status
    if not status:
        raise DatabricksOperationError("SQL execution returned no status.")

    if status.state == StatementState.SUCCEEDED:
        return

    error_message = status.error.message if status.error else "Unknown SQL error"
    raise DatabricksOperationError(
        f"SQL statement failed ({status.state.value}): {error_message}\n"
        f"Statement: {statement[:500]}"
    )


def _enum_value(value) -> str | None:
    if value is None:
        return None
    return value.value if hasattr(value, "value") else str(value)


class LoreGraph:
    """
    Manages story lore ingestion and continuity retrieval for Project Anubhuti.

    Uses OpenAI embeddings and Databricks Delta + Vector Search.
    """

    def __init__(
        self,
        profile: str = DEFAULT_PROFILE,
        warehouse_id: str | None = None,
        load_env_file: bool = True,
    ) -> None:
        if load_env_file:
            load_dotenv()

        self.host = os.getenv("DATABRICKS_HOST", "").strip() or None
        self.token = os.getenv("DATABRICKS_TOKEN", "").strip() or None
        self.openai_api_key = _require_env("OPENAI_API_KEY")

        try:
            self.openai_client = OpenAI(api_key=self.openai_api_key)
        except OpenAIError as exc:
            raise ConfigurationError(
                f"Failed to initialize OpenAI client: {exc}"
            ) from exc

        self.workspace = build_workspace_client(
            host=self.host,
            token=self.token,
            profile=profile,
        )
        self.warehouse_id = resolve_sql_warehouse_id(self.workspace, warehouse_id)

        logger.info(
            "LoreGraph initialized (table=%s, index=%s, warehouse=%s)",
            FULL_TABLE_NAME,
            INDEX_NAME,
            self.warehouse_id,
        )

    def _embed_text(self, text: str) -> list[float]:
        cleaned = text.strip()
        if not cleaned:
            raise EmbeddingError("Cannot embed empty text.")

        try:
            response = self.openai_client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=cleaned,
            )
        except OpenAIError as exc:
            raise EmbeddingError(
                f"OpenAI embedding request failed for model '{EMBEDDING_MODEL}': {exc}"
            ) from exc

        embedding = response.data[0].embedding
        if len(embedding) != EMBEDDING_DIMENSION:
            raise EmbeddingError(
                f"Expected {EMBEDDING_DIMENSION}-dim embedding, got {len(embedding)}."
            )
        return embedding

    @staticmethod
    def _sql_string_literal(value: str) -> str:
        return "'" + value.replace("'", "''") + "'"

    @staticmethod
    def _sql_array_literal(values: list[float]) -> str:
        return "array(" + ", ".join(repr(float(v)) for v in values) + ")"

    def _execute_sql(self, statement: str) -> None:
        execute_sql(self.workspace, self.warehouse_id, statement)

    def _trigger_index_sync(self) -> None:
        """Trigger a Delta Sync refresh so newly ingested lore is searchable."""
        deadline = time.time() + INDEX_SYNC_TIMEOUT_SECONDS
        sync_requested = False

        while time.time() < deadline:
            if not sync_requested:
                try:
                    self.workspace.vector_search_indexes.sync_index(
                        index_name=INDEX_NAME
                    )
                    sync_requested = True
                    logger.info("Triggered vector index sync for '%s'.", INDEX_NAME)
                except SdkDatabricksError as exc:
                    err = str(exc).lower()
                    if "not ready to sync" in err or "waiting_for_resources" in err:
                        logger.debug(
                            "Index sync not ready yet; waiting for pipeline: %s", exc
                        )
                    else:
                        raise DatabricksOperationError(
                            f"Failed to trigger vector index sync for '{INDEX_NAME}': {exc}"
                        ) from exc

            try:
                index = self.workspace.vector_search_indexes.get_index(
                    index_name=INDEX_NAME
                )
            except SdkDatabricksError as exc:
                raise DatabricksOperationError(
                    f"Failed to poll index status for '{INDEX_NAME}': {exc}"
                ) from exc

            status = index.status
            message = (status.message if status and status.message else "").lower()
            if "failed" in message and "creation succeeded" not in message:
                detail = status.message if status else "Unknown provisioning failure"
                raise DatabricksOperationError(
                    f"Vector index '{INDEX_NAME}' provisioning failed: {detail}"
                )

            if status and status.ready and sync_requested:
                if "re-syncing" not in message and "pending" not in message:
                    logger.info("Vector index '%s' is online after sync.", INDEX_NAME)
                    return

            time.sleep(INDEX_SYNC_POLL_INTERVAL_SECONDS)

        logger.warning(
            "Index sync for '%s' did not fully confirm within timeout; "
            "data is in Delta and will become searchable after pipeline completion.",
            INDEX_NAME,
        )

    def ingest_lore(
        self, character_id: str, lore_text: str, *, sync_index: bool = True
    ) -> str:
        """
        Embed lore text and persist it to the Delta table.

        Returns the generated lore record ID.
        """
        character_id = character_id.strip()
        lore_text = lore_text.strip()
        if not character_id:
            raise ValueError("character_id must not be empty.")
        if not lore_text:
            raise ValueError("lore_text must not be empty.")

        lore_id = str(uuid.uuid4())
        embedding = self._embed_text(lore_text)

        statement = (
            f"INSERT INTO {FULL_TABLE_NAME} "
            f"(id, character_id, lore_text, embedding) VALUES ("
            f"{self._sql_string_literal(lore_id)}, "
            f"{self._sql_string_literal(character_id)}, "
            f"{self._sql_string_literal(lore_text)}, "
            f"{self._sql_array_literal(embedding)}"
            f")"
        )

        logger.info("Ingesting lore for character_id=%s (id=%s)", character_id, lore_id)
        self._execute_sql(statement)
        if sync_index:
            self._trigger_index_sync()
        return lore_id

    def check_continuity(self, scene_text: str, top_k: int = 3) -> list[dict[str, Any]]:
        """
        Retrieve the most relevant established lore for a draft scene.

        Returns up to `top_k` records with id, character_id, lore_text, and score.
        """
        scene_text = scene_text.strip()
        if not scene_text:
            raise ValueError("scene_text must not be empty.")
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        query_vector = self._embed_text(scene_text)

        try:
            results = self.workspace.vector_search_indexes.query_index(
                index_name=INDEX_NAME,
                columns=["id", "character_id", "lore_text"],
                query_vector=query_vector,
                num_results=top_k,
            )
        except SdkDatabricksError as exc:
            raise DatabricksOperationError(
                f"Vector Search query failed for index '{INDEX_NAME}': {exc}"
            ) from exc

        if not results.result or not results.result.data_array:
            logger.info("No continuity matches found for scene.")
            return []

        matches: list[dict[str, Any]] = []
        for row in results.result.data_array:
            if len(row) < 4:
                logger.warning("Unexpected vector search row shape: %s", row)
                continue

            matches.append(
                {
                    "id": row[0],
                    "character_id": row[1],
                    "lore_text": row[2],
                    "score": float(row[3]),
                }
            )

        logger.info("Continuity check returned %d match(es).", len(matches))
        return matches

    def check_continuity_json(self, scene_text: str, top_k: int = 3) -> str:
        """Convenience helper that returns continuity matches as JSON."""
        return json.dumps(self.check_continuity(scene_text, top_k=top_k), indent=2)
