"""
Bootstrap Unity Catalog objects for Project Anubhuti Lore Engine.

Creates:
  - Delta table: main.anubhuti.story_lore
  - Vector Search endpoint: anubhuti-lore-vs-endpoint
  - Vector Search index: main.anubhuti.lore_index
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

from databricks.sdk.errors import DatabricksError as SdkDatabricksError
from databricks.sdk.service.vectorsearch import (
    DeltaSyncVectorIndexSpecRequest,
    EmbeddingVectorColumn,
    EndpointStatusState,
    EndpointType,
    PipelineType,
    VectorIndexType,
)
from dotenv import load_dotenv

try:
    from .lore_manager import (
        CATALOG,
        EMBEDDING_DIMENSION,
        ENDPOINT_NAME,
        FULL_TABLE_NAME,
        INDEX_NAME,
        SCHEMA,
        DatabricksOperationError,
        build_workspace_client,
        execute_sql,
        resolve_sql_warehouse_id,
    )
except ImportError:  # pragma: no cover - direct script execution
    from lore_manager import (
        CATALOG,
        EMBEDDING_DIMENSION,
        ENDPOINT_NAME,
        FULL_TABLE_NAME,
        INDEX_NAME,
        SCHEMA,
        DatabricksOperationError,
        build_workspace_client,
        execute_sql,
        resolve_sql_warehouse_id,
    )

logger = logging.getLogger(__name__)

ENDPOINT_POLL_INTERVAL_SECONDS = 15
ENDPOINT_TIMEOUT_SECONDS = 1800
INDEX_POLL_INTERVAL_SECONDS = 15
INDEX_TIMEOUT_SECONDS = 1800


def ensure_schema(workspace, warehouse_id: str) -> None:
    logger.info("Ensuring schema %s.%s exists...", CATALOG, SCHEMA)
    execute_sql(
        workspace,
        warehouse_id,
        f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}",
    )


def ensure_story_lore_table(workspace, warehouse_id: str) -> None:
    logger.info("Ensuring Delta table %s exists...", FULL_TABLE_NAME)
    statement = f"""
CREATE TABLE IF NOT EXISTS {FULL_TABLE_NAME} (
  id STRING NOT NULL,
  character_id STRING NOT NULL,
  lore_text STRING NOT NULL,
  embedding ARRAY<FLOAT> NOT NULL
)
USING DELTA
TBLPROPERTIES (
  'delta.enableChangeDataFeed' = 'true'
)
COMMENT 'Project Anubhuti canonical story lore with OpenAI embeddings'
"""
    execute_sql(workspace, warehouse_id, statement)


def _enum_value(value) -> str | None:
    if value is None:
        return None
    return value.value if hasattr(value, "value") else str(value)


def _get_endpoint_state(workspace, endpoint_name: str) -> str | None:
    try:
        endpoint = workspace.vector_search_endpoints.get_endpoint(endpoint_name=endpoint_name)
    except SdkDatabricksError as exc:
        if any(
            token in str(exc).lower()
            for token in ("resource_does_not_exist", "does not exist", "not found")
        ):
            return None
        raise DatabricksOperationError(
            f"Failed to read vector search endpoint '{endpoint_name}': {exc}"
        ) from exc

    if endpoint.endpoint_status and endpoint.endpoint_status.state:
        return _enum_value(endpoint.endpoint_status.state)
    return None


def ensure_vector_search_endpoint(workspace, endpoint_name: str = ENDPOINT_NAME) -> None:
    state = _get_endpoint_state(workspace, endpoint_name)
    if state == EndpointStatusState.ONLINE.value:
        logger.info("Vector Search endpoint '%s' is already ONLINE.", endpoint_name)
        return

    if state:
        logger.info(
            "Vector Search endpoint '%s' exists with state=%s; waiting for ONLINE...",
            endpoint_name,
            state,
        )
    else:
        logger.info("Creating Vector Search endpoint '%s'...", endpoint_name)
        try:
            workspace.vector_search_endpoints.create_endpoint(
                name=endpoint_name,
                endpoint_type=EndpointType.STANDARD,
            )
        except SdkDatabricksError as exc:
            if "already exists" not in str(exc).lower():
                raise DatabricksOperationError(
                    f"Failed to create vector search endpoint '{endpoint_name}': {exc}"
                ) from exc
            logger.info("Endpoint '%s' already exists.", endpoint_name)

    deadline = time.time() + ENDPOINT_TIMEOUT_SECONDS
    while time.time() < deadline:
        state = _get_endpoint_state(workspace, endpoint_name)
        if state == EndpointStatusState.ONLINE.value:
            logger.info("Vector Search endpoint '%s' is ONLINE.", endpoint_name)
            return
        if state == EndpointStatusState.PROVISIONING_FAILED.value:
            raise DatabricksOperationError(
                f"Vector Search endpoint '{endpoint_name}' provisioning failed."
            )
        time.sleep(ENDPOINT_POLL_INTERVAL_SECONDS)

    raise DatabricksOperationError(
        f"Timed out waiting for vector search endpoint '{endpoint_name}' to become ONLINE."
    )


def _get_index_status(workspace, index_name: str) -> dict[str, object] | None:
    try:
        index = workspace.vector_search_indexes.get_index(index_name=index_name)
    except SdkDatabricksError as exc:
        if any(
            token in str(exc).lower()
            for token in ("resource_does_not_exist", "does not exist", "not found")
        ):
            return None
        raise DatabricksOperationError(
            f"Failed to read vector search index '{index_name}': {exc}"
        ) from exc

    status = index.status
    if not status:
        return {"ready": False, "message": "missing status"}

    return {
        "ready": bool(status.ready),
        "message": status.message or "",
    }


def _index_is_online(status: dict[str, object] | None) -> bool:
    return bool(status and status.get("ready"))


def _index_failed(status: dict[str, object] | None) -> bool:
    if not status:
        return False
    message = str(status.get("message", "")).lower()
    return "failed" in message or "error" in message


def ensure_vector_search_index(
    workspace,
    index_name: str = INDEX_NAME,
    endpoint_name: str = ENDPOINT_NAME,
    source_table: str = FULL_TABLE_NAME,
) -> None:
    status = _get_index_status(workspace, index_name)
    if _index_is_online(status):
        logger.info("Vector Search index '%s' is already online.", index_name)
        return

    if status:
        logger.info(
            "Vector Search index '%s' exists (ready=%s); waiting for ONLINE... message=%s",
            index_name,
            status.get("ready"),
            status.get("message"),
        )
    else:
        logger.info("Creating Vector Search index '%s'...", index_name)
        try:
            workspace.vector_search_indexes.create_index(
                name=index_name,
                endpoint_name=endpoint_name,
                primary_key="id",
                index_type=VectorIndexType.DELTA_SYNC,
                delta_sync_index_spec=DeltaSyncVectorIndexSpecRequest(
                    source_table=source_table,
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
                raise DatabricksOperationError(
                    f"Failed to create vector search index '{index_name}': {exc}"
                ) from exc
            logger.info("Index '%s' already exists.", index_name)

    deadline = time.time() + INDEX_TIMEOUT_SECONDS
    while time.time() < deadline:
        status = _get_index_status(workspace, index_name)
        if _index_is_online(status):
            logger.info("Vector Search index '%s' is online.", index_name)
            return
        if _index_failed(status):
            raise DatabricksOperationError(
                f"Vector Search index '{index_name}' provisioning failed: "
                f"{status.get('message') if status else 'unknown error'}"
            )
        time.sleep(INDEX_POLL_INTERVAL_SECONDS)

    raise DatabricksOperationError(
        f"Timed out waiting for vector search index '{index_name}' to come online."
    )


def bootstrap(
    profile: str = "DEFAULT",
    warehouse_id: str | None = None,
    skip_endpoint: bool = False,
    skip_index: bool = False,
) -> None:
    load_dotenv()

    workspace = build_workspace_client(profile=profile)
    resolved_warehouse_id = resolve_sql_warehouse_id(workspace, warehouse_id)

    ensure_schema(workspace, resolved_warehouse_id)
    ensure_story_lore_table(workspace, resolved_warehouse_id)

    if not skip_endpoint:
        ensure_vector_search_endpoint(workspace)

    if not skip_index:
        ensure_vector_search_index(workspace)

    logger.info("Project Anubhuti vector database bootstrap complete.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Initialize Databricks Delta table and Vector Search index for Anubhuti lore."
    )
    parser.add_argument(
        "--profile",
        default="DEFAULT",
        help="Databricks CLI profile (default: DEFAULT)",
    )
    parser.add_argument(
        "--warehouse-id",
        default=None,
        help="SQL warehouse ID (default: DATABRICKS_WAREHOUSE_ID or first RUNNING warehouse)",
    )
    parser.add_argument(
        "--skip-endpoint",
        action="store_true",
        help="Skip vector search endpoint creation",
    )
    parser.add_argument(
        "--skip-index",
        action="store_true",
        help="Skip vector search index creation",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    try:
        bootstrap(
            profile=args.profile,
            warehouse_id=args.warehouse_id,
            skip_endpoint=args.skip_endpoint,
            skip_index=args.skip_index,
        )
    except Exception as exc:
        logger.error("Bootstrap failed: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
