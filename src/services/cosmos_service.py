"""Azure Cosmos DB service for investigation state persistence."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from azure.cosmos import CosmosClient, PartitionKey, exceptions

from src.config.settings import get_settings

logger = logging.getLogger(__name__)


class CosmosService:
    """Manages investigation state and history in Azure Cosmos DB."""

    def __init__(self) -> None:
        settings = get_settings()
        self._client = CosmosClient(settings.azure_cosmos_endpoint, settings.azure_cosmos_key)
        self._database = self._client.create_database_if_not_exists(settings.azure_cosmos_database)
        self._container = self._database.create_container_if_not_exists(
            id=settings.azure_cosmos_container,
            partition_key=PartitionKey(path="/investigation_id"),
        )

    def save_investigation(self, investigation_id: str, data: dict[str, Any]) -> dict[str, Any]:
        """Create or update an investigation record."""
        item = {
            "id": investigation_id,
            "investigation_id": investigation_id,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            **data,
        }
        result: dict[str, Any] = self._container.upsert_item(item)
        logger.info("Saved investigation: %s", investigation_id)
        return result

    def get_investigation(self, investigation_id: str) -> dict[str, Any] | None:
        """Retrieve an investigation by ID."""
        try:
            item: dict[str, Any] = self._container.read_item(
                investigation_id, partition_key=investigation_id
            )
            return item
        except exceptions.CosmosResourceNotFoundError:
            return None

    def list_investigations(self, limit: int = 50) -> list[dict[str, Any]]:
        """List recent investigations."""
        query = "SELECT * FROM c ORDER BY c.updated_at DESC OFFSET 0 LIMIT @limit"
        items = self._container.query_items(
            query=query,
            parameters=[{"name": "@limit", "value": limit}],
            enable_cross_partition_query=True,
        )
        return list(items)

    def save_agent_trace(
        self, investigation_id: str, agent_name: str, trace_data: dict[str, Any]
    ) -> None:
        """Save an agent execution trace for observability."""
        trace_item = {
            "id": f"{investigation_id}__trace__{agent_name}__{datetime.now(timezone.utc).timestamp()}",
            "investigation_id": investigation_id,
            "type": "agent_trace",
            "agent_name": agent_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **trace_data,
        }
        self._container.upsert_item(trace_item)
        logger.debug("Saved trace for agent %s in investigation %s", agent_name, investigation_id)

    def delete_investigation(self, investigation_id: str) -> None:
        """Delete an investigation and all its traces."""
        query = "SELECT c.id FROM c WHERE c.investigation_id = @inv_id"
        items = self._container.query_items(
            query=query,
            parameters=[{"name": "@inv_id", "value": investigation_id}],
            enable_cross_partition_query=True,
        )
        for item in items:
            self._container.delete_item(item["id"], partition_key=investigation_id)
        logger.info("Deleted investigation: %s", investigation_id)
