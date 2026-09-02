from __future__ import annotations

import os
from collections.abc import Iterable

from neo4j import Driver, GraphDatabase


class Neo4jStore:
    """Small persistence/query layer for the GraphGuard investigation graph."""

    def __init__(self, uri: str | None = None, user: str | None = None, password: str | None = None) -> None:
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.getenv("NEO4J_USER", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD")
        if not self.password:
            raise ValueError("NEO4J_PASSWORD must be set")
        self.driver: Driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))

    def close(self) -> None:
        self.driver.close()

    def create_schema(self) -> None:
        with self.driver.session() as session:
            session.run(
                "CREATE CONSTRAINT transaction_id IF NOT EXISTS "
                "FOR (t:Transaction) REQUIRE t.tx_id IS UNIQUE"
            )
            session.run(
                "CREATE INDEX transaction_risk IF NOT EXISTS "
                "FOR (t:Transaction) ON (t.risk_score)"
            )

    def upsert_transactions(self, rows: Iterable[dict], batch_size: int = 5000) -> None:
        batch: list[dict] = []
        with self.driver.session() as session:
            for row in rows:
                batch.append(row)
                if len(batch) >= batch_size:
                    self._write_transactions(session, batch)
                    batch.clear()
            if batch:
                self._write_transactions(session, batch)

    @staticmethod
    def _write_transactions(session, rows: list[dict]) -> None:
        session.run(
            """
            UNWIND $rows AS row
            MERGE (t:Transaction {tx_id: row.tx_id})
            SET t.time_step = row.time_step,
                t.risk_score = row.risk_score
            """,
            rows=rows,
        )

    def upsert_edges(self, rows: Iterable[dict], batch_size: int = 5000) -> None:
        batch: list[dict] = []
        with self.driver.session() as session:
            for row in rows:
                batch.append(row)
                if len(batch) >= batch_size:
                    self._write_edges(session, batch)
                    batch.clear()
            if batch:
                self._write_edges(session, batch)

    @staticmethod
    def _write_edges(session, rows: list[dict]) -> None:
        session.run(
            """
            UNWIND $rows AS row
            MATCH (source:Transaction {tx_id: row.source_tx_id})
            MATCH (target:Transaction {tx_id: row.target_tx_id})
            MERGE (source)-[:PAYS_TO]->(target)
            """,
            rows=rows,
        )

    def get_transaction(self, tx_id: int) -> dict | None:
        with self.driver.session() as session:
            record = session.run(
                "MATCH (t:Transaction {tx_id: $tx_id}) RETURN t{.*} AS transaction",
                tx_id=tx_id,
            ).single()
            return record["transaction"] if record else None

    def get_suspicious_neighbors(self, tx_id: int, limit: int = 20) -> list[dict]:
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (t:Transaction {tx_id: $tx_id})-[:PAYS_TO]-(neighbor:Transaction)
                RETURN neighbor{.*} AS transaction
                ORDER BY neighbor.risk_score DESC
                LIMIT $limit
                """,
                tx_id=tx_id,
                limit=limit,
            )
            return [record["transaction"] for record in result]
