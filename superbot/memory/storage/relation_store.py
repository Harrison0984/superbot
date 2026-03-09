"""Relation storage module: SQLite-based structured knowledge storage - hierarchical design"""
import sqlite3
import json
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime


class RelationStore:
    """Relation storage using SQLite - hierarchical design (raw_logs -> memory_nodes -> relationships)"""

    def __init__(self, db_path: str = "./data/memmory.db"):
        self.db_path = db_path
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """Get a new database connection for each call.

        Note: SQLite connections are not thread-safe, so we create a new connection
        for each operation. This is the recommended approach for SQLite.
        """
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL")  # Enable WAL mode for better concurrency
        return conn

    def _init_db(self):
        """初始化数据库表"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 原始日志表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS raw_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                content TEXT NOT NULL,
                source TEXT,
                incremental_density REAL
            )
        """)

        # 记忆节点表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memory_nodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                raw_id INTEGER,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                tag TEXT,
                summary TEXT NOT NULL,
                vector_id TEXT,
                entities TEXT,
                facts TEXT,
                FOREIGN KEY (raw_id) REFERENCES raw_logs(id)
            )
        """)

        # 关系表 (head - relation - tail)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS relationships (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                head TEXT NOT NULL,
                relation TEXT NOT NULL,
                tail TEXT NOT NULL,
                strength REAL DEFAULT 1.0,
                last_seen DATETIME DEFAULT CURRENT_TIMESTAMP,
                ref_id INTEGER,
                UNIQUE(head, relation, tail)
            )
        """)

        # 索引
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_raw_logs_id ON raw_logs(id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_mem_nodes_vector ON memory_nodes(vector_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_rel_head_tail ON relationships(head, tail)")

        conn.commit()
        conn.close()

    def add_raw_log(
        self,
        content: str,
        source: str = None,
        incremental_density: float = None
    ) -> int:
        """Add raw log"""
        conn = self._get_conn()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO raw_logs (content, source, incremental_density)
                VALUES (?, ?, ?)
            """, (content, source, incremental_density))
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()

    def add_memory_node(
        self,
        raw_id: int,
        tag: str,
        summary: str,
        vector_id: str,
        entities: List[Dict] = None,
        facts: List[str] = None
    ) -> int:
        """Add memory node"""
        conn = self._get_conn()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO memory_nodes (raw_id, tag, summary, vector_id, entities, facts)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                raw_id,
                tag,
                summary,
                vector_id,
                json.dumps(entities) if entities else None,
                json.dumps(facts) if facts else None
            ))
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()

    def add_relation(
        self,
        head: str,
        relation: str,
        tail: str,
        ref_id: int = None,
        strength: float = 1.0
    ) -> None:
        """Add entity relation"""
        conn = self._get_conn()
        try:
            cursor = conn.cursor()
            now = datetime.now().isoformat()
            cursor.execute("""
                INSERT INTO relationships (head, relation, tail, strength, last_seen, ref_id)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (head, relation, tail, strength, now, ref_id))
            conn.commit()
        finally:
            conn.close()

    def get_memory(self, memory_id: int) -> Optional[Dict[str, Any]]:
        """Get memory node"""
        conn = self._get_conn()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM memory_nodes WHERE id = ?", (memory_id,))
            row = cursor.fetchone()

            if not row:
                return None

            return {
                "id": row[0],
                "raw_id": row[1],
                "timestamp": row[2],
                "tag": row[3],
                "summary": row[4],
                "vector_id": row[5],
                "entities": json.loads(row[6]) if row[6] else [],
                "facts": json.loads(row[7]) if row[7] else []
            }
        finally:
            conn.close()

    def get_raw_log(self, raw_id: int) -> Optional[Dict[str, Any]]:
        """Get raw log"""
        conn = self._get_conn()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM raw_logs WHERE id = ?", (raw_id,))
            row = cursor.fetchone()

            if not row:
                return None

            return {
                "id": row[0],
                "timestamp": row[1],
                "content": row[2],
                "source": row[3],
                "incremental_density": row[4]
            }
        finally:
            conn.close()

    def get_relations(self, entity: str) -> List[Dict[str, Any]]:
        """Get relations for an entity"""
        conn = self._get_conn()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM relationships
                WHERE head = ? OR tail = ?
            """, (entity, entity))

            rows = cursor.fetchall()

            results = []
            for row in rows:
                results.append({
                    "id": row[0],
                    "head": row[1],
                    "relation": row[2],
                    "tail": row[3],
                    "strength": row[4],
                    "last_seen": row[5],
                    "ref_id": row[6]
                })

            return results
        finally:
            conn.close()


class EnhancedRelationStore(RelationStore):
    """Enhanced relation storage with trigger-based decay"""

    # Exponential decay coefficient, half-life ~14 days
    LAMBDA = 0.05

    def upsert_relation(
        self,
        head: str,
        relation: str,
        tail: str,
        ref_id: int = None,
        increment: float = 0.5
    ) -> None:
        """
        Trigger-based decay upsert operation

        Logic:
        1. Find existing relation
        2. If exists: decay old strength first, then add increment
           S_new = S_old * exp(-λ * days) + increment
        3. If not exists: create new relation with strength = increment
        """
        conn = self._get_conn()
        try:
            cursor = conn.cursor()
            now = datetime.now().isoformat()

            # Use trigger-based decay upsert
            sql = f"""
                INSERT INTO relationships (head, relation, tail, strength, last_seen, ref_id)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(head, relation, tail) DO UPDATE SET
                    strength = (
                        SELECT strength * EXP(-{self.LAMBDA} * (
                            julianday('now') - julianday(COALESCE(last_seen, created_at))
                        ))
                        FROM relationships
                        WHERE head = excluded.head
                          AND relation = excluded.relation
                          AND tail = excluded.tail
                    ) + ?,
                    last_seen = ?,
                    ref_id = COALESCE(excluded.ref_id, ref_id);
            """

            try:
                cursor.execute(sql, (
                    head, relation, tail,
                    increment, now, ref_id,  # INSERT values
                    increment, now            # UPDATE values
                ))
                conn.commit()
            except Exception as e:
                # Fallback to regular insert if UPSERT fails
                cursor.execute("""
                    INSERT INTO relationships (head, relation, tail, strength, last_seen, ref_id)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (head, relation, tail, increment, now, ref_id))
                conn.commit()
        finally:
            conn.close()

    def get_entity_relations(self, entity: str) -> List[Dict[str, Any]]:
        """Get all relations for an entity (with time decay)"""
        conn = self._get_conn()
        try:
            cursor = conn.cursor()

            # Use exponential decay to calculate effective strength
            sql = """
                SELECT id, head, relation, tail,
                       strength * EXP(-? * (julianday('now') - julianday(COALESCE(last_seen, created_at)))) as effective_strength,
                       last_seen, ref_id
                FROM relationships
                WHERE head = ? OR tail = ?
                ORDER BY effective_strength DESC
                LIMIT 10;
            """

            cursor.execute(sql, (self.LAMBDA, entity, entity))
            rows = cursor.fetchall()

            results = []
            for row in rows:
                results.append({
                    "id": row[0],
                    "head": row[1],
                    "relation": row[2],
                    "tail": row[3],
                    "effective_strength": row[4],
                    "last_seen": row[5],
                    "ref_id": row[6]
                })

            return results
        finally:
            conn.close()

    def get_memory_with_raw(self, memory_id: int) -> Optional[Dict[str, Any]]:
        """Get memory node with its raw log"""
        memory = self.get_memory(memory_id)
        if not memory:
            return None

        raw_id = memory.get("raw_id")
        if raw_id:
            raw_log = self.get_raw_log(raw_id)
            memory["raw_log"] = raw_log

        return memory
