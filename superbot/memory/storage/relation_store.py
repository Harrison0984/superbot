"""关系存储模块：使用 SQLite 存储结构化知识 - 分层设计"""
import sqlite3
import json
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime
from threading import local


class RelationStore:
    """关系存储：使用 SQLite - 分层设计 (raw_logs → memory_nodes → relationships)"""

    def __init__(self, db_path: str = "./data/memmory.db"):
        self.db_path = db_path
        self._local = local()  # 线程本地存储
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """获取线程本地的数据库连接"""
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self._local.conn = sqlite3.connect(self.db_path)
        return self._local.conn

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
        """添加原始日志"""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO raw_logs (content, source, incremental_density)
            VALUES (?, ?, ?)
        """, (content, source, incremental_density))

        conn.commit()
        return cursor.lastrowid

    def add_memory_node(
        self,
        raw_id: int,
        tag: str,
        summary: str,
        vector_id: str,
        entities: List[Dict] = None,
        facts: List[str] = None
    ) -> int:
        """添加记忆节点"""
        conn = self._get_conn()
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

    def add_relation(
        self,
        head: str,
        relation: str,
        tail: str,
        ref_id: int = None,
        strength: float = 1.0
    ) -> None:
        """添加实体关系"""
        conn = self._get_conn()
        cursor = conn.cursor()
        now = datetime.now().isoformat()

        cursor.execute("""
            INSERT INTO relationships (head, relation, tail, strength, last_seen, ref_id)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (head, relation, tail, strength, now, ref_id))

        conn.commit()

    def get_memory(self, memory_id: int) -> Optional[Dict[str, Any]]:
        """获取记忆节点"""
        conn = self._get_conn()
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

    def get_raw_log(self, raw_id: int) -> Optional[Dict[str, Any]]:
        """获取原始日志"""
        conn = self._get_conn()
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

    def get_relations(self, entity: str) -> List[Dict[str, Any]]:
        """获取与实体相关的关系"""
        conn = self._get_conn()
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


class EnhancedRelationStore(RelationStore):
    """增强版关系存储：添加触发式衰减"""

    # 指数衰减系数 λ=0.05，半衰期约14天
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
        触发式衰减的 Upsert 操作

        逻辑：
        1. 查找已存在的关系
        2. 如果存在：先衰减旧强度，再叠加增量
           S_new = S_old * exp(-λ * days) + increment
        3. 如果不存在：创建新关系，强度 = increment
        """
        conn = self._get_conn()
        cursor = conn.cursor()
        now = datetime.now().isoformat()

        # 使用触发式衰减的 Upsert
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
                increment, now, ref_id,  # INSERT 值
                increment, now            # UPDATE 值
            ))
            conn.commit()
        except Exception as e:
            # 如果 UPSERT 失败，回退到普通插入
            cursor.execute("""
                INSERT INTO relationships (head, relation, tail, strength, last_seen, ref_id)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (head, relation, tail, increment, now, ref_id))
            conn.commit()

    def get_entity_relations(self, entity: str) -> List[Dict[str, Any]]:
        """获取实体的所有关系（带时间衰减）"""
        conn = self._get_conn()
        cursor = conn.cursor()

        # 使用指数衰减计算有效强度
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

    def get_memory_with_raw(self, memory_id: int) -> Optional[Dict[str, Any]]:
        """获取记忆节点及其原始日志"""
        memory = self.get_memory(memory_id)
        if not memory:
            return None

        raw_id = memory.get("raw_id")
        if raw_id:
            raw_log = self.get_raw_log(raw_id)
            memory["raw_log"] = raw_log

        return memory
