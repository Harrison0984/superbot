"""Experience store: action logs for tool execution history."""
import sqlite3
import json
from typing import List, Dict, Any, Optional
from datetime import datetime


class ExperienceStore:
    """Store for action logs - simple success rate calculation."""

    def __init__(self, db_path: str = "./data/experience.db"):
        self.db_path = db_path
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """Get a new database connection."""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self):
        """Initialize database tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Action logs table - records each tool execution
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS action_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                action_type TEXT NOT NULL,
                method TEXT NOT NULL,
                success INTEGER NOT NULL,
                quality REAL,
                time_cost REAL,
                context TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_action_logs_type ON action_logs(action_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_action_logs_timestamp ON action_logs(timestamp)")

        conn.commit()
        conn.close()

    # ==================== Action Logs ====================

    def record_action(
        self,
        action_type: str,
        method: str,
        success: bool,
        quality: float = None,
        time_cost: float = None,
        context: Dict = None,
    ) -> int:
        """Record a tool execution result."""
        conn = self._get_conn()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO action_logs (action_type, method, success, quality, time_cost, context)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                action_type,
                method,
                1 if success else 0,
                quality,
                time_cost,
                json.dumps(context) if context else None,
            ))
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()

    def get_action_logs(
        self,
        action_type: str = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get action logs, optionally filtered by action_type."""
        conn = self._get_conn()
        try:
            cursor = conn.cursor()
            if action_type:
                cursor.execute("""
                    SELECT id, action_type, method, success, quality, time_cost, context, timestamp
                    FROM action_logs
                    WHERE action_type = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (action_type, limit))
            else:
                cursor.execute("""
                    SELECT id, action_type, method, success, quality, time_cost, context, timestamp
                    FROM action_logs
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (limit,))

            rows = cursor.fetchall()
            results = []
            for row in rows:
                results.append({
                    "id": row[0],
                    "action_type": row[1],
                    "method": row[2],
                    "success": bool(row[3]),
                    "quality": row[4],
                    "time_cost": row[5],
                    "context": json.loads(row[6]) if row[6] else None,
                    "timestamp": row[7],
                })
            return results
        finally:
            conn.close()

    # ==================== Success Rate Calculation ====================

    def get_success_rate(self, action_type: str) -> Dict[str, Any]:
        """Calculate success rate for an action type directly from logs.

        Returns:
            {
                "action_type": str,
                "success_rate": float,  # success_count / total_count
                "total_count": int,
                "success_count": int,
                "last_success": str or None,  # timestamp of last success
            }
        """
        conn = self._get_conn()
        try:
            cursor = conn.cursor()

            # Get counts
            cursor.execute("""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as success_count,
                    MAX(CASE WHEN success = 1 THEN timestamp END) as last_success
                FROM action_logs
                WHERE action_type = ?
            """, (action_type,))

            row = cursor.fetchone()
            if not row or row[0] == 0:
                return {
                    "action_type": action_type,
                    "success_rate": 0.0,
                    "total_count": 0,
                    "success_count": 0,
                    "last_success": None,
                }

            total_count, success_count, last_success = row
            success_rate = success_count / total_count if total_count > 0 else 0.0

            return {
                "action_type": action_type,
                "success_rate": success_rate,
                "total_count": total_count,
                "success_count": success_count,
                "last_success": last_success,
            }
        finally:
            conn.close()

    def get_all_success_rates(self) -> Dict[str, Dict[str, Any]]:
        """Get success rates for all action types."""
        conn = self._get_conn()
        try:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT
                    action_type,
                    COUNT(*) as total,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as success_count,
                    MAX(CASE WHEN success = 1 THEN timestamp END) as last_success
                FROM action_logs
                GROUP BY action_type
            """)

            rows = cursor.fetchall()
            results = {}
            for row in rows:
                action_type, total, success_count, last_success = row
                success_rate = success_count / total if total > 0 else 0.0

                results[action_type] = {
                    "action_type": action_type,
                    "success_rate": success_rate,
                    "total_count": total,
                    "success_count": success_count,
                    "last_success": last_success,
                }
            return results
        finally:
            conn.close()
