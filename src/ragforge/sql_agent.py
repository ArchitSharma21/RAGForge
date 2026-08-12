from __future__ import annotations

import re
from typing import Any

import duckdb
import pandas as pd

from .llm import GeminiGateway
from .security import validate_readonly_sql


class SQLWorkspace:
    def __init__(self):
        self.conn = duckdb.connect(database=":memory:")
        self.tables: list[str] = []

    @staticmethod
    def _safe_table_name(name: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9_]", "_", name).strip("_")
        if not cleaned or cleaned[0].isdigit():
            cleaned = "t_" + cleaned
        return cleaned[:80].lower()

    def add_dataframe(self, name: str, df: pd.DataFrame) -> str:
        table = self._safe_table_name(name)
        base = table
        suffix = 2
        while table in self.tables:
            table = f"{base}_{suffix}"
            suffix += 1
        view = f"_df_{len(self.tables)}"
        self.conn.register(view, df)
        self.conn.execute(f'CREATE TABLE "{table}" AS SELECT * FROM "{view}"')
        self.conn.unregister(view)
        self.tables.append(table)
        return table

    def schema_text(self) -> str:
        pieces = []
        for table in self.tables:
            rows = self.conn.execute(f'DESCRIBE "{table}"').fetchall()
            cols = ", ".join(f"{r[0]} {r[1]}" for r in rows)
            pieces.append(f"{table}({cols})")
        return "\n".join(pieces)


    def analytics_context(self, max_rows: int = 20) -> tuple[str, list[dict[str, Any]]]:
        """Build deterministic table evidence for corpus-level analytical synthesis.

        This uses no LLM call. It exposes schema, bounded rows, numeric ranges and
        categorical/boolean distributions so the normal grounded generation step
        can synthesize trends across documents and structured data together.
        """
        if not self.tables:
            return "", []
        blocks: list[str] = []
        sources: list[dict[str, Any]] = []
        for idx, table in enumerate(self.tables, start=1):
            df = self.conn.execute(f'SELECT * FROM "{table}" LIMIT {max(1, int(max_rows))}').fetchdf()
            desc_rows = self.conn.execute(f'DESCRIBE "{table}"').fetchall()
            schema = ", ".join(f"{row[0]} {row[1]}" for row in desc_rows)
            insights: list[str] = []
            for col in df.columns:
                series = df[col].dropna()
                if series.empty:
                    continue
                if pd.api.types.is_bool_dtype(series):
                    counts = series.astype(str).value_counts().to_dict()
                    insights.append(f"{col}: values={counts}")
                elif pd.api.types.is_numeric_dtype(series):
                    insights.append(
                        f"{col}: min={series.min()}, max={series.max()}, mean={round(float(series.mean()), 3)}"
                    )
                elif series.nunique(dropna=True) <= 8:
                    counts = series.astype(str).value_counts().to_dict()
                    insights.append(f"{col}: values={counts}")
            preview = df.to_markdown(index=False) if len(df) else "(no rows)"
            summary = "; ".join(insights[:12]) or "No compact descriptive statistics available."
            block = (
                f"[T{idx}] TABLE: {table}\n"
                f"SCHEMA: {schema}\n"
                f"DESCRIPTIVE SIGNALS: {summary}\n"
                f"ROWS (bounded preview):\n{preview}"
            )
            blocks.append(block)
            sources.append(
                {
                    "id": f"T{idx}",
                    "type": "table",
                    "title": table,
                    "rows": int(self.conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]),
                    "schema": schema,
                    "snippet": f"{summary}\n{preview}"[:1600],
                }
            )
        return "\n\n".join(blocks), sources

    def generate_sql(self, question: str, gateway: GeminiGateway) -> str:
        if not self.tables:
            raise ValueError("No CSV/XLSX tables are loaded in this session")
        prompt = f"""You write DuckDB SQL for a read-only analytics assistant.
Available tables:\n{self.schema_text()}
Question: {question}
Return JSON with keys sql and rationale. The SQL must be a single SELECT or WITH query. Never modify data."""
        data = gateway.complete_json(prompt, {"sql": "", "rationale": ""})
        return validate_readonly_sql(str(data.get("sql", "")))

    def execute_sql(self, sql: str) -> pd.DataFrame:
        validated = validate_readonly_sql(sql)
        return self.conn.execute(validated).fetchdf()

    def benchmark_query(self, question: str, gateway: GeminiGateway) -> tuple[str, pd.DataFrame]:
        """Generate and execute SQL with one LLM call for component evaluation.

        Routing is evaluated separately by the semantic-planner benchmark. The
        Text2SQL component benchmark therefore avoids an extra planner call and
        a second natural-language answer-generation call.
        """
        sql = self.generate_sql(question, gateway)
        return sql, self.execute_sql(sql)

    def ask(self, question: str, gateway: GeminiGateway) -> tuple[str, str, list[dict[str, Any]]]:
        sql = self.generate_sql(question, gateway)
        result = self.execute_sql(sql)
        preview = result.head(200)
        result_md = preview.to_markdown(index=False) if len(preview) else "(no rows)"
        answer_prompt = f"""Answer the user's data question using the SQL result below.
Question: {question}\nSQL: {sql}\nResult:\n{result_md}
Mention the computed result clearly. Do not invent values outside the table."""
        answer = gateway.complete(answer_prompt)
        sources = [{"id": "SQL1", "type": "sql", "title": "DuckDB query", "sql": sql, "rows": len(result)}]
        return answer, sql, sources
