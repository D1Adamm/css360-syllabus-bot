"""The migration files and schema.sql agree with each other.

`schema.sql` creates the current schema in full for a fresh database;
`db/migrations/*.sql` upgrade a database that already has data. The README
promises that the two produce the same result and that every migration is
idempotent. Those are properties of text files, and text files drift, so they
are asserted here rather than remembered.

No database is involved. The checks are syntactic: every statement in a
migration is of a form that is safe to re-run, and every object a migration
creates is also created by schema.sql. Whether the SQL *runs* is proven by
applying it to a real PostgreSQL, which the deployment runbook does and which
the opt-in integration suite does when TEST_DATABASE_URL is set.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

DB_DIR = Path(__file__).resolve().parent.parent / "db"
SCHEMA = DB_DIR / "schema.sql"
MIGRATIONS = sorted((DB_DIR / "migrations").glob("*.sql"))

CREATE_TABLE = re.compile(r"CREATE TABLE IF NOT EXISTS (\w+)", re.IGNORECASE)
CREATE_INDEX = re.compile(r"CREATE (?:UNIQUE )?INDEX IF NOT EXISTS (\w+)", re.IGNORECASE)
ADD_COLUMN = re.compile(
    r"ALTER TABLE (\w+)\s+ADD COLUMN IF NOT EXISTS (\w+)", re.IGNORECASE
)

IDEMPOTENT_STATEMENT = re.compile(
    r"^\s*(?:"
    r"CREATE TABLE IF NOT EXISTS"
    r"|CREATE (?:UNIQUE )?INDEX IF NOT EXISTS"
    r"|ALTER TABLE \w+\s+ADD COLUMN IF NOT EXISTS"
    r")",
    re.IGNORECASE,
)


def _strip_comments(sql: str) -> str:
    return "\n".join(
        line for line in sql.splitlines() if not line.strip().startswith("--")
    )


def _statements(sql: str) -> list[str]:
    return [part.strip() for part in _strip_comments(sql).split(";") if part.strip()]


class MigrationIdempotencyTests(unittest.TestCase):
    def test_there_is_at_least_one_migration(self) -> None:
        self.assertGreaterEqual(len(MIGRATIONS), 2)

    def test_every_migration_statement_is_safe_to_re_run(self) -> None:
        """Re-running a migration on the VM must be a no-op, never an error."""
        for path in MIGRATIONS:
            for statement in _statements(path.read_text(encoding="utf-8")):
                with self.subTest(migration=path.name, statement=statement[:60]):
                    self.assertRegex(statement, IDEMPOTENT_STATEMENT)

    def test_no_migration_drops_or_rewrites_anything(self) -> None:
        forbidden = re.compile(
            r"\b(DROP|TRUNCATE|DELETE\s+FROM|UPDATE\s+\w+\s+SET|ALTER\s+COLUMN)\b",
            re.IGNORECASE,
        )
        for path in MIGRATIONS:
            with self.subTest(migration=path.name):
                self.assertIsNone(
                    forbidden.search(_strip_comments(path.read_text(encoding="utf-8")))
                )


class SchemaMirrorsMigrationsTests(unittest.TestCase):
    """A fresh database from schema.sql must equal an upgraded one."""

    def setUp(self) -> None:
        self.schema_sql = _strip_comments(SCHEMA.read_text(encoding="utf-8"))

    def test_every_migrated_table_is_in_schema_sql(self) -> None:
        schema_tables = set(CREATE_TABLE.findall(self.schema_sql))
        for path in MIGRATIONS:
            for table in CREATE_TABLE.findall(_strip_comments(path.read_text())):
                with self.subTest(migration=path.name, table=table):
                    self.assertIn(table, schema_tables)

    def test_every_migrated_index_is_in_schema_sql(self) -> None:
        schema_indexes = set(CREATE_INDEX.findall(self.schema_sql))
        for path in MIGRATIONS:
            for index in CREATE_INDEX.findall(_strip_comments(path.read_text())):
                with self.subTest(migration=path.name, index=index):
                    self.assertIn(index, schema_indexes)

    def test_every_migrated_column_exists_in_schema_sql(self) -> None:
        """Either inline in the CREATE TABLE or as the same ALTER at the end."""
        for path in MIGRATIONS:
            for table, column in ADD_COLUMN.findall(_strip_comments(path.read_text())):
                with self.subTest(migration=path.name, table=table, column=column):
                    inline = re.search(
                        rf"CREATE TABLE IF NOT EXISTS {table}\s*\((.*?)\n\);",
                        self.schema_sql,
                        re.IGNORECASE | re.DOTALL,
                    )
                    inline_has_it = bool(
                        inline and re.search(rf"^\s*{column}\s", inline.group(1), re.M)
                    )
                    altered = re.search(
                        rf"ALTER TABLE {table}\s+ADD COLUMN IF NOT EXISTS {column}\b",
                        self.schema_sql,
                        re.IGNORECASE,
                    )
                    self.assertTrue(inline_has_it or altered)


class AuthSchemaShapeTests(unittest.TestCase):
    """The identity tables carry the constraints the design relies on."""

    def setUp(self) -> None:
        self.sql = _strip_comments(
            (DB_DIR / "migrations" / "002_auth_identity.sql").read_text()
        )

    def test_identity_tables_exist(self) -> None:
        for table in (
            "users",
            "course_memberships",
            "invitations",
            "invitation_course_grants",
            "participants",
            "auth_sessions",
            "admin_actions",
        ):
            with self.subTest(table=table):
                self.assertIn(f"CREATE TABLE IF NOT EXISTS {table}", self.sql)

    def test_participants_store_nothing_identifying(self) -> None:
        block = re.search(
            r"CREATE TABLE IF NOT EXISTS participants\s*\((.*?)\n\);", self.sql, re.S
        )
        assert block is not None
        # A column declaration is `name TYPE ...`; continuation lines such as
        # `REFERENCES ...` and `ON DELETE ...` carry no type and are skipped.
        column_line = re.compile(
            r"^\s*(\w+)\s+(UUID|TEXT|TIMESTAMPTZ|INTEGER|BOOLEAN|JSONB|BIGSERIAL)\b",
            re.IGNORECASE,
        )
        columns = {
            match.group(1).lower()
            for match in (column_line.match(line) for line in block.group(1).splitlines())
            if match
        }
        self.assertEqual(
            columns,
            {"participant_id", "course_id", "invitation_id", "created_at", "last_seen_at"},
        )
        for forbidden in ("email", "name", "netid", "ip", "user_agent"):
            self.assertNotIn(forbidden, columns)

    def test_secrets_are_stored_hashed_or_not_at_all(self) -> None:
        self.assertIn("token_hash", self.sql)
        self.assertIn("password_hash", self.sql)
        self.assertNotRegex(self.sql, r"\bpassword\s+TEXT\b")
        self.assertNotRegex(self.sql, r"\btoken\s+TEXT\b")

    def test_a_session_is_exactly_one_principal(self) -> None:
        self.assertIn("auth_sessions_exactly_one_principal", self.sql)

    def test_attribution_columns_are_nullable_and_anonymise_on_delete(self) -> None:
        for table in ("evaluations", "seed_examples"):
            with self.subTest(table=table):
                match = re.search(
                    rf"ALTER TABLE {table}\s+ADD COLUMN IF NOT EXISTS participant_id UUID\s+"
                    r"REFERENCES participants\(participant_id\)\s+ON DELETE SET NULL",
                    self.sql,
                )
                self.assertIsNotNone(match)
                self.assertNotRegex(
                    self.sql, rf"ALTER TABLE {table}[^;]*participant_id UUID NOT NULL"
                )


if __name__ == "__main__":
    unittest.main()
