"""Fill every table of a freshly bootstrapped PostgreSQL database with a few synthetic rows.

usage: python -I seed_db.py <db_url> [rows_per_table]

The rows are made up from the column types alone, so the same schema always gets
the same rows: the API snapshot then compares two code versions on data that has
something in it, without any real data leaving anyone's machine. Foreign keys
point at the rows seeded earlier in the referenced table. A row a constraint
refuses is skipped and counted, never forced.
"""
from __future__ import annotations

import sys
from collections import defaultdict

import psycopg

COLUMNS = """
SELECT c.table_name, c.column_name, c.data_type, c.udt_name, c.is_nullable = 'YES', c.column_default
FROM information_schema.columns c
JOIN information_schema.tables t USING (table_schema, table_name)
WHERE c.table_schema = 'public' AND t.table_type = 'BASE TABLE'
ORDER BY c.table_name, c.ordinal_position
"""
FOREIGN_KEYS = """
SELECT tc.table_name, kcu.column_name, ccu.table_name, ccu.column_name
FROM information_schema.table_constraints tc
JOIN information_schema.key_column_usage kcu
  ON tc.constraint_name = kcu.constraint_name AND tc.table_schema = kcu.table_schema
JOIN information_schema.constraint_column_usage ccu
  ON tc.constraint_name = ccu.constraint_name AND tc.table_schema = ccu.table_schema
WHERE tc.constraint_type = 'FOREIGN KEY' AND tc.table_schema = 'public'
"""


def _value(table: str, col: str, dtype: str, udt: str, i: int):
    if dtype in ("integer", "bigint", "smallint"):
        return i
    if dtype in ("real", "double precision", "numeric"):
        return i * 0.5
    if dtype == "boolean":
        return i % 2 == 0
    if dtype.startswith("timestamp"):
        return f"2026-01-0{i} 00:00:00"
    if dtype == "date":
        return f"2026-01-0{i}"
    if dtype in ("json", "jsonb"):
        return "{}"
    if dtype == "ARRAY":
        return []
    if udt == "vector":
        return None
    return f"{col}-{i}"


def seed(url: str, rows: int = 3) -> None:
    with psycopg.connect(url, autocommit=False) as conn:
        cols = defaultdict(list)
        for table, col, dtype, udt, nullable, default in conn.execute(COLUMNS):
            cols[table].append((col, dtype, udt, nullable, default))
        fks = defaultdict(dict)
        for table, col, ref_table, ref_col in conn.execute(FOREIGN_KEYS):
            fks[table][col] = (ref_table, ref_col)

        order, seen = [], set()

        def visit(table, path=()):
            if table in seen or table in path:
                return
            for ref_table, _ in fks[table].values():
                if ref_table != table:
                    visit(ref_table, path + (table,))
            seen.add(table)
            order.append(table)

        for table in sorted(cols):
            visit(table)

        inserted: dict[str, list[dict]] = defaultdict(list)
        skipped = 0
        for table in order:
            for i in range(1, rows + 1):
                names, values = [], []
                for col, dtype, udt, nullable, default in cols[table]:
                    if default and default.startswith("nextval("):
                        continue
                    if col in fks[table]:
                        ref_table, ref_col = fks[table][col]
                        refs = inserted.get(ref_table) or []
                        value = refs[(i - 1) % len(refs)].get(ref_col) if refs else None
                        if value is None and nullable:
                            continue
                    else:
                        value = _value(table, col, dtype, udt, i)
                    if value is None:
                        continue
                    names.append(f'"{col}"')
                    values.append(value)
                sql = (f'INSERT INTO "{table}" ({", ".join(names)}) VALUES ({", ".join(["%s"] * len(values))}) RETURNING *'
                       if names else f'INSERT INTO "{table}" DEFAULT VALUES RETURNING *')
                try:
                    with conn.transaction():
                        cur = conn.execute(sql, values)
                        desc = [d.name for d in cur.description]
                        inserted[table].append(dict(zip(desc, cur.fetchone())))
                except psycopg.Error:
                    skipped += 1
        conn.commit()
    total = sum(len(v) for v in inserted.values())
    print(f"[seed] {total} rows in {len(inserted)}/{len(cols)} tables, {skipped} refused by constraints")


if __name__ == "__main__":
    seed(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 3)
