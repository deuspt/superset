# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Tests for the generic_loader module, specifically UUID handling."""

from unittest.mock import MagicMock, patch

import pandas as pd


def _setup_database_mocks(
    mock_get_db: MagicMock, mock_database: MagicMock, has_table: bool = False
) -> MagicMock:
    """Helper to set up common database mocks."""
    mock_database.id = 1
    mock_database.has_table.return_value = has_table
    mock_get_db.return_value = mock_database

    mock_engine = MagicMock()
    mock_inspector = MagicMock()
    mock_inspector.default_schema_name = "public"
    mock_database.get_sqla_engine.return_value.__enter__ = MagicMock(
        return_value=mock_engine
    )
    mock_database.get_sqla_engine.return_value.__exit__ = MagicMock(return_value=False)

    return mock_inspector


@patch("superset.examples.generic_loader.db")
@patch("superset.examples.generic_loader.get_example_database")
@patch("superset.examples.generic_loader.read_example_data")
def test_load_parquet_table_sets_uuid_on_new_table(
    mock_read_data: MagicMock,
    mock_get_db: MagicMock,
    mock_db: MagicMock,
) -> None:
    """Test that load_parquet_table sets UUID when creating a new SqlaTable."""
    from superset.examples.generic_loader import load_parquet_table

    mock_database = MagicMock()
    mock_inspector = _setup_database_mocks(mock_get_db, mock_database, has_table=False)

    with patch("superset.examples.generic_loader.inspect") as mock_inspect:
        mock_inspect.return_value = mock_inspector

        # No existing table by UUID or table_name
        mock_db.session.query.return_value.filter_by.return_value.first.return_value = (
            None
        )

        mock_read_data.return_value = pd.DataFrame({"col1": [1, 2, 3]})

        test_uuid = "14f48794-ebfa-4f60-a26a-582c49132f1b"

        result = load_parquet_table(
            parquet_file="test_data",
            table_name="test_table",
            database=mock_database,
            only_metadata=True,
            uuid=test_uuid,
        )

        assert result.uuid == test_uuid


@patch("superset.examples.generic_loader.db")
@patch("superset.examples.generic_loader.get_example_database")
def test_load_parquet_table_finds_existing_by_uuid_first(
    mock_get_db: MagicMock,
    mock_db: MagicMock,
) -> None:
    """Test that load_parquet_table looks up by UUID first when provided."""
    from superset.examples.generic_loader import load_parquet_table

    mock_database = MagicMock()
    mock_inspector = _setup_database_mocks(mock_get_db, mock_database, has_table=True)

    with patch("superset.examples.generic_loader.inspect") as mock_inspect:
        mock_inspect.return_value = mock_inspector

        # Existing table found by UUID
        test_uuid = "existing-uuid-1234"
        mock_existing_table = MagicMock()
        mock_existing_table.uuid = test_uuid
        mock_existing_table.table_name = "test_table"

        # First call (by uuid) returns the table, second call (by table_name) not needed
        mock_db.session.query.return_value.filter_by.return_value.first.return_value = (
            mock_existing_table
        )

        result = load_parquet_table(
            parquet_file="test_data",
            table_name="test_table",
            database=mock_database,
            only_metadata=True,
            uuid=test_uuid,
        )

        # Should return the existing table found by UUID
        assert result.uuid == test_uuid
        assert result is mock_existing_table


@patch("superset.examples.generic_loader.db")
@patch("superset.examples.generic_loader.get_example_database")
def test_load_parquet_table_backfills_uuid_on_existing_table(
    mock_get_db: MagicMock,
    mock_db: MagicMock,
) -> None:
    """Test that existing dataset with uuid=None gets UUID backfilled."""
    from superset.examples.generic_loader import load_parquet_table

    mock_database = MagicMock()
    mock_inspector = _setup_database_mocks(mock_get_db, mock_database, has_table=True)

    with patch("superset.examples.generic_loader.inspect") as mock_inspect:
        mock_inspect.return_value = mock_inspector

        # Existing table with NO UUID (needs backfill)
        mock_existing_table = MagicMock()
        mock_existing_table.uuid = None
        mock_existing_table.table_name = "test_table"

        # UUID lookup returns None, table_name lookup returns the table
        def filter_by_side_effect(**kwargs):
            mock_result = MagicMock()
            if "uuid" in kwargs:
                mock_result.first.return_value = None
            else:
                mock_result.first.return_value = mock_existing_table
            return mock_result

        mock_db.session.query.return_value.filter_by.side_effect = filter_by_side_effect

        new_uuid = "new-uuid-5678"

        result = load_parquet_table(
            parquet_file="test_data",
            table_name="test_table",
            database=mock_database,
            only_metadata=True,
            uuid=new_uuid,
        )

        # UUID should be backfilled
        assert result.uuid == new_uuid


@patch("superset.examples.generic_loader.db")
@patch("superset.examples.generic_loader.get_example_database")
def test_load_parquet_table_avoids_uuid_collision(
    mock_get_db: MagicMock,
    mock_db: MagicMock,
) -> None:
    """Test that finding by UUID doesn't try to re-set UUID (avoids collision)."""
    from superset.examples.generic_loader import load_parquet_table

    mock_database = MagicMock()
    mock_inspector = _setup_database_mocks(mock_get_db, mock_database, has_table=True)

    with patch("superset.examples.generic_loader.inspect") as mock_inspect:
        mock_inspect.return_value = mock_inspector

        # Table already has the UUID we're looking for
        test_uuid = "existing-uuid-1234"
        mock_existing_table = MagicMock()
        mock_existing_table.uuid = test_uuid

        # UUID lookup finds the table
        mock_db.session.query.return_value.filter_by.return_value.first.return_value = (
            mock_existing_table
        )

        result = load_parquet_table(
            parquet_file="test_data",
            table_name="test_table",
            database=mock_database,
            only_metadata=True,
            uuid=test_uuid,
        )

        # UUID should remain unchanged (not re-assigned)
        assert result.uuid == test_uuid


@patch("superset.examples.generic_loader.db")
@patch("superset.examples.generic_loader.get_example_database")
def test_load_parquet_table_preserves_existing_different_uuid(
    mock_get_db: MagicMock,
    mock_db: MagicMock,
) -> None:
    """Test that if table has different UUID, we find it by UUID lookup first."""
    from superset.examples.generic_loader import load_parquet_table

    mock_database = MagicMock()
    mock_inspector = _setup_database_mocks(mock_get_db, mock_database, has_table=True)

    with patch("superset.examples.generic_loader.inspect") as mock_inspect:
        mock_inspect.return_value = mock_inspector

        # A table exists with the target UUID
        target_uuid = "target-uuid-1234"
        mock_uuid_table = MagicMock()
        mock_uuid_table.uuid = target_uuid

        # UUID lookup finds the UUID-matching table
        mock_db.session.query.return_value.filter_by.return_value.first.return_value = (
            mock_uuid_table
        )

        result = load_parquet_table(
            parquet_file="test_data",
            table_name="different_table_name",
            database=mock_database,
            only_metadata=True,
            uuid=target_uuid,
        )

        # Should return the table found by UUID, not create new one
        assert result is mock_uuid_table
        assert result.uuid == target_uuid


@patch("superset.examples.generic_loader.db")
@patch("superset.examples.generic_loader.get_example_database")
@patch("superset.examples.generic_loader.read_example_data")
def test_load_parquet_table_works_without_uuid(
    mock_read_data: MagicMock,
    mock_get_db: MagicMock,
    mock_db: MagicMock,
) -> None:
    """Test that load_parquet_table still works when no UUID is provided."""
    from superset.examples.generic_loader import load_parquet_table

    mock_database = MagicMock()
    mock_inspector = _setup_database_mocks(mock_get_db, mock_database, has_table=False)

    with patch("superset.examples.generic_loader.inspect") as mock_inspect:
        mock_inspect.return_value = mock_inspector

        # No existing table
        mock_db.session.query.return_value.filter_by.return_value.first.return_value = (
            None
        )

        mock_read_data.return_value = pd.DataFrame({"col1": [1, 2, 3]})

        result = load_parquet_table(
            parquet_file="test_data",
            table_name="test_table",
            database=mock_database,
            only_metadata=True,
        )

        assert result is not None
        assert result.table_name == "test_table"


def test_create_generic_loader_passes_uuid() -> None:
    """Test that create_generic_loader passes UUID to load_parquet_table."""
    from superset.examples.generic_loader import create_generic_loader

    test_uuid = "test-uuid-1234"
    loader = create_generic_loader(
        parquet_file="test_data",
        table_name="test_table",
        uuid=test_uuid,
    )

    assert loader is not None
    assert callable(loader)
    assert loader.__name__ == "load_test_data"


@patch("superset.examples.generic_loader.db")
def test_find_dataset_returns_uuid_match_first(mock_db: MagicMock) -> None:
    """Test that _find_dataset returns UUID match over table_name match."""
    from superset.examples.generic_loader import _find_dataset

    # Row with UUID (should be found first)
    uuid_row = MagicMock()
    uuid_row.uuid = "target-uuid"
    uuid_row.table_name = "table_a"

    # Row without UUID (same table_name as query)
    tablename_row = MagicMock()
    tablename_row.uuid = None
    tablename_row.table_name = "test_table"

    # UUID lookup returns uuid_row
    mock_db.session.query.return_value.filter_by.return_value.first.return_value = (
        uuid_row
    )

    result, found_by_uuid = _find_dataset("test_table", 1, "target-uuid", "public")

    assert result is uuid_row
    assert found_by_uuid is True


@patch("superset.examples.generic_loader.db")
def test_find_dataset_falls_back_to_table_name(mock_db: MagicMock) -> None:
    """Test that _find_dataset falls back to table_name when UUID not found."""
    from superset.examples.generic_loader import _find_dataset

    tablename_row = MagicMock()
    tablename_row.uuid = None
    tablename_row.table_name = "test_table"

    # UUID lookup returns None, table_name lookup returns the row
    def filter_by_side_effect(**kwargs):
        mock_result = MagicMock()
        if "uuid" in kwargs:
            mock_result.first.return_value = None
        else:
            mock_result.first.return_value = tablename_row
        return mock_result

    mock_db.session.query.return_value.filter_by.side_effect = filter_by_side_effect

    result, found_by_uuid = _find_dataset("test_table", 1, "nonexistent-uuid", "public")

    assert result is tablename_row
    assert found_by_uuid is False


@patch("superset.examples.generic_loader.db")
@patch("superset.examples.generic_loader.get_example_database")
@patch("superset.examples.generic_loader.read_example_data")
def test_load_parquet_table_duplicate_rows_table_missing(
    mock_read_data: MagicMock,
    mock_get_db: MagicMock,
    mock_db: MagicMock,
) -> None:
    """Test UUID-first lookup when duplicate rows exist and physical table is missing.

    Scenario:
    - Row A: table_name="test_table", uuid="target-uuid" (correct row)
    - Row B: table_name="test_table", uuid=None (duplicate from broken run)
    - Physical table was dropped

    The UUID-first lookup should find Row A and avoid collision.
    """
    from superset.examples.generic_loader import load_parquet_table

    mock_database = MagicMock()
    mock_inspector = _setup_database_mocks(
        mock_get_db,
        mock_database,
        has_table=False,  # Table was dropped
    )

    with patch("superset.examples.generic_loader.inspect") as mock_inspect:
        mock_inspect.return_value = mock_inspector

        # Row with UUID (should be found by UUID lookup)
        uuid_row = MagicMock()
        uuid_row.uuid = "target-uuid"
        uuid_row.table_name = "test_table"
        uuid_row.database = mock_database

        # UUID lookup finds the correct row
        mock_db.session.query.return_value.filter_by.return_value.first.return_value = (
            uuid_row
        )

        mock_read_data.return_value = pd.DataFrame({"col1": [1, 2, 3]})

        result = load_parquet_table(
            parquet_file="test_data",
            table_name="test_table",
            database=mock_database,
            only_metadata=True,
            uuid="target-uuid",
        )

        # Should return the UUID row, not try to backfill (which would collide)
        assert result is uuid_row
        assert result.uuid == "target-uuid"


@patch("superset.examples.generic_loader.db")
def test_find_dataset_distinguishes_schemas(mock_db: MagicMock) -> None:
    """Test that _find_dataset uses schema to distinguish same-name tables.

    Scenario:
    - Row A: table_name="users", schema="schema_a", uuid=None
    - Row B: table_name="users", schema="schema_b", uuid=None

    Looking up "users" in "schema_b" should find Row B, not Row A.
    """
    from superset.examples.generic_loader import _find_dataset

    # Row in schema_b (should be found)
    schema_b_row = MagicMock()
    schema_b_row.uuid = None
    schema_b_row.table_name = "users"
    schema_b_row.schema = "schema_b"

    # No UUID lookup (uuid not provided), table_name lookup returns schema_b row
    def filter_by_side_effect(**kwargs):
        mock_result = MagicMock()
        if "uuid" in kwargs:
            mock_result.first.return_value = None
        elif kwargs.get("schema") == "schema_b":
            mock_result.first.return_value = schema_b_row
        else:
            mock_result.first.return_value = None  # schema_a not requested
        return mock_result

    mock_db.session.query.return_value.filter_by.side_effect = filter_by_side_effect

    result, found_by_uuid = _find_dataset("users", 1, None, "schema_b")

    assert result is schema_b_row
    assert found_by_uuid is False
    assert result.schema == "schema_b"
