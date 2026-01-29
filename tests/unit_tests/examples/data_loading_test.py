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
"""Tests for the data_loading module, specifically UUID extraction from YAML."""

from pathlib import Path


def test_get_dataset_config_from_yaml_extracts_uuid(tmp_path: Path) -> None:
    """Test that get_dataset_config_from_yaml extracts UUID from YAML."""
    from superset.examples.data_loading import get_dataset_config_from_yaml

    # Create a temporary dataset.yaml with UUID
    yaml_content = """
table_name: birth_names
schema: public
data_file: data.parquet
uuid: 14f48794-ebfa-4f60-a26a-582c49132f1b
"""
    dataset_yaml = tmp_path / "dataset.yaml"
    dataset_yaml.write_text(yaml_content)

    result = get_dataset_config_from_yaml(tmp_path)

    assert result["uuid"] == "14f48794-ebfa-4f60-a26a-582c49132f1b"
    assert result["table_name"] == "birth_names"
    assert result["schema"] == "public"


def test_get_dataset_config_from_yaml_handles_missing_uuid(tmp_path: Path) -> None:
    """Test that missing UUID returns None."""
    from superset.examples.data_loading import get_dataset_config_from_yaml

    # Create a temporary dataset.yaml without UUID
    yaml_content = """
table_name: birth_names
schema: public
"""
    dataset_yaml = tmp_path / "dataset.yaml"
    dataset_yaml.write_text(yaml_content)

    result = get_dataset_config_from_yaml(tmp_path)

    assert result["uuid"] is None
    assert result["table_name"] == "birth_names"


def test_get_dataset_config_from_yaml_handles_missing_file(tmp_path: Path) -> None:
    """Test that missing dataset.yaml returns None for all fields."""
    from superset.examples.data_loading import get_dataset_config_from_yaml

    result = get_dataset_config_from_yaml(tmp_path)

    assert result["uuid"] is None
    assert result["table_name"] is None
    assert result["schema"] is None


def test_get_multi_dataset_config_extracts_uuid(tmp_path: Path) -> None:
    """Test that _get_multi_dataset_config extracts UUID from datasets/*.yaml."""
    from superset.examples.data_loading import _get_multi_dataset_config

    # Create datasets directory and YAML file
    datasets_dir = tmp_path / "datasets"
    datasets_dir.mkdir()

    yaml_content = """
table_name: cleaned_sales_data
schema: null
uuid: e8623bb9-5e00-f531-506a-19607f5f8005
"""
    dataset_yaml = datasets_dir / "cleaned_sales_data.yaml"
    dataset_yaml.write_text(yaml_content)

    data_file = tmp_path / "data" / "cleaned_sales_data.parquet"

    result = _get_multi_dataset_config(tmp_path, "cleaned_sales_data", data_file)

    assert result["uuid"] == "e8623bb9-5e00-f531-506a-19607f5f8005"
    assert result["table_name"] == "cleaned_sales_data"


def test_get_multi_dataset_config_handles_missing_uuid(tmp_path: Path) -> None:
    """Test that missing UUID in multi-dataset config returns None."""
    from superset.examples.data_loading import _get_multi_dataset_config

    # Create datasets directory and YAML file without UUID
    datasets_dir = tmp_path / "datasets"
    datasets_dir.mkdir()

    yaml_content = """
table_name: my_dataset
schema: null
"""
    dataset_yaml = datasets_dir / "my_dataset.yaml"
    dataset_yaml.write_text(yaml_content)

    data_file = tmp_path / "data" / "my_dataset.parquet"

    result = _get_multi_dataset_config(tmp_path, "my_dataset", data_file)

    assert result["uuid"] is None
    assert result["table_name"] == "my_dataset"


def test_get_multi_dataset_config_handles_missing_file(tmp_path: Path) -> None:
    """Test that missing datasets/*.yaml returns None for UUID."""
    from superset.examples.data_loading import _get_multi_dataset_config

    data_file = tmp_path / "data" / "my_dataset.parquet"

    result = _get_multi_dataset_config(tmp_path, "my_dataset", data_file)

    assert result["uuid"] is None
    # Falls back to dataset_name when no YAML
    assert result["table_name"] == "my_dataset"
