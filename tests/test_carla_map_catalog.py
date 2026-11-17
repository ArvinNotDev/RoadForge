import pytest

from datacollector.config import CollectionConfig, TOWNS
from datacollector.map_catalog import (
    build_map_reference_index,
    normalize_map_name,
    require_map_reference,
    validate_selected_map_names,
)


def test_collector_has_no_unverified_default_map_names():
    assert TOWNS == ()
    assert CollectionConfig().towns == {}


def test_map_index_contains_only_server_reported_maps_and_preserves_references():
    references = [
        "/Game/Carla/Maps/EmptyMap",
        "/Game/Carla/Maps/Town10HD_Opt",
        "/Game/Carla/Maps/Town12",
        "/Game/Carla/Maps/Town15",
    ]

    indexed = build_map_reference_index(references)

    assert list(indexed) == ["EmptyMap", "Town10HD_Opt", "Town12", "Town15"]
    assert indexed["Town12"] == "/Game/Carla/Maps/Town12"
    assert "Town01_Opt" not in indexed


def test_map_name_normalizes_full_package_references():
    assert normalize_map_name("/Game/Carla/Maps/Town15/") == "Town15"
    assert normalize_map_name("Town10HD_Opt") == "Town10HD_Opt"
    assert normalize_map_name("") == ""


def test_stale_or_uninstalled_map_selection_is_rejected():
    indexed = build_map_reference_index(["Town10HD_Opt", "Town12"])

    with pytest.raises(ValueError, match="not installed"):
        validate_selected_map_names(["Town01_Opt"], indexed)


def test_valid_selection_is_deduplicated_and_preserved():
    indexed = build_map_reference_index(["/Game/Carla/Maps/Town12", "Town15"])

    assert validate_selected_map_names(["Town12", "Town15", "Town12"], indexed) == [
        "Town12", "Town15"
    ]


def test_loading_requires_an_exact_server_reference():
    indexed = build_map_reference_index(["/Game/Carla/Maps/Town12"])

    assert require_map_reference("Town12", indexed) == "/Game/Carla/Maps/Town12"
    with pytest.raises(ValueError, match="Town01_Opt.*not discovered"):
        require_map_reference("Town01_Opt", indexed)


def test_duplicate_display_names_with_different_references_are_rejected():
    with pytest.raises(ValueError, match="same name"):
        build_map_reference_index([
            "/Game/Carla/Maps/Town12",
            "/AnotherPackage/Town12",
        ])
