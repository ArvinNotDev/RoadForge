import pytest

from datacollector.config import CollectionConfig, TOWNS
from datacollector.map_catalog import (
    build_map_reference_index,
    normalize_map_name,
    require_map_reference,
    validate_selected_map_names,
    select_collection_towns,
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



def test_current_world_mode_uses_only_the_loaded_town():
    selected = select_collection_towns(
        {
            "Town10HD_Opt": {"enabled": True, "target": 100},
            "Town12": {"enabled": True, "target": 200},
        },
        {
            "Town10HD_Opt": "/Game/Carla/Maps/Town10HD_Opt",
            "Town12": "/Game/Carla/Maps/Town12",
        },
        use_current_world=True,
        current_map_reference="/Game/Carla/Maps/Town10HD_Opt",
    )

    assert selected == ["Town10HD_Opt"]


def test_current_world_mode_requires_current_town_to_be_enabled():
    with pytest.raises(ValueError, match="Enable the currently loaded map"):
        select_collection_towns(
            {"Town12": {"enabled": False, "target": 100}},
            {"Town12": "/Game/Carla/Maps/Town12"},
            use_current_world=True,
            current_map_reference="/Game/Carla/Maps/Town12",
        )


def test_current_world_mode_requires_positive_target():
    with pytest.raises(ValueError, match="positive image target"):
        select_collection_towns(
            {"Town12": {"enabled": True, "target": 0}},
            {"Town12": "/Game/Carla/Maps/Town12"},
            use_current_world=True,
            current_map_reference="/Game/Carla/Maps/Town12",
        )


def test_current_world_mode_rejects_unknown_current_map():
    with pytest.raises(ValueError, match="was not discovered"):
        select_collection_towns(
            {"Town12": {"enabled": True, "target": 100}},
            {"Town10HD_Opt": "/Game/Carla/Maps/Town10HD_Opt"},
            use_current_world=True,
            current_map_reference="/Game/Carla/Maps/Town12",
        )
