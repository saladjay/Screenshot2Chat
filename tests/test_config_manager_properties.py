"""
Property-based tests for ConfigManager
Task 6.2, 6.4, 6.6: ConfigManager property tests
Property 16: Configuration Layer Priority
Property 21: Configuration Export-Import Round-Trip
Property 19: Configuration Validation Rejection
Validates: Requirements 9.1, 9.4, 9.7
"""

import pytest
import tempfile
import os
import yaml
import json
from hypothesis import given, strategies as st, settings, assume
from src.screenshot2chat.config.config_manager import ConfigManager


@settings(max_examples=100, deadline=None)
@given(
    key=st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=('Lu', 'Ll'))),
    default_value=st.integers(),
    user_value=st.integers(),
    runtime_value=st.integers()
)
def test_property_16_configuration_layer_priority(key, default_value, user_value, runtime_value):
    """
    Feature: screenshot-analysis-library-refactor
    Property 16: Configuration Layer Priority
    
    For any configuration key present in multiple layers, the get() method 
    should return the value from the highest priority layer (runtime > user > default).
    """
    assume(default_value != user_value and user_value != runtime_value)
    
    config_manager = ConfigManager()
    
    # Set values in all three layers
    config_manager.set(key, default_value, layer='default')
    config_manager.set(key, user_value, layer='user')
    config_manager.set(key, runtime_value, layer='runtime')
    
    # Get should return runtime value (highest priority)
    result = config_manager.get(key)
    assert result == runtime_value, f"Expected runtime value {runtime_value}, got {result}"
    
    # Remove runtime layer value
    config_manager.configs['runtime'] = {}
    
    # Now should return user value
    result = config_manager.get(key)
    assert result == user_value, f"Expected user value {user_value}, got {result}"
    
    # Remove user layer value
    config_manager.configs['user'] = {}
    
    # Now should return default value
    result = config_manager.get(key)
    assert result == default_value, f"Expected default value {default_value}, got {result}"


@settings(max_examples=100, deadline=None)
@given(
    config_dict=st.dictionaries(
        keys=st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=('Lu', 'Ll'))),
        values=st.one_of(
            st.integers(),
            st.floats(allow_nan=False, allow_infinity=False),
            st.text(max_size=50),
            st.booleans()
        ),
        min_size=1,
        max_size=10
    )
)
def test_property_21_configuration_export_import_roundtrip(config_dict):
    """
    Feature: screenshot-analysis-library-refactor
    Property 21: Configuration Export-Import Round-Trip
    
    For any configuration state, exporting to a file and importing back 
    should restore the exact same configuration.
    """
    config_manager = ConfigManager()
    
    # Set configuration values
    for key, value in config_dict.items():
        config_manager.set(key, value, layer='user')
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test YAML round-trip
        yaml_path = os.path.join(tmpdir, "config.yaml")
        config_manager.save(yaml_path, layer='user')
        
        # Create new config manager and load
        new_config_manager = ConfigManager()
        new_config_manager.load(yaml_path, layer='user')
        
        # Verify all values match
        for key, expected_value in config_dict.items():
            actual_value = new_config_manager.get(key)
            if isinstance(expected_value, float):
                assert abs(actual_value - expected_value) < 1e-6, f"Key {key}: expected {expected_value}, got {actual_value}"
            else:
                assert actual_value == expected_value, f"Key {key}: expected {expected_value}, got {actual_value}"
        
        # Test JSON round-trip
        json_path = os.path.join(tmpdir, "config.json")
        config_manager.save(json_path, layer='user')
        
        json_config_manager = ConfigManager()
        json_config_manager.load(json_path, layer='user')
        
        for key, expected_value in config_dict.items():
            actual_value = json_config_manager.get(key)
            if isinstance(expected_value, float):
                assert abs(actual_value - expected_value) < 1e-6
            else:
                assert actual_value == expected_value


@settings(max_examples=100, deadline=None)
@given(
    key=st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=('Lu', 'Ll'))),
    invalid_value=st.one_of(
        st.text(max_size=5),  # When expecting int
        st.integers(min_value=1000, max_value=9999)  # Out of range
    )
)
def test_property_19_configuration_validation_rejection(key, invalid_value):
    """
    Feature: screenshot-analysis-library-refactor
    Property 19: Configuration Validation Rejection
    
    For any invalid configuration value (wrong type, out of range, missing required field), 
    the validation function should reject it with a descriptive error.
    """
    config_manager = ConfigManager()
    
    # Define a schema that will reject the invalid value
    if isinstance(invalid_value, str):
        # Schema expects integer
        schema = {
            key: {
                "type": "integer",
                "required": True
            }
        }
    else:
        # Schema expects value in range [0, 100]
        schema = {
            key: {
                "type": "integer",
                "min": 0,
                "max": 100,
                "required": True
            }
        }
    
    # Set invalid value
    config_manager.set(key, invalid_value, layer='user')
    
    # Validation should fail
    is_valid, errors = config_manager.validate(schema)
    assert not is_valid, f"Validation should fail for invalid value {invalid_value}"
    assert errors is not None and len(errors) > 0, "Should provide error messages"
    assert key in str(errors), f"Error should mention the invalid key {key}"


@settings(max_examples=100, deadline=None)
@given(
    nested_key=st.lists(
        st.text(min_size=1, max_size=10, alphabet=st.characters(whitelist_categories=('Lu', 'Ll'))),
        min_size=2,
        max_size=4
    ),
    value=st.integers()
)
def test_nested_configuration_keys(nested_key, value):
    """
    Test that nested configuration keys work correctly with dot notation
    """
    config_manager = ConfigManager()
    
    # Create dot-separated key
    key_str = '.'.join(nested_key)
    
    # Set nested value
    config_manager.set(key_str, value, layer='user')
    
    # Get nested value
    result = config_manager.get(key_str)
    assert result == value, f"Expected {value}, got {result}"
    
    # Verify nested structure was created
    config = config_manager.configs['user']
    for part in nested_key[:-1]:
        assert part in config, f"Missing nested key part: {part}"
        config = config[part]
    assert nested_key[-1] in config, f"Missing final key: {nested_key[-1]}"


@settings(max_examples=100, deadline=None)
@given(
    key=st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=('Lu', 'Ll'))),
    default_return=st.integers()
)
def test_configuration_default_value(key, default_return):
    """
    Test that get() returns default value when key doesn't exist
    """
    config_manager = ConfigManager()
    
    # Get non-existent key with default
    result = config_manager.get(key, default=default_return)
    assert result == default_return, f"Expected default {default_return}, got {result}"


@settings(max_examples=100, deadline=None)
@given(
    parent_dict=st.dictionaries(
        keys=st.text(min_size=1, max_size=10, alphabet=st.characters(whitelist_categories=('Lu', 'Ll'))),
        values=st.integers(),
        min_size=2,
        max_size=5
    ),
    child_dict=st.dictionaries(
        keys=st.text(min_size=1, max_size=10, alphabet=st.characters(whitelist_categories=('Lu', 'Ll'))),
        values=st.integers(),
        min_size=1,
        max_size=3
    )
)
def test_configuration_inheritance_and_override(parent_dict, child_dict):
    """
    Test that child configuration overrides parent while preserving non-overridden values
    """
    config_manager = ConfigManager()
    
    # Set parent (default) configuration
    for key, value in parent_dict.items():
        config_manager.set(key, value, layer='default')
    
    # Set child (user) configuration
    for key, value in child_dict.items():
        config_manager.set(key, value, layer='user')
    
    # Check overridden values
    for key, expected_value in child_dict.items():
        actual_value = config_manager.get(key)
        assert actual_value == expected_value, f"Child value should override: {key}"
    
    # Check non-overridden parent values
    for key, expected_value in parent_dict.items():
        if key not in child_dict:
            actual_value = config_manager.get(key)
            assert actual_value == expected_value, f"Parent value should be preserved: {key}"
