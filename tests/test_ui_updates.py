"""Tests for UI update functions and session management.

This module tests the behavioral contracts of session-based file management
without testing implementation details or library code.
"""

import pytest

from image_to_midi.ui_updates import (
    get_or_create_file_manager,
    cleanup_session,
    cleanup_cache,
    _file_managers,
)


@pytest.fixture(autouse=True)
def clean_registry():
    """Clean the file manager registry before and after each test.

    This ensures test isolation by clearing any state from previous tests.
    """
    _file_managers.clear()
    yield
    _file_managers.clear()


def test_get_or_create_returns_consistent_manager():
    """Behavior: Same session ID should return the same file manager instance."""
    session_id = "test-session-123"

    manager1 = get_or_create_file_manager(session_id)
    manager2 = get_or_create_file_manager(session_id)

    assert manager1 is manager2


def test_different_sessions_get_different_managers():
    """Behavior: Different session IDs should return different file managers."""
    session_a = "session-a"
    session_b = "session-b"

    manager_a = get_or_create_file_manager(session_a)
    manager_b = get_or_create_file_manager(session_b)

    assert manager_a is not manager_b


def test_cleanup_session_removes_from_registry():
    """Behavior: Cleanup should remove the session from the active registry."""
    session_id = "test-session-456"

    # Create a file manager for the session
    get_or_create_file_manager(session_id)
    assert session_id in _file_managers

    # Cleanup should remove it
    cleanup_session(session_id)
    assert session_id not in _file_managers


def test_cleanup_session_is_idempotent():
    """Behavior: Cleaning up a non-existent session should not raise errors."""
    session_id = "never-existed"

    # Should not raise any exception
    cleanup_session(session_id)


def test_cleanup_cache_removes_all_sessions():
    """Behavior: Cache cleanup should remove all active sessions from registry."""
    session_1 = "session-1"
    session_2 = "session-2"
    session_3 = "session-3"

    # Create multiple sessions
    get_or_create_file_manager(session_1)
    get_or_create_file_manager(session_2)
    get_or_create_file_manager(session_3)

    assert len(_file_managers) == 3

    # Cleanup all
    cleanup_cache(session_id=None)

    assert len(_file_managers) == 0


def test_cleanup_cache_with_specific_session():
    """Behavior: Cache cleanup with session ID should only remove that session."""
    session_a = "keep-this"
    session_b = "remove-this"

    get_or_create_file_manager(session_a)
    get_or_create_file_manager(session_b)

    # Cleanup only session_b
    cleanup_cache(session_id=session_b)

    assert session_a in _file_managers
    assert session_b not in _file_managers


def test_multiple_sessions_are_isolated():
    """Behavior: Multiple concurrent sessions should maintain isolation."""
    sessions = [f"user-{i}" for i in range(5)]
    managers = []

    # Create managers for all sessions
    for session_id in sessions:
        manager = get_or_create_file_manager(session_id)
        managers.append(manager)

    # All managers should be unique
    assert len(set(managers)) == 5

    # All sessions should be tracked
    assert len(_file_managers) == 5
