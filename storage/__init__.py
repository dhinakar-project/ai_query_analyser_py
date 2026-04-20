"""Storage module for SQLite-backed persistence."""

from storage.db import Database, get_database

__all__ = ["Database", "get_database"]