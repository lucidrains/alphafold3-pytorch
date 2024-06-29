"""General-purpose errors used throughout the data pipeline."""

class Error(Exception):
    """Base class for exceptions."""


class MultipleChainsError(Error):
    """An error indicating that multiple chains were found for a given ID."""
