class RiceFertilizerError(Exception):
    """Base project exception."""

class ConfigurationError(RiceFertilizerError):
    """Missing or invalid scientific/technical configuration."""

class DataValidationError(RiceFertilizerError):
    """Source data failed validation."""

class DomainError(RiceFertilizerError):
    """Candidate cannot be evaluated safely."""
