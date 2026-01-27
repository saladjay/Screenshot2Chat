"""Custom exceptions for analysis pipeline."""


class AnalysisError(Exception):
    """Base class for analysis errors."""


class ConfigError(AnalysisError):
    """Raised when analysis configuration is invalid or missing."""


class NicknameAnalysisError(AnalysisError):
    """Base class for nickname detection errors."""


class NicknameNotFoundError(NicknameAnalysisError):
    """Raised when no nickname candidate is found."""


class NicknameScoreTooLowError(NicknameAnalysisError):
    """Raised when nickname score is below the configured threshold."""


class DialogAnalysisError(AnalysisError):
    """Base class for dialog analysis errors."""


class DialogCountTooLowError(DialogAnalysisError):
    """Raised when dialog bubble count is below the configured threshold."""


class UnknownSpeakerTooHighError(DialogAnalysisError):
    """Raised when unknown speaker ratio exceeds the configured threshold."""
