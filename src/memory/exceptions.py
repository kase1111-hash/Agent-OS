"""
Agent OS Memory Vault Exceptions

Custom exceptions for memory vault operations providing consistent error handling.
"""

from typing import Optional


class VaultError(Exception):
    """Base exception for vault operations."""

    pass


class VaultNotInitializedError(VaultError):
    """Operation attempted on an uninitialized vault."""

    def __init__(self, operation: str = "access"):
        super().__init__(f"Cannot {operation}: Vault not initialized")
        self.operation = operation


class VaultInitializationError(VaultError):
    """Vault initialization failed."""

    def __init__(
        self,
        message: str,
        component: Optional[str] = None,
        original_error: Optional[Exception] = None,
    ):
        self.component = component
        self.original_error = original_error
        super().__init__(message)


class ConsentError(VaultError):
    """Base exception for consent-related errors."""

    pass


class ConsentDeniedError(ConsentError):
    """Consent was not granted for the requested operation."""

    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        reason: Optional[str] = None,
    ):
        self.operation = operation
        self.reason = reason
        super().__init__(message)


class ConsentNotFoundError(ConsentError):
    """Referenced consent record does not exist."""

    def __init__(self, consent_id: str):
        self.consent_id = consent_id
        super().__init__(f"Consent not found: {consent_id}")


class ConsentExpiredError(ConsentError):
    """Consent has expired."""

    def __init__(self, consent_id: str):
        self.consent_id = consent_id
        super().__init__(f"Consent expired: {consent_id}")


class BlobError(VaultError):
    """Base exception for blob storage errors."""

    pass


class BlobNotFoundError(BlobError):
    """Requested blob does not exist."""

    def __init__(self, blob_id: str):
        self.blob_id = blob_id
        super().__init__(f"Blob not found: {blob_id}")


class BlobSealedError(BlobError):
    """Blob is sealed and cannot be accessed."""

    def __init__(self, blob_id: str):
        self.blob_id = blob_id
        super().__init__(f"Blob is sealed: {blob_id}")


class StorageError(VaultError):
    """Error during storage operations."""

    def __init__(
        self,
        message: str,
        operation: str,
        original_error: Optional[Exception] = None,
    ):
        self.operation = operation
        self.original_error = original_error
        super().__init__(message)


class EncryptionError(VaultError):
    """Error during encryption/decryption."""

    def __init__(
        self,
        message: str,
        blob_id: Optional[str] = None,
        original_error: Optional[Exception] = None,
    ):
        self.blob_id = blob_id
        self.original_error = original_error
        super().__init__(message)


class GenesisVerificationError(VaultError):
    """Genesis record verification failed."""

    def __init__(self, message: str):
        super().__init__(message)


class IntegrityError(VaultError):
    """Vault integrity check failed."""

    def __init__(self, message: str, details: Optional[str] = None):
        self.details = details
        super().__init__(message)
