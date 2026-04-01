"""
Custom exceptions for the NID OCR application.

All service-layer errors originate from one of these three classes so
that callers can catch them with a single `except OCRServiceError` or
drill down to a specific subtype.
"""


class OCRServiceError(Exception):
    """
    Base class for all OCR-pipeline errors.

    Raise this (or a subclass) whenever the service layer encounters a
    condition that should be surfaced to the API layer as a 4xx / 5xx
    response rather than propagated as an unhandled Python exception.
    """


class PreprocessingError(OCRServiceError):
    """
    Raised by :class:`~nid_app.image_preprocessor.NIDImagePreprocessor`
    when an image cannot be loaded, decoded, or prepared for OCR.

    Typical causes:
    - File not found or inaccessible path.
    - Corrupted / unsupported image format.
    - Image resolution below the minimum threshold (< 200 × 200 px).
    """


class OCRTimeoutError(OCRServiceError):
    """
    Raised when the OCR engine exceeds the per-request time budget
    (default 60 seconds).

    This protects the web worker from being blocked indefinitely by a
    very large or adversarial image.
    """
