"""
Agent OS Messaging Exceptions

Custom exceptions for the messaging subsystem to enable precise error handling.
"""

from typing import Optional


class MessageBusError(Exception):
    """Base exception for message bus operations."""

    def __init__(self, message: str, channel: Optional[str] = None):
        self.channel = channel
        super().__init__(message)


class MessageDeliveryError(MessageBusError):
    """Failed to deliver a message to one or more subscribers."""

    def __init__(
        self,
        message: str,
        channel: str,
        subscriber_name: Optional[str] = None,
        partial_delivery: bool = False,
    ):
        self.subscriber_name = subscriber_name
        self.partial_delivery = partial_delivery
        super().__init__(message, channel)


class HandlerExecutionError(MessageBusError):
    """A message handler raised an exception during execution."""

    def __init__(
        self,
        message: str,
        channel: str,
        handler_name: str,
        original_error: Optional[Exception] = None,
    ):
        self.handler_name = handler_name
        self.original_error = original_error
        super().__init__(message, channel)


class SubscriptionError(MessageBusError):
    """Error during subscription or unsubscription."""

    pass


class ChannelNotFoundError(MessageBusError):
    """Requested channel does not exist."""

    pass


class MessageValidationError(MessageBusError):
    """Message failed validation before publishing."""

    pass


class BusShutdownError(MessageBusError):
    """Operation attempted on a shutdown message bus."""

    def __init__(self, operation: str = "publish"):
        super().__init__(f"Cannot {operation}: message bus is shutting down")
