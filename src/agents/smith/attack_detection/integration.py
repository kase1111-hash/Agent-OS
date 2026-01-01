"""
Attack Detection Integration Module

Provides utilities for wiring up the attack detection system with
other Agent-OS components. This module makes it easy to:

1. Connect the boundary daemon to the attack detector
2. Set up event routing between components
3. Configure the full attack detection pipeline

Example usage:

    from src.agents.smith.agent import SmithAgent
    from src.boundary.daemon.boundary_daemon import SmithDaemon
    from src.agents.smith.attack_detection.integration import (
        connect_boundary_to_smith,
        setup_attack_detection_pipeline,
    )

    # Create components
    smith = SmithAgent()
    smith.initialize({"attack_detection_enabled": True})

    daemon = SmithDaemon()
    daemon.start()

    # Connect them
    connect_boundary_to_smith(daemon, smith)
"""

import logging
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.agents.smith.agent import SmithAgent
    from src.boundary.daemon.boundary_daemon import SmithDaemon

logger = logging.getLogger(__name__)


def connect_boundary_to_smith(
    daemon: "SmithDaemon",
    smith: "SmithAgent",
) -> Callable[[], None]:
    """
    Connect boundary daemon events to Smith's attack detector.

    This sets up a subscription so that all boundary daemon events
    (tripwires, security violations, enforcement actions, etc.)
    are forwarded to Smith for attack detection analysis.

    Args:
        daemon: The boundary daemon instance
        smith: The Smith agent instance

    Returns:
        A cleanup function to disconnect
    """
    def event_handler(event_type: str, event_data: Dict[str, Any]) -> None:
        """Handle boundary daemon events."""
        # Route to appropriate Smith method based on event type
        if event_type == "tripwire":
            smith.process_tripwire_event(event_data)
        else:
            # General boundary event
            event_data["event_type"] = event_type
            smith.process_boundary_event(event_data)

    # Subscribe to daemon events
    daemon.subscribe(event_handler)

    logger.info("Connected boundary daemon to Smith attack detector")

    # Return cleanup function
    def disconnect() -> None:
        daemon.unsubscribe(event_handler)
        logger.info("Disconnected boundary daemon from Smith attack detector")

    return disconnect


class AttackDetectionPipeline:
    """
    Coordinates all components of the attack detection system.

    This class manages the lifecycle and connections between:
    - SmithAgent (with attack detection enabled)
    - BoundaryDaemon
    - Attack Detector
    - Analyzer
    - Remediation Engine
    - Recommendation System
    """

    def __init__(
        self,
        smith: "SmithAgent",
        daemon: Optional["SmithDaemon"] = None,
        on_attack: Optional[Callable[[Any], None]] = None,
        on_recommendation: Optional[Callable[[Any], None]] = None,
    ):
        """
        Initialize the attack detection pipeline.

        Args:
            smith: Smith agent (must have attack detection enabled)
            daemon: Optional boundary daemon to connect
            on_attack: Optional callback for attack events
            on_recommendation: Optional callback for recommendations
        """
        self.smith = smith
        self.daemon = daemon
        self._disconnect_fn: Optional[Callable[[], None]] = None
        self._running = False

        # Register callbacks
        if on_attack:
            smith.register_attack_callback(on_attack)

    def start(self) -> bool:
        """
        Start the attack detection pipeline.

        Returns:
            True if started successfully
        """
        if self._running:
            logger.warning("Pipeline already running")
            return True

        # Connect boundary daemon if provided
        if self.daemon:
            self._disconnect_fn = connect_boundary_to_smith(
                self.daemon,
                self.smith,
            )

        self._running = True
        logger.info("Attack detection pipeline started")
        return True

    def stop(self) -> None:
        """Stop the attack detection pipeline."""
        if not self._running:
            return

        # Disconnect boundary daemon
        if self._disconnect_fn:
            self._disconnect_fn()
            self._disconnect_fn = None

        self._running = False
        logger.info("Attack detection pipeline stopped")

    def get_status(self) -> Dict[str, Any]:
        """Get pipeline status."""
        status = {
            "running": self._running,
            "smith_status": self.smith.get_attack_detection_status(),
            "daemon_connected": self.daemon is not None and self._disconnect_fn is not None,
        }

        if self.daemon:
            status["daemon_mode"] = self.daemon.mode.name
            status["daemon_running"] = self.daemon.is_running

        return status


def setup_attack_detection_pipeline(
    smith: "SmithAgent",
    daemon: Optional["SmithDaemon"] = None,
    auto_start: bool = True,
    on_attack: Optional[Callable[[Any], None]] = None,
    on_recommendation: Optional[Callable[[Any], None]] = None,
) -> AttackDetectionPipeline:
    """
    Set up the complete attack detection pipeline.

    This is a convenience function that creates and optionally starts
    the attack detection pipeline.

    Args:
        smith: Smith agent (must have attack detection enabled)
        daemon: Optional boundary daemon to connect
        auto_start: Whether to start the pipeline immediately
        on_attack: Optional callback for attack events
        on_recommendation: Optional callback for recommendations

    Returns:
        Configured AttackDetectionPipeline

    Example:
        smith = create_smith(config={
            "attack_detection_enabled": True,
            "attack_detection_config": {
                "enable_boundary_events": True,
                "auto_lockdown_on_critical": True,
            }
        })

        daemon = create_smith_daemon()
        daemon.start()

        pipeline = setup_attack_detection_pipeline(
            smith=smith,
            daemon=daemon,
            on_attack=lambda a: print(f"Attack: {a.attack_id}"),
        )
    """
    pipeline = AttackDetectionPipeline(
        smith=smith,
        daemon=daemon,
        on_attack=on_attack,
        on_recommendation=on_recommendation,
    )

    if auto_start:
        pipeline.start()

    return pipeline


def create_attack_alert_handler(
    notify_channels: Optional[List[str]] = None,
    log_to_file: Optional[str] = None,
    trigger_lockdown_severity: int = 5,
) -> Callable[[Any], None]:
    """
    Create an attack alert handler for notifications.

    This creates a callback function that can be passed to the
    attack detection system to handle alerts.

    Args:
        notify_channels: List of notification channels (future)
        log_to_file: Path to log attacks to
        trigger_lockdown_severity: Severity at which to recommend lockdown

    Returns:
        Callback function for attack events
    """
    def handler(attack: Any) -> None:
        """Handle attack alerts."""
        severity = attack.severity.value if hasattr(attack.severity, 'value') else attack.severity

        # Log the attack
        logger.warning(
            f"ATTACK ALERT: {attack.attack_id} - "
            f"Type: {attack.attack_type.name} - "
            f"Severity: {attack.severity.name}"
        )

        # Log to file if configured
        if log_to_file:
            try:
                import json
                with open(log_to_file, "a") as f:
                    f.write(json.dumps({
                        "timestamp": datetime.now().isoformat(),
                        "attack_id": attack.attack_id,
                        "type": attack.attack_type.name,
                        "severity": attack.severity.name,
                        "description": attack.description,
                    }) + "\n")
            except Exception as e:
                logger.error(f"Failed to log attack to file: {e}")

        # Check if lockdown should be recommended
        if severity >= trigger_lockdown_severity:
            logger.critical(
                f"LOCKDOWN RECOMMENDED for attack {attack.attack_id}"
            )

    return handler


# Export key classes/functions
__all__ = [
    "connect_boundary_to_smith",
    "AttackDetectionPipeline",
    "setup_attack_detection_pipeline",
    "create_attack_alert_handler",
]
