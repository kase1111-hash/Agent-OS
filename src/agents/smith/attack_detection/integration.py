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

import asyncio
import logging
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.agents.smith.agent import SmithAgent
    from src.boundary.daemon.boundary_daemon import SmithDaemon

from .notifications import (
    NotificationManager,
    NotificationConfig,
    NotificationChannel,
    NotificationPriority,
    SecurityAlert,
    create_notification_manager,
    create_alert_from_attack,
    create_console_channel,
    create_slack_channel,
    create_email_channel,
    create_pagerduty_channel,
    create_webhook_channel,
)
from .config import (
    AttackDetectionConfig,
    ConfigLoader,
    load_config,
    SeverityLevel,
    NotificationChannelType,
)

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
    - Notification System
    """

    def __init__(
        self,
        smith: "SmithAgent",
        daemon: Optional["SmithDaemon"] = None,
        notification_manager: Optional[NotificationManager] = None,
        on_attack: Optional[Callable[[Any], None]] = None,
        on_recommendation: Optional[Callable[[Any], None]] = None,
        enable_console_notifications: bool = True,
    ):
        """
        Initialize the attack detection pipeline.

        Args:
            smith: Smith agent (must have attack detection enabled)
            daemon: Optional boundary daemon to connect
            notification_manager: Optional notification manager for alerts
            on_attack: Optional callback for attack events
            on_recommendation: Optional callback for recommendations
            enable_console_notifications: Enable console output for testing
        """
        self.smith = smith
        self.daemon = daemon
        self._disconnect_fn: Optional[Callable[[], None]] = None
        self._running = False

        # Set up notifications
        self.notification_manager = notification_manager or create_notification_manager()

        # Add console channel if enabled (for development/testing)
        if enable_console_notifications:
            name, config = create_console_channel()
            self.notification_manager.add_channel(name, config)

        # Register attack callback that sends notifications
        self._user_attack_callback = on_attack
        smith.register_attack_callback(self._handle_attack)

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

    def _handle_attack(self, attack: Any) -> None:
        """Handle attack events by sending notifications and calling user callback."""
        try:
            # Create alert from attack
            alert = create_alert_from_attack(
                attack_id=attack.attack_id,
                attack_type=attack.attack_type.name if hasattr(attack.attack_type, 'name') else str(attack.attack_type),
                severity=attack.severity.name if hasattr(attack.severity, 'name') else str(attack.severity),
                description=attack.description,
                source=attack.source if hasattr(attack, 'source') else None,
                target=attack.target if hasattr(attack, 'target') else None,
                indicators=attack.indicators if hasattr(attack, 'indicators') else [],
                mitre_tactics=attack.mitre_techniques if hasattr(attack, 'mitre_techniques') else [],
                recommendations=[],
            )

            # Send notifications (async in background)
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.ensure_future(self.notification_manager.send_alert(alert))
                else:
                    loop.run_until_complete(self.notification_manager.send_alert(alert))
            except RuntimeError:
                # No event loop, use sync method
                self.notification_manager.send_alert_sync(alert)

        except Exception as e:
            logger.error(f"Failed to send attack notification: {e}")

        # Call user callback if provided
        if self._user_attack_callback:
            try:
                self._user_attack_callback(attack)
            except Exception as e:
                logger.error(f"User attack callback error: {e}")

    def add_notification_channel(
        self,
        name: str,
        config: NotificationConfig,
    ) -> bool:
        """Add a notification channel to the pipeline."""
        return self.notification_manager.add_channel(name, config)

    def get_status(self) -> Dict[str, Any]:
        """Get pipeline status."""
        status = {
            "running": self._running,
            "smith_status": self.smith.get_attack_detection_status(),
            "daemon_connected": self.daemon is not None and self._disconnect_fn is not None,
            "notification_channels": self.notification_manager.get_channel_status(),
        }

        if self.daemon:
            status["daemon_mode"] = self.daemon.mode.name
            status["daemon_running"] = self.daemon.is_running

        return status


def setup_attack_detection_pipeline(
    smith: "SmithAgent",
    daemon: Optional["SmithDaemon"] = None,
    notification_manager: Optional[NotificationManager] = None,
    auto_start: bool = True,
    on_attack: Optional[Callable[[Any], None]] = None,
    on_recommendation: Optional[Callable[[Any], None]] = None,
    enable_console_notifications: bool = True,
) -> AttackDetectionPipeline:
    """
    Set up the complete attack detection pipeline.

    This is a convenience function that creates and optionally starts
    the attack detection pipeline.

    Args:
        smith: Smith agent (must have attack detection enabled)
        daemon: Optional boundary daemon to connect
        notification_manager: Optional notification manager for alerts
        auto_start: Whether to start the pipeline immediately
        on_attack: Optional callback for attack events
        on_recommendation: Optional callback for recommendations
        enable_console_notifications: Enable console output for testing

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

        # Add Slack notifications for critical alerts
        from src.agents.smith.attack_detection import create_slack_channel
        name, config = create_slack_channel("slack", "https://hooks.slack.com/...")
        pipeline.add_notification_channel(name, config)
    """
    pipeline = AttackDetectionPipeline(
        smith=smith,
        daemon=daemon,
        notification_manager=notification_manager,
        on_attack=on_attack,
        on_recommendation=on_recommendation,
        enable_console_notifications=enable_console_notifications,
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


def setup_pipeline_from_config(
    smith: "SmithAgent",
    config: Optional[AttackDetectionConfig] = None,
    config_path: Optional[str] = None,
    daemon: Optional["SmithDaemon"] = None,
    auto_start: bool = True,
    on_attack: Optional[Callable[[Any], None]] = None,
    on_recommendation: Optional[Callable[[Any], None]] = None,
) -> AttackDetectionPipeline:
    """
    Set up attack detection pipeline from configuration.

    This is a convenience function that loads configuration and creates
    a fully configured pipeline with notifications based on config.

    Args:
        smith: Smith agent (must have attack detection enabled)
        config: Pre-loaded configuration (overrides config_path)
        config_path: Path to YAML configuration file
        daemon: Optional boundary daemon to connect
        auto_start: Whether to start the pipeline immediately
        on_attack: Optional callback for attack events
        on_recommendation: Optional callback for recommendations

    Returns:
        Configured AttackDetectionPipeline

    Example:
        from src.agents.smith.attack_detection import setup_pipeline_from_config

        pipeline = setup_pipeline_from_config(
            smith=smith,
            config_path="config/attack_detection.yaml",
            daemon=daemon,
        )
    """
    # Load configuration
    if config is None:
        if config_path:
            config = load_config(path=config_path)
        else:
            config = load_config()

    # Create notification manager and configure channels from config
    notification_manager = create_notification_manager()

    # Add console channel if enabled in config
    if config.notifications.include_console:
        name, channel_config = create_console_channel()
        notification_manager.add_channel(name, channel_config)

    # Configure notification channels from config
    for channel_cfg in config.notifications.channels:
        if not channel_cfg.enabled:
            continue

        try:
            # Map severity
            severity_map = {
                SeverityLevel.LOW: NotificationPriority.LOW,
                SeverityLevel.MEDIUM: NotificationPriority.MEDIUM,
                SeverityLevel.HIGH: NotificationPriority.HIGH,
                SeverityLevel.CRITICAL: NotificationPriority.URGENT,
                SeverityLevel.CATASTROPHIC: NotificationPriority.CRITICAL,
            }
            min_severity = severity_map.get(channel_cfg.min_severity, NotificationPriority.MEDIUM)

            if channel_cfg.type == NotificationChannelType.SLACK:
                if channel_cfg.webhook_url:
                    name, channel = create_slack_channel(
                        name=channel_cfg.name,
                        webhook_url=channel_cfg.webhook_url,
                        channel=channel_cfg.channel,
                        min_severity=min_severity,
                    )
                    notification_manager.add_channel(name, channel)

            elif channel_cfg.type == NotificationChannelType.EMAIL:
                if channel_cfg.smtp_host and channel_cfg.from_address and channel_cfg.to_addresses:
                    name, channel = create_email_channel(
                        name=channel_cfg.name,
                        smtp_host=channel_cfg.smtp_host,
                        from_address=channel_cfg.from_address,
                        to_addresses=channel_cfg.to_addresses,
                        smtp_port=channel_cfg.smtp_port,
                        use_tls=channel_cfg.use_tls,
                        username=channel_cfg.username,
                        password=channel_cfg.password,
                        min_severity=min_severity,
                    )
                    notification_manager.add_channel(name, channel)

            elif channel_cfg.type == NotificationChannelType.PAGERDUTY:
                if channel_cfg.routing_key:
                    name, channel = create_pagerduty_channel(
                        name=channel_cfg.name,
                        routing_key=channel_cfg.routing_key,
                        min_severity=min_severity,
                    )
                    notification_manager.add_channel(name, channel)

            elif channel_cfg.type == NotificationChannelType.WEBHOOK:
                if channel_cfg.webhook_url:
                    name, channel = create_webhook_channel(
                        name=channel_cfg.name,
                        url=channel_cfg.webhook_url,
                        headers=channel_cfg.headers,
                        min_severity=min_severity,
                    )
                    notification_manager.add_channel(name, channel)

            elif channel_cfg.type == NotificationChannelType.CONSOLE:
                name, channel = create_console_channel(
                    name=channel_cfg.name,
                    min_severity=min_severity,
                )
                notification_manager.add_channel(name, channel)

            logger.info(f"Configured notification channel: {channel_cfg.name} ({channel_cfg.type.value})")

        except Exception as e:
            logger.error(f"Failed to configure notification channel {channel_cfg.name}: {e}")

    # Create pipeline with configured notification manager
    pipeline = AttackDetectionPipeline(
        smith=smith,
        daemon=daemon,
        notification_manager=notification_manager,
        on_attack=on_attack,
        on_recommendation=on_recommendation,
        enable_console_notifications=False,  # Already handled above
    )

    if auto_start:
        pipeline.start()

    logger.info(f"Attack detection pipeline configured from {'file' if config_path else 'defaults'}")

    return pipeline


# Export key classes/functions
__all__ = [
    "connect_boundary_to_smith",
    "AttackDetectionPipeline",
    "setup_attack_detection_pipeline",
    "setup_pipeline_from_config",
    "create_attack_alert_handler",
]
