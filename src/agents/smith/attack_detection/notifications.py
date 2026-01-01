"""
Notification System for Attack Alerting

This module provides multi-channel alerting for security incidents
detected by the attack detection system.

Supported Channels:
- Email (SMTP)
- Webhook (generic HTTP POST)
- Slack (via webhook or API)
- PagerDuty (via Events API)
- Console (for development/testing)

Features:
- Severity-based filtering
- Rate limiting and throttling
- Alert aggregation
- Notification templates
- Delivery tracking and retry
"""

import asyncio
import hashlib
import json
import logging
import smtplib
import threading
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin

logger = logging.getLogger(__name__)


class NotificationChannel(Enum):
    """Available notification channels."""
    CONSOLE = auto()
    EMAIL = auto()
    WEBHOOK = auto()
    SLACK = auto()
    PAGERDUTY = auto()
    TEAMS = auto()


class NotificationPriority(Enum):
    """Notification priority levels."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    URGENT = 4
    CRITICAL = 5


class DeliveryStatus(Enum):
    """Notification delivery status."""
    PENDING = auto()
    SENT = auto()
    DELIVERED = auto()
    FAILED = auto()
    THROTTLED = auto()
    SUPPRESSED = auto()


@dataclass
class NotificationConfig:
    """Configuration for a notification channel."""
    channel: NotificationChannel
    enabled: bool = True

    # Severity filtering
    min_severity: NotificationPriority = NotificationPriority.MEDIUM

    # Throttling
    rate_limit: int = 10  # Max notifications per window
    rate_window_seconds: int = 60  # Window size in seconds

    # Aggregation
    aggregate_window_seconds: int = 300  # 5 minutes
    aggregate_similar: bool = True

    # Retry
    max_retries: int = 3
    retry_delay_seconds: int = 30

    # Channel-specific config
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SecurityAlert:
    """A security alert to be sent."""
    alert_id: str
    title: str
    description: str
    severity: NotificationPriority
    timestamp: datetime

    # Attack context
    attack_id: Optional[str] = None
    attack_type: Optional[str] = None
    source: Optional[str] = None
    target: Optional[str] = None

    # Additional info
    indicators: List[str] = field(default_factory=list)
    mitre_tactics: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "title": self.title,
            "description": self.description,
            "severity": self.severity.name,
            "timestamp": self.timestamp.isoformat(),
            "attack_id": self.attack_id,
            "attack_type": self.attack_type,
            "source": self.source,
            "target": self.target,
            "indicators": self.indicators,
            "mitre_tactics": self.mitre_tactics,
            "recommendations": self.recommendations,
            "metadata": self.metadata,
        }

    def get_fingerprint(self) -> str:
        """Get fingerprint for deduplication/aggregation."""
        key = f"{self.attack_type}:{self.source}:{self.target}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]


@dataclass
class NotificationRecord:
    """Record of a notification attempt."""
    record_id: str
    alert: SecurityAlert
    channel: NotificationChannel
    status: DeliveryStatus
    created_at: datetime
    sent_at: Optional[datetime] = None
    delivered_at: Optional[datetime] = None
    error_message: Optional[str] = None
    retry_count: int = 0


class NotificationSender(ABC):
    """Abstract base class for notification senders."""

    @abstractmethod
    async def send(
        self,
        alert: SecurityAlert,
        config: NotificationConfig,
    ) -> Tuple[bool, Optional[str]]:
        """
        Send a notification.

        Returns:
            Tuple of (success, error_message)
        """
        pass

    @abstractmethod
    def validate_config(self, config: NotificationConfig) -> bool:
        """Validate channel configuration."""
        pass


class ConsoleSender(NotificationSender):
    """Console/log notification sender for development."""

    SEVERITY_COLORS = {
        NotificationPriority.LOW: "\033[32m",      # Green
        NotificationPriority.MEDIUM: "\033[33m",   # Yellow
        NotificationPriority.HIGH: "\033[91m",     # Light red
        NotificationPriority.URGENT: "\033[31m",   # Red
        NotificationPriority.CRITICAL: "\033[35m", # Magenta
    }
    RESET = "\033[0m"

    async def send(
        self,
        alert: SecurityAlert,
        config: NotificationConfig,
    ) -> Tuple[bool, Optional[str]]:
        """Print alert to console."""
        color = self.SEVERITY_COLORS.get(alert.severity, "")

        output = [
            f"\n{color}{'=' * 60}{self.RESET}",
            f"{color}🚨 SECURITY ALERT: {alert.title}{self.RESET}",
            f"{color}{'=' * 60}{self.RESET}",
            f"Severity: {alert.severity.name}",
            f"Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Alert ID: {alert.alert_id}",
        ]

        if alert.attack_id:
            output.append(f"Attack ID: {alert.attack_id}")
        if alert.attack_type:
            output.append(f"Attack Type: {alert.attack_type}")
        if alert.source:
            output.append(f"Source: {alert.source}")
        if alert.target:
            output.append(f"Target: {alert.target}")

        output.append(f"\nDescription:\n{alert.description}")

        if alert.indicators:
            output.append(f"\nIndicators: {', '.join(alert.indicators)}")
        if alert.mitre_tactics:
            output.append(f"MITRE Tactics: {', '.join(alert.mitre_tactics)}")
        if alert.recommendations:
            output.append("\nRecommendations:")
            for rec in alert.recommendations:
                output.append(f"  • {rec}")

        output.append(f"{color}{'=' * 60}{self.RESET}\n")

        print("\n".join(output))
        logger.info(f"Console alert sent: {alert.alert_id}")

        return True, None

    def validate_config(self, config: NotificationConfig) -> bool:
        return True


class EmailSender(NotificationSender):
    """Email notification sender via SMTP."""

    async def send(
        self,
        alert: SecurityAlert,
        config: NotificationConfig,
    ) -> Tuple[bool, Optional[str]]:
        """Send alert via email."""
        smtp_config = config.config

        required = ["smtp_host", "from_address", "to_addresses"]
        for key in required:
            if key not in smtp_config:
                return False, f"Missing required config: {key}"

        try:
            # Build email
            msg = MIMEMultipart("alternative")
            msg["Subject"] = self._build_subject(alert)
            msg["From"] = smtp_config["from_address"]
            msg["To"] = ", ".join(smtp_config["to_addresses"])

            # Plain text version
            text_body = self._build_text_body(alert)
            msg.attach(MIMEText(text_body, "plain"))

            # HTML version
            html_body = self._build_html_body(alert)
            msg.attach(MIMEText(html_body, "html"))

            # Send email
            smtp_host = smtp_config["smtp_host"]
            smtp_port = smtp_config.get("smtp_port", 587)
            use_tls = smtp_config.get("use_tls", True)

            # Run SMTP in executor to not block
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self._send_smtp,
                msg,
                smtp_host,
                smtp_port,
                use_tls,
                smtp_config.get("username"),
                smtp_config.get("password"),
                smtp_config["to_addresses"],
            )

            logger.info(f"Email alert sent: {alert.alert_id}")
            return True, None

        except Exception as e:
            logger.error(f"Email send failed: {e}")
            return False, str(e)

    def _send_smtp(
        self,
        msg: MIMEMultipart,
        host: str,
        port: int,
        use_tls: bool,
        username: Optional[str],
        password: Optional[str],
        recipients: List[str],
    ) -> None:
        """Synchronous SMTP send."""
        with smtplib.SMTP(host, port, timeout=30) as server:
            if use_tls:
                server.starttls()
            if username and password:
                server.login(username, password)
            server.sendmail(msg["From"], recipients, msg.as_string())

    def _build_subject(self, alert: SecurityAlert) -> str:
        """Build email subject line."""
        severity_prefix = {
            NotificationPriority.LOW: "[LOW]",
            NotificationPriority.MEDIUM: "[MEDIUM]",
            NotificationPriority.HIGH: "[HIGH]",
            NotificationPriority.URGENT: "[URGENT]",
            NotificationPriority.CRITICAL: "🚨 [CRITICAL]",
        }
        prefix = severity_prefix.get(alert.severity, "")
        return f"{prefix} Security Alert: {alert.title}"

    def _build_text_body(self, alert: SecurityAlert) -> str:
        """Build plain text email body."""
        lines = [
            f"SECURITY ALERT: {alert.title}",
            "=" * 50,
            "",
            f"Severity: {alert.severity.name}",
            f"Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"Alert ID: {alert.alert_id}",
        ]

        if alert.attack_id:
            lines.append(f"Attack ID: {alert.attack_id}")
        if alert.attack_type:
            lines.append(f"Attack Type: {alert.attack_type}")
        if alert.source:
            lines.append(f"Source: {alert.source}")
        if alert.target:
            lines.append(f"Target: {alert.target}")

        lines.extend(["", "DESCRIPTION:", alert.description, ""])

        if alert.indicators:
            lines.extend(["INDICATORS:", *[f"  - {i}" for i in alert.indicators], ""])

        if alert.mitre_tactics:
            lines.extend(["MITRE ATT&CK:", *[f"  - {t}" for t in alert.mitre_tactics], ""])

        if alert.recommendations:
            lines.extend(["RECOMMENDATIONS:", *[f"  - {r}" for r in alert.recommendations], ""])

        lines.extend([
            "",
            "---",
            "This alert was generated by Agent Smith Attack Detection System.",
        ])

        return "\n".join(lines)

    def _build_html_body(self, alert: SecurityAlert) -> str:
        """Build HTML email body."""
        severity_colors = {
            NotificationPriority.LOW: "#28a745",
            NotificationPriority.MEDIUM: "#ffc107",
            NotificationPriority.HIGH: "#fd7e14",
            NotificationPriority.URGENT: "#dc3545",
            NotificationPriority.CRITICAL: "#6f42c1",
        }
        color = severity_colors.get(alert.severity, "#6c757d")

        indicators_html = ""
        if alert.indicators:
            indicators_html = "<h3>Indicators</h3><ul>"
            for ind in alert.indicators:
                indicators_html += f"<li><code>{ind}</code></li>"
            indicators_html += "</ul>"

        tactics_html = ""
        if alert.mitre_tactics:
            tactics_html = "<h3>MITRE ATT&CK Tactics</h3><ul>"
            for tactic in alert.mitre_tactics:
                tactics_html += f"<li>{tactic}</li>"
            tactics_html += "</ul>"

        recommendations_html = ""
        if alert.recommendations:
            recommendations_html = "<h3>Recommendations</h3><ul>"
            for rec in alert.recommendations:
                recommendations_html += f"<li>{rec}</li>"
            recommendations_html += "</ul>"

        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }}
                .alert-header {{ background: {color}; color: white; padding: 20px; border-radius: 8px 8px 0 0; }}
                .alert-body {{ padding: 20px; background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 0 0 8px 8px; }}
                .meta {{ color: #6c757d; margin: 10px 0; }}
                code {{ background: #e9ecef; padding: 2px 6px; border-radius: 4px; }}
                h3 {{ color: #495057; border-bottom: 1px solid #dee2e6; padding-bottom: 8px; }}
            </style>
        </head>
        <body>
            <div class="alert-header">
                <h1>🚨 {alert.title}</h1>
                <p><strong>Severity:</strong> {alert.severity.name}</p>
            </div>
            <div class="alert-body">
                <div class="meta">
                    <p><strong>Time:</strong> {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
                    <p><strong>Alert ID:</strong> <code>{alert.alert_id}</code></p>
                    {f'<p><strong>Attack ID:</strong> <code>{alert.attack_id}</code></p>' if alert.attack_id else ''}
                    {f'<p><strong>Attack Type:</strong> {alert.attack_type}</p>' if alert.attack_type else ''}
                    {f'<p><strong>Source:</strong> {alert.source}</p>' if alert.source else ''}
                    {f'<p><strong>Target:</strong> {alert.target}</p>' if alert.target else ''}
                </div>

                <h3>Description</h3>
                <p>{alert.description}</p>

                {indicators_html}
                {tactics_html}
                {recommendations_html}

                <hr>
                <p style="color: #6c757d; font-size: 12px;">
                    This alert was generated by Agent Smith Attack Detection System.
                </p>
            </div>
        </body>
        </html>
        """

    def validate_config(self, config: NotificationConfig) -> bool:
        required = ["smtp_host", "from_address", "to_addresses"]
        return all(key in config.config for key in required)


class WebhookSender(NotificationSender):
    """Generic webhook notification sender."""

    async def send(
        self,
        alert: SecurityAlert,
        config: NotificationConfig,
    ) -> Tuple[bool, Optional[str]]:
        """Send alert via HTTP webhook."""
        webhook_config = config.config

        if "url" not in webhook_config:
            return False, "Missing required config: url"

        try:
            import aiohttp
        except ImportError:
            # Fall back to requests
            return await self._send_with_requests(alert, webhook_config)

        try:
            url = webhook_config["url"]
            headers = webhook_config.get("headers", {})
            headers.setdefault("Content-Type", "application/json")

            # Build payload
            payload = self._build_payload(alert, webhook_config)

            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    url,
                    json=payload,
                    headers=headers,
                    ssl=webhook_config.get("verify_ssl", True),
                ) as response:
                    if response.status >= 200 and response.status < 300:
                        logger.info(f"Webhook alert sent: {alert.alert_id}")
                        return True, None
                    else:
                        error = f"Webhook returned {response.status}"
                        logger.error(error)
                        return False, error

        except Exception as e:
            logger.error(f"Webhook send failed: {e}")
            return False, str(e)

    async def _send_with_requests(
        self,
        alert: SecurityAlert,
        webhook_config: Dict[str, Any],
    ) -> Tuple[bool, Optional[str]]:
        """Fallback to requests library."""
        try:
            import requests
        except ImportError:
            return False, "Neither aiohttp nor requests available"

        try:
            url = webhook_config["url"]
            headers = webhook_config.get("headers", {})
            headers.setdefault("Content-Type", "application/json")

            payload = self._build_payload(alert, webhook_config)

            response = requests.post(
                url,
                json=payload,
                headers=headers,
                verify=webhook_config.get("verify_ssl", True),
                timeout=30,
            )

            if response.status_code >= 200 and response.status_code < 300:
                return True, None
            else:
                return False, f"Webhook returned {response.status_code}"

        except Exception as e:
            return False, str(e)

    def _build_payload(
        self,
        alert: SecurityAlert,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build webhook payload."""
        template = config.get("payload_template")

        if template:
            # Simple template substitution
            payload = json.loads(
                json.dumps(template)
                .replace("{{alert_id}}", alert.alert_id)
                .replace("{{title}}", alert.title)
                .replace("{{description}}", alert.description)
                .replace("{{severity}}", alert.severity.name)
                .replace("{{timestamp}}", alert.timestamp.isoformat())
                .replace("{{attack_type}}", alert.attack_type or "")
            )
            return payload
        else:
            return alert.to_dict()

    def validate_config(self, config: NotificationConfig) -> bool:
        return "url" in config.config


class SlackSender(NotificationSender):
    """Slack notification sender."""

    SEVERITY_COLORS = {
        NotificationPriority.LOW: "#36a64f",
        NotificationPriority.MEDIUM: "#daa038",
        NotificationPriority.HIGH: "#ff9800",
        NotificationPriority.URGENT: "#dc3545",
        NotificationPriority.CRITICAL: "#6f42c1",
    }

    async def send(
        self,
        alert: SecurityAlert,
        config: NotificationConfig,
    ) -> Tuple[bool, Optional[str]]:
        """Send alert to Slack."""
        slack_config = config.config

        if "webhook_url" not in slack_config:
            return False, "Missing required config: webhook_url"

        try:
            import aiohttp
        except ImportError:
            try:
                import requests
            except ImportError:
                return False, "Neither aiohttp nor requests available"
            return self._send_with_requests(alert, slack_config)

        try:
            payload = self._build_slack_message(alert, slack_config)

            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    slack_config["webhook_url"],
                    json=payload,
                ) as response:
                    if response.status == 200:
                        logger.info(f"Slack alert sent: {alert.alert_id}")
                        return True, None
                    else:
                        text = await response.text()
                        return False, f"Slack returned {response.status}: {text}"

        except Exception as e:
            logger.error(f"Slack send failed: {e}")
            return False, str(e)

    def _send_with_requests(
        self,
        alert: SecurityAlert,
        slack_config: Dict[str, Any],
    ) -> Tuple[bool, Optional[str]]:
        """Synchronous fallback."""
        import requests

        try:
            payload = self._build_slack_message(alert, slack_config)
            response = requests.post(
                slack_config["webhook_url"],
                json=payload,
                timeout=30,
            )

            if response.status_code == 200:
                return True, None
            else:
                return False, f"Slack returned {response.status_code}"
        except Exception as e:
            return False, str(e)

    def _build_slack_message(
        self,
        alert: SecurityAlert,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build Slack Block Kit message."""
        color = self.SEVERITY_COLORS.get(alert.severity, "#6c757d")

        # Build fields
        fields = [
            {"type": "mrkdwn", "text": f"*Severity:*\n{alert.severity.name}"},
            {"type": "mrkdwn", "text": f"*Time:*\n{alert.timestamp.strftime('%Y-%m-%d %H:%M')}"},
        ]

        if alert.attack_type:
            fields.append({"type": "mrkdwn", "text": f"*Attack Type:*\n{alert.attack_type}"})
        if alert.source:
            fields.append({"type": "mrkdwn", "text": f"*Source:*\n{alert.source}"})

        blocks = [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": f"🚨 {alert.title}", "emoji": True}
            },
            {
                "type": "section",
                "fields": fields[:4]  # Max 4 fields per section
            },
        ]

        # Description
        if alert.description:
            blocks.append({
                "type": "section",
                "text": {"type": "mrkdwn", "text": alert.description[:3000]}
            })

        # Indicators
        if alert.indicators:
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Indicators:* `{', '.join(alert.indicators[:5])}`"
                }
            })

        # MITRE Tactics
        if alert.mitre_tactics:
            blocks.append({
                "type": "context",
                "elements": [
                    {"type": "mrkdwn", "text": f"*MITRE ATT&CK:* {', '.join(alert.mitre_tactics)}"}
                ]
            })

        # Alert ID footer
        blocks.append({
            "type": "context",
            "elements": [
                {"type": "mrkdwn", "text": f"Alert ID: `{alert.alert_id}`"}
            ]
        })

        payload = {
            "attachments": [
                {
                    "color": color,
                    "blocks": blocks,
                }
            ]
        }

        # Add channel if specified
        if "channel" in config:
            payload["channel"] = config["channel"]

        return payload

    def validate_config(self, config: NotificationConfig) -> bool:
        return "webhook_url" in config.config


class PagerDutySender(NotificationSender):
    """PagerDuty notification sender via Events API v2."""

    EVENTS_URL = "https://events.pagerduty.com/v2/enqueue"

    SEVERITY_MAP = {
        NotificationPriority.LOW: "info",
        NotificationPriority.MEDIUM: "warning",
        NotificationPriority.HIGH: "error",
        NotificationPriority.URGENT: "critical",
        NotificationPriority.CRITICAL: "critical",
    }

    async def send(
        self,
        alert: SecurityAlert,
        config: NotificationConfig,
    ) -> Tuple[bool, Optional[str]]:
        """Send alert to PagerDuty."""
        pd_config = config.config

        if "routing_key" not in pd_config:
            return False, "Missing required config: routing_key"

        try:
            import aiohttp
        except ImportError:
            try:
                import requests
            except ImportError:
                return False, "Neither aiohttp nor requests available"
            return self._send_with_requests(alert, pd_config)

        try:
            payload = self._build_pd_event(alert, pd_config)

            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    self.EVENTS_URL,
                    json=payload,
                ) as response:
                    if response.status in (200, 201, 202):
                        logger.info(f"PagerDuty alert sent: {alert.alert_id}")
                        return True, None
                    else:
                        text = await response.text()
                        return False, f"PagerDuty returned {response.status}: {text}"

        except Exception as e:
            logger.error(f"PagerDuty send failed: {e}")
            return False, str(e)

    def _send_with_requests(
        self,
        alert: SecurityAlert,
        pd_config: Dict[str, Any],
    ) -> Tuple[bool, Optional[str]]:
        """Synchronous fallback."""
        import requests

        try:
            payload = self._build_pd_event(alert, pd_config)
            response = requests.post(self.EVENTS_URL, json=payload, timeout=30)

            if response.status_code in (200, 201, 202):
                return True, None
            else:
                return False, f"PagerDuty returned {response.status_code}"
        except Exception as e:
            return False, str(e)

    def _build_pd_event(
        self,
        alert: SecurityAlert,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build PagerDuty Events API v2 payload."""
        severity = self.SEVERITY_MAP.get(alert.severity, "warning")

        # Build custom details
        custom_details = {
            "attack_id": alert.attack_id,
            "attack_type": alert.attack_type,
            "source": alert.source,
            "target": alert.target,
            "indicators": alert.indicators,
            "mitre_tactics": alert.mitre_tactics,
        }

        return {
            "routing_key": config["routing_key"],
            "event_action": "trigger",
            "dedup_key": alert.get_fingerprint(),
            "payload": {
                "summary": f"{alert.title} - {alert.severity.name}",
                "source": alert.source or "agent-smith",
                "severity": severity,
                "timestamp": alert.timestamp.isoformat(),
                "custom_details": custom_details,
            },
            "links": config.get("links", []),
            "images": config.get("images", []),
        }

    def validate_config(self, config: NotificationConfig) -> bool:
        return "routing_key" in config.config


class TeamsSender(NotificationSender):
    """Microsoft Teams notification sender via webhook."""

    SEVERITY_COLORS = {
        NotificationPriority.LOW: "28a745",
        NotificationPriority.MEDIUM: "ffc107",
        NotificationPriority.HIGH: "fd7e14",
        NotificationPriority.URGENT: "dc3545",
        NotificationPriority.CRITICAL: "6f42c1",
    }

    async def send(
        self,
        alert: SecurityAlert,
        config: NotificationConfig,
    ) -> Tuple[bool, Optional[str]]:
        """Send alert to Microsoft Teams."""
        teams_config = config.config

        if "webhook_url" not in teams_config:
            return False, "Missing required config: webhook_url"

        try:
            import aiohttp
        except ImportError:
            try:
                import requests
            except ImportError:
                return False, "Neither aiohttp nor requests available"
            return self._send_with_requests(alert, teams_config)

        try:
            payload = self._build_teams_card(alert, teams_config)

            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    teams_config["webhook_url"],
                    json=payload,
                ) as response:
                    if response.status == 200:
                        logger.info(f"Teams alert sent: {alert.alert_id}")
                        return True, None
                    else:
                        text = await response.text()
                        return False, f"Teams returned {response.status}: {text}"

        except Exception as e:
            logger.error(f"Teams send failed: {e}")
            return False, str(e)

    def _send_with_requests(
        self,
        alert: SecurityAlert,
        teams_config: Dict[str, Any],
    ) -> Tuple[bool, Optional[str]]:
        """Synchronous fallback."""
        import requests

        try:
            payload = self._build_teams_card(alert, teams_config)
            response = requests.post(
                teams_config["webhook_url"],
                json=payload,
                timeout=30,
            )

            if response.status_code == 200:
                return True, None
            else:
                return False, f"Teams returned {response.status_code}"
        except Exception as e:
            return False, str(e)

    def _build_teams_card(
        self,
        alert: SecurityAlert,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build Teams Adaptive Card message."""
        color = self.SEVERITY_COLORS.get(alert.severity, "6c757d")

        # Build facts
        facts = [
            {"name": "Severity", "value": alert.severity.name},
            {"name": "Time", "value": alert.timestamp.strftime("%Y-%m-%d %H:%M:%S")},
            {"name": "Alert ID", "value": alert.alert_id},
        ]

        if alert.attack_type:
            facts.append({"name": "Attack Type", "value": alert.attack_type})
        if alert.source:
            facts.append({"name": "Source", "value": alert.source})
        if alert.target:
            facts.append({"name": "Target", "value": alert.target})

        sections = [
            {
                "activityTitle": f"🚨 {alert.title}",
                "facts": facts,
                "markdown": True,
            }
        ]

        if alert.description:
            sections.append({
                "text": alert.description,
                "markdown": True,
            })

        return {
            "@type": "MessageCard",
            "@context": "http://schema.org/extensions",
            "themeColor": color,
            "summary": f"Security Alert: {alert.title}",
            "sections": sections,
        }

    def validate_config(self, config: NotificationConfig) -> bool:
        return "webhook_url" in config.config


class NotificationManager:
    """
    Central notification manager.

    Handles routing, throttling, and delivery of security alerts
    across multiple notification channels.
    """

    SENDERS: Dict[NotificationChannel, type] = {
        NotificationChannel.CONSOLE: ConsoleSender,
        NotificationChannel.EMAIL: EmailSender,
        NotificationChannel.WEBHOOK: WebhookSender,
        NotificationChannel.SLACK: SlackSender,
        NotificationChannel.PAGERDUTY: PagerDutySender,
        NotificationChannel.TEAMS: TeamsSender,
    }

    def __init__(
        self,
        on_notification: Optional[Callable[[NotificationRecord], None]] = None,
    ):
        """
        Initialize notification manager.

        Args:
            on_notification: Callback when notification is sent
        """
        self._channels: Dict[str, NotificationConfig] = {}
        self._senders: Dict[NotificationChannel, NotificationSender] = {}
        self._on_notification = on_notification

        # Rate limiting
        self._rate_tracker: Dict[str, List[datetime]] = defaultdict(list)
        self._rate_lock = threading.Lock()

        # Aggregation
        self._pending_aggregation: Dict[str, List[SecurityAlert]] = defaultdict(list)
        self._aggregation_lock = threading.Lock()

        # Delivery tracking
        self._records: Dict[str, NotificationRecord] = {}

        # Initialize senders
        for channel, sender_class in self.SENDERS.items():
            self._senders[channel] = sender_class()

    def add_channel(
        self,
        name: str,
        config: NotificationConfig,
    ) -> bool:
        """Add a notification channel."""
        # Validate config
        sender = self._senders.get(config.channel)
        if not sender:
            logger.error(f"No sender for channel: {config.channel}")
            return False

        if not sender.validate_config(config):
            logger.error(f"Invalid config for channel: {name}")
            return False

        self._channels[name] = config
        logger.info(f"Added notification channel: {name} ({config.channel.name})")
        return True

    def remove_channel(self, name: str) -> bool:
        """Remove a notification channel."""
        if name in self._channels:
            del self._channels[name]
            logger.info(f"Removed notification channel: {name}")
            return True
        return False

    def enable_channel(self, name: str) -> bool:
        """Enable a notification channel."""
        if name in self._channels:
            self._channels[name].enabled = True
            return True
        return False

    def disable_channel(self, name: str) -> bool:
        """Disable a notification channel."""
        if name in self._channels:
            self._channels[name].enabled = False
            return True
        return False

    async def send_alert(
        self,
        alert: SecurityAlert,
        channels: Optional[List[str]] = None,
    ) -> Dict[str, Tuple[bool, Optional[str]]]:
        """
        Send an alert to specified or all channels.

        Args:
            alert: The security alert to send
            channels: Specific channels to use (None = all enabled)

        Returns:
            Dict mapping channel name to (success, error_message)
        """
        results: Dict[str, Tuple[bool, Optional[str]]] = {}

        target_channels = channels or list(self._channels.keys())

        for channel_name in target_channels:
            config = self._channels.get(channel_name)
            if not config:
                results[channel_name] = (False, "Channel not found")
                continue

            if not config.enabled:
                results[channel_name] = (False, "Channel disabled")
                continue

            # Check severity threshold
            if alert.severity.value < config.min_severity.value:
                results[channel_name] = (False, "Below severity threshold")
                continue

            # Check rate limiting
            if not self._check_rate_limit(channel_name, config):
                results[channel_name] = (False, "Rate limited")
                self._record_notification(
                    alert, config.channel, DeliveryStatus.THROTTLED
                )
                continue

            # Check aggregation
            if config.aggregate_similar:
                should_send = self._check_aggregation(alert, channel_name, config)
                if not should_send:
                    results[channel_name] = (False, "Aggregated")
                    continue

            # Send notification
            sender = self._senders.get(config.channel)
            if not sender:
                results[channel_name] = (False, "No sender")
                continue

            success, error = await sender.send(alert, config)
            results[channel_name] = (success, error)

            # Record
            status = DeliveryStatus.SENT if success else DeliveryStatus.FAILED
            self._record_notification(alert, config.channel, status, error)

            # Update rate tracker
            self._update_rate_tracker(channel_name)

        return results

    def send_alert_sync(
        self,
        alert: SecurityAlert,
        channels: Optional[List[str]] = None,
    ) -> Dict[str, Tuple[bool, Optional[str]]]:
        """Synchronous wrapper for send_alert."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(self.send_alert(alert, channels))

    def _check_rate_limit(
        self,
        channel_name: str,
        config: NotificationConfig,
    ) -> bool:
        """Check if channel is within rate limit."""
        with self._rate_lock:
            now = datetime.now()
            window_start = now - timedelta(seconds=config.rate_window_seconds)

            # Clean old entries
            self._rate_tracker[channel_name] = [
                t for t in self._rate_tracker[channel_name]
                if t > window_start
            ]

            # Check limit
            return len(self._rate_tracker[channel_name]) < config.rate_limit

    def _update_rate_tracker(self, channel_name: str) -> None:
        """Update rate tracker after successful send."""
        with self._rate_lock:
            self._rate_tracker[channel_name].append(datetime.now())

    def _check_aggregation(
        self,
        alert: SecurityAlert,
        channel_name: str,
        config: NotificationConfig,
    ) -> bool:
        """
        Check if alert should be sent or aggregated.

        Returns True if should send now, False if aggregated.
        """
        with self._aggregation_lock:
            fingerprint = alert.get_fingerprint()
            key = f"{channel_name}:{fingerprint}"

            pending = self._pending_aggregation.get(key, [])

            if not pending:
                # First alert of this type, send it
                self._pending_aggregation[key] = [alert]

                # Schedule aggregation window cleanup
                asyncio.get_event_loop().call_later(
                    config.aggregate_window_seconds,
                    self._flush_aggregation,
                    key,
                )

                return True
            else:
                # Add to pending, don't send
                pending.append(alert)
                return False

    def _flush_aggregation(self, key: str) -> None:
        """Flush aggregated alerts after window expires."""
        with self._aggregation_lock:
            if key in self._pending_aggregation:
                del self._pending_aggregation[key]

    def _record_notification(
        self,
        alert: SecurityAlert,
        channel: NotificationChannel,
        status: DeliveryStatus,
        error: Optional[str] = None,
    ) -> None:
        """Record a notification attempt."""
        record_id = hashlib.sha256(
            f"{alert.alert_id}:{channel.name}:{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]

        record = NotificationRecord(
            record_id=record_id,
            alert=alert,
            channel=channel,
            status=status,
            created_at=datetime.now(),
            sent_at=datetime.now() if status == DeliveryStatus.SENT else None,
            error_message=error,
        )

        self._records[record_id] = record

        if self._on_notification:
            try:
                self._on_notification(record)
            except Exception as e:
                logger.error(f"Notification callback error: {e}")

    def get_channel_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all channels."""
        status = {}
        for name, config in self._channels.items():
            with self._rate_lock:
                recent_count = len(self._rate_tracker.get(name, []))

            status[name] = {
                "channel": config.channel.name,
                "enabled": config.enabled,
                "min_severity": config.min_severity.name,
                "rate_limit": config.rate_limit,
                "recent_notifications": recent_count,
            }
        return status

    def get_recent_records(
        self,
        limit: int = 100,
        channel: Optional[NotificationChannel] = None,
        status: Optional[DeliveryStatus] = None,
    ) -> List[NotificationRecord]:
        """Get recent notification records."""
        records = list(self._records.values())

        if channel:
            records = [r for r in records if r.channel == channel]
        if status:
            records = [r for r in records if r.status == status]

        records.sort(key=lambda r: r.created_at, reverse=True)
        return records[:limit]


# Factory functions

def create_notification_manager(
    on_notification: Optional[Callable[[NotificationRecord], None]] = None,
) -> NotificationManager:
    """Create a notification manager."""
    return NotificationManager(on_notification=on_notification)


def create_console_channel(
    name: str = "console",
    min_severity: NotificationPriority = NotificationPriority.LOW,
) -> Tuple[str, NotificationConfig]:
    """Create a console notification channel config."""
    return name, NotificationConfig(
        channel=NotificationChannel.CONSOLE,
        min_severity=min_severity,
    )


def create_slack_channel(
    name: str,
    webhook_url: str,
    channel: Optional[str] = None,
    min_severity: NotificationPriority = NotificationPriority.HIGH,
) -> Tuple[str, NotificationConfig]:
    """Create a Slack notification channel config."""
    config = {
        "webhook_url": webhook_url,
    }
    if channel:
        config["channel"] = channel

    return name, NotificationConfig(
        channel=NotificationChannel.SLACK,
        min_severity=min_severity,
        config=config,
    )


def create_email_channel(
    name: str,
    smtp_host: str,
    from_address: str,
    to_addresses: List[str],
    smtp_port: int = 587,
    use_tls: bool = True,
    username: Optional[str] = None,
    password: Optional[str] = None,
    min_severity: NotificationPriority = NotificationPriority.HIGH,
) -> Tuple[str, NotificationConfig]:
    """Create an email notification channel config."""
    config = {
        "smtp_host": smtp_host,
        "smtp_port": smtp_port,
        "from_address": from_address,
        "to_addresses": to_addresses,
        "use_tls": use_tls,
    }
    if username:
        config["username"] = username
    if password:
        config["password"] = password

    return name, NotificationConfig(
        channel=NotificationChannel.EMAIL,
        min_severity=min_severity,
        config=config,
    )


def create_pagerduty_channel(
    name: str,
    routing_key: str,
    min_severity: NotificationPriority = NotificationPriority.URGENT,
) -> Tuple[str, NotificationConfig]:
    """Create a PagerDuty notification channel config."""
    return name, NotificationConfig(
        channel=NotificationChannel.PAGERDUTY,
        min_severity=min_severity,
        config={"routing_key": routing_key},
    )


def create_webhook_channel(
    name: str,
    url: str,
    headers: Optional[Dict[str, str]] = None,
    min_severity: NotificationPriority = NotificationPriority.MEDIUM,
) -> Tuple[str, NotificationConfig]:
    """Create a webhook notification channel config."""
    config = {"url": url}
    if headers:
        config["headers"] = headers

    return name, NotificationConfig(
        channel=NotificationChannel.WEBHOOK,
        min_severity=min_severity,
        config=config,
    )


def create_alert_from_attack(
    attack_id: str,
    attack_type: str,
    severity: str,
    description: str,
    source: Optional[str] = None,
    target: Optional[str] = None,
    indicators: Optional[List[str]] = None,
    mitre_tactics: Optional[List[str]] = None,
    recommendations: Optional[List[str]] = None,
) -> SecurityAlert:
    """Create a SecurityAlert from attack detection data."""
    # Map severity string to NotificationPriority
    severity_map = {
        "low": NotificationPriority.LOW,
        "medium": NotificationPriority.MEDIUM,
        "high": NotificationPriority.HIGH,
        "critical": NotificationPriority.URGENT,
        "catastrophic": NotificationPriority.CRITICAL,
    }
    priority = severity_map.get(severity.lower(), NotificationPriority.MEDIUM)

    alert_id = hashlib.sha256(
        f"{attack_id}:{datetime.now().isoformat()}".encode()
    ).hexdigest()[:16]

    return SecurityAlert(
        alert_id=f"ALERT-{alert_id.upper()}",
        title=f"{attack_type} Attack Detected",
        description=description,
        severity=priority,
        timestamp=datetime.now(),
        attack_id=attack_id,
        attack_type=attack_type,
        source=source,
        target=target,
        indicators=indicators or [],
        mitre_tactics=mitre_tactics or [],
        recommendations=recommendations or [],
    )
