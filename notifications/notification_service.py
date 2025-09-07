"""
Real-time Notification System for Maersk Shipment AI System

This module provides comprehensive notification capabilities for high-risk predictions,
status changes, and system alerts with multiple delivery channels.
"""

import asyncio
import logging
import smtplib
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
import aiohttp
from pathlib import Path

logger = logging.getLogger(__name__)


class NotificationChannel(Enum):
    EMAIL = "email"
    WEBHOOK = "webhook"
    SLACK = "slack"
    TEAMS = "teams"
    SMS = "sms"


class NotificationPriority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class NotificationType(Enum):
    DELAY_PREDICTION = "delay_prediction"
    STATUS_CHANGE = "status_change"
    HIGH_RISK_ALERT = "high_risk_alert"
    SYSTEM_ALERT = "system_alert"
    ML_MODEL_ALERT = "ml_model_alert"
    OPERATIONAL_INSIGHT = "operational_insight"


@dataclass
class NotificationRecipient:
    """Notification recipient configuration"""
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    webhook_url: Optional[str] = None
    slack_channel: Optional[str] = None
    teams_webhook: Optional[str] = None
    active: bool = True
    preferences: Dict[str, bool] = None
    
    def __post_init__(self):
        if self.preferences is None:
            self.preferences = {
                NotificationType.DELAY_PREDICTION.value: True,
                NotificationType.STATUS_CHANGE.value: True,
                NotificationType.HIGH_RISK_ALERT.value: True,
                NotificationType.SYSTEM_ALERT.value: True,
                NotificationType.ML_MODEL_ALERT.value: True,
                NotificationType.OPERATIONAL_INSIGHT.value: False,
            }


@dataclass
class Notification:
    """Notification data structure"""
    id: str
    type: NotificationType
    priority: NotificationPriority
    title: str
    message: str
    data: Dict[str, Any] = None
    recipients: List[str] = None
    channels: List[NotificationChannel] = None
    created_at: datetime = None
    scheduled_for: Optional[datetime] = None
    sent_at: Optional[datetime] = None
    status: str = "pending"
    retry_count: int = 0
    max_retries: int = 3
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.data is None:
            self.data = {}
        if self.recipients is None:
            self.recipients = []
        if self.channels is None:
            self.channels = [NotificationChannel.EMAIL]


class NotificationTemplates:
    """Notification message templates"""
    
    @staticmethod
    def delay_prediction_template(data: Dict[str, Any]) -> Dict[str, str]:
        """Template for delay prediction notifications"""
        shipment_id = data.get('shipment_id', 'Unknown')
        asset_id = data.get('asset_id', 'Unknown')
        delay_probability = data.get('delay_probability', 0) * 100
        risk_level = data.get('risk_level', 'Unknown')
        
        title = f"🚨 High Delay Risk Alert - {asset_id}"
        
        message = f"""
        ⚠️ HIGH DELAY RISK DETECTED
        
        Shipment: {shipment_id}
        Asset: {asset_id}
        Delay Probability: {delay_probability:.1f}%
        Risk Level: {risk_level}
        
        📊 Predicted Factors:
        """
        
        # Add risk factors if available
        risk_factors = data.get('top_risk_factors', [])
        for i, (factor, importance) in enumerate(risk_factors[:3]):
            message += f"\n  {i+1}. {factor}: {importance:.3f}"
        
        message += f"""
        
        🎯 Recommendations:
        • Monitor this shipment closely
        • Check weather and traffic conditions
        • Verify asset status and route
        • Consider alternative routing if possible
        
        Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        return {"title": title, "message": message}
    
    @staticmethod
    def status_change_template(data: Dict[str, Any]) -> Dict[str, str]:
        """Template for status change notifications"""
        shipment_id = data.get('shipment_id', 'Unknown')
        asset_id = data.get('asset_id', 'Unknown')
        old_status = data.get('old_status', 'Unknown')
        new_status = data.get('new_status', 'Unknown')
        
        status_emoji = {
            'In Transit': '🚚',
            'Delivered': '✅',
            'Delayed': '⚠️',
            'Cancelled': '❌'
        }
        
        emoji = status_emoji.get(new_status, '📦')
        title = f"{emoji} Status Update - {asset_id}"
        
        message = f"""
        📋 SHIPMENT STATUS CHANGE
        
        Shipment: {shipment_id}
        Asset: {asset_id}
        Status: {old_status} → {new_status}
        
        Updated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        return {"title": title, "message": message}
    
    @staticmethod
    def high_risk_alert_template(data: Dict[str, Any]) -> Dict[str, str]:
        """Template for high-risk alerts"""
        risk_count = data.get('high_risk_count', 0)
        total_shipments = data.get('total_shipments', 0)
        risk_percentage = (risk_count / total_shipments * 100) if total_shipments > 0 else 0
        
        title = f"🚨 Multiple High-Risk Shipments Detected ({risk_count})"
        
        message = f"""
        ⚠️ HIGH RISK SHIPMENT ALERT
        
        High-Risk Shipments: {risk_count}
        Total Active Shipments: {total_shipments}
        Risk Percentage: {risk_percentage:.1f}%
        
        🎯 Immediate Actions Required:
        • Review all high-risk shipments
        • Implement preventive measures
        • Notify operations team
        • Monitor closely for next 24 hours
        
        Alert generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        return {"title": title, "message": message}
    
    @staticmethod
    def system_alert_template(data: Dict[str, Any]) -> Dict[str, str]:
        """Template for system alerts"""
        alert_type = data.get('alert_type', 'System Alert')
        severity = data.get('severity', 'Medium')
        component = data.get('component', 'System')
        details = data.get('details', 'No details available')
        
        title = f"🔧 {alert_type} - {component}"
        
        message = f"""
        🚨 SYSTEM ALERT
        
        Component: {component}
        Severity: {severity}
        Alert Type: {alert_type}
        
        Details: {details}
        
        Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        return {"title": title, "message": message}
    
    @staticmethod
    def ml_model_alert_template(data: Dict[str, Any]) -> Dict[str, str]:
        """Template for ML model alerts"""
        model_name = data.get('model_name', 'Unknown')
        alert_reason = data.get('alert_reason', 'Model issue')
        accuracy = data.get('accuracy', 0)
        
        title = f"🤖 ML Model Alert - {model_name}"
        
        message = f"""
        🧠 MACHINE LEARNING ALERT
        
        Model: {model_name}
        Issue: {alert_reason}
        Current Accuracy: {accuracy*100:.1f}%
        
        🎯 Recommended Actions:
        • Review model performance
        • Consider retraining with fresh data
        • Check data quality
        • Validate prediction accuracy
        
        Alert time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        return {"title": title, "message": message}


class EmailNotificationChannel:
    """Email notification delivery channel"""
    
    def __init__(self, smtp_server: str, smtp_port: int, username: str, password: str, from_email: str):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_email = from_email
    
    async def send(self, notification: Notification, recipient: NotificationRecipient) -> bool:
        """Send email notification"""
        if not recipient.email:
            logger.warning(f"No email address for recipient {recipient.name}")
            return False
        
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.from_email
            msg['To'] = recipient.email
            msg['Subject'] = f"[Maersk AI] {notification.title}"
            
            # Add body
            body = MIMEText(notification.message, 'plain')
            msg.attach(body)
            
            # Send email
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.username, self.password)
            text = msg.as_string()
            server.sendmail(self.from_email, recipient.email, text)
            server.quit()
            
            logger.info(f"Email sent to {recipient.email} for notification {notification.id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email to {recipient.email}: {e}")
            return False


class WebhookNotificationChannel:
    """Webhook notification delivery channel"""
    
    async def send(self, notification: Notification, recipient: NotificationRecipient) -> bool:
        """Send webhook notification"""
        if not recipient.webhook_url:
            logger.warning(f"No webhook URL for recipient {recipient.name}")
            return False
        
        try:
            payload = {
                "notification_id": notification.id,
                "type": notification.type.value,
                "priority": notification.priority.value,
                "title": notification.title,
                "message": notification.message,
                "data": notification.data,
                "timestamp": notification.created_at.isoformat()
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    recipient.webhook_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        logger.info(f"Webhook sent to {recipient.webhook_url} for notification {notification.id}")
                        return True
                    else:
                        logger.error(f"Webhook failed with status {response.status}")
                        return False
                        
        except Exception as e:
            logger.error(f"Failed to send webhook to {recipient.webhook_url}: {e}")
            return False


class SlackNotificationChannel:
    """Slack notification delivery channel"""
    
    def __init__(self, bot_token: str):
        self.bot_token = bot_token
    
    async def send(self, notification: Notification, recipient: NotificationRecipient) -> bool:
        """Send Slack notification"""
        if not recipient.slack_channel:
            logger.warning(f"No Slack channel for recipient {recipient.name}")
            return False
        
        try:
            # Format message for Slack
            slack_message = {
                "channel": recipient.slack_channel,
                "text": notification.title,
                "attachments": [{
                    "color": self._get_color_for_priority(notification.priority),
                    "title": notification.title,
                    "text": notification.message,
                    "footer": "Maersk Shipment AI System",
                    "ts": int(notification.created_at.timestamp())
                }]
            }
            
            headers = {
                "Authorization": f"Bearer {self.bot_token}",
                "Content-Type": "application/json"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://slack.com/api/chat.postMessage",
                    json=slack_message,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    result = await response.json()
                    if result.get('ok'):
                        logger.info(f"Slack message sent to {recipient.slack_channel} for notification {notification.id}")
                        return True
                    else:
                        logger.error(f"Slack API error: {result.get('error')}")
                        return False
                        
        except Exception as e:
            logger.error(f"Failed to send Slack message: {e}")
            return False
    
    def _get_color_for_priority(self, priority: NotificationPriority) -> str:
        """Get color for notification priority"""
        colors = {
            NotificationPriority.LOW: "good",
            NotificationPriority.MEDIUM: "warning", 
            NotificationPriority.HIGH: "danger",
            NotificationPriority.CRITICAL: "#ff0000"
        }
        return colors.get(priority, "good")


class NotificationService:
    """Main notification service orchestrator"""
    
    def __init__(self):
        self.recipients: Dict[str, NotificationRecipient] = {}
        self.channels: Dict[NotificationChannel, Any] = {}
        self.notification_queue: asyncio.Queue = asyncio.Queue()
        self.notification_history: List[Notification] = []
        self.templates = NotificationTemplates()
        self.running = False
        
        # Default system recipients
        self._setup_default_recipients()
    
    def _setup_default_recipients(self):
        """Setup default system recipients"""
        # Add default admin recipient
        admin = NotificationRecipient(
            name="System Administrator",
            email="admin@maersk.com",
            webhook_url="http://localhost:3000/webhook",  # Example webhook
            preferences={
                NotificationType.DELAY_PREDICTION.value: True,
                NotificationType.STATUS_CHANGE.value: False,
                NotificationType.HIGH_RISK_ALERT.value: True,
                NotificationType.SYSTEM_ALERT.value: True,
                NotificationType.ML_MODEL_ALERT.value: True,
                NotificationType.OPERATIONAL_INSIGHT.value: True,
            }
        )
        self.recipients["admin"] = admin
    
    def configure_email_channel(self, smtp_server: str, smtp_port: int, username: str, password: str, from_email: str):
        """Configure email notification channel"""
        self.channels[NotificationChannel.EMAIL] = EmailNotificationChannel(
            smtp_server, smtp_port, username, password, from_email
        )
        logger.info("Email notification channel configured")
    
    def configure_slack_channel(self, bot_token: str):
        """Configure Slack notification channel"""
        self.channels[NotificationChannel.SLACK] = SlackNotificationChannel(bot_token)
        logger.info("Slack notification channel configured")
    
    def configure_webhook_channel(self):
        """Configure webhook notification channel"""
        self.channels[NotificationChannel.WEBHOOK] = WebhookNotificationChannel()
        logger.info("Webhook notification channel configured")
    
    def add_recipient(self, recipient_id: str, recipient: NotificationRecipient):
        """Add notification recipient"""
        self.recipients[recipient_id] = recipient
        logger.info(f"Added recipient: {recipient.name}")
    
    async def send_notification(self, notification: Notification) -> bool:
        """Send notification immediately"""
        success_count = 0
        total_attempts = 0
        
        # Use default recipients if none specified
        recipient_ids = notification.recipients if notification.recipients else list(self.recipients.keys())
        
        for recipient_id in recipient_ids:
            recipient = self.recipients.get(recipient_id)
            if not recipient or not recipient.active:
                continue
            
            # Check recipient preferences
            if not recipient.preferences.get(notification.type.value, True):
                logger.info(f"Skipping notification {notification.id} for {recipient.name} (preferences)")
                continue
            
            # Send via each configured channel
            for channel in notification.channels:
                if channel not in self.channels:
                    logger.warning(f"Channel {channel.value} not configured")
                    continue
                
                total_attempts += 1
                try:
                    channel_handler = self.channels[channel]
                    success = await channel_handler.send(notification, recipient)
                    if success:
                        success_count += 1
                except Exception as e:
                    logger.error(f"Error sending notification via {channel.value}: {e}")
        
        # Update notification status
        if success_count > 0:
            notification.status = "sent"
            notification.sent_at = datetime.now()
        else:
            notification.status = "failed"
            notification.retry_count += 1
        
        # Store in history
        self.notification_history.append(notification)
        
        logger.info(f"Notification {notification.id}: {success_count}/{total_attempts} successful deliveries")
        return success_count > 0
    
    async def queue_notification(self, notification: Notification):
        """Add notification to processing queue"""
        await self.notification_queue.put(notification)
        logger.info(f"Queued notification {notification.id}")
    
    async def start_notification_processor(self):
        """Start the notification processing worker"""
        self.running = True
        logger.info("Starting notification processor...")
        
        while self.running:
            try:
                # Get notification from queue with timeout
                notification = await asyncio.wait_for(
                    self.notification_queue.get(),
                    timeout=1.0
                )
                
                # Check if it's scheduled for future delivery
                if notification.scheduled_for and notification.scheduled_for > datetime.now():
                    logger.info(f"Notification {notification.id} scheduled for {notification.scheduled_for}")
                    # Put it back in queue for later (simplified approach)
                    await asyncio.sleep(1)
                    await self.notification_queue.put(notification)
                    continue
                
                # Send notification
                await self.send_notification(notification)
                
            except asyncio.TimeoutError:
                # No notifications in queue, continue
                continue
            except Exception as e:
                logger.error(f"Error processing notification: {e}")
                await asyncio.sleep(1)
    
    def stop_notification_processor(self):
        """Stop the notification processing worker"""
        self.running = False
        logger.info("Stopped notification processor")
    
    async def create_delay_prediction_alert(self, shipment_data: Dict[str, Any]) -> str:
        """Create and queue a delay prediction alert"""
        template_data = self.templates.delay_prediction_template(shipment_data)
        
        notification = Notification(
            id=f"delay_{shipment_data.get('shipment_id', 'unknown')}_{int(datetime.now().timestamp())}",
            type=NotificationType.DELAY_PREDICTION,
            priority=NotificationPriority.HIGH,
            title=template_data["title"],
            message=template_data["message"],
            data=shipment_data,
            channels=[NotificationChannel.EMAIL, NotificationChannel.WEBHOOK]
        )
        
        await self.queue_notification(notification)
        return notification.id
    
    async def create_status_change_alert(self, shipment_data: Dict[str, Any]) -> str:
        """Create and queue a status change alert"""
        template_data = self.templates.status_change_template(shipment_data)
        
        notification = Notification(
            id=f"status_{shipment_data.get('shipment_id', 'unknown')}_{int(datetime.now().timestamp())}",
            type=NotificationType.STATUS_CHANGE,
            priority=NotificationPriority.MEDIUM,
            title=template_data["title"],
            message=template_data["message"],
            data=shipment_data,
            channels=[NotificationChannel.EMAIL]
        )
        
        await self.queue_notification(notification)
        return notification.id
    
    async def create_high_risk_alert(self, risk_data: Dict[str, Any]) -> str:
        """Create and queue a high-risk shipments alert"""
        template_data = self.templates.high_risk_alert_template(risk_data)
        
        notification = Notification(
            id=f"highrisk_{int(datetime.now().timestamp())}",
            type=NotificationType.HIGH_RISK_ALERT,
            priority=NotificationPriority.CRITICAL,
            title=template_data["title"],
            message=template_data["message"],
            data=risk_data,
            channels=[NotificationChannel.EMAIL, NotificationChannel.WEBHOOK, NotificationChannel.SLACK]
        )
        
        await self.queue_notification(notification)
        return notification.id
    
    async def create_system_alert(self, alert_data: Dict[str, Any]) -> str:
        """Create and queue a system alert"""
        template_data = self.templates.system_alert_template(alert_data)
        
        priority = NotificationPriority.HIGH if alert_data.get('severity') == 'Critical' else NotificationPriority.MEDIUM
        
        notification = Notification(
            id=f"system_{int(datetime.now().timestamp())}",
            type=NotificationType.SYSTEM_ALERT,
            priority=priority,
            title=template_data["title"],
            message=template_data["message"],
            data=alert_data,
            channels=[NotificationChannel.EMAIL, NotificationChannel.WEBHOOK]
        )
        
        await self.queue_notification(notification)
        return notification.id
    
    def get_notification_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get notification history"""
        recent_notifications = self.notification_history[-limit:]
        return [asdict(notification) for notification in recent_notifications]
    
    def get_notification_stats(self) -> Dict[str, Any]:
        """Get notification statistics"""
        total_notifications = len(self.notification_history)
        sent_notifications = len([n for n in self.notification_history if n.status == "sent"])
        failed_notifications = len([n for n in self.notification_history if n.status == "failed"])
        
        return {
            "total_notifications": total_notifications,
            "sent_notifications": sent_notifications,
            "failed_notifications": failed_notifications,
            "success_rate": (sent_notifications / total_notifications * 100) if total_notifications > 0 else 0,
            "active_recipients": len([r for r in self.recipients.values() if r.active]),
            "configured_channels": list(self.channels.keys())
        }


# Global notification service instance
_notification_service = None

def get_notification_service() -> NotificationService:
    """Get or create the notification service singleton"""
    global _notification_service
    if _notification_service is None:
        _notification_service = NotificationService()
    return _notification_service


if __name__ == "__main__":
    # Example usage
    import asyncio
    
    async def test_notification_service():
        service = get_notification_service()
        
        # Configure email (example - use real credentials in production)
        service.configure_email_channel(
            smtp_server="smtp.gmail.com",
            smtp_port=587,
            username="your-email@gmail.com",
            password="your-password",
            from_email="your-email@gmail.com"
        )
        
        # Configure webhook
        service.configure_webhook_channel()
        
        # Start processor
        processor_task = asyncio.create_task(service.start_notification_processor())
        
        # Create test notification
        test_data = {
            "shipment_id": "SHIP123",
            "asset_id": "Truck_1",
            "delay_probability": 0.85,
            "risk_level": "High",
            "top_risk_factors": [
                ("weather_conditions", 0.45),
                ("traffic_conditions", 0.32),
                ("fuel_efficiency", 0.18)
            ]
        }
        
        notification_id = await service.create_delay_prediction_alert(test_data)
        print(f"Created notification: {notification_id}")
        
        # Let it process for a moment
        await asyncio.sleep(2)
        
        # Stop processor
        service.stop_notification_processor()
        processor_task.cancel()
        
        # Show stats
        stats = service.get_notification_stats()
        print(f"Notification stats: {stats}")
    
    # Run test
    asyncio.run(test_notification_service())
