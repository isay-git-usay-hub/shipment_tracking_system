"""
Integration Service for Notification System

This module provides integration between the notification system and other components
of the Maersk Shipment AI System, including ML predictions, shipment tracking, and
system monitoring.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
import json

from notification_service import get_notification_service
from alert_monitor import get_alert_monitor

logger = logging.getLogger(__name__)


class NotificationIntegrationService:
    """
    Integration service that connects notifications with other system components
    """
    
    def __init__(self):
        self.notification_service = get_notification_service()
        self.alert_monitor = get_alert_monitor()
        self.integration_active = False
        
        # Configuration
        self.config = {
            'auto_start_monitoring': True,
            'enable_ml_predictions': True,
            'enable_status_monitoring': True,
            'enable_system_health': True,
            'webhook_timeout': 30,
            'retry_failed_notifications': True
        }
    
    def configure(self, config_updates: Dict[str, Any]):
        """Update integration configuration"""
        self.config.update(config_updates)
        logger.info(f"Integration configuration updated: {config_updates}")
    
    async def start_integration(self):
        """Start the notification integration service"""
        self.integration_active = True
        
        # Start notification processor
        notification_processor = asyncio.create_task(
            self.notification_service.start_notification_processor()
        )
        
        # Start alert monitoring if configured
        if self.config['auto_start_monitoring']:
            self.alert_monitor.start_monitoring()
        
        logger.info("🚀 Notification integration service started")
        return notification_processor
    
    def stop_integration(self):
        """Stop the notification integration service"""
        self.integration_active = False
        self.notification_service.stop_notification_processor()
        self.alert_monitor.stop_monitoring()
        
        logger.info("🛑 Notification integration service stopped")
    
    async def handle_shipment_update(self, shipment_data: Dict[str, Any]):
        """Handle shipment update event and trigger appropriate notifications"""
        if not self.integration_active:
            return
        
        try:
            # Monitor the shipment for alerts using simplified check
            delay_probability = shipment_data.get('delay_probability', 0)
            if delay_probability >= 0.8:
                await self.alert_monitor.check_high_delay_risk(shipment_data)
                    
        except Exception as e:
            logger.error(f"Error handling shipment update: {e}")
    
    async def handle_ml_prediction(self, prediction_data: Dict[str, Any]):
        """Handle ML prediction event and create notifications for high-risk predictions"""
        if not self.integration_active or not self.config['enable_ml_predictions']:
            return
        
        try:
            # Check if this is a high-risk prediction
            delay_probability = prediction_data.get('delay_probability', 0)
            risk_level = prediction_data.get('risk_level', 'Low')
            
            # Create notification for high-risk predictions
            if delay_probability >= 0.8 and risk_level in ['High', 'Critical']:
                await self.notification_service.create_delay_prediction_alert(prediction_data)
                logger.info(f"High-risk prediction alert created for {prediction_data.get('shipment_id', 'unknown')}")
                    
        except Exception as e:
            logger.error(f"Error handling ML prediction: {e}")
    
    async def handle_system_health_check(self, health_metrics: Dict[str, Any]):
        """Handle system health check event"""
        if not self.integration_active or not self.config['enable_system_health']:
            return
        
        try:
            # Check for system health issues
            issues = []
            
            # API response time
            api_response_time = health_metrics.get('api_response_time', 0)
            if api_response_time > 5.0:
                issues.append(f"High API response time: {api_response_time:.2f}s")
            
            # Database errors
            db_errors = health_metrics.get('database_errors_per_hour', 0)
            if db_errors > 5:
                issues.append(f"Database errors: {db_errors}/hour")
            
            # ML service availability
            ml_available = health_metrics.get('ml_service_available', True)
            if not ml_available:
                issues.append("ML service unavailable")
            
            if issues:
                alert_data = {
                    'alert_type': 'System Health',
                    'severity': 'High' if len(issues) > 2 else 'Medium',
                    'component': 'System Infrastructure',
                    'details': '; '.join(issues),
                    'metrics': health_metrics
                }
                
                await self.notification_service.create_system_alert(alert_data)
                logger.warning(f"System health alert created: {len(issues)} issues detected")
                    
        except Exception as e:
            logger.error(f"Error handling system health check: {e}")
    
    def configure_webhook_notifications(self):
        """Configure webhook notifications"""
        try:
            self.notification_service.configure_webhook_channel()
            logger.info("🔗 Webhook notifications configured")
        except Exception as e:
            logger.error(f"Error configuring webhook notifications: {e}")
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Get comprehensive integration status"""
        try:
            return {
                'integration_active': self.integration_active,
                'configuration': self.config,
                'notification_stats': self.notification_service.get_notification_stats(),
                'monitoring_status': self.alert_monitor.get_monitoring_status(),
                'system_health': {
                    'notification_service': True,
                    'alert_monitor': self.alert_monitor.monitoring_active,
                    'configured_channels': list(self.notification_service.channels.keys())
                }
            }
        except Exception as e:
            logger.error(f"Error getting integration status: {e}")
            return {
                'integration_active': False,
                'error': str(e)
            }
    
    async def send_startup_notification(self):
        """Send notification when system starts up"""
        try:
            startup_data = {
                'alert_type': 'System Startup',
                'severity': 'Low',
                'component': 'Maersk Shipment AI System',
                'details': f'System started successfully at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
            }
            
            await self.notification_service.create_system_alert(startup_data)
            logger.info("🚀 System startup notification sent")
        except Exception as e:
            logger.error(f"Error sending startup notification: {e}")


# Global integration service instance
_integration_service = None

def get_integration_service() -> NotificationIntegrationService:
    """Get or create the integration service singleton"""
    global _integration_service
    if _integration_service is None:
        _integration_service = NotificationIntegrationService()
    return _integration_service
