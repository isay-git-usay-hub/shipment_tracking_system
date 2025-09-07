"""
Smart Alert Monitoring System

This module monitors shipment data, ML predictions, and system events to automatically
trigger appropriate notifications based on predefined rules and thresholds.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import json

from notification_service import (
    get_notification_service, 
    NotificationPriority, 
    NotificationType
)

logger = logging.getLogger(__name__)


class AlertTrigger(Enum):
    HIGH_DELAY_RISK = "high_delay_risk"
    MULTIPLE_HIGH_RISKS = "multiple_high_risks"
    STATUS_CHANGE = "status_change"
    ML_MODEL_PERFORMANCE = "ml_model_performance"
    SYSTEM_HEALTH = "system_health"
    OPERATIONAL_THRESHOLD = "operational_threshold"


@dataclass
class AlertRule:
    """Configuration for alert rules"""
    name: str
    trigger: AlertTrigger
    enabled: bool = True
    conditions: Dict[str, Any] = None
    cooldown_minutes: int = 60
    priority: NotificationPriority = NotificationPriority.MEDIUM
    description: str = ""
    
    def __post_init__(self):
        if self.conditions is None:
            self.conditions = {}


class AlertMonitor:
    """Smart monitoring system for automated notifications"""
    
    def __init__(self):
        self.notification_service = get_notification_service()
        self.alert_rules: Dict[str, AlertRule] = {}
        self.monitoring_active = False
        self.last_alerts: Dict[str, datetime] = {}
        
        # Monitoring state
        self.previous_shipment_states: Dict[str, Dict] = {}
        self.ml_performance_history: List[Dict] = []
        
        # Setup default alert rules
        self._setup_default_rules()
    
    def _setup_default_rules(self):
        """Setup default monitoring rules"""
        
        # High delay risk rule
        high_risk_rule = AlertRule(
            name="high_delay_risk",
            trigger=AlertTrigger.HIGH_DELAY_RISK,
            conditions={
                "min_delay_probability": 0.8,
                "risk_level": "High"
            },
            priority=NotificationPriority.HIGH,
            cooldown_minutes=30,
            description="Alert when a shipment has high delay probability (>80%)"
        )
        self.alert_rules["high_delay_risk"] = high_risk_rule
        
        # Multiple high risks rule
        multiple_risks_rule = AlertRule(
            name="multiple_high_risks",
            trigger=AlertTrigger.MULTIPLE_HIGH_RISKS,
            conditions={
                "min_high_risk_count": 3,
                "max_risk_percentage": 30.0
            },
            priority=NotificationPriority.CRITICAL,
            cooldown_minutes=120,
            description="Alert when multiple shipments (≥3) have high delay risk"
        )
        self.alert_rules["multiple_high_risks"] = multiple_risks_rule
        
        # Status change rule
        status_change_rule = AlertRule(
            name="status_change",
            trigger=AlertTrigger.STATUS_CHANGE,
            conditions={
                "notify_on_delay": True,
                "notify_on_delivery": False,
                "notify_on_cancellation": True
            },
            priority=NotificationPriority.MEDIUM,
            cooldown_minutes=5,
            description="Alert on important status changes"
        )
        self.alert_rules["status_change"] = status_change_rule
        
        # ML model performance rule
        ml_performance_rule = AlertRule(
            name="ml_model_performance",
            trigger=AlertTrigger.ML_MODEL_PERFORMANCE,
            conditions={
                "min_accuracy": 0.7,
                "max_prediction_errors": 10
            },
            priority=NotificationPriority.HIGH,
            cooldown_minutes=180,
            description="Alert when ML model performance degrades"
        )
        self.alert_rules["ml_model_performance"] = ml_performance_rule
        
        # System health rule
        system_health_rule = AlertRule(
            name="system_health",
            trigger=AlertTrigger.SYSTEM_HEALTH,
            conditions={
                "api_response_time_threshold": 5.0,  # seconds
                "database_error_threshold": 5,       # errors per hour
                "ml_service_availability": True
            },
            priority=NotificationPriority.HIGH,
            cooldown_minutes=60,
            description="Alert on system health issues"
        )
        self.alert_rules["system_health"] = system_health_rule
    
    def add_custom_rule(self, rule: AlertRule):
        """Add a custom alert rule"""
        self.alert_rules[rule.name] = rule
        logger.info(f"Added custom alert rule: {rule.name}")
    
    def enable_rule(self, rule_name: str, enabled: bool = True):
        """Enable or disable an alert rule"""
        if rule_name in self.alert_rules:
            self.alert_rules[rule_name].enabled = enabled
            logger.info(f"Rule '{rule_name}' {'enabled' if enabled else 'disabled'}")
    
    def _is_in_cooldown(self, rule_name: str, cooldown_minutes: int) -> bool:
        """Check if rule is in cooldown period"""
        if rule_name not in self.last_alerts:
            return False
        
        last_alert = self.last_alerts[rule_name]
        cooldown_end = last_alert + timedelta(minutes=cooldown_minutes)
        return datetime.now() < cooldown_end
    
    def _update_cooldown(self, rule_name: str):
        """Update the last alert time for a rule"""
        self.last_alerts[rule_name] = datetime.now()
    
    async def check_high_delay_risk(self, shipment_data: Dict[str, Any]) -> bool:
        """Check for high delay risk alerts"""
        rule = self.alert_rules.get("high_delay_risk")
        if not rule or not rule.enabled:
            return False
        
        delay_probability = shipment_data.get('delay_probability', 0)
        risk_level = shipment_data.get('risk_level', 'Unknown')
        
        # Check conditions
        min_prob = rule.conditions.get('min_delay_probability', 0.8)
        required_risk = rule.conditions.get('risk_level', 'High')
        
        if delay_probability >= min_prob and risk_level == required_risk:
            # Check cooldown for this specific shipment
            shipment_id = shipment_data.get('shipment_id', 'unknown')
            cooldown_key = f"{rule.name}_{shipment_id}"
            
            if self._is_in_cooldown(cooldown_key, rule.cooldown_minutes):
                return False
            
            # Create alert
            await self.notification_service.create_delay_prediction_alert(shipment_data)
            self._update_cooldown(cooldown_key)
            
            logger.info(f"High delay risk alert triggered for shipment {shipment_id}")
            return True
        
        return False
    
    def start_monitoring(self):
        """Start the monitoring system"""
        self.monitoring_active = True
        logger.info("Alert monitoring system started")
    
    def stop_monitoring(self):
        """Stop the monitoring system"""
        self.monitoring_active = False
        logger.info("Alert monitoring system stopped")
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get current monitoring status and statistics"""
        active_rules = [rule.name for rule in self.alert_rules.values() if rule.enabled]
        
        return {
            'monitoring_active': self.monitoring_active,
            'total_rules': len(self.alert_rules),
            'active_rules': len(active_rules),
            'rule_names': active_rules,
            'tracked_shipments': len(self.previous_shipment_states),
            'ml_performance_entries': len(self.ml_performance_history),
            'recent_alerts': len(self.last_alerts)
        }
    
    def get_alert_rules_summary(self) -> List[Dict[str, Any]]:
        """Get summary of all alert rules"""
        return [
            {
                'name': rule.name,
                'trigger': rule.trigger.value,
                'enabled': rule.enabled,
                'priority': rule.priority.value,
                'cooldown_minutes': rule.cooldown_minutes,
                'description': rule.description,
                'conditions': rule.conditions
            }
            for rule in self.alert_rules.values()
        ]


# Global alert monitor instance
_alert_monitor = None

def get_alert_monitor() -> AlertMonitor:
    """Get or create the alert monitor singleton"""
    global _alert_monitor
    if _alert_monitor is None:
        _alert_monitor = AlertMonitor()
    return _alert_monitor
