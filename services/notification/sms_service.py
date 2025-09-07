"""
SMS notification service
"""
import logging
from datetime import datetime
from typing import Dict, Any, List
import asyncio
import httpx
from config.settings import settings

logger = logging.getLogger(__name__)


class SMSService:
    """Service for sending SMS notifications"""

    def __init__(self):
        # In a real implementation, you would use services like Twilio, AWS SNS, etc.
        # For this demo, we'll simulate SMS sending
        self.provider = "simulation"  # Could be "twilio", "aws_sns", etc.

    async def send_sms(self, to_phone: str, message: str) -> Dict[str, Any]:
        """Send SMS message"""
        try:
            # Validate phone number format
            if not self._validate_phone_number(to_phone):
                return {
                    'status': 'failed',
                    'error': 'Invalid phone number format',
                    'timestamp': datetime.now().isoformat()
                }

            # Limit message length for SMS
            if len(message) > 160:
                message = message[:157] + "..."
                logger.warning(f"Message truncated to 160 characters for SMS to {to_phone}")

            if self.provider == "simulation":
                result = await self._simulate_sms_send(to_phone, message)
            else:
                result = await self._send_via_provider(to_phone, message)

            return result

        except Exception as e:
            logger.error(f"Error sending SMS to {to_phone}: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def _validate_phone_number(self, phone: str) -> bool:
        """Validate phone number format"""
        # Basic validation - in production, use proper phone number validation
        if not phone:
            return False

        # Remove common formatting
        cleaned = phone.replace(" ", "").replace("-", "").replace("(", "").replace(")", "")

        # Check if it contains only digits and plus sign
        if not all(c.isdigit() or c == "+" for c in cleaned):
            return False

        # Check length (basic check)
        if len(cleaned) < 10 or len(cleaned) > 15:
            return False

        return True

    async def _simulate_sms_send(self, to_phone: str, message: str) -> Dict[str, Any]:
        """Simulate SMS sending for demo purposes"""
        # Simulate network delay
        await asyncio.sleep(0.1)

        logger.info(f"[SMS SIMULATION] To: {to_phone}, Message: {message}")

        # Simulate 95% success rate
        import random
        if random.random() < 0.95:
            return {
                'status': 'delivered',
                'message_id': f"sim_{hash(to_phone + message + str(datetime.now()))}",
                'timestamp': datetime.now().isoformat(),
                'provider': 'simulation'
            }
        else:
            return {
                'status': 'failed',
                'error': 'Simulated delivery failure',
                'timestamp': datetime.now().isoformat(),
                'provider': 'simulation'
            }

    async def _send_via_provider(self, to_phone: str, message: str) -> Dict[str, Any]:
        """Send SMS via actual provider (Twilio, AWS SNS, etc.)"""
        # This would be implemented based on your chosen SMS provider
        # Example implementations:

        if self.provider == "twilio":
            return await self._send_via_twilio(to_phone, message)
        elif self.provider == "aws_sns":
            return await self._send_via_aws_sns(to_phone, message)
        else:
            raise ValueError(f"Unsupported SMS provider: {self.provider}")

    async def _send_via_twilio(self, to_phone: str, message: str) -> Dict[str, Any]:
        """Send SMS via Twilio (example implementation)"""
        # Example Twilio implementation (requires twilio package and credentials)
        try:
            # from twilio.rest import Client
            # client = Client(account_sid, auth_token)
            # 
            # message = client.messages.create(
            #     body=message,
            #     from_='+1234567890',  # Your Twilio number
            #     to=to_phone
            # )
            # 
            # return {
            #     'status': 'delivered',
            #     'message_id': message.sid,
            #     'timestamp': datetime.now().isoformat(),
            #     'provider': 'twilio'
            # }

            # Placeholder for demo
            return {
                'status': 'not_implemented',
                'error': 'Twilio integration not configured',
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Twilio SMS error: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'provider': 'twilio'
            }

    async def _send_via_aws_sns(self, to_phone: str, message: str) -> Dict[str, Any]:
        """Send SMS via AWS SNS (example implementation)"""
        try:
            # Example AWS SNS implementation (requires boto3 and AWS credentials)
            # import boto3
            # 
            # sns = boto3.client('sns')
            # response = sns.publish(
            #     PhoneNumber=to_phone,
            #     Message=message
            # )
            # 
            # return {
            #     'status': 'delivered',
            #     'message_id': response['MessageId'],
            #     'timestamp': datetime.now().isoformat(),
            #     'provider': 'aws_sns'
            # }

            # Placeholder for demo
            return {
                'status': 'not_implemented',
                'error': 'AWS SNS integration not configured',
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"AWS SNS SMS error: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'provider': 'aws_sns'
            }

    async def send_bulk_sms(self, sms_data: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Send multiple SMS messages concurrently"""
        tasks = []
        for data in sms_data:
            task = self.send_sms(
                to_phone=data['to_phone'],
                message=data['message']
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    'status': 'failed',
                    'error': str(result),
                    'sms_index': i
                })
            else:
                processed_results.append(result)

        return processed_results

    def create_delay_sms(self, shipment_id: str, delay_hours: float, new_eta: str) -> str:
        """Create SMS message for shipment delay"""
        return f"Maersk Alert: Shipment {shipment_id} delayed by {delay_hours:.1f}h. New ETA: {new_eta}. Track: maersk.com/track"

    def create_update_sms(self, shipment_id: str, status: str, eta: str) -> str:
        """Create SMS message for shipment update"""
        return f"Maersk Update: {shipment_id} is {status}. ETA: {eta}. Track: maersk.com/track"

    def create_arrival_sms(self, shipment_id: str, location: str) -> str:
        """Create SMS message for shipment arrival"""
        return f"Maersk: Shipment {shipment_id} has arrived at {location}. Contact: +1-800-MAERSK-1"


# Global SMS service instance
sms_service = SMSService()
