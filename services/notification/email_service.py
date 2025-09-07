"""
Email notification service using SendGrid
"""
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
import asyncio
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, From, To, Subject, Content, Attachment, FileContent, FileName, FileType, Disposition
from config.settings import settings

logger = logging.getLogger(__name__)


class EmailService:
    """Service for sending emails via SendGrid"""

    def __init__(self):
        self.sendgrid_client = SendGridAPIClient(api_key=settings.SENDGRID_API_KEY)
        self.from_email = settings.FROM_EMAIL

    async def send_email(self, to_email: str, subject: str, content: str, 
                        content_type: str = "text/plain", attachments: List[Dict] = None) -> Dict[str, Any]:
        """Send email via SendGrid"""
        try:
            message = Mail(
                from_email=From(self.from_email, "Maersk Shipping"),
                to_emails=To(to_email),
                subject=Subject(subject),
                plain_text_content=Content("text/plain", content) if content_type == "text/plain" else None,
                html_content=Content("text/html", content) if content_type == "text/html" else None
            )

            # Add attachments if provided
            if attachments:
                for attachment_data in attachments:
                    attachment = Attachment(
                        FileContent(attachment_data.get('content', '')),
                        FileName(attachment_data.get('filename', 'attachment.txt')),
                        FileType(attachment_data.get('type', 'text/plain')),
                        Disposition('attachment')
                    )
                    message.attachment = attachment

            response = self.sendgrid_client.send(message)

            return {
                'status': 'delivered' if response.status_code == 202 else 'failed',
                'status_code': response.status_code,
                'message_id': response.headers.get('X-Message-Id', ''),
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error sending email to {to_email}: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    async def send_html_email(self, to_email: str, subject: str, html_content: str,
                             text_content: Optional[str] = None) -> Dict[str, Any]:
        """Send HTML email with optional text fallback"""
        try:
            message = Mail(
                from_email=From(self.from_email, "Maersk Shipping"),
                to_emails=To(to_email),
                subject=Subject(subject)
            )

            # Add HTML content
            message.content = Content("text/html", html_content)

            # Add text fallback if provided
            if text_content:
                message.content = [
                    Content("text/plain", text_content),
                    Content("text/html", html_content)
                ]

            response = self.sendgrid_client.send(message)

            return {
                'status': 'delivered' if response.status_code == 202 else 'failed',
                'status_code': response.status_code,
                'message_id': response.headers.get('X-Message-Id', ''),
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error sending HTML email to {to_email}: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    async def send_bulk_emails(self, email_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Send multiple emails concurrently"""
        tasks = []
        for data in email_data:
            task = self.send_email(
                to_email=data['to_email'],
                subject=data['subject'],
                content=data['content'],
                content_type=data.get('content_type', 'text/plain'),
                attachments=data.get('attachments')
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    'status': 'failed',
                    'error': str(result),
                    'email_index': i
                })
            else:
                processed_results.append(result)

        return processed_results

    def create_shipment_delay_email(self, customer_name: str, shipment_id: str,
                                  delay_hours: float, new_eta: str, reason: str) -> str:
        """Create HTML email for shipment delay notification"""
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Shipment Delay Notification</title>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background-color: #003f7f; color: white; padding: 20px; text-align: center; }}
                .content {{ padding: 20px; background-color: #f9f9f9; }}
                .alert {{ background-color: #fff3cd; border: 1px solid #ffeaa7; padding: 15px; margin: 20px 0; border-radius: 4px; }}
                .details {{ background-color: white; padding: 15px; margin: 20px 0; border-radius: 4px; }}
                .footer {{ background-color: #e9ecef; padding: 15px; text-align: center; font-size: 12px; }}
                .button {{ display: inline-block; background-color: #003f7f; color: white; padding: 10px 20px; text-decoration: none; border-radius: 4px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Maersk</h1>
                    <h2>Shipment Update</h2>
                </div>

                <div class="content">
                    <p>Dear {customer_name},</p>

                    <div class="alert">
                        <strong>Important Update:</strong> Your shipment {shipment_id} has been updated with a revised delivery estimate.
                    </div>

                    <div class="details">
                        <h3>Shipment Details</h3>
                        <p><strong>Shipment ID:</strong> {shipment_id}</p>
                        <p><strong>Estimated Delay:</strong> {delay_hours} hours</p>
                        <p><strong>Reason:</strong> {reason}</p>
                        <p><strong>New Estimated Arrival:</strong> {new_eta}</p>
                    </div>

                    <p>We sincerely apologize for any inconvenience this may cause. Our team is actively monitoring the situation and working to minimize any further delays.</p>

                    <p>For real-time tracking and updates, please visit our website or contact our customer service team.</p>

                    <p style="text-align: center; margin: 30px 0;">
                        <a href="https://www.maersk.com/tracking" class="button">Track Shipment</a>
                    </p>
                </div>

                <div class="footer">
                    <p>© 2024 Maersk. All rights reserved.</p>
                    <p>Customer Service: customerservice@maersk.com | Phone: +1-800-MAERSK-1</p>
                </div>
            </div>
        </body>
        </html>
        """
        return html_template

    def create_shipment_update_email(self, customer_name: str, shipment_id: str,
                                   status: str, location: str, eta: str) -> str:
        """Create HTML email for general shipment update"""
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Shipment Status Update</title>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background-color: #003f7f; color: white; padding: 20px; text-align: center; }}
                .content {{ padding: 20px; background-color: #f9f9f9; }}
                .status {{ background-color: #d4edda; border: 1px solid #c3e6cb; padding: 15px; margin: 20px 0; border-radius: 4px; }}
                .details {{ background-color: white; padding: 15px; margin: 20px 0; border-radius: 4px; }}
                .footer {{ background-color: #e9ecef; padding: 15px; text-align: center; font-size: 12px; }}
                .button {{ display: inline-block; background-color: #003f7f; color: white; padding: 10px 20px; text-decoration: none; border-radius: 4px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Maersk</h1>
                    <h2>Shipment Status Update</h2>
                </div>

                <div class="content">
                    <p>Dear {customer_name},</p>

                    <div class="status">
                        <strong>Status Update:</strong> Your shipment {shipment_id} is progressing as planned.
                    </div>

                    <div class="details">
                        <h3>Current Status</h3>
                        <p><strong>Shipment ID:</strong> {shipment_id}</p>
                        <p><strong>Status:</strong> {status}</p>
                        <p><strong>Current Location:</strong> {location}</p>
                        <p><strong>Estimated Arrival:</strong> {eta}</p>
                    </div>

                    <p>Your shipment is progressing well. We will continue to monitor and provide updates as needed.</p>

                    <p style="text-align: center; margin: 30px 0;">
                        <a href="https://www.maersk.com/tracking" class="button">Track Shipment</a>
                    </p>
                </div>

                <div class="footer">
                    <p>© 2024 Maersk. All rights reserved.</p>
                    <p>Customer Service: customerservice@maersk.com | Phone: +1-800-MAERSK-1</p>
                </div>
            </div>
        </body>
        </html>
        """
        return html_template


# Global email service instance
email_service = EmailService()
