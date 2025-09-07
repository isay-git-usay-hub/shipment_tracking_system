"""
AI-powered communication service for customer notifications
"""
import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from jinja2 import Template
from sqlalchemy.orm import Session

from models.database.connection import get_db
from models.database.models import CommunicationLog, Customer, Shipment
from models.schemas import (
    CommunicationGenerationRequest, CommunicationGenerationResponse,
    CommunicationTypeEnum
)
from services.notification.email_service import EmailService
from services.notification.sms_service import SMSService
from services.llm.service import llm_service
from config.settings import settings

logger = logging.getLogger(__name__)


class CommunicationService:
    """Service for AI-powered customer communication"""

    def __init__(self):
        self.email_service = EmailService()
        self.sms_service = SMSService()
        self.templates = self._load_templates()
        self.local_llm = self._initialize_local_llm()

    def _initialize_local_llm(self):
        """Initialize local LLM model"""
        try:
            # Use our local LLM service
            logger.info("Local LLM service initialized")
            return llm_service
        except Exception as e:
            logger.error(f"Error initializing local LLM: {e}")
            return None

    def _load_templates(self) -> Dict[str, Dict[str, Template]]:
        """Load communication templates"""
        return {
            'delay_notification': {
                'email': Template("""
Subject: Important Update: Shipment {{ shipment_id }} Delay Notification

Dear {{ customer_name }},

We hope this message finds you well. We are writing to inform you about an important update regarding your shipment {{ shipment_id }}.

**Shipment Details:**
- Shipment ID: {{ shipment_id }}
- Origin: {{ origin_port }}
- Destination: {{ destination_port }}
- Original ETA: {{ original_eta }}

**Delay Information:**
We have identified that your shipment may experience a delay of approximately {{ delay_hours }} hours due to {{ delay_reason }}.

**Revised ETA:** {{ new_eta }}

**What We're Doing:**
{{ actions_taken }}

**Alternative Options:**
{{ alternative_options if alternative_options else "Our team is working to minimize any impact to your delivery schedule." }}

We sincerely apologize for any inconvenience this may cause and appreciate your understanding. Our team is actively monitoring the situation and will provide updates as they become available.

If you have any questions or concerns, please don't hesitate to contact our customer service team at {{ contact_info }}.

Thank you for your continued trust in Maersk.

Best regards,
The Maersk Team
                """),
                'sms': Template("""
Maersk Update: Shipment {{ shipment_id }} may be delayed by {{ delay_hours }}h due to {{ delay_reason }}. New ETA: {{ new_eta }}. We apologize for any inconvenience. Questions? Call {{ contact_phone }}.
                """)
            },
            'shipment_update': {
                'email': Template("""
Subject: Shipment {{ shipment_id }} Status Update

Dear {{ customer_name }},

We wanted to provide you with the latest update on your shipment {{ shipment_id }}.

**Current Status:** {{ current_status }}
**Location:** {{ current_location }}
**Updated ETA:** {{ updated_eta }}

{{ additional_details }}

We will continue to monitor your shipment and provide updates as needed.

Best regards,
The Maersk Team
                """),
                'sms': Template("""
Maersk Update: Shipment {{ shipment_id }} status: {{ current_status }}. Location: {{ current_location }}. ETA: {{ updated_eta }}.
                """)
            },
            'proactive_communication': {
                'email': Template("""
Subject: Proactive Update: Your Shipment {{ shipment_id }}

Dear {{ customer_name }},

We're pleased to provide you with a proactive update on your shipment {{ shipment_id }}.

**Good News:** Your shipment is progressing well and remains on schedule for delivery on {{ eta }}.

**Current Status:**
- Location: {{ current_location }}
- Progress: {{ progress_percentage }}% complete
- No delays anticipated at this time

{{ additional_info }}

Thank you for choosing Maersk for your shipping needs.

Best regards,
The Maersk Team
                """)
            }
        }

    async def generate_communication(self, request: CommunicationGenerationRequest) -> CommunicationGenerationResponse:
        """Generate AI-powered communication"""
        try:
            # Get shipment and customer data
            db = next(get_db())
            shipment = db.query(Shipment).filter(Shipment.id == request.shipment_id).first()
            customer = db.query(Customer).filter(Customer.id == request.customer_id).first()

            if not shipment or not customer:
                raise ValueError("Shipment or customer not found")

            # Prepare context data
            context_data = self._prepare_context_data(shipment, customer, request)

            # Generate personalized content using AI
            ai_content = await self._generate_ai_content(request, context_data)

            # Apply template and personalization
            final_content = self._apply_template_and_personalization(
                ai_content, request, context_data
            )

            db.close()

            return CommunicationGenerationResponse(
                subject=final_content['subject'],
                message=final_content['message'],
                recipient=customer.email if request.communication_type == CommunicationTypeEnum.EMAIL else customer.phone,
                type=request.communication_type,
                personalization_used=context_data,
                model_used="local_llm_template",
                generated_at=datetime.now()
            )

        except Exception as e:
            logger.error(f"Error generating communication: {e}")
            # Return fallback communication
            return self._generate_fallback_communication(request)

    def _prepare_context_data(self, shipment: Shipment, customer: Customer, 
                            request: CommunicationGenerationRequest) -> Dict[str, Any]:
        """Prepare context data for AI generation"""
        context = {
            'customer_name': customer.name,
            'customer_company': customer.company or '',
            'shipment_id': shipment.shipment_id,
            'origin_port': shipment.origin_port,
            'destination_port': shipment.destination_port,
            'scheduled_arrival': shipment.scheduled_arrival.strftime('%Y-%m-%d %H:%M'),
            'current_status': shipment.status.value,
            'cargo_type': shipment.cargo_type or 'general cargo',
            'language': request.language,
            'tone': request.tone,
            'communication_context': request.context
        }

        # Add delay-specific information if provided
        if request.delay_info:
            context.update({
                'delay_hours': request.delay_info.get('predicted_delay_hours', 0),
                'delay_reason': request.delay_info.get('primary_reason', 'operational factors'),
                'delay_probability': request.delay_info.get('delay_probability', 0),
                'risk_level': request.delay_info.get('risk_level', 'low'),
                'recommendations': request.delay_info.get('recommendations', [])
            })

            # Calculate new ETA
            if 'predicted_delay_hours' in request.delay_info:
                from datetime import timedelta
                new_eta = shipment.scheduled_arrival + timedelta(hours=request.delay_info['predicted_delay_hours'])
                context['new_eta'] = new_eta.strftime('%Y-%m-%d %H:%M')

        return context

    async def _generate_ai_content(self, request: CommunicationGenerationRequest, 
                                 context_data: Dict[str, Any]) -> Dict[str, str]:
        """Generate content using local LLM"""
        try:
            # Build prompt based on context
            prompt = self._build_ai_prompt(request, context_data)

            # Use our local LLM service for text generation
            if self.local_llm is not None:
                generated_text = self.local_llm.generate_text(prompt)
                # Parse the response to extract subject and message
                return self._parse_ai_response(generated_text, request.communication_type)
            else:
                # Fallback to template-based generation
                logger.info("Using template-based generation as local LLM fallback")
                return self._generate_template_content(request, context_data)

        except Exception as e:
            logger.error(f"Error with local LLM: {e}")
            # Return template-based content as fallback
            return self._generate_template_content(request, context_data)

    def _build_ai_prompt(self, request: CommunicationGenerationRequest, 
                        context_data: Dict[str, Any]) -> str:
        """Build prompt for AI content generation"""
        base_prompt = f"""
Generate a professional {request.communication_type.value} communication for a maritime shipping customer with the following details:

Customer: {context_data['customer_name']} from {context_data['customer_company']}
Shipment: {context_data['shipment_id']} ({context_data['origin_port']} to {context_data['destination_port']})
Context: {request.context}
Tone: {request.tone}
Language: {request.language}
        """

        if request.delay_info:
            delay_prompt = f"""
Delay Information:
- Predicted delay: {context_data.get('delay_hours', 0)} hours
- Reason: {context_data.get('delay_reason', 'operational factors')}
- Risk level: {context_data.get('risk_level', 'low')}
- New estimated arrival: {context_data.get('new_eta', 'TBD')}
            """
            base_prompt += delay_prompt

        if request.custom_instructions:
            base_prompt += f"\n\nAdditional instructions: {request.custom_instructions}"

        format_instructions = """
Please format your response as follows:
SUBJECT: [subject line]
MESSAGE: [main message content]

The communication should be professional, empathetic, and provide clear information. 
Include relevant details and next steps where appropriate.
        """

        return base_prompt + "\n\n" + format_instructions

    def _get_system_prompt(self, comm_type: CommunicationTypeEnum, tone: str) -> str:
        """Get system prompt for local LLM"""
        base_system = """You are an AI assistant specialized in creating professional maritime shipping communications for Maersk, a leading global logistics company. You excel at creating personalized, clear, and empathetic communications that maintain customer trust while providing essential information."""

        if comm_type == CommunicationTypeEnum.EMAIL:
            system_prompt = base_system + """ 

Create professional email communications that are:
- Clear and informative
- Appropriately formal but friendly
- Include all relevant shipping details
- Provide helpful next steps
- Show empathy for any inconvenience
- Maintain Maersk's professional brand voice
            """
        elif comm_type == CommunicationTypeEnum.SMS:
            system_prompt = base_system + """ 

Create concise SMS messages that are:
- Under 160 characters when possible
- Include essential information only
- Clear and direct
- Professional but brief
- Include contact information for follow-up
            """
        else:
            system_prompt = base_system

        if tone == "urgent":
            system_prompt += " Use an urgent but professional tone that conveys the importance of immediate attention."
        elif tone == "friendly":
            system_prompt += " Use a warm, friendly tone while maintaining professionalism."
        elif tone == "formal":
            system_prompt += " Use formal business language appropriate for corporate communications."

        return system_prompt

    def _parse_ai_response(self, content: str, comm_type: CommunicationTypeEnum) -> Dict[str, str]:
        """Parse AI response to extract subject and message"""
        lines = content.strip().split('\n')
        subject = ""
        message = ""

        current_section = None
        for line in lines:
            line = line.strip()
            if line.startswith('SUBJECT:'):
                current_section = 'subject'
                subject = line.replace('SUBJECT:', '').strip()
            elif line.startswith('MESSAGE:'):
                current_section = 'message'
                message = line.replace('MESSAGE:', '').strip()
            elif current_section == 'message' and line:
                message += "\n" + line

        # Clean up the content
        subject = subject.strip()
        message = message.strip()

        # For SMS, combine subject and message if needed
        if comm_type == CommunicationTypeEnum.SMS:
            if not message and subject:
                message = subject
                subject = ""

        return {
            'subject': subject,
            'message': message
        }

    def _generate_template_content(self, request: CommunicationGenerationRequest,
                                 context_data: Dict[str, Any]) -> Dict[str, str]:
        """Generate content using templates as fallback"""
        template_key = request.context.lower().replace(' ', '_')
        comm_type = 'email' if request.communication_type == CommunicationTypeEnum.EMAIL else 'sms'

        if template_key in self.templates and comm_type in self.templates[template_key]:
            template = self.templates[template_key][comm_type]
            content = template.render(**context_data)

            # For email templates, extract subject
            if comm_type == 'email' and 'Subject:' in content:
                lines = content.split('\n')
                subject = lines[0].replace('Subject:', '').strip()
                message = '\n'.join(lines[1:]).strip()
            else:
                subject = f"Shipment {context_data['shipment_id']} Update"
                message = content.strip()

            return {'subject': subject, 'message': message}
        else:
            # Ultimate fallback
            return {
                'subject': f"Shipment {context_data['shipment_id']} Update",
                'message': f"Dear {context_data['customer_name']}, we have an update regarding your shipment {context_data['shipment_id']}. Please contact our customer service team for more information."
            }

    def _apply_template_and_personalization(self, ai_content: Dict[str, str],
                                          request: CommunicationGenerationRequest,
                                          context_data: Dict[str, Any]) -> Dict[str, str]:
        """Apply final template formatting and personalization"""
        # Apply any final personalization
        subject = ai_content['subject']
        message = ai_content['message']

        # Add standard Maersk contact information
        if request.communication_type == CommunicationTypeEnum.EMAIL:
            message += "\n\n---\nFor questions or concerns, contact us at:"
            message += "\nCustomer Service: customerservice@maersk.com"
            message += "\nPhone: +1-800-MAERSK-1"
            message += "\nWebsite: www.maersk.com"

        return {
            'subject': subject,
            'message': message
        }

    def _generate_fallback_communication(self, request: CommunicationGenerationRequest) -> CommunicationGenerationResponse:
        """Generate fallback communication when AI fails"""
        return CommunicationGenerationResponse(
            subject="Shipment Update",
            message="We have an update regarding your shipment. Please contact our customer service team for more information.",
            recipient="customer@example.com",
            type=request.communication_type,
            personalization_used={},
            model_used="fallback",
            generated_at=datetime.now()
        )

    async def send_communication(self, communication: CommunicationGenerationResponse) -> Dict[str, Any]:
        """Send the generated communication"""
        try:
            if communication.type == CommunicationTypeEnum.EMAIL:
                result = await self.email_service.send_email(
                    to_email=communication.recipient,
                    subject=communication.subject,
                    content=communication.message
                )
            elif communication.type == CommunicationTypeEnum.SMS:
                result = await self.sms_service.send_sms(
                    to_phone=communication.recipient,
                    message=communication.message
                )
            else:
                raise ValueError(f"Unsupported communication type: {communication.type}")

            return result

        except Exception as e:
            logger.error(f"Error sending communication: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    async def store_communication_log(self, request: CommunicationGenerationRequest,
                                    communication: CommunicationGenerationResponse,
                                    send_result: Dict[str, Any]) -> None:
        """Store communication in database"""
        try:
            db = next(get_db())

            log_entry = CommunicationLog(
                shipment_id=request.shipment_id,
                customer_id=request.customer_id,
                type=communication.type,
                subject=communication.subject,
                message=communication.message,
                recipient=communication.recipient,
                ai_generated=True,
                model_used=communication.model_used,
                template_used=request.context,
                personalization_data=communication.personalization_used,
                delivery_status=send_result.get('status', 'unknown'),
                sent_at=datetime.now()
            )

            if send_result.get('status') == 'delivered':
                log_entry.delivered_at = datetime.now()

            db.add(log_entry)
            db.commit()
            db.close()

            logger.info(f"Stored communication log for shipment {request.shipment_id}")

        except Exception as e:
            logger.error(f"Error storing communication log: {e}")

    async def generate_and_send_communication(self, request: CommunicationGenerationRequest) -> Dict[str, Any]:
        """Generate and send communication in one operation"""
        try:
            # Generate communication
            communication = await self.generate_communication(request)

            # Send communication
            send_result = await self.send_communication(communication)

            # Store in database
            await self.store_communication_log(request, communication, send_result)

            return {
                'communication': communication.model_dump(),
                'send_result': send_result,
                'status': 'success'
            }

        except Exception as e:
            logger.error(f"Error in generate_and_send_communication: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    async def batch_generate_communications(self, requests: List[CommunicationGenerationRequest]) -> List[Dict[str, Any]]:
        """Generate communications for multiple requests"""
        tasks = [self.generate_and_send_communication(request) for request in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch communication failed for request {i}: {result}")
                processed_results.append({
                    'status': 'failed',
                    'error': str(result),
                    'request_index': i
                })
            else:
                processed_results.append(result)

        return processed_results


# Global communication service instance
communication_service = CommunicationService()
