import os
import logging
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, Email, To, Content, HtmlContent
from app.core.config import settings

logger = logging.getLogger(__name__)

class EmailService:
    def __init__(self):
        self.sender_email = settings.EMAIL_SENDER
        self.sendgrid_api_key = settings.SENDGRID_API_KEY

    def send_email(self, to_email, subject, html_content):
        try:
            # Create a SendGrid client
            print("Sending email...")
            sg = SendGridAPIClient(self.sendgrid_api_key)
            
            # Create the email
            from_email = Email(self.sender_email)
            to_email = To(to_email)
            content = HtmlContent(html_content)
            mail = Mail(from_email, to_email, subject, content)
            
            # Send the email
            response = sg.send(mail)
            
            # Log the response
            logger.info(f"Email sent to {to_email.email} with status code: {response.status_code}")
            return True
        except Exception as e:
            logger.error(f"Failed to send email to {to_email}: {str(e)}")
            return False

    def send_verification_email(self, to_email, verification_code):
        subject = "Husslify - Email Verification Code"
        html_content = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background: linear-gradient(90deg, #3b82f6, #6366f1); color: white; padding: 20px; text-align: center; border-radius: 5px 5px 0 0; }}
                .content {{ background-color: #f9fafb; padding: 20px; border-radius: 0 0 5px 5px; }}
                .code {{ font-size: 32px; font-weight: bold; text-align: center; margin: 30px 0; letter-spacing: 5px; color: #4f46e5; }}
                .footer {{ text-align: center; margin-top: 20px; font-size: 12px; color: #6b7280; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Husslify</h1>
                </div>
                <div class="content">
                    <p>Hello,</p>
                    <p>Thank you for registering with Husslify. To verify your email address, please use the following verification code:</p>
                    <div class="code">{verification_code}</div>
                    <p>This code will expire in 30 minutes.</p>
                    <p>If you didn't request this verification, please ignore this email.</p>
                </div>
                <div class="footer">
                    <p>&copy; {2024} Husslify. All rights reserved.</p>
                </div>
            </div>
        </body>
        </html>
        """
        return self.send_email(to_email, subject, html_content)


email_service = EmailService() 