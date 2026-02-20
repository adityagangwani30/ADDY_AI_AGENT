from fastapi import APIRouter, Request, Form
from fastapi.responses import Response
from brain.ai_brain import run_agent
from twilio.twiml.messaging_response import MessagingResponse

router = APIRouter()

@router.post("/webhook")
async def whatsapp_webhook(
    request: Request,
    Body: str = Form(...)
):
    # Use phone number as session id
    form_data = await request.form()
    sender = form_data.get("From", "unknown")

    result = run_agent(
        user_message=Body,
        session_id=sender
    )

    reply = result.message

    twilio_response = MessagingResponse()
    twilio_response.message(reply)

    return Response(
        content=str(twilio_response),
        media_type="application/xml"
    )