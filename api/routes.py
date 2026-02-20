from fastapi import APIRouter, Request, Form
from fastapi.responses import PlainTextResponse
from brain.ai_brain import run_agent
from twilio.twiml.messaging_response import MessagingResponse

router = APIRouter()

@router.post("/webhook")
async def whatsapp_webhook(
    request: Request,
    Body: str = Form(...)
):
    session_id = request.client.host  # simple session mapping

    result = run_agent(
        user_message=Body,
        session_id=session_id
    )

    reply = result.message

    twilio_response = MessagingResponse()
    twilio_response.message(reply)

    return PlainTextResponse(str(twilio_response))