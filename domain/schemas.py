from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class WebhookRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=4000)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentDecision(BaseModel):
    type: Literal["tool_call", "response"]
    tool: str | None = None
    account: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    response: str | None = None

    @model_validator(mode="after")
    def validate_shape(self) -> "AgentDecision":
        if self.type == "tool_call":
            if not self.tool:
                raise ValueError("tool is required when type=tool_call")
            if not self.account:
                raise ValueError("account is required when type=tool_call")
        if self.type == "response" and not self.response:
            raise ValueError("response is required when type=response")
        return self


class AgentResult(BaseModel):
    request_id: str
    status: Literal["ok", "confirmation_required", "error"]
    message: str
    tool_name: str | None = None
    account: str | None = None
    latency_ms: int | None = None
    data: Any | None = None
    error_type: str | None = None


class SendEmailParams(BaseModel):
    to: str = Field(min_length=3)
    subject: str = Field(min_length=1)
    body: str = Field(min_length=1)


class ListEmailsParams(BaseModel):
    max_results: int = Field(default=5, ge=1, le=50)


class SearchEmailParams(BaseModel):
    query: str = Field(min_length=1)
    max_results: int = Field(default=10, ge=1, le=50)


class DeleteEmailParams(BaseModel):
    message_id: str = Field(min_length=1)


class ListEventsParams(BaseModel):
    max_results: int = Field(default=10, ge=1, le=50)


class CreateEventParams(BaseModel):
    summary: str = Field(min_length=1)
    start_time: str = Field(min_length=1)
    end_time: str = Field(min_length=1)
    time_zone: str = Field(default="UTC", min_length=1)


class DeleteEventParams(BaseModel):
    event_id: str = Field(min_length=1)


class ListFilesParams(BaseModel):
    page_size: int = Field(default=10, ge=1, le=100)


class UploadFileParams(BaseModel):
    file_path: str = Field(min_length=1)
    mime_type: str | None = None


class DeleteFileParams(BaseModel):
    file_id: str = Field(min_length=1)
