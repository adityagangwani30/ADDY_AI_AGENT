from __future__ import annotations

from agent.assistant import PhaseOneAssistant, run_agent

GLOBAL_FALLBACK_RESPONSE = "⚠️ I couldn’t process that request properly. Please try again."
SecureHybridAgent = PhaseOneAssistant

__all__ = ["run_agent", "GLOBAL_FALLBACK_RESPONSE", "SecureHybridAgent"]
