"""Actionable diagnostics without exposing request URLs, credentials or response bodies."""

import httpx


def describe_failure(exc: Exception) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        advice = {
            400: "The service rejected the request. Check the model and request format.",
            401: "Authentication failed. Use an AI Gateway API key for Vercel, or the selected service's API key.",
            402: "The service requires available credits or billing. Check your provider account.",
            403: "Access denied. Check the key's permissions and your account's model access.",
            404: "The endpoint or model was not found. Check the configured service and model ID.",
            408: "The service timed out. Try again shortly.",
            413: "The ticket is too large. Shorten it and try again.",
            422: "The service could not process the request. Check the model and request format.",
            429: "The service rate or usage limit was reached. Check your quota and try again later.",
        }.get(status)
        if advice is None:
            advice = (
                "The service is unavailable. Try again shortly."
                if status >= 500
                else "The service rejected the request. Check your provider configuration."
            )
        return f"HTTP {status}: {advice}"
    if isinstance(exc, httpx.TimeoutException):
        return "Request timed out. Check your connection and try again."
    if isinstance(exc, httpx.RequestError):
        return "Could not reach the service. Check your network connection and try again."
    return type(exc).__name__
