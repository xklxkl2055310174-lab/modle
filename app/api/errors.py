from dataclasses import dataclass

from fastapi import Request
from fastapi.responses import JSONResponse


@dataclass
class Failure:
    status: int
    code: str
    message: str


class ServiceError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.failure = Failure(status, code, message)


def error_response(request: Request, error: ServiceError) -> JSONResponse:
    failure = error.failure
    return JSONResponse(
        {
            "code": failure.code,
            "message": failure.message,
            "request_id": getattr(request.state, "request_id", "unknown"),
        },
        status_code=failure.status,
        headers={"Retry-After": "1"} if failure.status in (429, 503) else {},
    )
