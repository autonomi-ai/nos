from fastapi import status
from fastapi.responses import JSONResponse
from loguru import logger
from starlette.requests import Request


async def default_exception_middleware(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as exc:
        base_error_message = f"Internal server error: [method={request.method}], url={request.url}]"
        logger.error(f"Internal server error: [method={request.method}, url={request.url}, exc={exc}]")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"message": base_error_message},
        )


async def default_exception_handler(request, error):
    """Default exception handler for all routes."""
    base_error_message = f"Internal server error: [method={request.method}], url={request.url}]"
    logger.error(f"Internal server error: [method={request.method}, url={request.url}, error={error}]")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"message": f"{base_error_message}."},
    )
