import os
from typing import Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import APIKeyHeader, HTTPBearer
from loguru import logger


valid_m2m_keys = {}
for key in os.getenv("NOS_M2M_API_KEYS", "").split(","):
    if len(key) > 0:
        logger.debug(f"Adding valid_m2m_keys [key={key}]")
        valid_m2m_keys[key] = key

api_key_header = APIKeyHeader(name="X-Api-Key", auto_error=False)
api_key_bearer = HTTPBearer(auto_error=False)


async def get_api_key(request: Request) -> Optional[str]:
    api_key: Optional[str] = None
    api_key_header_value = request.headers.get("X-Api-Key")
    if api_key_header_value:
        api_key = api_key_header_value
    else:
        authorization: Optional[str] = request.headers.get("Authorization")
        if authorization:
            scheme, credentials = authorization.split()
            if scheme.lower() == "bearer":
                api_key = credentials
    return api_key


async def validate_m2m_key(request: Request, api_key: Optional[str] = Depends(get_api_key)) -> bool:
    logger.debug(f"validate_m2m_key [api_key={api_key}]")
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication token (use `X-Api-Key` or `Authorization: Bearer <token>`)",
        )
    if api_key not in valid_m2m_keys:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid authentication token (use `X-Api-Key` or `Authorization: Bearer <token>`)",
        )
    assert isinstance(api_key, str)
    return True


if valid_m2m_keys:
    ValidMachineToMachine = Depends(validate_m2m_key)
else:
    ValidMachineToMachine = None
