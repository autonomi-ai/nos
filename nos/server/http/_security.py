import os

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import APIKeyHeader
from loguru import logger


valid_m2m_keys = {}
for key in os.getenv("NOS_M2M_API_KEYS", "").split(","):
    if len(key) > 0:
        logger.debug(f"Adding valid_m2m_keys [key={key}]")
        valid_m2m_keys[key] = key
api_key_header = APIKeyHeader(name="X-Api-Key", auto_error=False)


async def validate_m2m_key(request: Request, api_key: str = Depends(api_key_header)) -> bool:
    logger.debug(f"validate_m2m_key [api_key={api_key}]")

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-Api-Key Key header",
        )

    if api_key not in valid_m2m_keys:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid Machine-to-Machine Key",
        )

    assert isinstance(api_key, str)
    return True


if valid_m2m_keys:
    ValidMachineToMachine = Depends(validate_m2m_key)
else:
    ValidMachineToMachine = None
