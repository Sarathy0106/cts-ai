"""
Health check utilities.
Checks reachability of all remote ML services via HTTPS GET before each request.
"""
from __future__ import annotations
import asyncio
import logging
from typing import NamedTuple
import httpx

logger = logging.getLogger(__name__)

REMOTE_SERVICES = [
    {"name": "Risk Prediction Model", "url": "https://api.sidanex.com/cts-ml1/"},
    {"name": "Prioritization Model",  "url": "https://api.sidanex.com/cts-ml3/"},
    {"name": "Star Impact Model",     "url": "https://api.sidanex.com/cts-ml4/"},
]

HEALTH_TIMEOUT = 5.0


class ServiceStatus(NamedTuple):
    name:      str
    url:       str
    reachable: bool
    detail:    str


async def _check_service(name: str, url: str) -> ServiceStatus:
    try:
        async with httpx.AsyncClient(timeout=HEALTH_TIMEOUT) as client:
            resp = await client.get(url)
        reachable = resp.status_code < 500
        detail = f"HTTP {resp.status_code}"
        logger.info("Health check -- %s (%s): %s", name, url, detail)
        return ServiceStatus(name=name, url=url, reachable=reachable, detail=detail)
    except httpx.TimeoutException:
        logger.warning("Health check -- %s: timeout", name)
        return ServiceStatus(name=name, url=url, reachable=False, detail="timeout")
    except Exception as exc:
        logger.warning("Health check -- %s: %s", name, exc)
        return ServiceStatus(name=name, url=url, reachable=False, detail=str(exc)[:120])


async def check_remote_services() -> list[ServiceStatus]:
    """Checks all remote ML services in parallel."""
    tasks = [_check_service(s["name"], s["url"]) for s in REMOTE_SERVICES]
    return list(await asyncio.gather(*tasks))


async def is_reachable(host: str, port: int, timeout: float = 3.0) -> bool:
    """Legacy TCP check retained for backward compatibility."""
    try:
        _, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port), timeout=timeout)
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass
        return True
    except Exception:
        return False
