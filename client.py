"""
ClaimWatch environment client.

Connects to a running ClaimWatch server over WebSocket.

Async usage:
    import asyncio, json, websockets
    from client import ClaimWatchClient
    from models import ClaimAction

    async def main():
        async with ClaimWatchClient("ws://localhost:8000/ws") as client:
            obs = await client.reset(task=1, seed=42)
            result = await client.step(ClaimAction(action="auto_approve"))
            print(result["observation"]["procedure_name"])
            print(result["reward"])

    asyncio.run(main())

Sync usage:
    from client import ClaimWatchClientSync
    from models import ClaimAction

    with ClaimWatchClientSync("ws://localhost:8000/ws") as client:
        obs = client.reset(task=1, seed=42)
        result = client.step(ClaimAction(action="auto_approve"))
"""

import json
import asyncio
import websockets
from models import ClaimAction


class ClaimWatchClient:
    """Async WebSocket client for ClaimWatch."""

    def __init__(self, url: str = "ws://localhost:8000/ws"):
        self.url = url
        self._ws = None

    async def __aenter__(self):
        self._ws = await websockets.connect(self.url)
        return self

    async def __aexit__(self, *args):
        if self._ws:
            await self._ws.close()

    async def reset(self, task: int = 1, seed: int = 42) -> dict:
        await self._ws.send(json.dumps({
            "method": "reset",
            "task": task,
            "seed": seed,
        }))
        return json.loads(await self._ws.recv())

    async def step(self, action: ClaimAction) -> dict:
        await self._ws.send(json.dumps({
            "method": "step",
            "action": action.action,
        }))
        return json.loads(await self._ws.recv())

    async def state(self) -> dict:
        await self._ws.send(json.dumps({"method": "state"}))
        return json.loads(await self._ws.recv())


class ClaimWatchClientSync:
    """Synchronous wrapper around ClaimWatchClient."""

    def __init__(self, url: str = "ws://localhost:8000/ws"):
        self.url = url
        self._loop = asyncio.new_event_loop()
        self._client = ClaimWatchClient(url)

    def __enter__(self):
        self._loop.run_until_complete(self._client.__aenter__())
        return self

    def __exit__(self, *args):
        self._loop.run_until_complete(self._client.__aexit__(*args))
        self._loop.close()

    def reset(self, task: int = 1, seed: int = 42) -> dict:
        return self._loop.run_until_complete(self._client.reset(task, seed))

    def step(self, action: ClaimAction) -> dict:
        return self._loop.run_until_complete(self._client.step(action))

    def state(self) -> dict:
        return self._loop.run_until_complete(self._client.state())
