import asyncio


class ControlledModel:
    def __init__(self):
        self.requests = asyncio.Queue()

    async def decide(self, snapshot, tools):
        future = asyncio.get_running_loop().create_future()
        self.requests.put_nowait((snapshot, future))
        return await future

    async def next(self):
        return await asyncio.wait_for(self.requests.get(), 1)


async def event_of(agent, run_id, kind):
    sub = agent.subscribe(run_id, after=0)
    try:
        async with asyncio.timeout(1):
            async for event in sub:
                if event.type == kind:
                    return event
    finally:
        sub.close()
    raise AssertionError(f"missing event {kind}")
