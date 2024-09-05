import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial


async def loop_with_function(function, *args):
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as pool:
        if args is None:
            return await loop.run_in_executor(pool, function)
        else:
            return await loop.run_in_executor(pool, partial(function, *args))
