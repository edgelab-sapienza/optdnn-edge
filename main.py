import asyncio
from benchmarker_core.node import Node


async def main():
    n = Node()
    await n.serve()


if __name__ == "__main__":
    asyncio.run(main())
