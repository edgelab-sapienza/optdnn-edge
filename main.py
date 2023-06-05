from tf_optimizer_core.node import Node
import asyncio


async def main():
    n = Node()
    await n.serve()


if __name__ == "__main__":
    asyncio.run(main())
