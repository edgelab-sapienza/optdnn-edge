from tf_optimizer_core.node import Node
import asyncio
import argparse


async def main():
    parser = argparse.ArgumentParser(
        prog="tf_optimizer_core",
        description="Node use for benchmarking models",
    )
    parser.add_argument(
        "--port",
        type=int,
        help="Port number in which the software listens for connection (default 12300)",
        required=False,
        default=12300,
    )
    args = parser.parse_args()

    n = Node(args.port)
    await n.serve()


if __name__ == "__main__":
    asyncio.run(main())
