import asyncio
from app import analyze
import traceback

async def main():
    try:
        print("Testing analyze()...")
        # Simulating the text input
        # analyze takes text as a kwarg string
        res = await analyze(file=None, text="Apple Inc. reported $5 billion in revenue")
        print("Success:", res)
    except Exception as e:
        print("API endpoint failed with exception:")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
