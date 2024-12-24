import asyncio
from typing import Optional
from test_main import Pipe


class ResponseAccumulator:
    def __init__(self):
        self.current_response = ""
        self.is_accumulating = False

    def add_chunk(self, chunk: str) -> Optional[str]:
        if chunk and not self.is_accumulating:
            self.is_accumulating = True

        if self.is_accumulating:
            self.current_response += chunk
        return None

    def get_final_response(self) -> str:
        return self.current_response


async def accumulate_stream(stream):
    """აკუმულირებს async გენერატორის მიერ დაბრუნებულ მნიშვნელობებს"""
    accumulated = ""
    async for chunk in stream:
        if chunk:
            accumulated += chunk
    return accumulated


async def test_pipe():
    """ტესტავს Pipe კლასის ფუნქციონალს"""
    pipe = Pipe()

    events = []

    async def event_handler(event):
        events.append(event)
        print(f"\nEvent received: {event}")

    print("\nStarting pipe test...")
    try:
        stream = pipe.pipe(
            body={"messages": [{"role": "user", "content": "ხელის კრემის ყიდვა მინდა"}]},
            __user__={"id": "test", "name": "Test User"},
            __task__="chat",
            __event_emitter__=event_handler
        )

        response = await accumulate_stream(stream)
        print(f"\nAccumulated response: {response}")
        print(f"\nTotal events received: {len(events)}")

        # დეტალური ანალიზი ივენთების
        status_events = [e for e in events if e.get('type') == 'status']
        citation_events = [e for e in events if e.get('type') == 'citation']

        print(f"\nStatus events: {len(status_events)}")
        print(f"Citation events: {len(citation_events)}")

        return response, events

    except Exception as e:
        print(f"\nError in test_pipe: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return None, []


async def test_ailab_agent():
    """ტესტავს AILab აგენტის ფუნქციონალს"""
    pipe = Pipe()

    messages = [
        {"role": "system", "content": "You are an AI Lab assistant."},
        {"role": "user", "content": "ამიხსენი როგორ ხდება ბუნებრივი ენის დამუშავება"},
    ]

    body = {"messages": messages}
    user = {"id": "test_user", "name": "Test User"}

    events = []

    async def event_emitter(event):
        events.append(event)
        print("\nEvent:", event)

    print("\nStarting agent test...")
    try:
        stream = pipe.pipe(
            body=body,
            __user__=user,
            __task__="chat",
            __event_emitter__=event_emitter
        )

        response = await accumulate_stream(stream)
        print(f"\nAccumulated response from agent: {response}")
        print(f"\nTotal events from agent: {len(events)}")

        return response, events

    except Exception as e:
        print(f"\nError in test_ailab_agent: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return None, []


async def run_all_tests():
    """გაუშვებს ყველა ტესტს"""
    print("=== Starting All Tests ===")

    print("\n1. Testing basic pipe functionality")
    pipe_response, pipe_events = await test_pipe()

    print("\n2. Testing AILab agent functionality")
    agent_response, agent_events = await test_ailab_agent()

    print("\n=== Test Results Summary ===")
    print(f"Pipe test {'succeeded' if pipe_response else 'failed'}")
    print(f"Agent test {'succeeded' if agent_response else 'failed'}")

    print("\n=== Event Statistics ===")
    print(f"Pipe events: {len(pipe_events)}")
    print(f"Agent events: {len(agent_events)}")


if __name__ == "__main__":
    asyncio.run(run_all_tests())