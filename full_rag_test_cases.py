import asyncio
from typing import Optional, Callable, Any, Awaitable
from test_main import Pipe
import json


class TestAILabBot:
    def __init__(self):
        self.pipe = Pipe()

    async def test_conversation(self, messages, task="chat", user=None):
        """
        ტესტავს მთლიან საუბარს ბოტთან
        """
        if user is None:
            user = {
                "id": "test_user",
                "name": "Test User"
            }

        events = []

        async def event_emitter(event):
            events.append(event)
            print("\nEvent:", json.dumps(event, indent=2, ensure_ascii=False))

        try:
            response = ""
            async for chunk in self.pipe.pipe(
                    body={"messages": messages},
                    __user__=user,
                    __task__=task,
                    __event_emitter__=event_emitter
            ):
                if chunk:
                    print("Response chunk:", chunk)
                    response += chunk

            return {
                "success": True,
                "response": response,
                "events": events
            }

        except Exception as e:
            print(f"Error during conversation: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    async def run_test_scenarios(self):
        """
        ტესტავს სხვადასხვა სცენარებს
        """
        print("\n=== Starting Test Scenarios ===\n")

        test_scenarios = [
            {
                "name": "დანაკარგის ფუნქციის კითხვა",
                "messages": [
                    {"role": "system", "content": "You are an AI assistant."},
                    {"role": "user", "content": "დანაკარგის ფუნქცია რა არის?"}
                ]
            },
            {
                "name": "ნეირონული ქსელების კითხვა",
                "messages": [
                    {"role": "system", "content": "You are an AI assistant."},
                    {"role": "user", "content": "ნეირონული ქსელები როგორ მუშაობს?"}
                ]
            },
            {
                "name": "მანქანური სწავლების კითხვა",
                "messages": [
                    {"role": "system", "content": "You are an AI assistant."},
                    {"role": "user", "content": "მანქანური სწავლება რას ნიშნავს?"}
                ]
            },
            {
                "name": "მრავალნაბიჯიანი საუბარი",
                "messages": [
                    {"role": "system", "content": "You are an AI assistant."},
                    {"role": "user", "content": "რა არის მანქანური სწავლება?"},
                    {"role": "assistant",
                     "content": "მანქანური სწავლება არის ხელოვნური ინტელექტის ნაწილი, რომელიც საშუალებას აძლევს კომპიუტერებს ისწავლონ მონაცემებიდან."},
                    {"role": "user", "content": "და რა არის დანაკარგის ფუნქცია ამ კონტექსტში?"}
                ]
            }
        ]

        results = {}
        for scenario in test_scenarios:
            print(f"\nTesting Scenario: {scenario['name']}")
            print("-" * 50)

            result = await self.test_conversation(scenario['messages'])

            if result['success']:
                print("\nFinal Response:", result['response'])
                print("\nEvents Generated:", len(result['events']))
                print("Status Events:", sum(1 for e in result['events'] if e['type'] == 'status'))
                print("Citation Events:", sum(1 for e in result['events'] if e['type'] == 'citation'))
            else:
                print("\nTest Failed:", result['error'])

            results[scenario['name']] = result
            print("-" * 50)

        return results


async def run_tests():
    bot_tester = TestAILabBot()

    print("Starting Comprehensive Bot Tests...")
    results = await bot_tester.run_test_scenarios()

    print("\n=== Test Summary ===")
    for scenario_name, result in results.items():
        status = "✅ Passed" if result['success'] else "❌ Failed"
        print(f"{scenario_name}: {status}")


if __name__ == "__main__":
    asyncio.run(run_tests())