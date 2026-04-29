import os
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

def test_openai_connection():
    print("=" * 50)
    print("Testing Azure OpenAI connection...")
    print(f"  Endpoint : {os.environ.get('AZURE_OPENAI_ENDPOINT')}")
    print(f"  Deployment: {os.environ.get('AZURE_OPENAI_DEPLOYMENT')}")
    print(f"  API Key  : {os.environ.get('AZURE_OPENAI_API_KEY', '')[:8]}...")
    print("=" * 50)

    try:
        client = AzureOpenAI(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        )

        response = client.chat.completions.create(
            model=os.environ["AZURE_OPENAI_DEPLOYMENT"],
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user",   "content": "Reply with exactly: CONNECTION OK"}
            ],
            max_tokens=20,
        )

        reply = response.choices[0].message.content.strip()
        print(f"\nPASS LLM response : {reply}")
        print(f"   Model used   : {response.model}")
        print(f"   Tokens used  : {response.usage.total_tokens}")

        ### uncomment if u want to test Run a multi-turn conversation and Stream the output ###
        # response = client.chat.completions.create(
        #     stream=True,
        #     messages=[
        #         {
        #             "role": "system",
        #             "content": "You are a helpful assistant.",
        #         },
        #         {
        #             "role": "user",
        #             "content": "I am going to Paris, what should I see?",
        #         },
        #         {
        #             "role": "assistant",
        #             "content": "Paris, the capital of France, is known for its stunning architecture, art museums, historical landmarks, and romantic atmosphere. Here are some of the top attractions to see in Paris:\n\n1. The Eiffel Tower: The iconic Eiffel Tower is one of the most recognizable landmarks in the world and offers breathtaking views of the city.\n2. The Louvre Museum: The Louvre is one of the worlds largest and most famous museums, housing an impressive collection of art and artifacts, including the Mona Lisa.\n3. Notre-Dame Cathedral: This beautiful cathedral is one of the most famous landmarks in Paris and is known for its Gothic architecture and stunning stained glass windows.\n\nThese are just a few of the many attractions that Paris has to offer. With so much to see and do, its no wonder that Paris is one of the most popular tourist destinations in the world.",
        #         },
        #         {
        #             "role": "user",
        #             "content": "What is so great about #1?",
        #         }
        #     ],
        #     max_tokens=4096,
        #     temperature=1.0,
        #     top_p=1.0,
        #     model=os.environ.get("AZURE_OPENAI_DEPLOYMENT")
        # )

        # for update in response:
        #     if update.choices:
        #         print(update.choices[0].delta.content or "", end="")

        client.close()

    except Exception as e:
        print(f"\nFAILED: {e}")

if __name__ == "__main__":
    test_openai_connection()