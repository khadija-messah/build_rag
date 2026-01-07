import requests

URL = "http://localhost:11434/api/generate"

print("FREE AI Chatbot 🤖 (type 'bye' to exit)")

while True:
    user_input = input("You: ")

    if user_input.lower() == "bye":
        print("Bot: Bye 👋")
        break

    response = requests.post(URL,
        json={
            "model": "mistral",
            "prompt": user_input,
            "stream": False
        }
    )

    data = response.json()
    print("Bot:", data["response"])
