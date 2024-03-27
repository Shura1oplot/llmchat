# LLMChat

A simple python library, a wrapper to the well-known LLM APIs (OpenAI, Anthropic, Google AI, Mistral).

```python
import llmchat

# API keys can be imported from the environment as well
api_keys = {
    "openai": "sk-XXX",
    "anthropic": "sk-ant-api03-XXX",
    "mistralai": "XXX",
    "googleai": "XXX"
}

create_chat = chat_factory(api_keys=api_keys,
                           async_=False)
chat = create_chat(model="gpt-4",  # gpt-4, claude-3, gemini-pro, mistral-large
                   system="You are a business analyst.",  # optional
                   temperature=0.1,  # optional
                   log_dir="logs")  # optional
response = chat.send("Hi!")
print(response)
response = chat.send("How are you?")
print(response)
```
