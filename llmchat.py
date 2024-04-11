import os
import re
import uuid
import time
import datetime
import json
import logging
import asyncio

import httpx
import openai
import anthropic
import cohere


def extract_tripple_quotes(s):
    if len(s.split('"""')) < 2:
        return s

    last_3q = s.rfind('"""')
    prev_3q = s[:last_3q].rfind('"""')
    s = s[prev_3q+3:last_3q]
    s = s.strip()
    return s


def extract_answer_tag(s):
    m = re.search(r"<answer>(.*)</answer>", s, flags=re.DOTALL)

    if not m:
        raise ValueError(s)

    return m.group(1)


class ChatError(Exception):
    pass


class AsyncChatBase:

    models = None
    provider = None

    _temperature_default = 0.5
    _api_key_env_var = None

    def __init__(self,
                 api_key=None,
                 system=None,
                 temperature=None,
                 log_dir=None):
        super().__init__()

        if api_key is not None:
            self.api_key = api_key

        else:
            if self._api_key_env_var is None:
                raise ValueError("API key is not specified")

            self.api_key = os.getenv(self._api_key_env_var)

            if not self.api_key:
                raise ValueError("API key is not specified")

        self.system = system

        self.temperature = temperature

        if self.temperature is None:
            self.temperature = self._temperature_default

        self.log_dir = log_dir
        self.log_file = "{}-{}.txt".format(
            datetime.datetime.today().strftime("%Y%m%d%H%M%S"),
            str(uuid.uuid4())[:6])

    async def send(self, prompt):
        response = await self._send(prompt)

        if self.log_dir:
            with open(os.path.join(self.log_dir, self.log_file), "a",
                      encoding="utf-8") as fp:
                fp.write(f"User: {prompt}\n\n")
                fp.write(f"Assistant: {response}\n\n")
                fp.flush()

        return response


    async def _send(self, prompt):
        raise NotImplementedError()

    def _process_output(self, s):
        return s.strip()


class ChatBase:

    models = None
    provider = None

    _temperature_default = 0.5
    _api_key_env_var = None

    def __init__(self,
                 api_key=None,
                 system=None,
                 temperature=None,
                 log_dir=None):
        super().__init__()

        if api_key is not None:
            self.api_key = api_key

        else:
            if self._api_key_env_var is None:
                raise ValueError("API key is not specified")

            self.api_key = os.getenv(self._api_key_env_var)

            if not self.api_key:
                raise ValueError("API key is not specified")

        self.system = system

        self.temperature = temperature

        if self.temperature is None:
            self.temperature = self._temperature_default

        self.log_dir = log_dir
        self.log_file = "{}-{}.txt".format(
            datetime.datetime.today().strftime("%Y%m%d%H%M%S"),
            str(uuid.uuid4())[:6])

    def send(self, prompt):
        response = self._send(prompt)

        if self.log_dir:
            with open(os.path.join(self.log_dir, self.log_file), "a",
                      encoding="utf-8") as fp:
                fp.write(f"User: {prompt}\n\n")
                fp.write(f"Assistant: {response}\n\n")
                fp.flush()

        return response


    def _send(self, prompt):
        raise NotImplementedError()

    def _process_output(self, s):
        return s.strip()


class AsyncOpenAIChatBase(AsyncChatBase):

    _model = None
    _base_url = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        kw = {"api_key": self.api_key}

        if self._base_url:
            kw["base_url"] = self._base_url

        self.client = openai.AsyncOpenAI(**kw)

        self.messages = []

        if self.system:
            self.messages.append({"role": "system",
                                  "content": self.system})

    async def _send(self, prompt):
        user_message = {"role": "user",
                        "content": prompt}

        messages = self.messages[:]

        messages.append(user_message)

        tries = 5

        while True:
            try:
                response = await self.client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    temperature=self.temperature)

            # FIXME: proper error handling
            except openai.BadRequestError:
                tries -= 1

                if tries == 0:
                    raise

                await asyncio.sleep(1)

            else:
                break

        if hasattr(response, "error"):
            logging.error("%s: response=%s", self.__class__.__name__, response)
            raise ChatError(response.error)

        model_message = response.choices[0].message
        self.messages.append(user_message)
        self.messages.append({"role": model_message.role,
                              "content": model_message.content})
        return self._process_output(model_message.content)


class OpenAIChatBase(ChatBase):

    _model = None
    _base_url = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        kw = {"api_key": self.api_key}

        if self._base_url:
            kw["base_url"] = self._base_url

        self.client = openai.OpenAI(**kw)

        self.messages = []

        if self.system:
            self.messages.append({"role": "system",
                                  "content": self.system})

    def _send(self, prompt):
        user_message = {"role": "user",
                        "content": prompt}

        messages = self.messages[:]

        messages.append(user_message)

        tries = 5

        while True:
            try:
                response = self.client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    temperature=self.temperature)

            # FIXME: proper error handling
            except openai.BadRequestError:
                tries -= 1

                if tries == 0:
                    raise

                time.sleep(1)

            else:
                break

        if hasattr(response, "error"):
            logging.error("%s: response=%s", self.__class__.__name__, response)
            raise ChatError(response.error)

        model_message = response.choices[0].message
        self.messages.append(user_message)
        self.messages.append({"role": model_message.role,
                              "content": model_message.content})
        return self._process_output(model_message.content)


class AsyncOpenAIChat(AsyncOpenAIChatBase):

    models = ("chatgpt", "gpt-4", "gpt-4-turbo")
    provider = "openai"

    _api_key_env_var = "OPENAI_API_KEY"

    _model = "gpt-4-turbo"


class OpenAIChat(OpenAIChatBase):

    models = ("chatgpt", "gpt-4", "gpt-4-turbo")
    provider = "openai"

    _api_key_env_var = "OPENAI_API_KEY"

    _model = "gpt-4-turbo"


# class AsyncAnthropicChat(AsyncOpenAIChatBase):

#     models = ("claude", "claude-2", )
#     provider = "openrouter"

#     _api_key_env_var = "OPENROUTER_API_KEY"

#     _model = "anthropic/claude-2"
#     _base_url = "https://openrouter.ai/api/v1"


# class AnthropicChat(OpenAIChatBase):

#     models = ("claude", "claude-2", )
#     provider = "openrouter"

#     _api_key_env_var = "OPENROUTER_API_KEY"

#     _model = "anthropic/claude-2"
#     _base_url = "https://openrouter.ai/api/v1"


class AsyncMistralAIChat(AsyncOpenAIChatBase):

    models = ("mistral", "mistral-large", )
    provider = "mistralai"

    _api_key_env_var = "MISTRALAI_API_KEY"

    _model = "mistral-large-latest"
    _base_url = "https://api.mistral.ai/v1"


class MistralAIChat(OpenAIChatBase):

    models = ("mistral", "mistral-large", )
    provider = "mistralai"

    _api_key_env_var = "MISTRALAI_API_KEY"

    _model = "mistral-large-latest"
    _base_url = "https://api.mistral.ai/v1"


class AsyncGoogleAIChat(AsyncChatBase):

    models = ("gemini-pro", )
    provider = "googleai"

    _api_key_env_var = "GOOGLEAI_API_KEY"

    _model = "gemini-1.5-pro-latest"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.contents = []

        if self.system:
            system = self.system
            system += "\n\nAnswer OK if you understand."
            self.contents.append({"role": "user",
                                  "parts": [{"text": system}]})
            self.contents.append({"role": "model",
                                  "parts": [{"text": "OK"}]})

    # https://ai.google.dev/models/gemini
    async def _send(self, prompt):
        url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
               f"{self._model}:generateContent?key={self.api_key}")

        user_msg = {"role": "user",
                    "parts": [{"text": prompt}]}

        contents = self.contents[:]

        contents.append(user_msg)

        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": 2048,
            }
        }

        headers = {"Content-Type": "application/json"}

        tries = 5

        async with httpx.AsyncClient() as client:
            while True:
                try:
                    response = await client.post(
                        url=url,
                        headers=headers,
                        data=json.dumps(payload),
                        timeout=60)
                    response.raise_for_status()

                except httpx.HTTPError:  # FIXME: proper error handling
                    tries -= 1

                    if tries == 0:
                        raise

                    await asyncio.sleep(1)

                else:
                    break

        model_msg = response.json()["candidates"][0]["content"]
        self.contents.append(user_msg)
        self.contents.append(model_msg)
        return self._process_output(model_msg["parts"][0]["text"])


class GoogleAIChat(ChatBase):

    models = ("gemini-pro", )
    provider = "googleai"

    _api_key_env_var = "GOOGLEAI_API_KEY"

    _model = "gemini-1.0-pro"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.contents = []

        if self.system:
            system = self.system
            system += "\n\nAnswer OK if you understand."
            self.contents.append({"role": "user",
                                  "parts": [{"text": system}]})
            self.contents.append({"role": "model",
                                  "parts": [{"text": "OK"}]})

    # https://ai.google.dev/models/gemini
    def _send(self, prompt):
        url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
               f"{self._model}:generateContent?key={self.api_key}")

        user_msg = {"role": "user",
                    "parts": [{"text": prompt}]}

        contents = self.contents[:]

        contents.append(user_msg)

        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": 2048,
            }
        }

        headers = {"Content-Type": "application/json"}

        tries = 5

        with httpx.Client() as client:
            while True:
                try:
                    response = client.post(
                        url=url,
                        headers=headers,
                        data=json.dumps(payload),
                        timeout=60)
                    response.raise_for_status()

                except httpx.HTTPError:  # FIXME: proper error handling
                    tries -= 1

                    if tries == 0:
                        raise

                    time.sleep(1)

                else:
                    break

        model_msg = response.json()["candidates"][0]["content"]
        self.contents.append(user_msg)
        self.contents.append(model_msg)
        return self._process_output(model_msg["parts"][0]["text"])


class AsyncAnthropicChat(AsyncChatBase):

    models = ("claude", "claude-3", "claude-3-opus")
    provider = "anthropic"

    _api_key_env_var = "ANTHROPIC_API_KEY"

    _model = "claude-3-opus-20240229"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.client = anthropic.AsyncAnthropic(
            api_key=self.api_key,
            timeout=120.0)  # 2 minutes, default - 10 minutes

        self.messages = []

    async def _send(self, prompt):
        user_message = {"role": "user",
                        "content": prompt}

        messages = self.messages[:]

        messages.append(user_message)

        tries = 5

        while True:
            kw = {}

            if self.system:
                kw["system"] = self.system

            try:
                response = await self.client.messages.create(
                    model=self._model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=4096,
                    **kw)

            except anthropic.AnthropicError:  # FIXME: proper error handling
                tries -= 1

                if tries == 0:
                    raise

                await asyncio.sleep(1)

            else:
                break

        if hasattr(response, "error"):
            logging.error("%s: response=%s", self.__class__.__name__, response)
            raise ChatError(response.error.message)

        model_message = response.content[0].text
        self.messages.append(user_message)
        self.messages.append({"role": response.role,
                              "content": model_message})
        return self._process_output(model_message)


class AnthropicChat(ChatBase):

    models = ("claude", "claude-3", "claude-3-opus")
    provider = "anthropic"

    _api_key_env_var = "ANTHROPIC_API_KEY"

    _model = "claude-3-opus-20240229"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.client = anthropic.Anthropic(
            api_key=self.api_key,
            timeout=120.0)  # 2 minutes, default - 10 minutes

        self.messages = []

    def _send(self, prompt):
        user_message = {"role": "user",
                        "content": prompt}

        messages = self.messages[:]

        messages.append(user_message)

        tries = 5

        while True:
            kw = {}

            if self.system:
                kw["system"] = self.system

            try:
                response = self.client.messages.create(
                    model=self._model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=4096,
                    **kw)

            except anthropic.AnthropicError:  # FIXME: proper error handling
                tries -= 1

                if tries == 0:
                    raise

                time.sleep(1)

            else:
                break

        if hasattr(response, "error"):
            logging.error("%s: response=%s", self.__class__.__name__, response)
            raise ChatError(response.error.message)

        model_message = response.content[0].text
        self.messages.append(user_message)
        self.messages.append({"role": response.role,
                              "content": model_message})
        return self._process_output(model_message)


class AsyncCohereChat(ChatBase):

    models = ("command-r-plus", )
    provider = "cohere"

    _api_key_env_var = "COHERE_API_KEY"

    _model = "command-r-plus"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.client = cohere.AsyncClient(
            api_key=self.api_key,
            timeout=120.0)  # 2 minutes, default - 10 minutes

        self.messages = []

    async def _send(self, prompt):
        tries = 5
        exception = None

        while True:
            kw = {}

            if self.system:
                kw["preamble"] = self.system

            try:
                response = await self.client.chat(
                    model=self._model,
                    chat_history=self.messages,
                    temperature=self.temperature,
                    **kw)

            # FIXME: proper error handling
            except cohere.core.api_error.ApiError as e:
                exception = e

                tries -= 1

                if tries == 0:
                    raise

                time.sleep(1)

            else:
                exception = None

        if exception:
            logging.error("%s: response=%s", self.__class__.__name__,
                          exception.body)
            raise ChatError(exception.body["message"])

        model_message = response.text
        self.messages.append({"role": "USER",
                              "content": prompt})
        self.messages.append({"role": "CHATBOT",
                              "content": model_message})
        return self._process_output(model_message)


class CohereChat(ChatBase):

    models = ("command-r-plus", )
    provider = "cohere"

    _api_key_env_var = "COHERE_API_KEY"

    _model = "command-r-plus"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.client = cohere.Client(
            api_key=self.api_key,
            timeout=120.0)  # 2 minutes, default - 10 minutes

        self.messages = []

    def _send(self, prompt):
        tries = 5
        exception = None

        while True:
            kw = {}

            if self.system:
                kw["preamble"] = self.system

            try:
                response = self.client.chat(
                    model=self._model,
                    chat_history=self.messages,
                    temperature=self.temperature,
                    **kw)

            # FIXME: proper error handling
            except cohere.core.api_error.ApiError as e:
                exception = e

                tries -= 1

                if tries == 0:
                    raise

                time.sleep(1)

            else:
                exception = None

        if exception:
            logging.error("%s: response=%s", self.__class__.__name__,
                          exception.body)
            raise ChatError(exception.body["message"])

        model_message = response.text
        self.messages.append({"role": "USER",
                              "content": prompt})
        self.messages.append({"role": "CHATBOT",
                              "content": model_message})
        return self._process_output(model_message)


def chat_factory(model=None,
                 async_=False,
                 api_keys=None,
                 **kwargs):
    chat_classes_async = (AsyncOpenAIChat,
                          AsyncAnthropicChat,
                          AsyncMistralAIChat,
                          AsyncGoogleAIChat,
                          AsyncCohereChat)
    chat_classes_sync = (OpenAIChat,
                         AnthropicChat,
                         MistralAIChat,
                         GoogleAIChat,
                         CohereChat)

    if async_:
        chat_classes = chat_classes_async
    else:
        chat_classes = chat_classes_sync

    model_meta = model

    def create_chat(model=None, api_key=None, **kw):
        if model is None:
            model = model_meta

        if model is None:
            model = "gpt-4"

        chat_class = None

        for cls in chat_classes:
            if model in cls.models:
                chat_class = cls
                break

        if chat_class is None:
            raise ValueError(model)

        if api_key is None:
            api_key = api_keys.get(chat_class.provider)

        kw_ = {}
        kw_.update(kwargs)
        kw_.update(kw)

        return chat_class(api_key=api_key, **kw_)

    return create_chat


create_chat = chat_factory(async_=False)
create_chat_async = chat_factory(async_=True)
