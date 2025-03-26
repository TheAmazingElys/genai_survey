import argparse
from llama_cpp import Llama


from typing import Protocol


class LLM(Protocol):

    def __init__(self, system_prompt = None): ...

    def request(self, content) -> str: ...


LLAMA_3B = "models/Hermes-3-Llama-3.2-3B-Q6_K.gguf"
LLAMA_3B_CHAT_FORMAT = "llama-3"

QWEN_0_5_B = "models/qwen2.5-0.5b-instruct-fp16.gguf"
QWEN_CHAT_FORMAT = "qwen"

class LlamaCppLLM:

    def __init__(self, system_prompt = None):
        self.llm = Llama(
            model_path=LLAMA_3B,
            chat_format=LLAMA_3B_CHAT_FORMAT,
            # n_gpu_layers=-1, # Uncomment to use GPU acceleration
            # seed=1337, # Uncomment to set a specific seed
            # n_ctx=2048, # Uncomment to increase the context window,
            verbose=False,
        )

        self.system_prompt = system_prompt
        self.clear()

    def clear(self):
        self.messages = []
        if self.system_prompt:
            self.messages.append({"role": "system", "content": self.system_prompt},)


    def request(self, content):

        if content:
            self.messages.append({"role": "user", "content": content},)

        result = self.llm.create_chat_completion(
            messages = self.messages
        )

        self.messages.append(result["choices"][0]["message"])

        return result["choices"][0]["message"]["content"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chat with a model.")
    args = parser.parse_args()

    llm = LlamaCppLLM("You are a LLM chatbot. Be nice and helpful. Answer in 1 or 2 lines max if possible.")
    # llm = OllamaLLM()

    print("Chat with the model. Type 'exit' to end the chat.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        if user_input.lower() == "clear":
            llm.clear()
            continue
        if user_input.lower() == "debug":
            print(llm.messages)
            continue
        response = llm.request(user_input)
        print(f"Model: {response}")