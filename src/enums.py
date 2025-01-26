from enum import Enum

class LlmModels(Enum):
    LLAMA_2_13B_CHAT = "meta-llama/Llama-2-13b-chat-hf"
    LLAMA_2_13B_INSTRUCT = "codellama/CodeLlama-13b-Instruct-hf"

    def __str__(self):
        return self.value