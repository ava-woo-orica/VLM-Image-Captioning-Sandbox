from abc import ABC, abstractmethod
from google import genai
from loguru import logger


class LLMInterface(ABC):
    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def inference(self, prompt: str, **kwargs) -> str:
        pass

    @abstractmethod
    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
        return False


class LLMInterfaceFactory:
    def __init__(self):
        self.__model_mapping = {
            "gemini": GeminiInterface,
        }

    def get_model(self, model_type: str, **kwargs) -> LLMInterface:
        if model_type not in self.__model_mapping:
            raise ValueError(f"Unknown model type '{model_type}'. Available: {list(self.__model_mapping)}")
        return self.__model_mapping[model_type](**kwargs)


class GeminiInterface(LLMInterface):
    def __init__(self, **kwargs):
        self.client = genai.Client(vertexai=True, project=kwargs.get("project_id"), location=kwargs.get("location"))
        self.model_name = kwargs.get("model_name")

    def inference(self, prompt: str, **kwargs) -> str:
        try:
            resp = self.client.models.generate_content(model=self.model_name, contents=prompt)
            return (resp.text or "").strip()
        except KeyError as e:
            logger.error(f"Missing required argument: {e}")
            return ""
        except Exception as e:
            logger.error(f"Error during inference: {e}")
            return ""

    def close(self):
        if hasattr(self, "client"):
            self.client.close()

    def __del__(self):
        self.close()


if __name__ == "__main__":
    factory = LLMInterfaceFactory()
    with factory.get_model(model_type="gemini", model_name="gemini-3.1-flash-lite-preview") as model:
        prompt = "What is the capital of France?"
        response = model.inference(prompt)
        print(f"Response: {response}")
