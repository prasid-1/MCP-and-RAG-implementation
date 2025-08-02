
import argparse
from email import parser
from langchain_ollama import OllamaLLM as Ollama
from getAiModel import getAiModel


prompt = """The provided text is extracted from an image using OCR. The image has word for GRE exam preparation. each words has a definition, example, and mnemonic if available. Your job is to keep these information in json format with correct spelling. (output only json format) use format:
```json
[
  {
    "word": "<word1>",
    "definition": "<definition1>",
    "example": "<example1>",
    "mnemonic": "<mnemonic1>"
  },
  {
    "word": "<word2>",
    "definition": "<definition2>",
    "example": "<example2>",
    "mnemonic": "<mnemonic2>"
  }
]
```
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    # query_text = prompt.join(args.query_text.split("\n"))
    query_text = prompt + "\n" + " ".join(args.query_text.split("_"))
    # print(f"Query Text: {query_text}")
    query_output(query_text)


def query_output(query_text: str):
    model = Ollama(model=getAiModel())
    response_text = model.invoke(query_text)
    print(response_text)


if __name__ == "__main__":
    main()