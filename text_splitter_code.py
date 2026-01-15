from langchain_text_splitters import RecursiveCharacterTextSplitter,Language

text="""
class Calculator:
    def __init__(self, name):
        self.name = name

    def add(self, a, b):
        return a + b

    def subtract(self, a, b):
        return a - b


def calculate_operations():
    calc = Calculator("Simple Calculator")
    result1 = calc.add(10, 5)
    result2 = calc.subtract(10, 3)
    print(result1, result2)


if __name__ == "__main__":
    calculate_operations()

"""

splitter=RecursiveCharacterTextSplitter.from_language(
    Language=Language.PYTHON,
    chunk_size=376,
    chunk_overlap=0
)

chunks=splitter.split_text(text)
print(len(chunks))
print(chunks)