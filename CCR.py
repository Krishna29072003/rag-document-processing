from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
'''from langchain_community.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors import LLMChainExtractor'''
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
load_dotenv()

# Recreate the document objects from the previous data
docs = [
    Document(page_content=(
        """The Grand Canyon is one of the most visited natural wonders in the world.
        Photosynthesis is the process by which green plants convert sunlight into energy.
        Millions of tourists travel to see it every year. The rocks date back millions of years."""
    ), metadata={"source": "Doc1"}),

    Document(page_content=(
        """In medieval Europe, castles were built primarily for defense.
        The chlorophyll in plant cells captures sunlight during photosynthesis.
        Knights wore armor made of metal. Siege weapons were often used to breach castle walls."""
    ), metadata={"source": "Doc2"}),

    Document(page_content=(
        """Basketball was invented by Dr. James Naismith in the late 19th century.
        It was originally played with a soccer ball and peach baskets. NBA is now a global league."""
    ), metadata={"source": "Doc3"}),

    Document(page_content=(
        """The history of cinema began in the late 1800s. Silent films were the earliest form.
        Thomas Edison was among the pioneers. Photosynthesis does not occur in animal cells.
        Modern filmmaking involves complex CGI and sound design."""
    ), metadata={"source": "Doc4"})
]

embeddings=HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vector_store=FAISS.from_documents(
    documents=docs,
    embedding=embeddings
)

#base_retriever=vector_store.as_retriever(search_kwargs={"k": 5})

llm=ChatGroq(model="llama-3.1-8b-instant",temperature=0)

# Create the contextual compression retriever
'''compression_retriever = ContextualCompressionRetriever(
    base_retriever=base_retriever,
    base_compressor=compressor
)'''


# Query the retriever
query = "What is photosynthesis?"

compressed_docs = []

for i in docs:
    prompt = f"""
    Question:
    {query}

    Document:
    {i.page_content}

    Extract ONLY the parts relevant to the question.
    """
    
    compressed_text = llm.invoke(prompt)
    compressed_docs.append(Document(
            page_content=compressed_text.content,
            metadata=i.metadata
        ))

for j, doc in enumerate(compressed_docs):
    print(f"\n--- Result {j+1} ---")
    print(doc.page_content)
