from langchain_community.document_loaders import TextLoader
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os 
load_dotenv()
loader= TextLoader("poem.txt")
docs=loader.load()
#print(docs[0].page_content)


model=ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)

prompt= PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']

)

parser=StrOutputParser()

chain= prompt | model | parser
result=chain.invoke({'topic':docs[0].page_content})
print(result)

#chain.get_graph().print_ascii()