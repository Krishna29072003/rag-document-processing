from langchain_text_splitters import RecursiveCharacterTextSplitter
text="""
Regression is a supervised learning technique used to predict continuous values based on input data. It helps model the relationship between independent variables and a dependent variable by finding patterns in historical data. Common applications of regression include price prediction, demand forecasting, and trend analysis.
"""

splitter=RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=15
)

chunks=splitter.split_text(text)
print(len(chunks))
print(chunks)