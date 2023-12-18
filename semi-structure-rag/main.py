import uuid
from typing import Any

from langchain.retrievers import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel
from unstructured.partition.pdf import partition_pdf

import dotenv

dotenv.load_dotenv()
path = "./public/"
file = "2307.09288.pdf"

# Get elements
raw_pdf_elements = partition_pdf(
    # filename=path + "DS Batch Tracker.pdf",
    filename=path + file,
    # Unstructured first finds embedded image blocks
    extract_images_in_pdf=False,
    # Use layout model (YOLOX) to get bounding boxes (for tables) and find titles
    # Titles are any subsection of the document
    infer_table_structure=True,
    # Post processing to aggregate text once we have the title
    chunking_strategy="by_title",
    # Chunking params to aggregate text blocks
    # Attempt to create a new chunk 3800 chars
    # Attempt to keep chunks > 2000 chars
    max_characters=4000,
    new_after_n_chars=3800,
    combine_text_under_n_chars=2000,
    image_output_dir_path=path,
    url=None,
)


class Element(BaseModel):
    type: str
    text: Any


# Categorize by type
categorized_elements = []
categorized_elements_text = []
categorized_elements_table = []
for element in raw_pdf_elements:
    if "unstructured.documents.elements.Table" in str(type(element)):
        # categorized_elements.append(Element(type="table", text=str(element)))
        categorized_elements_table.append(str(element))
    elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
        # categorized_elements.append(Element(type="text", text=str(element)))
        categorized_elements_text.append(str(element))

# Prompt
prompt_text = """You are an assistant tasked with summarizing tables and text. \ 
Give a concise summary of the table or text. Table or text chunk: {element} """
prompt = ChatPromptTemplate.from_template(prompt_text)

# Summary chain
model = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

table_summaries = summarize_chain.batch(categorized_elements_table, {"max_concurrency": 5})

text_summaries = summarize_chain.batch(categorized_elements_text, {"max_concurrency": 5})

# The vectorstore to use to index the child chunks
vectorstore = Chroma(collection_name="summaries", embedding_function=OpenAIEmbeddings())

# The storage layer for the parent documents
store = InMemoryStore()
id_key = "doc_id"

# The retriever (empty to start)
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    docstore=store,
    id_key=id_key,
)

# Add texts
doc_ids = [str(uuid.uuid4()) for _ in categorized_elements_text]
summary_texts = [
    Document(page_content=s, metadata={id_key: doc_ids[i]})
    for i, s in enumerate(text_summaries)
]
retriever.vectorstore.add_documents(summary_texts)
retriever.docstore.mset(list(zip(doc_ids, categorized_elements_text)))

# Add tables
table_ids = [str(uuid.uuid4()) for _ in categorized_elements_table]
summary_tables = [
    Document(page_content=s, metadata={id_key: table_ids[i]})
    for i, s in enumerate(table_summaries)
]
retriever.vectorstore.add_documents(summary_tables)
retriever.docstore.mset(list(zip(table_ids, categorized_elements_table)))

# Prompt template
template = """Answer the question based only on the following context, which can include text and tables:
{context}
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# LLM
model = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

# RAG pipeline
chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
)

res = chain.invoke("What is the number of training tokens for LLaMA2?")
print(res)
