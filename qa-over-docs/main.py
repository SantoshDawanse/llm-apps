from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.output_parsers import PydanticOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

import dotenv
from langchain_core.tracers.context import tracing_v2_enabled

import models

dotenv.load_dotenv()

pdf_loader = PyPDFLoader("./public/2305.05976.pdf")
docs = pdf_loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

template = """You are an expert in the Machine Learning and Artificial Intelligence. You have deep knowledge on 
large language models and your research includes Negative Common Sense Knowledge.

Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}
Question: {question}
Answer the user query: {format_instructions}"""

parser = PydanticOutputParser(pydantic_object=models.Response)
rag_prompt_custom = PromptTemplate(
    template=template,
    input_variables=["question"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# rag_chain_from_docs = (
#     {
#         "context": lambda input: format_docs(input["documents"]),
#         "question": itemgetter("question"),
#     }
#     | rag_prompt_custom
#     | llm
#     | StrOutputParser()
# )
# rag_chain_with_source = RunnableParallel(
#     {"documents": retriever, "question": RunnablePassthrough()}
# ) | {
#     "documents": lambda input_doc: [doc.metadata for doc in input_doc["documents"]],
#     "answer": rag_chain_from_docs,
# }

rag_chain = (
    {
        "context": retriever | (lambda in_docs: "\n\n".join(doc.page_content for doc in in_docs)),
        "question": RunnablePassthrough()
    }
    | rag_prompt_custom
    | llm
    | parser
)

with tracing_v2_enabled(project_name="qa-over-docs"):
    output = rag_chain.invoke("What are the thing that we should be careful while designing prompts?")
    print(output)
    print(type(output))
