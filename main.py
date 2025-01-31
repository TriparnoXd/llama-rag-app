from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers.multi_query import MultiQueryRetriever

paths = [
    "./data/jemh101.pdf",
    "./data/jemh102.pdf",
    "./data/jemh103.pdf",
    "./data/jemh104.pdf",
    "./data/jemh105.pdf",
    "./data/jemh106.pdf",
    "./data/jemh107.pdf",
    "./data/jemh108.pdf",
    "./data/jemh109.pdf",
    "./data/jemh110.pdf",
    "./data/jemh111.pdf",
    "./data/jemh112.pdf",
    "./data/jemh113.pdf",
    "./data/jemh114.pdf",
]

docs_list = []

for path in paths:
    loader = PyPDFLoader(path)
    docs = loader.lazy_load() 
    docs_list.extend(docs) 

if docs_list:
    print(docs_list[0].page_content)

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size = 250,
    chunk_overlap = 0
)

doc_splits = text_splitter.split_documents(docs_list)

vectorstore = SKLearnVectorStore.from_documents(
    documents=doc_splits,
    embedding=OpenAIEmbeddings
)

prompt = PromptTemplate(
    template="""You are an AI tutor for question-answering tasks.
    Use the following documents to answer the question.
    If you don't know the answer, just say that you don't know.
    Give full explaination so that the student doesnt have any doubt left:
    Question: {question}
    Documents: {documents}
    Answer:
    """,
    input_variables=["question", "documents"],
)

llm = ChatOllama(model="llama3.2", temperature=0)
rag_chain = prompt | llm | StrOutputParser()


class RAGApplication:
    def __init__(self, retriever, rag_chain):
        self.retriever = retriever
        self.rag_chain = rag_chain
    def run(self, question):
        documents = self.retriever.invoke(question)
        doc_texts = "\\n".join([doc.page_content for doc in documents])
        answer = self.rag_chain.invoke({"question": question, "documents": doc_texts})
        return answer


retriever= MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(), llm=llm
)
rag_application = RAGApplication(retriever, rag_chain)
question = "What is prompt engineering"
answer = rag_application.run(question)
print("Question:", question)
print("Answer:", answer)