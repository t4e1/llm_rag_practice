from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from operator import itemgetter
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
import os

load_dotenv()

# 1. RAG를 만들기 위해 준비해야할 것 

# Document Loader 
# 파일경로를 매개변수로 받아 외부 문서를 읽어서 반환하는 함수 
def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    return docs

# Text Splitter
# docs를 받아서 split 하고 chunck 묶음으로 반환하는 함수
def split_text(docs):
    # text_splitter = CharacterTextSplitter(
    text_splitter = RecursiveCharacterTextSplitter(
        # separator="\n",
        separators=["\n\n", ". ", " "],  # 문단 -> 줄 -> 문장 -> 단어 순 
        chunk_size=200,     # 각 청크 최대 200자
        chunk_overlap=20   # 20자씩 겹침
    )

    chunks = text_splitter.split_documents(docs)
    return chunks

# Embedding 
def make_embeddings(chunks):
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",  
    )
    return embeddings

# Vector DB -> Retriever 
def make_database(chunks, embeddings):
    db = FAISS.from_documents(documents=chunks, embedding=embeddings)
    return db

def make_retriever(db, llm):
    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k":4,
            "lambda_mult": 0.7
            }
        )
    
    compressor = LLMChainExtractor.from_llm(llm)

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=retriever
    )
    return compression_retriever

# # 프롬프트
# prompt_template = PromptTemplate.from_template(
#     """당신은 질문에 답변하는 작업을 수행하는 어시스턴트입니다.
# 다음에 제공된 문맥 정보를 바탕으로 질문에 답하세요.
# 정답을 모를 경우, 모른다고만 말하세요.
# 답변은 반드시 한국어로 작성하세요.

# #문맥:
# {context}

# #질문:
# {question}

# #답변:"""
# )

# LLM 
def make_llm():
    llm = init_chat_model("gpt-4o-mini", model_provider="openai", temperature=0.0)
    return llm

# 체인 
def make_qa_chain(retriever, llm):

    # 기존에 사용하던 프롬프트 수정
    prompt = """
    당신은 질문에 답변하는 작업을 수행하는 어시스턴트입니다.
    다음에 제공된 문맥 정보와 이전 채팅 기록을 바탕으로 질문에 답하세요.
    정답을 모를 경우, 모른다고만 말하세요.
    답변은 반드시 한국어로 작성하세요.

    #문맥:
    {context}
    """

    # 이전 대화내역들을 받아옴
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )

    chain = (
        RunnablePassthrough.assign(context=itemgetter("input") | retriever)
        | qa_prompt
        | llm
        | StrOutputParser()
    )
    return chain

# 전체 시스템을 엮어서 내보내기
def setup_everything(pdf_file):
    
    docs = load_pdf(pdf_file)
    
    chunks = split_text(docs)

    embeddings = make_embeddings(chunks)

    db = make_database(chunks, embeddings)

    llm = make_llm()

    retriever = make_retriever(db, llm)

    qa_chain = make_qa_chain(retriever, llm)

    return qa_chain


# 질문하고 답변받는 함수
def ask_question(qa_chain, question, chat_history):
    answer = qa_chain.invoke({"input": question, "chat_history": chat_history})

    return answer