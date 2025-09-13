import os
import pandas as pd
import faiss
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers import ContextualCompressionRetriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from operator import itemgetter

load_dotenv()

# Embedding 
def make_embeddings():
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",  
    )
    return embeddings

# Vector DB -> Retriever 
def make_database_from_parquet(embedding, file_path):
    """
    Parquet 파일에서 임베딩 데이터와 텍스트를 로드하여 FAISS 벡터 데이터베이스를 생성합니다.
    """
    if not os.path.exists(file_path):
        print(f"❌ '{file_path}' 파일을 찾을 수 없습니다. 파일 경로를 확인해주세요.")
        return None

    print(f"⏳ '{file_path}' 파일을 로드 중입니다...")
    
    try:
        # Parquet 파일을 DataFrame으로 읽기
        df = pd.read_parquet(file_path)
        print("✅ 파일 로드 완료.")
        
        # DataFrame에서 임베딩 벡터와 텍스트 추출
        texts = df['text'].tolist()
        
        # Document 객체 생성
        docs = [Document(page_content=text) for text in texts]

        # FAISS.from_documents()를 사용하여 벡터 DB 생성
        faiss_db = FAISS.from_documents(docs, embedding)
        
        print("✅ FAISS 벡터 데이터베이스 생성이 완료되었습니다.")
        return faiss_db
    
    except Exception as e:
        print(f"❌ 벡터 DB 생성 중 오류 발생: {e}")
        return None

def make_retriever(faiss_db, llm):
    retriever = faiss_db.as_retriever(
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
def setup_everything(file_path):

    embeddings = make_embeddings()
    db = make_database_from_parquet(embeddings, file_path)
    llm = make_llm()
    retriever = make_retriever(db, llm)
    qa_chain = make_qa_chain(retriever, llm)

    return qa_chain

 
# 질문하고 답변받는 함수
def ask_question(qa_chain, question, chat_history):
    answer = qa_chain.invoke({"input": question, "chat_history": chat_history})

    return answer