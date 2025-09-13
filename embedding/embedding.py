import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# 파일 경로를 매개변수로 받아 외부 문서를 읽어서 반환하는 함수
def load_pdf(file_path):
    """
    지정된 PDF 파일을 로드하여 문서를 반환합니다.
    """
    try:
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        print(f"✅ '{file_path}' 파일을 성공적으로 로드했습니다.")
        return docs
    except Exception as e:
        print(f"❌ PDF 파일 로드 중 오류 발생: {e}")
        return None

# 문서를 받아서 청크 단위로 분할하고 반환하는 함수
def split_text(docs):
    """
    문서를 작은 청크로 분할합니다.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", ". ", " "],
        chunk_size=700,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(docs)
    print(f"✅ 텍스트를 {len(chunks)}개의 청크로 분할했습니다.")
    return chunks

# 임베딩을 생성하는 함수
def make_embeddings():
    """
    OpenAI embeddings 모델을 초기화하여 반환합니다.
    """
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
    )
    return embeddings

# 청크 데이터를 임베딩하고 Pandas DataFrame으로 변환하여 저장하는 함수
def process_and_save_data(chunks, embeddings, output_file_path):
    """
    각 텍스트 청크를 임베딩하고, 데이터를 DataFrame으로 변환하여 Parquet 파일로 저장합니다.
    """
    if not chunks:
        print("❌ 저장할 청크 데이터가 없습니다.")
        return

    # 청크의 텍스트를 임베딩
    print("⏳ 텍스트 청크를 임베딩 중입니다...")
    embedding_vectors = embeddings.embed_documents([chunk.page_content for chunk in chunks])
    print("✅ 모든 임베딩 생성이 완료되었습니다.")

    # 저장할 데이터 구조 생성
    data = []
    for i, chunk in enumerate(chunks):
        # 문서의 원본 경로와 메타데이터를 저장
        source_path = chunk.metadata.get("source", "unknown")
        data.append({
            "chunk_id": i,
            "text": chunk.page_content,
            "source": source_path,
            "embedding_vector": embedding_vectors[i]
        })

    # Pandas DataFrame으로 변환
    df = pd.DataFrame(data)

    # DataFrame을 Parquet 파일로 저장
    try:
        table = pa.Table.from_pandas(df)
        pq.write_table(table, output_file_path)
        print(f"✅ 데이터를 '{output_file_path}'에 성공적으로 저장했습니다.")
    except Exception as e:
        print(f"❌ Parquet 파일 저장 중 오류 발생: {e}")

# 전체 워크플로우를 실행하는 메인 함수
def main():
    pdf_file_path = "../data/High-Performance-Browser-Network.pdf"
    output_file = 'files/HPBN_embeddings.parquet'

    # 예시 파일을 생성합니다 (실제 사용 시에는 본인의 PDF 파일 경로를 지정하세요).
    if not os.path.exists(pdf_file_path):
        from fpdf import FPDF
        pdf = FPDF()
        pdf.add_page()
        pdf.add_font("Pretendard-Regular", "", r"Pretendard-Regular.ttf", uni=True)
        pdf.set_font("Pretendard-Regular", "", 12)
        pdf.cell(200, 10, txt="이것은 예시 PDF 파일입니다.", ln=1, align="C")
        pdf.cell(200, 10, txt="이 파일은 RAG를 위한 임베딩 테스트에 사용됩니다.", ln=1, align="C")
        pdf.output(pdf_file_path)
        print(f"ℹ️ '{pdf_file_path}' 예시 파일을 생성했습니다. 실제 PDF 파일로 경로를 변경하세요.")

    docs = load_pdf(pdf_file_path)
    if docs:
        chunks = split_text(docs)
        embeddings_model = make_embeddings()
        process_and_save_data(chunks, embeddings_model, output_file)
        print("✅ 모든 작업이 완료되었습니다.")

if __name__ == "__main__":
    main()
