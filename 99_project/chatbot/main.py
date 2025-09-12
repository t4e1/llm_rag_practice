import streamlit as st
from datetime import datetime
from rag_system import setup_everything, ask_question

# streamlit 페이지 설정 
st.set_page_config(page_title="AI Chatbot", page_icon="🐻", layout="wide")

# 세션 상태관리

# 시스템 준비완료 상태 초기화
if "is_ready" not in st.session_state:
    st.session_state.is_ready = False

# 채팅 메세지 목록
if "messages" not in st.session_state:
    st.session_state.messages = []  # 채팅 메세지 들을 저장할 리스트 

# RAG 시스템 초기화 
if "qa_system" not in st.session_state:
    st.session_state.qa_system = None  # RAG 시스템 객체를 저장할 변수


PDF_FILES = "../../data/[AI.GOV_해외동향]_2025-1호.pdf"

@st.cache_resource  # 한번 실행하고 결과를 캐시에 저장
def auto_start_system():
    qa_system = setup_everything(PDF_FILES)
    return qa_system

# AI에게 질문을 전달하고 답변을 받는 함수 
def get_ai_answer(question):
    return ask_question(st.session_state.qa_system, question)

# 메인 채팅 인터페이스 
if not st.session_state.is_ready:
    with st.spinner("AI 챗봇을 준비하는 중입니다.. 잠시만 기다려주세요 !"):
        try:
            # RAG 시스템 구성요소 초기화
            # RAG 구성요소 준비하는 코드 들어갈 자리 
            qa_system = auto_start_system()

            st.session_state.qa_system = qa_system

            # 첫 메세지 객체
            welcome = {
                "role":"assistant",
                "content":"안녕하세요 챗봇입니다!"
                + "AI 해외동향에 관련한 궁금한 것을 물어보세요!",
                "timestamp": datetime.now().strftime("%H:%M:%S")
            }

            st.session_state.messages.append(welcome)

            st.session_state.is_ready = True # 준비완료 

            st.success("AI 챗봇 준비완료!")
            st.rerun()

        except Exception as e:
            st.error(f"시스템 준비중 문제가 발생했습니다 : {str(e)}")

# 메인 채팅 인터페이스 
if st.session_state.is_ready: 
    chat_box = st.container(height=600)

    # 채팅 메시지 표현
    with chat_box:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                st.caption(f"⏰️{msg['timestamp']}")

    # 사용자 질문 입력
    if question := st.chat_input("질문을 입력하세요...."):
    
        # 현재 시간
        time_now = datetime.now().strftime("%H:%M:%S")

        # 사용자 질문을 메시지 객체로
        user_msg = {"role":"user", "content":question, "timestamp":time_now}

        # 세션에 추가 
        st.session_state.messages.append(user_msg)

        # AI 답변
        with st.spinner("답변을 생성하는 중...."):

            # RAG 와 연결 필요
            answer = get_ai_answer(question)

        ai_time = datetime.now().strftime("%H:%M:%S")
        ai_msg = {"role":"assistant", "content": answer, "timestamp":ai_time}
        st.session_state.messages.append(ai_msg)

        # 새로운 메시지 화면 반영을 위한 새로고침
        st.rerun()