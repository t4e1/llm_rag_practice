import streamlit as st

# 실행 
# streamlit run [파일명]

# 제목
st.title("streamlit 기본문법 실습")

# 1. 텍스트 표시 
st.header("텍스트 표시")
st.write("__안녕하세요__")  # 마크다운 지원 
st.write("안녕하세요")  # 마크다운 지원 
st.text("__안녕하세요__")   # 고정된 폰트로 텍스트 출력

# 입력 (text_input : 사용자가 입력한 값을 변수에 담을 수 있음)
st.header("사용자 입력")
name = st.text_input("이름을 입력하세요: ")

# 조건문
if name:
    st.write(f"{name}님 안녕하세요!")

# 버튼 
if st.button("클릭해보세요!"):
    st.success("버튼이 클릭되었습니다")

# 선택
color = st.selectbox("좋아하는 색깔:", ["빨강", "파랑", "초록"])
st.write(f"선택한 색: {color}")

# 레이아웃
col1, col2 = st.columns(2)

with col1:
    st.write("왼쪽 컬럼")
    st.button("왼쪽 버튼")
with col2:
    st.write("오른쪽 컬럼")
    st.button("오른쪽 버튼")

with st.sidebar:
    st.write("**사이드바 영역**")