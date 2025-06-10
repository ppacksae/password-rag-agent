pythonimport streamlit as st

st.title("🤖 AI 채팅 도우미")
st.write("OpenAI를 활용한 간단한 채팅봇입니다!")

# API 키 입력
api_key = st.text_input("OpenAI API 키를 입력하세요:", type="password")

if api_key:
    # 간단한 채팅 인터페이스
    user_input = st.text_input("질문을 입력하세요:")
    
    if user_input:
        st.write(f"**사용자:** {user_input}")
        st.write("**AI:** 곧 답변 기능이 추가될 예정입니다!")

st.info("💡 다음 단계에서 실제 OpenAI 연결을 추가하겠습니다!")
