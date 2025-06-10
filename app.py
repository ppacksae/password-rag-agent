import streamlit as st
import google.generativeai as genai

st.title("🤖 Gemini AI 채팅 도우미")
st.write("Google Gemini를 활용한 AI 채팅봇입니다!")

# API 키 입력
api_key = st.text_input("Google Gemini API 키를 입력하세요:", type="password")

if api_key:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-pro')
    
    # 채팅 인터페이스
    user_input = st.text_input("질문을 입력하세요:")
    
    if user_input:
        st.write(f"**사용자:** {user_input}")
        
        try:
            with st.spinner('AI가 생각 중...'):
                response = model.generate_content(user_input)
                st.write(f"**Gemini:** {response.text}")
        except Exception as e:
            st.error(f"오류가 발생했습니다: {e}")

st.info("💡 Google AI Studio에서 무료 API 키를 발급받으세요!")
