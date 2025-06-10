import streamlit as st

st.title("🎉 Hello Streamlit!")
st.write("성공적으로 배포되었습니다!")

name = st.text_input("이름을 입력하세요:")
if name:
    st.write(f"안녕하세요, {name}님!")
