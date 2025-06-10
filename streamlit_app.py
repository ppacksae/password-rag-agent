# streamlit_app.py - 개선된 RAG 시스템
# 🤖 Retrieval-Augmented Generation (검색 증강 생성) 시스템
# 개선사항: 다크모드, 파일업로드, 정확한 답변, 깔끔한 UI

import streamlit as st
import os
from sentence_transformers import SentenceTransformer
import chromadb
from docx import Document
import tempfile

# =====================================================
# 🎨 다크모드 CSS 스타일
# =====================================================

def load_css():
    st.markdown("""
    <style>
    /* 전체 배경 다크모드 */
    .stApp {
        background-color: #0e1117 !important;
        color: #ffffff !important;
    }
    
    /* 사이드바 완전 다크모드 */
    .css-1d391kg, .css-1lcbmhc, .css-17lntkn, 
    .stSidebar > div, .stSidebar, 
    section[data-testid="stSidebar"] {
        background-color: #1a1a1a !important;
        color: #ffffff !important;
    }
    
    /* 사이드바 내부 요소들 */
    .stSidebar .stMarkdown, .stSidebar h3, .stSidebar p, 
    .stSidebar label, .stSidebar .stSelectbox label {
        color: #ffffff !important;
    }
    
    /* 파일 업로더 다크모드 */
    .stFileUploader > div {
        background-color: #262730 !important;
        border: 2px dashed #4a4a4a !important;
        border-radius: 10px !important;
    }
    
    .stFileUploader label {
        color: #ffffff !important;
    }
    
    /* 채팅 입력창 완전 다크모드 */
    .stChatInput > div {
        background-color: #262730 !important;
        border-radius: 25px !important;
        border: 1px solid #4a4a4a !important;
    }
    
    .stChatInput input {
        background-color: #262730 !important;
        color: #ffffff !important;
        border: none !important;
    }
    
    .stChatInput input::placeholder {
        color: #888888 !important;
    }
    
    /* 채팅 메시지 스타일 개선 */
    .stChatMessage {
        background-color: #1e1e1e !important;
        border-radius: 15px !important;
        margin: 15px 0 !important;
        padding: 20px !important;
        color: #ffffff !important;
    }
    
    /* 사용자 메시지 */
    .stChatMessage[data-testid="user-message"] {
        background-color: #2d4356 !important;
        margin-left: 15% !important;
        border-left: 4px solid #4a9eff !important;
    }
    
    /* AI 메시지 */
    .stChatMessage[data-testid="assistant-message"] {
        background-color: #1a1a1a !important;
        margin-right: 15% !important;
        border-left: 4px solid #00d4aa !important;
    }
    
    /* 메시지 내 텍스트 색상 */
    .stChatMessage p, .stChatMessage div, .stChatMessage span {
        color: #ffffff !important;
    }
    
    /* 버튼 스타일 */
    .stButton > button {
        background-color: #4a9eff !important;
        color: white !important;
        border: none !important;
        border-radius: 20px !important;
        padding: 12px 24px !important;
        transition: all 0.3s !important;
        font-weight: 600 !important;
    }
    
    .stButton > button:hover {
        background-color: #357abd !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(74, 158, 255, 0.3) !important;
    }
    
    /* 제목 스타일 */
    h1 {
        color: #4a9eff !important;
        text-align: center !important;
        font-weight: 700 !important;
        margin-bottom: 30px !important;
    }
    
    h3, h4 {
        color: #ffffff !important;
    }
    
    /* 모든 텍스트를 흰색으로 */
    .stMarkdown, .stText, p, div, span, label {
        color: #ffffff !important;
    }
    
    /* 카드 스타일 */
    .info-card {
        background-color: #1a1a1a !important;
        padding: 20px !important;
        border-radius: 15px !important;
        border-left: 4px solid #4a9eff !important;
        margin: 10px 0 !important;
        color: #ffffff !important;
    }
    
    /* 상태 메시지 스타일 */
    .stError {
        background-color: #3d1a1a !important;
        border: 1px solid #ff6b6b !important;
        color: #ffffff !important;
    }
    
    .stWarning {
        background-color: #3d3d1a !important;
        border: 1px solid #ffa500 !important;
        color: #ffffff !important;
    }
    
    .stSuccess {
        background-color: #1a3d1a !important;
        border: 1px solid #4caf50 !important;
        color: #ffffff !important;
    }
    
    .stInfo {
        background-color: #1a2d3d !important;
        border: 1px solid #4a9eff !important;
        color: #ffffff !important;
    }
    
    /* Spinner 스타일 */
    .stSpinner {
        color: #4a9eff !important;
    }
    
    /* 숨기기: Streamlit 기본 요소들 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* 스크롤바 다크모드 */
    ::-webkit-scrollbar {
        width: 8px;
        background-color: #1a1a1a;
    }
    
    ::-webkit-scrollbar-thumb {
        background-color: #4a4a4a;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background-color: #666666;
    }
    </style>
    """, unsafe_allow_html=True)

# =====================================================
# 🧠 개선된 RAG 시스템 클래스
# =====================================================

class ImprovedRAG:
    def __init__(self):
        if 'rag_initialized' not in st.session_state:
            st.session_state.rag_initialized = True
            
            with st.spinner('🧠 AI 시스템 초기화 중...'):
                st.session_state.model = SentenceTransformer('all-MiniLM-L6-v2')
                st.session_state.client = chromadb.PersistentClient(path="./improved_chroma_db")
                st.session_state.collection = st.session_state.client.get_or_create_collection("company_info")
            
            st.success('✅ AI 시스템 준비 완료!')
    
    def load_word_file(self, file_path):
        """Word 파일을 더 정확하게 파싱 - 모든 내용 포함"""
        if not os.path.exists(file_path):
            return []
        
        doc = Document(file_path)
        texts = []
        current_section = ""
        section_content = []
        
        for i, para in enumerate(doc.paragraphs):
            text = para.text.strip()
            
            if text and len(text) > 0:  # 모든 텍스트 포함
                # 섹션 제목 감지 (** 로 감싸진 제목 또는 숫자. 로 시작)
                is_section_title = (
                    text.startswith('**') and text.endswith('**') or
                    text.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.'))
                )
                
                if is_section_title:
                    # 이전 섹션의 완전한 내용 저장
                    if current_section and section_content:
                        complete_section = current_section + "\n" + "\n".join(section_content)
                        texts.append({
                            'id': f'complete_section_{len(texts)}',
                            'text': complete_section,
                            'type': 'complete_section'
                        })
                    
                    # 새 섹션 시작
                    current_section = text.replace('**', '').strip()  # ** 제거
                    section_content = []
                    
                    # 섹션 제목도 검색 가능하게 저장
                    texts.append({
                        'id': f'title_{i}',
                        'text': current_section,
                        'type': 'title'
                    })
                
                # 섹션 내용 (ID:, PW: 등 모든 내용)
                elif current_section:
                    section_content.append(text)
                    
                    # 개별 항목도 섹션과 함께 저장
                    combined_text = f"{current_section}\n{text}"
                    texts.append({
                        'id': f'item_{i}',
                        'text': combined_text,
                        'type': 'item'
                    })
                
                # 독립적인 내용
                else:
                    texts.append({
                        'id': f'standalone_{i}',
                        'text': text,
                        'type': 'standalone'
                    })
        
        # 마지막 섹션의 완전한 내용 저장
        if current_section and section_content:
            complete_section = current_section + "\n" + "\n".join(section_content)
            texts.append({
                'id': f'complete_final_section',
                'text': complete_section,
                'type': 'complete_section'
            })
        
        return texts
    
    def add_documents(self, texts):
        """문서 추가 with 메타데이터"""
        try:
            st.session_state.client.delete_collection("company_info")
            st.session_state.collection = st.session_state.client.create_collection("company_info")
        except:
            pass
        
        if not texts:
            return 0
        
        documents = [item['text'] for item in texts]
        ids = [item['id'] for item in texts]
        metadatas = [{'type': item.get('type', 'unknown')} for item in texts]
        
        st.session_state.collection.add(
            documents=documents,
            ids=ids,
            metadatas=metadatas
        )
        
        return len(documents)
    
    def search(self, query, top_k=10):
        """더 정확한 검색"""
        results = st.session_state.collection.query(
            query_texts=[query],
            n_results=top_k
        )
        
        if results['documents'][0]:
            # 결과와 메타데이터를 함께 반환
            return list(zip(
                results['documents'][0],
                results['metadatas'][0] if results['metadatas'][0] else [{}] * len(results['documents'][0])
            ))
        return []
    
    def generate_precise_answer(self, question, search_results):
        """더 정확하고 완전한 답변 생성 - 모든 정보 포함"""
        if not search_results:
            return "❌ 관련 정보를 찾을 수 없습니다."
        
        q = question.lower()
        
        # 완전한 섹션 정보를 우선적으로 찾기
        complete_sections = []
        other_results = []
        
        for doc, metadata in search_results:
            if metadata.get('type') == 'complete_section':
                complete_sections.append(doc)
            else:
                other_results.append(doc)
        
        # 가장 완전한 정보 선택
        best_result = complete_sections[0] if complete_sections else other_results[0]
        
        # Adobe 관련 질문
        if any(word in q for word in ['adobe', '어도비']):
            # Adobe 섹션에서 모든 계정 정보 추출
            lines = best_result.split('\n')
            account_lines = []
            
            for line in lines:
                line = line.strip()
                # ID, PW 관련 모든 라인 포함
                if any(keyword in line for keyword in ['1)', '2)', 'ID:', 'PW:', 'id:', 'pw:', '@', 'perfect']):
                    account_lines.append(line)
            
            # 완전한 계정 정보가 있으면 사용, 없으면 전체 섹션 사용
            if account_lines:
                formatted_info = '\n'.join(account_lines)
            else:
                # 전체 섹션에서 adobe 관련 부분만 추출
                formatted_info = best_result
            
            return f"""🎨 **Adobe 계정 정보**

{formatted_info}

💡 위 계정으로 Adobe 제품에 로그인하세요!"""
        
        # Gmail 관련 질문  
        elif any(word in q for word in ['gmail', '구글', 'google']):
            lines = best_result.split('\n')
            account_lines = []
            
            for line in lines:
                line = line.strip()
                if any(keyword in line.lower() for keyword in ['1)', '2)', 'id:', 'pw:', '@gmail', 'pstorm']):
                    account_lines.append(line)
            
            if account_lines:
                formatted_info = '\n'.join(account_lines)
            else:
                formatted_info = best_result
            
            return f"""📧 **Gmail 계정 정보**

{formatted_info}

💡 위 계정으로 Gmail에 로그인하세요!"""
        
        # 와이파이 관련 질문
        elif any(word in q for word in ['와이파이', 'wifi', '무선']):
            return f"""📶 **와이파이 정보**

{best_result}

💡 위 정보로 무선 네트워크에 연결하세요!"""
        
        # 기본 답변 - 가장 완전한 정보 제공
        else:
            return f"""📋 **요청하신 정보**

{best_result}"""

# =====================================================
# 🌐 메인 애플리케이션
# =====================================================

def main():
    st.set_page_config(
        page_title="🤖 AI 도우미",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # CSS 로드
    load_css()
    
    # 헤더
    st.markdown("<h1>🤖 퍼펙트스톰 정보 도우미</h1>", unsafe_allow_html=True)
    
    # RAG 시스템 초기화
    rag = ImprovedRAG()
    
    # 사이드바
    with st.sidebar:
        st.markdown("### ⚙️ 설정")
        
        st.markdown("### 📁 문서 업로드")
        
        # 파일 업로드 기능
        uploaded_file = st.file_uploader(
            "Word 문서를 업로드하세요",
            type=['docx'],
            help="회사 정보가 담긴 .docx 파일을 선택하세요"
        )
        
        if uploaded_file is not None:
            if st.button("📄 문서 처리하기", type="primary"):
                with st.spinner('📖 문서 분석 중...'):
                    # 임시 파일로 저장
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        temp_path = tmp_file.name
                    
                    try:
                        # 문서 처리
                        texts = rag.load_word_file(temp_path)
                        if texts:
                            doc_count = rag.add_documents(texts)
                            st.success(f'✅ {doc_count}개 정보 로드 완료!')
                            st.session_state.docs_loaded = True
                        else:
                            st.error('❌ 문서를 읽을 수 없습니다')
                    finally:
                        # 임시 파일 삭제
                        os.unlink(temp_path)
        
        # 기존 파일 로드 (개발용)
        st.markdown("---")
        st.markdown("### 🔧 개발자 옵션")
        if st.button("📄 기본 파일 로드"):
            file_path = "pstorm_pw.docx"
            if os.path.exists(file_path):
                with st.spinner('📖 문서 분석 중...'):
                    texts = rag.load_word_file(file_path)
                    if texts:
                        doc_count = rag.add_documents(texts)
                        st.success(f'✅ {doc_count}개 정보 로드 완료!')
                        st.session_state.docs_loaded = True
            else:
                st.error(f'❌ {file_path} 파일을 찾을 수 없습니다')
        
        # 상태 표시
        st.markdown("---")
        st.markdown("### 📊 상태")
        if st.session_state.get('docs_loaded', False):
            st.markdown("🟢 **문서 로드됨**")
        else:
            st.markdown("🔴 **문서 필요**")
    
    # 메인 채팅 영역
    st.markdown("### 💬 AI와 대화하기")
    
    # 채팅 히스토리 초기화
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # 채팅 기록 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # 사용자 입력
    if prompt := st.chat_input("무엇이든 물어보세요! (예: adobe 계정 정보)"):
        
        # 문서 로드 확인
        if not st.session_state.get('docs_loaded', False):
            st.error("❌ 먼저 문서를 업로드해주세요!")
            return
        
        # 사용자 메시지 추가
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # AI 응답
        with st.chat_message("assistant"):
            with st.spinner("🔍 검색 중..."):
                # 검색 및 답변 생성
                results = rag.search(prompt, top_k=5)
                response = rag.generate_precise_answer(prompt, results)
                
                # 답변 표시
                st.markdown(response)
        
        # 응답을 히스토리에 추가
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    # 하단 안내
    if not st.session_state.get('docs_loaded', False):
        st.markdown("---")
        st.markdown("""
        <div class="info-card">
        <h3>🚀 시작하기</h3>
        <p>1. 왼쪽 사이드바에서 Word 문서를 업로드하세요</p>
        <p>2. '문서 처리하기' 버튼을 클릭하세요</p>
        <p>3. 궁금한 것을 물어보세요!</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="info-card">
            <h4>💡 사용 예시</h4>
            <ul>
            <li>adobe 계정 정보</li>
            <li>gmail 비밀번호</li>
            <li>와이파이 정보</li>
            <li>개발팀 연락처</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="info-card">
            <h4>🎯 팁</h4>
            <ul>
            <li>구체적인 키워드 사용</li>
            <li>한글/영어 모두 지원</li>
            <li>정확한 답변 제공</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()