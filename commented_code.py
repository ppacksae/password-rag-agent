# streamlit_app.py - RAG 시스템 완전 주석 버전
# 🤖 Retrieval-Augmented Generation (검색 증강 생성) 시스템
# 목적: 회사 문서를 기반으로 질문에 자동 답변하는 챗봇

# =====================================================
# 📦 필수 라이브러리 import
# =====================================================

import streamlit as st           # 웹 인터페이스 생성 (Flask보다 간단)
import os                       # 파일 시스템 접근 (파일 존재 확인 등)
from sentence_transformers import SentenceTransformer  # 🧠 텍스트→벡터 변환 AI
import chromadb                 # 🗄️ 벡터 데이터베이스 (빠른 유사도 검색)
from docx import Document       # 📄 Word 파일(.docx) 읽기

# =====================================================
# 🎨 웹페이지 기본 설정
# =====================================================

st.set_page_config(
    page_title="🤖 회사 정보 챗봇",    # 브라우저 탭에 표시될 제목
    page_icon="🔍",                  # 브라우저 탭 아이콘 (파비콘)
    layout="wide"                    # 화면을 넓게 사용 (기본값은 "centered")
)

# =====================================================
# 🧠 RAG 시스템 핵심 클래스
# =====================================================

class WebRAG:
    """
    RAG (Retrieval-Augmented Generation) 시스템
    
    작동 원리:
    1. 문서를 읽어서 작은 조각(chunk)으로 나눔
    2. 각 조각을 벡터(숫자 배열)로 변환해서 DB에 저장
    3. 질문이 오면 벡터로 변환해서 유사한 조각들 찾기
    4. 찾은 조각들을 바탕으로 예쁜 답변 만들기
    """
    
    def __init__(self):
        """
        🏗️ RAG 시스템 초기화
        
        주의: Streamlit은 코드가 바뀔 때마다 전체를 다시 실행하므로
        무거운 AI 모델을 매번 새로 로드하면 느려짐
        → session_state를 사용해서 한 번만 로드하도록 캐싱
        """
        
        # 🔍 이미 초기화했는지 확인 (중복 방지)
        if 'rag_initialized' not in st.session_state:
            st.session_state.rag_initialized = True  # 초기화 완료 표시
            
            # 💫 로딩 스피너 표시 (사용자 경험 개선)
            with st.spinner('🧠 AI 시스템 초기화 중...'):
                
                # 🤖 임베딩 모델 로드
                # 'all-MiniLM-L6-v2': 384차원, 빠른 속도, 적당한 정확도
                # 다운로드 크기: 약 90MB
                st.session_state.model = SentenceTransformer('all-MiniLM-L6-v2')
                
                # 🗃️ 벡터 데이터베이스 설정
                # PersistentClient: 컴퓨터를 껐다 켜도 데이터 유지
                # path="./web_chroma_db": 현재 폴더에 데이터베이스 파일 저장
                st.session_state.client = chromadb.PersistentClient(path="./web_chroma_db")
                
                # 📚 컬렉션 생성 (테이블 같은 개념)
                # get_or_create: 있으면 가져오고, 없으면 새로 만들기
                st.session_state.collection = st.session_state.client.get_or_create_collection("web_company_info")
                
                # 🚫 LLM 기능 비활성화 
                # 이유: GPT 같은 LLM은 비용이 들고 느릴 수 있음
                # 현재는 템플릿 기반 답변으로 충분
                st.session_state.llm_available = False
            
            # ✅ 초기화 완료 메시지
            st.success('✅ AI 시스템 (스마트 템플릿) 준비 완료!')
    
    def load_word_file(self, file_path):
        """
        📄 Word 파일을 읽어서 검색 가능한 형태로 변환
        
        Args:
            file_path (str): 읽을 Word 파일 경로
            
        Returns:
            list: 처리된 텍스트 조각들 [{'id': '...', 'text': '...'}, ...]
            
        핵심 아이디어:
        - 단순히 전체 문서를 하나로 저장하면 검색이 부정확
        - 의미 있는 단위로 나누되, 관련 정보는 함께 묶어야 함
        - 예: "회사구글계정" + "ID/PW 정보"를 하나로 묶기
        """
        
        # 🚨 파일 존재 여부 확인
        if not os.path.exists(file_path):
            return []  # 파일이 없으면 빈 리스트 반환
        
        # 📖 Word 문서 열기
        doc = Document(file_path)      # python-docx 라이브러리 사용
        texts = []                     # 처리된 텍스트들을 저장할 리스트
        current_section = ""           # 현재 섹션 제목 추적 변수
        
        # 📝 문서의 모든 문단(paragraph) 순회
        for i, para in enumerate(doc.paragraphs):
            text = para.text.strip()   # 앞뒤 공백/줄바꿈 제거
            
            # 🔍 의미 있는 텍스트만 처리 (너무 짧은 것 제외)
            if text and len(text) > 2:
                
                # 🏷️ 섹션 제목 감지
                # 조건: 50자 미만 + '-'나 'ID'로 시작하지 않음
                # 예: "회사구글계정", "와이파이 정보" 등
                if len(text) < 50 and not text.startswith('-') and not text.startswith('ID') and not text.startswith('PW'):
                    current_section = text  # 현재 섹션으로 설정
                    
                    # 🏷️ 섹션 제목 자체도 검색 가능하도록 저장
                    texts.append({
                        'id': f'section_{i}',
                        'text': text
                    })
                
                # 📄 섹션 내용 처리 (섹션이 있는 경우)
                elif current_section:
                    # 🔗 섹션 제목 + 내용을 결합한 버전
                    # 예: "회사구글계정\n-Id : pstorm2019@gmail.com"
                    combined_text = f"{current_section}\n{text}"
                    texts.append({
                        'id': f'para_{i}',
                        'text': combined_text  # 이게 검색에서 높은 점수를 받을 것
                    })
                    
                    # 📝 원본 내용도 별도 저장 (다양한 검색 패턴 지원)
                    texts.append({
                        'id': f'original_{i}',
                        'text': text
                    })
                
                # 📄 섹션이 없는 독립적인 내용
                else:
                    texts.append({
                        'id': f'para_{i}',
                        'text': text
                    })
        
        return texts
    
    def add_documents(self, texts):
        """
        📚 처리된 문서들을 벡터 데이터베이스에 추가
        
        Args:
            texts (list): load_word_file()에서 반환된 텍스트 리스트
            
        Returns:
            int: 추가된 문서 개수
            
        과정:
        1. 기존 데이터 삭제 (깔끔한 업데이트)
        2. 텍스트를 ChromaDB에 추가
        3. ChromaDB가 자동으로 벡터화 수행
        """
        
        try:
            # 🗑️ 기존 컬렉션 삭제 (새로운 데이터로 완전 교체)
            st.session_state.client.delete_collection("web_company_info")
            # 🆕 새 컬렉션 생성
            st.session_state.collection = st.session_state.client.create_collection("web_company_info")
        except:
            # 🤷‍♂️ 컬렉션이 없으면 에러 무시 (처음 실행할 때)
            pass
        
        # 📄 데이터 분리: 텍스트 내용과 ID를 각각 리스트로
        documents = [item['text'] for item in texts]  # 실제 텍스트 내용들
        ids = [item['id'] for item in texts]          # 각 텍스트의 고유 식별자
        
        # 🏗️ 벡터 데이터베이스에 추가
        # ChromaDB가 documents를 자동으로 임베딩(벡터화)해서 저장
        st.session_state.collection.add(
            documents=documents,  # 📝 텍스트들 (자동으로 벡터로 변환됨)
            ids=ids              # 🆔 각 텍스트의 고유 ID
        )
        
        return len(documents)  # 📊 추가된 문서 개수 반환
    
    def search(self, query, top_k=5):
        """
        🔍 질문과 유사한 문서 조각들을 찾기
        
        Args:
            query (str): 사용자 질문 ("회사구글계정은?")
            top_k (int): 반환할 최대 결과 개수 (기본 5개)
            
        Returns:
            list: 유사도 순으로 정렬된 문서 조각들
            
        벡터 검색 과정:
        1. 질문을 벡터로 변환: "회사구글계정" → [0.1, 0.5, -0.3, ...]
        2. DB의 모든 문서 벡터와 유사도 계산 (코사인 유사도)
        3. 가장 유사한 top_k개 문서 반환
        """
        
        # 🔍 ChromaDB에서 검색 수행
        results = st.session_state.collection.query(
            query_texts=[query],      # 🔍 검색할 질문 (리스트 형태로 전달)
            n_results=top_k          # 📊 반환할 결과 개수
        )
        
        # 📄 검색 결과가 있으면 문서들 반환
        if results['documents'][0]:
            return results['documents'][0]  # 첫 번째 쿼리의 결과 문서들
        return []                          # 🚫 결과 없으면 빈 리스트
    
    def generate_smart_answer(self, question, context_docs):
        """
        🎨 검색된 문서들을 바탕으로 예쁜 답변 생성
        
        Args:
            question (str): 사용자 질문
            context_docs (list): search()에서 반환된 관련 문서들
            
        Returns:
            str: 포맷팅된 최종 답변
            
        템플릿 전략:
        - 질문에서 핵심 키워드 추출 ('gmail', '와이파이' 등)
        - 키워드에 맞는 템플릿 선택
        - 검색된 문서 중 가장 관련성 높은 것 선택
        - 이모지와 구분선으로 예쁘게 포맷팅
        """
        
        # 🚨 검색 결과가 없으면 에러 메시지
        if not context_docs:
            return "❌ 관련 정보를 찾을 수 없습니다. 다른 키워드로 검색해보세요."
        
        # 📄 가장 관련성 높은 문서 선택 (첫 번째가 가장 유사)
        context = context_docs[0]
        q = question.lower()  # 🔤 대소문자 구분 없이 키워드 매칭
        
        # 🎯 Gmail/구글 관련 질문 감지
        if any(word in q for word in ['gmail', '구글', 'google', '지메일', '회사구글', '회사 구글']):
            
            # 🔍 여러 검색 결과 중에서 Gmail 관련 정보 우선 선택
            gmail_info = None
            for doc in context_docs:
                # 📧 Gmail 관련 키워드가 포함된 문서 찾기
                if any(keyword in doc.lower() for keyword in ['gmail', 'google', 'pstorm', '@gmail.com']):
                    gmail_info = doc
                    break  # 첫 번째로 찾은 것 사용
            
            # 📧 Gmail 정보가 있으면 그것을 우선 사용
            if gmail_info:
                context = gmail_info
            
            # 🎨 Gmail 전용 템플릿
            base_answer = f"""
👤 **구글 계정 정보**
━━━━━━━━━━━━━━━━━━━━
📧 **계정 정보**: 
{context}

💡 위 정보로 Gmail 및 구글 서비스에 로그인하시면 됩니다!
🔐 비밀번호는 안전하게 보관해주세요.
"""
        
        # 📶 와이파이 관련 질문
        elif any(word in q for word in ['와이파이', 'wifi', '무선', '인터넷']):
            base_answer = f"""
🔐 **와이파이 접속 정보**
━━━━━━━━━━━━━━━━━━━━
📡 **네트워크 정보**: 
{context}

💡 위 정보로 무선 네트워크에 연결하시면 됩니다!
📶 연결 후 인터넷 사용 가능합니다.
"""
        
        # 🔑 비밀번호 관련 질문
        elif any(word in q for word in ['비밀번호', '비번', 'password', '패스워드', 'pw']):
            base_answer = f"""
🔑 **비밀번호 정보**
━━━━━━━━━━━━━━━━━━━━
🛡️ **패스워드**: 
{context}

⚠️ 보안을 위해 타인과 공유하지 마세요!
🔒 정기적으로 변경하는 것을 권장합니다.
"""
        
        # 👤 계정/로그인 관련 질문
        elif any(word in q for word in ['아이디', 'id', '로그인', '계정']):
            base_answer = f"""
👤 **로그인 정보**
━━━━━━━━━━━━━━━━━━━━
🆔 **계정 정보**: 
{context}

💻 로그인 시 위 정보를 사용하시면 됩니다!
🔐 비밀번호와 함께 안전하게 보관하세요.
"""
        
        # 🛠️ 문제/오류 관련 질문
        elif any(word in q for word in ['문제', '오류', '안됨', '고장', '에러']):
            base_answer = f"""
🛠️ **문제 해결 정보**
━━━━━━━━━━━━━━━━━━━━
📋 **해결 방법**: 
{context}

📞 추가 도움이 필요하시면 IT팀에 연락해주세요!
"""
        
        # 🤷‍♂️ 기타 모든 질문 (기본 템플릿)
        else:
            base_answer = f"""
📋 **관련 정보**
━━━━━━━━━━━━━━━━━━━━
{context}

💡 더 구체적인 질문을 하시면 더 정확한 답변을 드릴 수 있습니다.
"""
        
        return base_answer

# =====================================================
# 🌐 웹 인터페이스 메인 함수
# =====================================================

def main():
    """
    🏠 Streamlit 웹앱의 메인 화면 구성
    
    구조:
    - 헤더 (제목, 설명)
    - 사이드바 (설정, 파일 로드)
    - 메인 영역 (채팅 인터페이스)
    - 하단 (사용법 안내)
    """
    
    # 🎨 페이지 헤더
    st.title("🤖 회사 정보 챗봇")
    st.markdown("---")  # 📏 구분선
    
    # 🎛️ 사이드바 설정 패널
    st.sidebar.header("⚙️ 설정")
    
    # 🧠 RAG 시스템 초기화
    rag = WebRAG()
    
    # 📁 파일 로드 섹션
    st.sidebar.subheader("📁 파일 설정")
    
    # 📄 파일 경로 설정 (실제 환경에서는 파일 업로드 기능 사용)
    file_path = "pstorm_pw.docx"  # 👈 실제 파일명으로 변경 필요
    
    # 📤 파일 로드 버튼
    if st.sidebar.button("📄 파일 로드", type="primary"):
        # 💫 로딩 스피너 표시
        with st.spinner(f'📖 {file_path} 파일 읽는 중...'):
            # 📖 Word 파일 읽기 및 처리
            texts = rag.load_word_file(file_path)
            
            if texts:
                # 🏗️ 벡터 DB에 추가
                doc_count = rag.add_documents(texts)
                st.sidebar.success(f'✅ {doc_count}개 정보 로드 완료!')
                st.session_state.docs_loaded = True  # 로드 완료 상태 저장
            else:
                # 🚨 파일 없음 에러
                st.sidebar.error(f'❌ 파일을 찾을 수 없습니다: {file_path}')
                st.session_state.docs_loaded = False
    
    # 📊 파일 로드 상태 표시
    if 'docs_loaded' in st.session_state and st.session_state.docs_loaded:
        st.sidebar.info("📚 문서가 로드되었습니다!")
    else:
        st.sidebar.warning("⚠️ 먼저 파일을 로드해주세요")
    
    # 🤖 AI 모델 상태 표시
    st.sidebar.markdown("---")
    st.sidebar.subheader("🤖 AI 상태")
    
    if st.session_state.get('llm_available', False):
        st.sidebar.success("🧠 AI 자연어 처리 활성화")
    else:
        st.sidebar.info("📋 스마트 템플릿 모드")
    
    # 💬 메인 채팅 영역
    st.header("💬 질문하기")
    
    # 📝 채팅 히스토리 초기화
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # 📜 이전 채팅 기록 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # 💬 사용자 입력 받기
    if prompt := st.chat_input("궁금한 것을 물어보세요! (예: 회사구글계정?)"):
        
        # 🚨 문서 로드 여부 확인
        if 'docs_loaded' not in st.session_state or not st.session_state.docs_loaded:
            st.error("❌ 먼저 사이드바에서 파일을 로드해주세요!")
            return  # 함수 종료
        
        # 👤 사용자 메시지 추가 및 표시
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # 🤖 AI 응답 생성
        with st.chat_message("assistant"):
            with st.spinner("🔍 검색 및 답변 생성 중..."):
                # 🔍 관련 문서 검색 (5개까지)
                results = rag.search(prompt, top_k=5)
                
                # 🎨 스마트 답변 생성
                response = rag.generate_smart_answer(prompt, results)
                
                # 📄 답변 표시
                st.markdown(response)
                
                # 📊 검색 결과 수 표시
                if results:
                    st.info(f"🎯 {len(results)}개의 관련 정보를 찾았습니다")
                    
                    # 🔍 디버깅용: 검색된 원본 정보 표시 (접을 수 있음)
                    with st.expander("🔍 검색된 원본 정보"):
                        for i, result in enumerate(results, 1):
                            st.write(f"**{i}.** {result}")
        
        # 💾 AI 응답을 채팅 기록에 추가
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    # 📄 하단 정보 및 사용법
    st.markdown("---")
    
    # 📊 2열 레이아웃
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 사용 예시")
        st.markdown("""
        - **회사구글계정** 또는 **gmail**
        - **와이파이 비밀번호**
        - **프린터 설정 방법**
        - **회사 시스템 접속**
        """)
    
    with col2:
        st.subheader("💡 팁")
        st.markdown("""
        - 구체적인 키워드 사용
        - 한글/영어 모두 가능
        - 여러 질문 연속 가능
        - **"🔍 검색된 원본 정보"**로 결과 확인
        """)

# =====================================================
# 🚀 프로그램 시작점
# =====================================================

if __name__ == "__main__":
    main()  # 메인 함수 실행