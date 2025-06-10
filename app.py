import streamlit as st
import google.generativeai as genai
import PyPDF2
from docx import Document
import io
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# 페이지 설정
st.set_page_config(
    page_title="🏢 AHN's AI 도우미",
    page_icon="🤖",
    layout="wide"
)

# 제목
st.title("🏢 AHN's AI 도우미")
st.write("문서를 업로드하고 AI와 대화하며 정보를 찾아보세요!")

# 사이드바 설정
with st.sidebar:
    st.header("⚙️ 설정")
    
    # API 키 입력
    api_key = st.text_input("Google Gemini API 키:", type="password")
    
    if api_key:
        genai.configure(api_key=api_key)
        st.success("✅ API 키 설정 완료!")
    
    st.divider()
    
    # 문서 업로드
    st.header("📄 문서 업로드")
    uploaded_files = st.file_uploader(
        "PDF, DOCX, TXT 파일을 업로드하세요",
        type=['pdf', 'docx', 'txt'],
        accept_multiple_files=True
    )

# 세션 상태 초기화
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'documents' not in st.session_state:
    st.session_state.documents = []

if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None

if 'encoder' not in st.session_state:
    st.session_state.encoder = None

# 문서 처리 함수들
def extract_text_from_pdf(file):
    """PDF에서 텍스트 추출"""
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file.read()))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"PDF 읽기 오류: {e}")
        return ""

def extract_text_from_docx(file):
    """DOCX에서 텍스트 추출"""
    try:
        doc = Document(io.BytesIO(file.read()))
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        st.error(f"DOCX 읽기 오류: {e}")
        return ""

def extract_text_from_txt(file):
    """TXT에서 텍스트 추출"""
    try:
        return file.read().decode('utf-8')
    except Exception as e:
        st.error(f"TXT 읽기 오류: {e}")
        return ""

def split_text_into_chunks(text, chunk_size=500):
    """텍스트를 청크로 분할"""
    if not text.strip():
        return []
    
    sentences = text.replace('\n', ' ').split('. ')
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        sentence_length = len(sentence)
        
        if current_length + sentence_length > chunk_size and current_chunk:
            chunks.append('. '.join(current_chunk) + '.')
            current_chunk = [sentence]
            current_length = sentence_length
        else:
            current_chunk.append(sentence)
            current_length += sentence_length
    
    if current_chunk:
        chunks.append('. '.join(current_chunk) + '.')
    
    return [chunk for chunk in chunks if len(chunk.strip()) > 50]  # 너무 짧은 청크 제거

def process_documents(files):
    """업로드된 문서들 처리"""
    documents = []
    
    for file in files:
        try:
            # 파일 타입별 텍스트 추출
            if file.type == "application/pdf":
                text = extract_text_from_pdf(file)
            elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                text = extract_text_from_docx(file)
            elif file.type == "text/plain":
                text = extract_text_from_txt(file)
            else:
                st.warning(f"지원하지 않는 파일 형식: {file.name}")
                continue
            
            if not text.strip():
                st.warning(f"파일에서 텍스트를 추출할 수 없습니다: {file.name}")
                continue
            
            # 텍스트를 청크로 분할
            chunks = split_text_into_chunks(text, chunk_size=500)
            
            for i, chunk in enumerate(chunks):
                documents.append({
                    'id': f"{file.name}_{i}",
                    'text': chunk,
                    'filename': file.name,
                    'chunk_id': i
                })
        
        except Exception as e:
            st.error(f"파일 {file.name} 처리 중 오류: {e}")
    
    return documents

@st.cache_resource
def load_sentence_transformer():
    """SentenceTransformer 모델 로드 (캐싱)"""
    try:
        return SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        st.error(f"SentenceTransformer 로드 오류: {e}")
        return None

def create_embeddings(documents):
    """문서 임베딩 생성"""
    if not documents:
        return None
    
    try:
        encoder = load_sentence_transformer()
        if encoder is None:
            return None
        
        texts = [doc['text'] for doc in documents]
        embeddings = encoder.encode(texts)
        
        return embeddings, encoder
    
    except Exception as e:
        st.error(f"임베딩 생성 중 오류: {e}")
        return None, None

def search_documents(query, documents, embeddings, encoder, n_results=3):
    """문서에서 관련 내용 검색"""
    try:
        if not documents or embeddings is None or encoder is None:
            return []
        
        # 쿼리 임베딩
        query_embedding = encoder.encode([query])
        
        # 유사도 계산
        similarities = cosine_similarity(query_embedding, embeddings)[0]
        
        # 상위 결과 선택
        top_indices = np.argsort(similarities)[::-1][:n_results]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.1:  # 최소 유사도 임계값
                results.append(documents[idx]['text'])
        
        return results
    
    except Exception as e:
        st.error(f"문서 검색 중 오류: {e}")
        return []

def generate_response(query, context_docs, api_key):
    """Gemini를 사용하여 응답 생성"""
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # 컨텍스트와 함께 프롬프트 구성
        if context_docs:
            context = "\n\n".join(context_docs)
            prompt = f"""
다음 문서 내용을 기반으로 질문에 답변해주세요.

문서 내용:
{context}

질문: {query}

답변 시 다음 규칙을 따라주세요:
1. 문서 내용을 기반으로 정확하게 답변하세요
2. 문서에 없는 내용은 추측하지 말고 "문서에서 해당 정보를 찾을 수 없습니다"라고 말하세요
3. 한국어로 친근하고 도움이 되는 톤으로 답변하세요
4. 가능하면 구체적인 예시나 세부 정보를 포함하세요
"""
        else:
            prompt = f"""
업로드된 문서가 없거나 관련 정보를 찾을 수 없습니다.

질문: {query}

일반적인 지식을 바탕으로 도움이 될 만한 답변을 제공하되, 
"업로드된 문서에서 관련 정보를 찾을 수 없어 일반적인 답변을 드립니다"라고 먼저 언급해주세요.
"""
        
        response = model.generate_content(prompt)
        return response.text
    
    except Exception as e:
        return f"응답 생성 중 오류가 발생했습니다: {e}"

# 메인 인터페이스
col1, col2 = st.columns([2, 1])

with col1:
    st.header("💬 AI 채팅")
    
    # 채팅 메시지 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # 사용자 입력
    if prompt := st.chat_input("궁금한 것을 물어보세요..."):
        # 사용자 메시지 추가
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # AI 응답 생성
        if api_key:
            with st.chat_message("assistant"):
                with st.spinner("AI가 답변을 생성하고 있습니다..."):
                    # 문서 검색
                    relevant_docs = search_documents(
                        prompt, 
                        st.session_state.documents, 
                        st.session_state.embeddings, 
                        st.session_state.encoder
                    )
                    
                    # 응답 생성
                    response = generate_response(prompt, relevant_docs, api_key)
                    
                    st.markdown(response)
                    
                    # 찾은 문서 정보 표시
                    if relevant_docs:
                        with st.expander(f"📚 참고한 문서 ({len(relevant_docs)}개)"):
                            for i, doc in enumerate(relevant_docs):
                                st.write(f"**문서 {i+1}:**")
                                st.write(doc[:200] + "..." if len(doc) > 200 else doc)
                                st.divider()
                    
                    # 응답을 세션에 저장
                    st.session_state.messages.append({"role": "assistant", "content": response})
        else:
            with st.chat_message("assistant"):
                st.error("API 키를 먼저 입력해주세요!")

with col2:
    st.header("📊 문서 현황")
    
    # 문서 업로드 처리
    if uploaded_files:
        if st.button("📤 문서 처리하기", type="primary"):
            with st.spinner("문서를 처리하고 있습니다..."):
                # 문서 처리
                documents = process_documents(uploaded_files)
                
                if documents:
                    st.session_state.documents = documents
                    
                    # 임베딩 생성
                    with st.spinner("문서 임베딩을 생성하고 있습니다..."):
                        embeddings, encoder = create_embeddings(documents)
                        if embeddings is not None:
                            st.session_state.embeddings = embeddings
                            st.session_state.encoder = encoder
                            st.success(f"✅ {len(documents)}개의 문서 청크가 처리되었습니다!")
                        else:
                            st.error("❌ 임베딩 생성에 실패했습니다.")
                else:
                    st.warning("⚠️ 처리할 수 있는 문서가 없습니다.")
                
                st.rerun()
    
    # 현재 문서 상태 표시
    if st.session_state.documents:
        st.info(f"📁 처리된 문서: {len(st.session_state.documents)}개 청크")
        
        # 파일별 청크 수 표시
        file_counts = {}
        for doc in st.session_state.documents:
            filename = doc['filename']
            file_counts[filename] = file_counts.get(filename, 0) + 1
        
        for filename, count in file_counts.items():
            st.write(f"• {filename}: {count}개 청크")
        
        # 임베딩 상태
        if st.session_state.embeddings is not None:
            st.success("🔍 검색 기능 활성화됨")
        else:
            st.warning("⚠️ 검색 기능 비활성화")
    
    # 채팅 히스토리 관리
    st.header("🗑️ 관리")
    
    col_clear1, col_clear2 = st.columns(2)
    
    with col_clear1:
        if st.button("🔄 채팅 초기화"):
            st.session_state.messages = []
            st.rerun()
    
    with col_clear2:
        if st.button("📂 문서 초기화"):
            st.session_state.documents = []
            st.session_state.embeddings = None
            st.session_state.encoder = None
            st.rerun()

# 사용법 안내
with st.expander("📖 사용법 안내"):
    st.markdown("""
    ### 🚀 시작하기
    1. **사이드바**에서 Google Gemini API 키를 입력하세요
    2. **문서를 업로드**하고 "문서 처리하기" 버튼을 클릭하세요
    3. **채팅창**에서 문서 내용에 대해 질문하세요
    
    ### 💡 팁
    - PDF, DOCX, TXT 파일을 지원합니다
    - 여러 파일을 동시에 업로드할 수 있습니다
    - 구체적인 질문을 하면 더 정확한 답변을 받을 수 있습니다
    - 답변 하단의 "참고한 문서" 섹션에서 출처를 확인할 수 있습니다
    
    ### ⚠️ 주의사항
    - API 키는 안전하게 관리하세요
    - 업로드된 문서는 세션이 끝나면 삭제됩니다
    - 첫 번째 문서 처리 시 모델 다운로드로 시간이 걸릴 수 있습니다
    """)

# 푸터
st.divider()
st.markdown("**🤖 Google Gemini 기반 RAG 챗봇** | 문서 기반 지능형 질의응답 시스템")
