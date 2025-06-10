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
    page_title="Corporate AI Assistant",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Gemini 스타일 다크 배경 CSS
st.markdown("""
<style>
    /* 전체 앱 다크 배경 (사이드바 제외) */
    .main {
        background-color: #1a1a1a !important;
        color: #ffffff !important;
    }
    
    .main .block-container {
        background-color: #1a1a1a !important;
        padding-top: 2rem;
        padding-bottom: 120px !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: none !important;
    }
    
    /* 사이드바는 화이트 유지 */
    .css-1d391kg {
        background-color: #f8f9fa !important;
        border-right: 1px solid #e9ecef !important;
    }
    
    /* 메인 컨텐츠 헤더 스타일 */
    .main h1, .main h2, .main h3 {
        color: #ffffff !important;
        font-weight: 600;
    }
    
    /* 메인 영역 텍스트 */
    .main p, .main span, .main div {
        color: #ffffff !important;
    }
    
    /* 구분선 스타일 */
    .main hr {
        border-color: #404040 !important;
        margin: 1.5rem 0;
    }
    
    /* 확장 섹션 다크 스타일 */
    .main .streamlit-expanderHeader {
        background-color: #2a2a2a !important;
        color: #ffffff !important;
        border: 1px solid #404040 !important;
        border-radius: 6px !important;
    }
    
    .main .streamlit-expanderContent {
        background-color: #262626 !important;
        border: 1px solid #404040 !important;
        color: #ffffff !important;
        border-radius: 0 0 6px 6px !important;
    }
    
    /* 채팅 입력창 다크 스타일 */
    [data-testid="stChatInput"] {
        background-color: #1a1a1a !important;
        border-top: 1px solid #404040 !important;
    }
    
    [data-testid="stChatInput"] textarea {
        background-color: #2a2a2a !important;
        color: #ffffff !important;
        border: 2px solid #404040 !important;
        border-radius: 25px !important;
        padding: 12px 20px !important;
        font-size: 1rem !important;
    }
    
    [data-testid="stChatInput"] textarea:focus {
        border-color: #ffffff !important;
        box-shadow: 0 0 0 1px #ffffff !important;
        outline: none !important;
    }
    
    [data-testid="stChatInput"] textarea::placeholder {
        color: #888888 !important;
    }
    
    /* 채팅 입력창 위치 조정 */
    .stChatInput {
        position: fixed;
        bottom: 0;
        left: 320px;
        right: 0;
        background-color: #1a1a1a !important;
        border-top: 1px solid #404040 !important;
        padding: 1rem 2rem !important;
        z-index: 999;
    }
    
    /* 사이드바 축소시 */
    .css-1lcbmhc.e1fqkh3o0 + .main .stChatInput {
        left: 60px;
    }
    
    /* 모바일 대응 */
    @media (max-width: 768px) {
        .stChatInput {
            left: 0;
            padding: 1rem;
        }
        
        .main .block-container {
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }
    }
    
    /* 스피너 색상 */
    .main .stSpinner {
        color: #ffffff !important;
    }
    
    /* 에러/성공 메시지 스타일 */
    .main .stError {
        background-color: #4a1a1a !important;
        border: 1px solid #6a2a2a !important;
        color: #ffffff !important;
    }
    
    .main .stSuccess {
        background-color: #1a4a1a !important;
        border: 1px solid #2a6a2a !important;
        color: #ffffff !important;
    }
    
    .main .stInfo {
        background-color: #1a3a4a !important;
        border: 1px solid #2a4a6a !important;
        color: #ffffff !important;
    }
    
    .main .stWarning {
        background-color: #4a3a1a !important;
        border: 1px solid #6a5a2a !important;
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

# 제목 및 헤더
st.title("AHN'S AI Assistant")
st.markdown("**Enterprise Document Intelligence Platform**")
st.markdown("---")

# 세션 상태 초기화
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'documents' not in st.session_state:
    st.session_state.documents = []

if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None

if 'encoder' not in st.session_state:
    st.session_state.encoder = None

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
        st.error(f"PDF reading error: {e}")
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
        st.error(f"DOCX reading error: {e}")
        return ""

def extract_text_from_txt(file):
    """TXT에서 텍스트 추출"""
    try:
        return file.read().decode('utf-8')
    except Exception as e:
        st.error(f"TXT reading error: {e}")
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
    
    return [chunk for chunk in chunks if len(chunk.strip()) > 50]

def process_documents(files):
    """업로드된 문서들 처리"""
    documents = []
    
    for file in files:
        try:
            if file.type == "application/pdf":
                text = extract_text_from_pdf(file)
            elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                text = extract_text_from_docx(file)
            elif file.type == "text/plain":
                text = extract_text_from_txt(file)
            else:
                st.warning(f"Unsupported file format: {file.name}")
                continue
            
            if not text.strip():
                st.warning(f"No text extracted from: {file.name}")
                continue
            
            chunks = split_text_into_chunks(text, chunk_size=500)
            
            for i, chunk in enumerate(chunks):
                documents.append({
                    'id': f"{file.name}_{i}",
                    'text': chunk,
                    'filename': file.name,
                    'chunk_id': i
                })
        
        except Exception as e:
            st.error(f"Error processing {file.name}: {e}")
    
    return documents

@st.cache_resource
def load_sentence_transformer():
    """SentenceTransformer 모델 로드"""
    try:
        return SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        st.error(f"SentenceTransformer loading error: {e}")
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
        st.error(f"Embedding generation error: {e}")
        return None, None

def search_documents(query, documents, embeddings, encoder, n_results=3):
    """문서에서 관련 내용 검색"""
    try:
        if not documents or embeddings is None or encoder is None:
            return []
        
        query_embedding = encoder.encode([query])
        similarities = cosine_similarity(query_embedding, embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:n_results]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.1:
                results.append(documents[idx]['text'])
        
        return results
    
    except Exception as e:
        st.error(f"Document search error: {e}")
        return []

def generate_response(query, context_docs, api_key):
    """Gemini를 사용하여 응답 생성"""
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        if context_docs:
            context = "\n\n".join(context_docs)
            prompt = f"""
Based on the following document content, please answer the question professionally.

Document Content:
{context}

Question: {query}

Please follow these guidelines:
1. Answer accurately based on the document content
2. If information is not in the documents, state "The requested information is not available in the uploaded documents"
3. Use a professional and helpful tone
4. Include specific examples or details when possible
5. Respond in Korean if the question is in Korean
"""
        else:
            prompt = f"""
No uploaded documents found or no relevant information available.

Question: {query}

Please provide a general response based on your knowledge, but first mention that "No relevant information was found in the uploaded documents, so I'm providing a general response."
Respond in Korean if the question is in Korean.
"""
        
        response = model.generate_content(prompt)
        return response.text
    
    except Exception as e:
        return f"Error generating response: {e}"

# 사이드바 설정
with st.sidebar:
    st.header("Configuration")
    
    # API 키 입력
    api_key = st.text_input("Google Gemini API Key:", type="password", help="Enter your API key to enable AI features")
    
    if api_key:
        genai.configure(api_key=api_key)
        st.success("API Connected")
    else:
        st.warning("API Key Required")
    
    st.markdown("---")
    
    # 문서 관리 섹션
    st.header("Document Management")
    
    # 문서 업로드
    st.subheader("Upload Documents")
    uploaded_files = st.file_uploader(
        "Supported formats: PDF, DOCX, TXT",
        type=['pdf', 'docx', 'txt'],
        accept_multiple_files=True,
        help="Upload company documents for AI analysis"
    )
    
    # 문서 처리 버튼
    if uploaded_files:
        if st.button("Process Documents", type="primary", use_container_width=True):
            with st.spinner("Processing documents..."):
                # 문서 처리 로직
                documents = process_documents(uploaded_files)
                
                if documents:
                    st.session_state.documents = documents
                    
                    with st.spinner("Generating embeddings..."):
                        embeddings, encoder = create_embeddings(documents)
                        if embeddings is not None:
                            st.session_state.embeddings = embeddings
                            st.session_state.encoder = encoder
                            st.success(f"Processed {len(documents)} document chunks")
                        else:
                            st.error("Embedding generation failed")
                else:
                    st.warning("No processable documents found")
                
                st.rerun()
    
    st.markdown("---")
    
    # 문서 현황
    st.subheader("Document Status")
    if st.session_state.get('documents'):
        st.metric("Total Chunks", len(st.session_state.documents))
        
        # 파일별 청크 수 표시
        file_counts = {}
        for doc in st.session_state.documents:
            filename = doc['filename']
            file_counts[filename] = file_counts.get(filename, 0) + 1
        
        for filename, count in file_counts.items():
            st.text(f"{filename}: {count} chunks")
        
        # 검색 기능 상태
        if st.session_state.get('embeddings') is not None:
            st.success("Search: Active")
        else:
            st.warning("Search: Inactive")
    else:
        st.info("No documents loaded")
    
    st.markdown("---")
    
    # 관리 기능
    st.subheader("System Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    with col2:
        if st.button("Clear Docs", use_container_width=True):
            st.session_state.documents = []
            st.session_state.embeddings = None
            st.session_state.encoder = None
            st.rerun()

# 사용법 안내를 사이드바로 이동
with st.sidebar:
    st.markdown("---")
    st.subheader("System Information")
    
    with st.expander("Getting Started"):
        st.markdown("""
        1. Enter your Google Gemini API key above
        2. Upload documents using the file uploader
        3. Click "Process Documents" to enable AI search
        4. Ask questions about your documents in the chat
        """)
    
    with st.expander("Features"):
        st.markdown("""
        - PDF, DOCX, TXT file support
        - Multiple file upload capability
        - Vector-based document search
        - Professional AI responses
        - Source document references
        """)
    
    with st.expander("Tips"):
        st.markdown("""
        - Use specific questions for better results
        - Multiple files can be processed together
        - First document processing may take time
        - Referenced sources shown below responses
        """)

# 메인 채팅 인터페이스
st.header("AI Chat Interface")

# 채팅 컨테이너 (사이드바 너비만큼 여백 추가)
chat_container = st.container()

with chat_container:
    # 초기 환영 메시지 (채팅이 비어있을 때만 표시)
    if not st.session_state.messages:
        st.markdown("""
        <div style="
            display: flex;
            justify-content: center;
            align-items: center;
            height: 300px;
            flex-direction: column;
            margin-left: 0;
        ">
            <h1 style="
                font-size: 3rem;
                font-weight: 300;
                color: #ffffff;
                margin-bottom: 2rem;
                text-align: center;
            ">안녕하세요</h1>
            <p style="
                font-size: 1.2rem;
                color: #cccccc;
                text-align: center;
                margin-bottom: 3rem;
            ">AHN'S AI Assistant가 도와드리겠습니다</p>
        </div>
        """, unsafe_allow_html=True)

    # 채팅 메시지 표시 (커스텀 스타일)
    if st.session_state.messages:
        for i, message in enumerate(st.session_state.messages):
            if message["role"] == "user":
                # 사용자 메시지 - 우측 정렬
                st.markdown(f"""
                <div style="
                    display: flex;
                    justify-content: flex-end;
                    margin: 1rem 0;
                    padding-right: 1rem;
                ">
                    <div style="
                        background-color: #e3f2fd;
                        color: #1565c0;
                        padding: 0.8rem 1.2rem;
                        border-radius: 18px 18px 4px 18px;
                        max-width: 70%;
                        font-size: 0.95rem;
                        line-height: 1.4;
                        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
                        word-wrap: break-word;
                    ">
                        {message["content"]}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                # AI 응답 - 좌측 정렬
                # HTML 태그 제거하고 깔끔하게 표시
                clean_content = message["content"].replace('<div>', '').replace('</div>', '').strip()
                
                st.markdown(f"""
                <div style="
                    display: flex;
                    justify-content: flex-start;
                    margin: 1rem 0;
                    align-items: flex-start;
                    padding-left: 1rem;
                ">
                    <div style="
                        background-color: #2a2a2a;
                        color: #ffffff;
                        padding: 0.8rem 1.2rem;
                        border-radius: 18px 18px 18px 4px;
                        max-width: 75%;
                        font-size: 0.95rem;
                        line-height: 1.5;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.3);
                        border: 1px solid #404040;
                        word-wrap: break-word;
                        white-space: pre-wrap;
                    ">
                        {clean_content}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # 참고 문서가 있을 경우 표시
                if "references" in message:
                    with st.expander(f"📚 참고한 문서 ({len(message['references'])}개)"):
                        for j, doc in enumerate(message["references"]):
                            st.write(f"**문서 {j+1}:**")
                            st.write(doc[:200] + "..." if len(doc) > 200 else doc)
                            if j < len(message["references"]) - 1:
                                st.markdown("---")

# 커스텀 채팅 입력창
st.markdown("""
<style>
    /* 메인 컨텐츠 영역 사이드바 겹침 방지 */
    .main .block-container {
        padding-bottom: 120px !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }
    
    /* 사이드바가 있을 때 메인 콘텐츠 여백 조정 */
    .main {
        margin-left: 0 !important;
    }
    
    /* 채팅 컨테이너 스타일 */
    .chat-container {
        margin-left: 0;
        width: 100%;
        max-width: none;
    }
    
    /* 채팅 입력창 커스터마이징 */
    .stChatInput {
        position: fixed;
        bottom: 0;
        left: 320px; /* 사이드바 너비만큼 여백 */
        right: 0;
        background: white;
        border-top: 1px solid #e9ecef;
        padding: 1rem;
        z-index: 999;
        box-shadow: 0 -2px 10px rgba(0,0,0,0.1);
    }
    
    /* 사이드바가 축소된 경우 */
    .css-1lcbmhc.e1fqkh3o0 + .main .stChatInput {
        left: 60px;
    }
    
    [data-testid="stChatInput"] {
        margin-bottom: 0;
        max-width: calc(100vw - 360px); /* 사이드바 고려한 최대 너비 */
    }
    
    [data-testid="stChatInput"] textarea {
        border: 2px solid #e9ecef !important;
        border-radius: 25px !important;
        padding: 12px 20px !important;
        font-size: 1rem !important;
        resize: none !important;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1) !important;
        transition: all 0.2s ease !important;
        width: 100% !important;
    }
    
    [data-testid="stChatInput"] textarea:focus {
        border-color: #3498db !important;
        box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.1), 0 2px 10px rgba(0,0,0,0.15) !important;
        outline: none !important;
    }
    
    [data-testid="stChatInput"] textarea::placeholder {
        color: #7f8c8d !important;
        font-size: 1rem !important;
    }
    
    /* 채팅 메시지 영역 여백 */
    .element-container:has([data-testid="stChatInput"]) {
        margin-bottom: 80px;
    }
    
    /* 모바일 대응 */
    @media (max-width: 768px) {
        .stChatInput {
            left: 0;
        }
        [data-testid="stChatInput"] {
            max-width: 100vw;
        }
    }
</style>
""", unsafe_allow_html=True)

# 사용자 입력 (커스텀 placeholder)
if prompt := st.chat_input("AHN'S AI 에게 물어보기"):
    # 사용자 메시지 추가
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # AI 응답 생성
    if api_key:
        with st.spinner("답변을 생성하고 있습니다..."):
            # 문서 검색
            relevant_docs = search_documents(
                prompt, 
                st.session_state.documents, 
                st.session_state.embeddings, 
                st.session_state.encoder
            )
            
            # 응답 생성
            response = generate_response(prompt, relevant_docs, api_key)
            
            # 응답을 세션에 저장 (참고 문서 포함)
            message_data = {"role": "assistant", "content": response}
            if relevant_docs:
                message_data["references"] = relevant_docs
            
            st.session_state.messages.append(message_data)
            
        # 페이지 새로고침으로 UI 업데이트
        st.rerun()
    else:
        st.error("사이드바에서 API 키를 먼저 입력해주세요!")

# 사용법 안내
with st.expander("System Information"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Getting Started:**
        1. Enter your Google Gemini API key in the sidebar
        2. Upload documents using the file uploader
        3. Click "Process Documents" to enable AI search
        4. Ask questions about your documents in the chat
        """)
    
    with col2:
        st.markdown("""
        **Features:**
        - PDF, DOCX, TXT file support
        - Multiple file upload capability
        - Vector-based document search
        - Professional AI responses
        - Source document references
        """)

# 푸터
st.markdown("---")
st.markdown("**AHN'S AI Assistant** | Enterprise Document Intelligence Platform | Powered by Google Gemini")
