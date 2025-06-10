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
    page_title="AHN'S AI Assistant",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 깔끔한 화이트 모드 CSS 스타일
st.markdown("""
<style>
    /* 메인 컨테이너 */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* 사이드바 스타일 */
    .css-1d391kg {
        background-color: #f8f9fa;
        border-right: 1px solid #e9ecef;
    }
    
    /* 헤더 스타일 */
    h1 {
        color: #2c3e50;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    h2, h3 {
        color: #34495e;
        font-weight: 600;
    }
    
    /* 기업용 부제목 */
    .stMarkdown p {
        color: #7f8c8d;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* 버튼 스타일 */
    .stButton > button {
        background-color: #3498db;
        color: white;
        border: none;
        border-radius: 6px;
        font-weight: 500;
        padding: 0.6rem 1.2rem;
        transition: all 0.2s ease;
        box-shadow: 0 2px 4px rgba(52, 152, 219, 0.2);
    }
    
    .stButton > button:hover {
        background-color: #2980b9;
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(52, 152, 219, 0.3);
    }
    
    /* Primary 버튼 (Process Documents) */
    .stButton > button[kind="primary"] {
        background-color: #27ae60;
        box-shadow: 0 2px 4px rgba(39, 174, 96, 0.2);
    }
    
    .stButton > button[kind="primary"]:hover {
        background-color: #229954;
        box-shadow: 0 4px 8px rgba(39, 174, 96, 0.3);
    }
    
    /* 입력 필드 스타일 */
    .stTextInput > div > div > input {
        border: 2px solid #e9ecef;
        border-radius: 6px;
        padding: 0.6rem;
        transition: border-color 0.2s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #3498db;
        box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.1);
    }
    
    /* 파일 업로더 스타일 */
    .stFileUploader {
        border: 2px dashed #bdc3c7;
        border-radius: 8px;
        padding: 1.5rem;
        background-color: #f8f9fa;
        transition: all 0.2s ease;
    }
    
    .stFileUploader:hover {
        border-color: #3498db;
        background-color: #ecf0f1;
    }
    
    /* 메시지 스타일 */
    .stSuccess {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        border-radius: 6px;
    }
    
    .stInfo {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        border-radius: 6px;
    }
    
    .stWarning {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        border-radius: 6px;
    }
    
    .stError {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        border-radius: 6px;
    }
    
    /* 채팅 메시지 스타일 */
    .stChatMessage {
        border-radius: 8px;
        margin: 0.5rem 0;
        border: 1px solid #e9ecef;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    
    /* 확장 가능한 섹션 */
    .streamlit-expanderHeader {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 6px;
        font-weight: 500;
    }
    
    .streamlit-expanderContent {
        border: 1px solid #e9ecef;
        border-top: none;
        border-radius: 0 0 6px 6px;
        background-color: #ffffff;
    }
    
    /* 메트릭 컨테이너 */
    [data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #e9ecef;
        border-radius: 6px;
        padding: 1rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    
    /* 구분선 */
    hr {
        border-color: #e9ecef;
        margin: 1.5rem 0;
    }
    
    /* 채팅 입력창 */
    [data-testid="stChatInput"] textarea {
        border: 2px solid #e9ecef;
        border-radius: 6px;
        transition: border-color 0.2s ease;
    }
    
    [data-testid="stChatInput"] textarea:focus {
        border-color: #3498db;
        box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.1);
    }
    
    /* 전체적인 텍스트 색상 */
    .stMarkdown, p, span, div {
        color: #2c3e50;
    }
    
    /* 링크 색상 */
    a {
        color: #3498db;
        text-decoration: none;
    }
    
    a:hover {
        color: #2980b9;
        text-decoration: underline;
    }
    
    /* 코드 블록 */
    .stCode, code, pre {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 4px;
    }
    
    /* 푸터 스타일 */
    .footer {
        text-align: center;
        color: #7f8c8d;
        font-size: 0.9rem;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #e9ecef;
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

# 메인 채팅 인터페이스
st.header("AI Chat Interface")

# 채팅 메시지 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력
if prompt := st.chat_input("Enter your question..."):
    # 사용자 메시지 추가
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # AI 응답 생성
    if api_key:
        with st.chat_message("assistant"):
            with st.spinner("Generating response..."):
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
                
                # 참고 문서 정보 표시
                if relevant_docs:
                    with st.expander(f"Referenced Documents ({len(relevant_docs)} sources)"):
                        for i, doc in enumerate(relevant_docs):
                            st.markdown(f"**Source {i+1}:**")
                            st.text(doc[:200] + "..." if len(doc) > 200 else doc)
                            if i < len(relevant_docs) - 1:
                                st.markdown("---")
                
                # 응답을 세션에 저장
                st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        with st.chat_message("assistant"):
            st.error("Please enter your API key in the sidebar to enable AI features.")

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
st.markdown("**Corporate AI Assistant** | Enterprise Document Intelligence Platform | Powered by Google Gemini")
