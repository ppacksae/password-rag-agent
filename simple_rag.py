# simple_rag.py - 간단한 RAG 시스템
import os
from docx import Document
from sentence_transformers import SentenceTransformer
import chromadb

class SimpleRAG:
    def __init__(self):
        print("🔧 RAG 시스템 초기화 중...")
        
        # 1. 임베딩 모델 로드
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ 임베딩 모델 로드 완료")
        
        # 2. 벡터 데이터베이스 설정
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.client.get_or_create_collection("passwords")
        print("✅ 벡터 데이터베이스 설정 완료")
    
    def load_word_file(self, file_path):
        """Word 파일을 읽어서 텍스트 추출"""
        print(f"📄 Word 파일 읽기: {file_path}")
        
        if not os.path.exists(file_path):
            print(f"❌ 파일을 찾을 수 없습니다: {file_path}")
            return []
        
        doc = Document(file_path)
        texts = []
        
        for i, para in enumerate(doc.paragraphs):
            if para.text.strip():  # 빈 줄 제외
                texts.append({
                    'id': f'para_{i}',
                    'text': para.text.strip()
                })
        
        print(f"✅ {len(texts)}개의 문단을 읽었습니다")
        return texts
    
    def add_documents(self, texts):
        """문서들을 벡터 데이터베이스에 추가"""
        print("🔄 문서를 벡터 데이터베이스에 추가 중...")
        
        documents = [item['text'] for item in texts]
        ids = [item['id'] for item in texts]
        
        # 임베딩 생성 및 저장
        self.collection.add(
            documents=documents,
            ids=ids
        )
        
        print(f"✅ {len(documents)}개 문서 추가 완료")
    
    def search(self, query, top_k=3):
        """질문에 대한 답변 검색"""
        print(f"🔍 검색 중: '{query}'")
        
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )
        
        if results['documents'][0]:
            print(f"✅ {len(results['documents'][0])}개의 관련 문서를 찾았습니다")
            return results['documents'][0]
        else:
            print("❌ 관련 문서를 찾지 못했습니다")
            return []

# 테스트 함수
def test_rag():
    print("🚀 RAG 시스템 테스트 시작!\n")
    
    # RAG 시스템 생성
    rag = SimpleRAG()
    
    # 예제 데이터로 테스트 (Word 파일이 없을 때)
    sample_data = [
        {'id': 'wifi_1', 'text': '사내 와이파이: CompanyWiFi, 비밀번호: Company2024!'},
        {'id': 'google_1', 'text': '구글 드라이브 계정: company@gmail.com, 비밀번호: Google123!'},
        {'id': 'printer_1', 'text': '프린터 설정: IP 192.168.1.100, 관리자 비번: printer2024'},
    ]
    
    print("\n📝 예제 데이터 추가 중...")
    rag.add_documents(sample_data)
    
    print("\n🔍 검색 테스트:")
    print("-" * 50)
    
    # 테스트 질문들
    queries = [
        "와이파이 비밀번호",
        "구글 계정",
        "프린터 설정"
    ]
    
    for query in queries:
        results = rag.search(query)
        print(f"\n질문: {query}")
        print(f"답변: {results[0] if results else '답변을 찾을 수 없습니다'}")

if __name__ == "__main__":
    test_rag()