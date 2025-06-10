# real_rag.py - 실제 파일을 읽는 RAG 시스템
import os
from sentence_transformers import SentenceTransformer
import chromadb
from docx import Document

class RealRAG:
    def __init__(self):
        print("🔧 실제 RAG 시스템 초기화 중...")
        
        # 임베딩 모델 로드
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ 임베딩 모델 로드 완료")
        
        # 벡터 데이터베이스 설정
        self.client = chromadb.PersistentClient(path="./real_chroma_db")
        self.collection = self.client.get_or_create_collection("company_info")
        print("✅ 벡터 데이터베이스 설정 완료")
    
    def load_text_file(self, file_path):
        """텍스트 파일을 읽어서 문단별로 분리"""
        print(f"📄 텍스트 파일 읽기: {file_path}")
        
        if not os.path.exists(file_path):
            print(f"❌ 파일을 찾을 수 없습니다: {file_path}")
            return []
        
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        lines = content.split('\n')
        texts = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            if line and len(line) > 5:
                texts.append({
                    'id': f'line_{i}',
                    'text': line
                })
        
        print(f"✅ {len(texts)}개의 정보를 읽었습니다")
        return texts
    
    def load_word_file(self, file_path):
        """Word 파일을 읽어서 문단별로 분리"""
        print(f"📄 Word 파일 읽기: {file_path}")
        
        if not os.path.exists(file_path):
            print(f"❌ 파일을 찾을 수 없습니다: {file_path}")
            return []
        
        doc = Document(file_path)
        texts = []
        
        for i, para in enumerate(doc.paragraphs):
            text = para.text.strip()
            if text and len(text) > 5:
                texts.append({
                    'id': f'para_{i}',
                    'text': text
                })
        
        print(f"✅ {len(texts)}개의 문단을 읽었습니다")
        return texts
    
    def add_documents(self, texts):
        """문서들을 벡터 데이터베이스에 추가"""
        print("🔄 정보를 벡터 데이터베이스에 추가 중...")
        
        try:
            self.client.delete_collection("company_info")
            self.collection = self.client.create_collection("company_info")
        except:
            pass
        
        documents = [item['text'] for item in texts]
        ids = [item['id'] for item in texts]
        
        self.collection.add(
            documents=documents,
            ids=ids
        )
        
        print(f"✅ {len(documents)}개 정보 추가 완료")
    
    def search(self, query, top_k=3):
        """질문에 대한 답변 검색"""
        print(f"🔍 검색 중: '{query}'")
        
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )
        
        if results['documents'][0]:
            found_docs = results['documents'][0]
            print(f"✅ {len(found_docs)}개의 관련 정보를 찾았습니다")
            return found_docs
        else:
            print("❌ 관련 정보를 찾지 못했습니다")
            return []
    
    def chat(self):
        """대화형 검색 시스템"""
        print("\n🤖 회사 정보 챗봇이 시작되었습니다!")
        print("💡 '종료' 또는 'quit'를 입력하면 종료됩니다.")
        print("-" * 50)
        
        while True:
            query = input("\n❓ 질문을 입력하세요: ").strip()
            
            if query.lower() in ['종료', 'quit', 'exit', '나가기']:
                print("👋 챗봇을 종료합니다!")
                break
            
            if not query:
                continue
            
            results = self.search(query)
            
            print("\n📋 검색 결과:")
            if results:
                for i, result in enumerate(results, 1):
                    print(f"{i}. {result}")
            else:
                print("관련 정보를 찾을 수 없습니다. 다른 키워드로 검색해보세요.")

def main():
    print("🚀 실제 RAG 시스템 시작!\n")
    
    rag = RealRAG()
    
    # 여기에 실제 파일명을 입력하세요
    file_path = "pstorm_pw.docx"  # 👈 실제 Word 파일명으로 변경
    
    # 파일 형식에 따라 다른 읽기 방식 사용
    if file_path.endswith('.docx'):
        texts = rag.load_word_file(file_path)
    else:
        texts = rag.load_text_file(file_path)
    
    if not texts:
        print("❌ 파일을 읽을 수 없습니다. 파일 경로를 확인해주세요.")
        return
    
    rag.add_documents(texts)
    
    print("\n🎯 빠른 테스트:")
    test_queries = ["와이파이 비번", "구글 계정", "프린터"]
    
    for query in test_queries:
        print(f"\n🔍 '{query}' 검색:")
        results = rag.search(query, top_k=1)
        if results:
            print(f"💡 답변: {results[0]}")
    
    rag.chat()

if __name__ == "__main__":
    main()