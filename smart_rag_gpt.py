# smart_rag_gpt.py - GPT 통합 RAG 시스템 (신버전)
import os
from sentence_transformers import SentenceTransformer
import chromadb
from docx import Document
from openai import OpenAI
from dotenv import load_dotenv

class SmartRAGWithGPT:
    def __init__(self):
        print("🧠 GPT 통합 RAG 시스템 초기화 중...")
        
        # 환경변수 로드
        load_dotenv()
        api_key = os.getenv('OPENAI_API_KEY')
        
        if not api_key:
            print("❌ OpenAI API 키가 설정되지 않았습니다!")
            print("💡 .env 파일에 OPENAI_API_KEY를 설정해주세요.")
            self.client = None
            return
        
        # OpenAI 클라이언트 초기화 (신버전)
        self.client = OpenAI(api_key=api_key)
        print("✅ OpenAI API 키 로드 완료")
        
        # 임베딩 모델 로드
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ 임베딩 모델 로드 완료")
        
        # 벡터 데이터베이스 설정
        self.chroma_client = chromadb.PersistentClient(path="./smart_chroma_db")
        self.collection = self.chroma_client.get_or_create_collection("smart_company_info")
        print("✅ 벡터 데이터베이스 설정 완료")
    
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
    
    def add_documents(self, texts):
        """문서들을 벡터 데이터베이스에 추가"""
        print("🔄 정보를 벡터 데이터베이스에 추가 중...")
        
        try:
            self.chroma_client.delete_collection("smart_company_info")
            self.collection = self.chroma_client.create_collection("smart_company_info")
        except:
            pass
        
        documents = [item['text'] for item in texts]
        ids = [item['id'] for item in texts]
        
        self.collection.add(
            documents=documents,
            ids=ids
        )
        
        print(f"✅ {len(documents)}개 정보 추가 완료")
    
    def search_documents(self, query, top_k=3):
        """문서 검색 (GPT 전 단계)"""
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )
        
        if results['documents'][0]:
            return results['documents'][0]
        return []
    
    def generate_answer_with_gpt(self, question, context_docs):
        """GPT를 사용해서 자연스러운 답변 생성"""
        
        if not self.client:
            return "OpenAI API가 설정되지 않았습니다."
        
        if not context_docs:
            return "죄송합니다. 관련 정보를 찾을 수 없습니다."
        
        # 컨텍스트 문서들을 하나의 문자열로 합치기
        context = "\n".join([f"- {doc}" for doc in context_docs])
        
        # GPT에게 보낼 프롬프트 구성
        prompt = f"""
당신은 회사의 IT 도우미입니다. 아래 회사 정보를 바탕으로 질문에 친절하고 정확하게 답변해주세요.

회사 정보:
{context}

질문: {question}

답변 규칙:
1. 위의 회사 정보만을 바탕으로 답변하세요
2. 정보가 없으면 "해당 정보가 없습니다"라고 말하세요
3. 친근하고 도움이 되는 톤으로 답변하세요
4. 구체적인 정보(비밀번호, IP주소 등)가 있으면 정확히 포함하세요

답변:"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"❌ GPT API 오류: {e}")
            return f"API 오류가 발생했습니다. 검색된 정보: {context_docs[0] if context_docs else '없음'}"
    
    def smart_search(self, query):
        """스마트 검색: RAG + GPT"""
        print(f"🔍 스마트 검색: '{query}'")
        
        # 1단계: 관련 문서 검색
        context_docs = self.search_documents(query, top_k=3)
        print(f"📋 {len(context_docs)}개의 관련 문서 발견")
        
        # 2단계: GPT로 자연스러운 답변 생성
        answer = self.generate_answer_with_gpt(query, context_docs)
        
        return answer, context_docs
    
    def chat(self):
        """대화형 스마트 검색 시스템"""
        print("\n🤖 GPT 통합 회사 정보 챗봇이 시작되었습니다!")
        print("💡 이제 자연스러운 대화가 가능합니다!")
        print("💡 '종료' 또는 'quit'를 입력하면 종료됩니다.")
        print("-" * 60)
        
        while True:
            query = input("\n❓ 무엇을 도와드릴까요? ").strip()
            
            if query.lower() in ['종료', 'quit', 'exit', '나가기']:
                print("👋 챗봇을 종료합니다!")
                break
            
            if not query:
                continue
            
            answer, context_docs = self.smart_search(query)
            
            print(f"\n🤖 답변:")
            print(answer)
            
            # 참고 문서도 표시 (옵션)
            if context_docs and len(context_docs) > 1:
                print(f"\n📚 참고 정보:")
                for i, doc in enumerate(context_docs[:2], 1):
                    print(f"  {i}. {doc}")

def main():
    print("🚀 GPT 통합 스마트 RAG 시스템 시작!\n")
    
    rag = SmartRAGWithGPT()
    
    if not rag.client:
        return
    
    # 실제 파일 경로 설정
    file_path = "pstorm_pw.docx"  # 👈 실제 Word 파일명으로 변경
    
    # 파일 읽기
    if file_path.endswith('.docx'):
        texts = rag.load_word_file(file_path)
    else:
        texts = rag.load_text_file(file_path)
    
    if not texts:
        print("❌ 파일을 읽을 수 없습니다. 파일 경로를 확인해주세요.")
        return
    
    rag.add_documents(texts)
    
    print("\n🎯 스마트 테스트:")
    test_queries = [
        "와이파이 비밀번호가 뭐야?",
        "구글 계정 정보 알려줘",
        "프린터가 안 되는데 어떻게 해?"
    ]
    
    for query in test_queries:
        print(f"\n🔍 질문: '{query}'")
        answer, _ = rag.smart_search(query)
        print(f"🤖 답변: {answer}")
    
    rag.chat()

if __name__ == "__main__":
    main()