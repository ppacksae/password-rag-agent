# test_installation.py
print("🔧 RAG 환경 테스트 시작...")

try:
    # 1. Word 파일 처리 테스트
    from docx import Document
    print("✅ python-docx 설치 완료")
    
    # 2. 임베딩 모델 테스트
    from sentence_transformers import SentenceTransformer
    print("✅ sentence-transformers 설치 완료")
    
    # 3. 벡터DB 테스트
    import chromadb
    print("✅ chromadb 설치 완료")
    
    # 4. 로컬 LLM 테스트
    import ollama
    print("✅ ollama 설치 완료")
    
    print("\n🎉 모든 필수 라이브러리 설치 완료!")
    print("이제 RAG 시스템을 구축할 수 있습니다!")
    
except ImportError as e:
    print(f"❌ 설치 오류: {e}")
