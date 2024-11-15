import faiss  # vector database
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch

# 사전 훈련된 임베딩 모델을 로드합니다
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
embedding_model = AutoModel.from_pretrained(model_name)

# 텍스트를 임베딩 벡터로 변환하는 함수
def generate_embedding(text):
    tokenized_input = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        model_output = embedding_model(**tokenized_input).last_hidden_state.mean(dim=1)  # 평균 풀링
    return model_output.cpu().numpy()

# 샘플 문서의 리스트 (코퍼스)
documents = [
    "영업시간은 주 5일, 월요일부터 금요일까지 오전 9시부터 오후 6시까지입니다.",
    "우리 회사는 최신 AI 기술을 사용하여 고객 맞춤형 서비스를 제공합니다.",
    "회사는 서울시 강남구 테헤란로에 위치하고 있으며, 12층에 있습니다.",
    "고객 서비스 센터는 24시간 운영되며, 긴급 지원이 가능합니다.",
    "저희 CTO는 최재철이며, 20년 이상의 IT 경력을 보유하고 있습니다.",
    "RAG 시스템은 실시간 데이터를 분석하여 빠르게 결과를 제공합니다.",
    "회사의 비전은 혁신을 통해 세상을 변화시키는 것입니다.",
    "제품 문의는 고객 지원팀에 전화로 요청하거나 이메일로 보내주세요.",
    "저희 본사는 부산에 위치하고 있으며, 추가 지점은 전국에 걸쳐 있습니다.",
    "다음 주부터는 신규 프로젝트에 대한 사전 등록이 시작될 예정입니다."
]

# FAISS에 문서 임베딩을 저장할 인덱스 생성
embedding_dimension = 384  # 사용된 모델의 임베딩 차원
faiss_index = faiss.IndexFlatL2(embedding_dimension)  # FAISS 인덱스 생성

# 각 문서의 임베딩 벡터를 생성하고 FAISS 인덱스에 추가
document_embeddings = np.vstack([generate_embedding(doc) for doc in documents])
faiss_index.add(document_embeddings)

# 사용자가 제공한 쿼리
user_query = "귀사의 사업장의 위치는 어디입니까?"
query_embedding = generate_embedding(user_query)

# 유사도에 따라 상위 2개 문서 검색
top_k = 2
_, retrieved_indices = faiss_index.search(query_embedding, top_k)
retrieved_documents = [documents[idx] for idx in retrieved_indices[0]]

# 결과 출력
print("사용자의 질의:", user_query)
print("검색된 문서:", retrieved_documents)
