# 코드 최상단에 추가
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import random
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
import numpy as np
from datetime import datetime
from openai import OpenAI  # 최신 OpenAI 패키지 임포트 방식으로 변경
import os

# 페이지 설정
st.set_page_config(page_title="광진구 착한가게 소개 챗봇", page_icon="🏪")

# 사이드바에 API 키 입력 필드 추가
with st.sidebar:
    st.header("ChatGPT API 설정")
    api_key = st.text_input("OpenAI API 키를 입력하세요", type="password")
    
    # API 키 저장
    if api_key:
        st.session_state.openai_api_key = api_key
        # 클라이언트 초기화는 실제 API 호출 시 수행
        st.success("API 키가 설정되었습니다!")
    else:
        st.warning("ChatGPT를 사용하기 위해 API 키를 입력하세요")

# 페이지 제목
st.title("🏪 광진구 착한가게 소개 챗봇")
st.write("광진구의 다양한 착한가게에 대한 정보를 물어보세요.")

# 임베딩 모델 설정 (세션 상태에 저장하여 재로딩 방지)
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')  # 다국어 지원 모델 사용

embedding_model = load_embedding_model()

# Chroma DB 클라이언트 설정
@st.cache_resource
def get_chroma_client():
    # 메모리에 저장하는 클라이언트 생성
    client = chromadb.Client()
    
    # 사용자 정의 임베딩 함수 설정
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )
    
    # 컬렉션 생성 또는 가져오기
    try:
        collection = client.get_collection(name="gwangjin_shops", embedding_function=embedding_function)
    except:
        collection = client.create_collection(name="gwangjin_shops", embedding_function=embedding_function)
        
        # 광진구 착한가게 데이터 (실제로는 공공데이터 등으로 수집)
        shops_data = [
            "착한식당 '맛있는 한끼'는 중곡동에 위치한 한식당으로, 지역 농산물을 활용한 건강한 식단을 합리적인 가격에 제공합니다. 특히 어르신들에게 10% 할인 혜택을 제공합니다.",
            "광진구 구의동의 '친환경 마트'는 지역 생산 제품과 친환경 제품을 판매하며, 매달 수익의 5%를 지역 취약계층에 기부하고 있습니다.",
            "화양동 '따뜻한 빵집'은 매일 신선한 빵을 구워 판매하며, 폐업시간에 남은 빵을 지역 아동센터에 기부하는 활동을 하고 있습니다.",
            "건대입구역 근처의 '착한 문구점'은 학생들에게 10% 할인을 제공하며, 학기 초에는 저소득층 학생들에게 무료로 학용품을 지원합니다.",
            "자양동의 '마을 세탁소'는 독거노인과 장애인 가정의 세탁물을 무료로 수거하여 세탁 서비스를 제공하는 착한가게입니다."
        ]
        
        # 데이터 추가
        for i, text in enumerate(shops_data):
            collection.add(
                documents=[text],
                metadatas=[{"source": f"shop_{i}"}],
                ids=[f"doc_{i}"]
            )
    
    return collection

# 벡터 데이터베이스 컬렉션 가져오기
collection = get_chroma_client()

# 벡터 유사도 검색 함수
def search_shops_data(query, n_results=3):
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    
    if results and results['documents'] and results['documents'][0]:
        return results['documents'][0]
    else:
        return ["관련 데이터를 찾을 수 없습니다."]

# ChatGPT를 이용한 응답 생성 함수 (최신 OpenAI API 사용)
def generate_chatgpt_response(query, context):
    if not context or context[0] == "관련 데이터를 찾을 수 없습니다.":
        return "죄송합니다, 질문에 관련된 정보를 찾을 수 없습니다."
    
    if not hasattr(st.session_state, 'openai_api_key') or not st.session_state.openai_api_key:
        return f"광진구 착한가게 정보: {' '.join(context)}\n\n(ChatGPT API 키를 입력하면 더 자연스러운 응답을 받을 수 있습니다.)"
    
    try:
        # OpenAI 클라이언트 초기화
        client = OpenAI(api_key=st.session_state.openai_api_key)
        
        # 프롬프트 구성
        prompt = f"""
        다음은 광진구 착한가게에 대한 정보입니다:
        
        {' '.join(context)}
        
        위 정보를 바탕으로 다음 질문에 친절하고 자세하게 답변해주세요:
        
        질문: {query}
        """
        
        # ChatGPT API 호출 (최신 방식)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "당신은 광진구 착한가게에 대한 정보를 제공하는 도우미입니다. 주어진 정보만을 기반으로 답변해주세요."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )
        
        # 응답 추출 (최신 API 형식)
        return response.choices[0].message.content
    
    except Exception as e:
        # API 오류 발생 시 기본 응답 제공
        return f"API 오류가 발생했습니다: {str(e)}\n\n기본 정보: {' '.join(context)}"

# 챗봇 응답 생성 함수
def chat_response(question):
    # 관련 데이터 검색
    relevant_data = search_shops_data(question)
    
    # ChatGPT API를 이용한 응답 생성
    return generate_chatgpt_response(question, relevant_data)

# 세션 상태 초기화
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 이전 대화 내용 표시
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력 받기
if prompt := st.chat_input("질문을 입력하세요 (예: 광진구에 어떤 착한가게가 있나요?)"):
    # 사용자 메시지 표시
    with st.chat_message("user"):
        st.markdown(prompt)

    # 사용자 메시지 저장
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    # 응답 생성
    with st.spinner("답변 생성 중..."):
        response = chat_response(prompt)

    # 응답 메시지 표시
    with st.chat_message("assistant"):
        st.markdown(response)

    # 응답 메시지 저장
    st.session_state.chat_history.append({"role": "assistant", "content": response})

# 예시 질문
st.sidebar.header("예시 질문")
example_questions = [
    "광진구에 어떤 착한가게들이 있나요?",
    "착한가게들이 제공하는 할인 혜택은 무엇인가요?",
    "착한가게들은 어떤 사회공헌 활동을 하고 있나요?",
    "중곡동 근처에 있는 착한가게를 알려주세요."
]

for question in example_questions:
    if st.sidebar.button(question):
        # 사용자 메시지 표시 및 저장
        with st.chat_message("user"):
            st.markdown(question)
        st.session_state.chat_history.append({"role": "user", "content": question})

        # 응답 생성
        with st.spinner("답변 생성 중..."):
            response = chat_response(question)

        # 응답 메시지 표시 및 저장
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.chat_history.append({"role": "assistant", "content": response})

        # 페이지 새로고침
        st.rerun()

# 사이드바 설정 (API 키 입력 이후의 나머지 부분)
with st.sidebar:
    st.header("데이터 관리")
    
    # 데이터 추가 섹션
    with st.expander("새 착한가게 데이터 추가"):
        new_data = st.text_area("새로운 착한가게 정보를 입력하세요")
        if st.button("데이터 추가"):
            if new_data:
                # 새 ID 생성
                new_id = f"doc_{int(random.random() * 10000)}"
                
                # 데이터 추가
                collection.add(
                    documents=[new_data],
                    metadatas=[{"source": "user_input", "date_added": str(datetime.now())}],
                    ids=[new_id]
                )
                st.success("착한가게 정보가 추가되었습니다!")
            else:
                st.error("정보를 입력해주세요.")
    
    # 대화 기록 초기화 버튼
    if st.button("대화 기록 초기화"):
        st.session_state.chat_history = []
        st.rerun()

    # 데이터 확인 섹션
    with st.expander("착한가게 데이터 확인"):
        # 모든 데이터 가져오기
        all_data = collection.get()
        if all_data and 'documents' in all_data:
            st.write("현재 분석에 사용 중인 착한가게 정보:")
            for idx, data in enumerate(all_data['documents']):
                st.write(f"{idx+1}. {data}")
        else:
            st.write("데이터가 없습니다.")
