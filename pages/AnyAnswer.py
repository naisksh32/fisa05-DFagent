import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
import re

from groq import Groq

load_dotenv()
api_key = os.getenv('KEY2')

# Groq 클라이언트 생성
client = Groq(api_key=api_key)

# 데이터베이스 불러오기(한번 불러오고 저장, 시간관리 측면에서 good!)
@st.cache_data
def load_data():
    df = pd.read_excel('AdidasSalesdata.xlsx')
    df=df.drop('Retailer ID', axis=1)
    return df

# 데이터 로드
df = load_data()

# ???
def table_definition_prompt():
    cols = ", ".join(df.columns.astype(str))
    return f"DataFrame df with columns: {cols}\n"

st.title("Pandas Chatbot (Gemma2 모델)")

# 이전 메시지 표시
# Streamlit은 기본적으로 매 요청마다 새로 실행되지만, st.session_state를 사용하면 사용자 세션 동안 유지되는 값을 저장할 수 있습니다.
# history라는 key가 세션 상태에 없다면 빈 리스트로 초기화합니다.
# 이 리스트는 사용자와 어시스턴트 간의 대화(메시지)를 차곡차곡 쌓아 나갑니다.
if "history" not in st.session_state:
    st.session_state.history = []

# 기록에 있는 메시지를 바탕으로 반복
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 사용자 입력
prompt = st.chat_input("질문을 입력하세요:")

# 만약 응답이 있으면
if prompt:
    # 세션 기록에 저장
    st.session_state.history.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.status("Gemma2 응답 생성 중..."):
        messages = [
            {"role": "system", "content": "너는 마케팅 전문가야. 이 데이터를 바탕으로 올바른 결과를 알려줘"},
            {"role": "user", "content": table_definition_prompt() + prompt}
        ] + st.session_state.history

        # 스트리밍 생성
        response_stream = client.chat.completions.create(
            model="gemma2-9b-it",
            messages=messages,
            max_tokens=300,
            temperature=0.7,
            stream=True,
            stop=None
        )

        # 응답 스트리밍 출력 및 저장
        full_response = ""
        with st.chat_message("assistant"):
            for chunk in response_stream:
                content_piece = chunk.choices[0].delta.content or ""
                full_response += content_piece
                st.write_stream(iter([content_piece]))  # 부분 출력

    # 대화 이력 저장
    st.session_state.history.append({"role": "assistant", "content": full_response})
    st.code(full_response)

