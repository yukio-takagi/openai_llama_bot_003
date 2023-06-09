### library
import streamlit as st
import openai
from llama_index import GPTSimpleVectorIndex, SimpleDirectoryReader, ServiceContext, PromptHelper
from llama_index.llm_predictor.chatgpt import ChatGPTLLMPredictor
import os

### API Key
# Streamlit Community Cloudの「Secrets」からOpenAI API keyを取得
os.environ["OPENAI_API_KEY"] = st.secrets.OpenAIAPI.openai_api_key # secrets に後ほどAPI Keyを保存する
openai.api_key = st.secrets.OpenAIAPI.openai_api_key # secrets に後ほどAPI Keyを保存する
openai_api_key = st.secrets.OpenAIAPI.openai_api_key

### 変数
# define prompt helper
# set maximum input size
max_input_size = 4096
# set number of output tokens
num_output = 256
# set maximum chunk overlap
max_chunk_overlap = 20
prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)
service_context = ServiceContext.from_defaults(llm_predictor=ChatGPTLLMPredictor(), prompt_helper=prompt_helper)

### 作成済みインデックスの読み込み
index = GPTSimpleVectorIndex.load_from_disk("index_01-05.json", service_context = service_context)

# st.session_state
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "system", "content": "あなたはコロナ感染症対策のとても優秀な担当者です"}  
        ]

# チャットボットとやりとりする関数
def communicate():
    messages = st.session_state["messages"]
    user_message = {"role": "user", "content": st.session_state["user_input"]}
    messages.append(user_message)

    question =  "日本語で答えてください。 "  + st.session_state["user_input"]
    res = index.query(question)

    answer_message = {"role": "answer", "content": res.response}
    messages.append(answer_message)

    text_message = {"role": "text", "content": res.source_nodes[0].node.text}
    messages.append(text_message)

    score_message = {"role": "score", "content": str(round(res.source_nodes[0].score, 3))}
    messages.append(score_message)

#    st.write("質問：" + st.session_state["user_input"])
#    st.write("====================================================================================")
#    st.write("回答： " + res.response)
#    st.write("====================================================================================")
#    st.write("関連FAQ： " + res.source_nodes[0].node.text)
#    st.write("====================================================================================")
#    st.write("スコア： " + str(round(res.source_nodes[0].score, 3)))

    st.session_state["user_input"] = ""  # 入力欄を消去

# ユーザーインターフェイスの構築
st.title("コロナ感染症対策")
st.write("ChatGPT APIを使ったコロナ感染症対策のFAQボットです。")

user_input = st.text_input("メッセージを入力してください。", key="user_input", on_change=communicate) # 入力値はsession_stateにkey引数で指定した名前で保管

if st.session_state["messages"]:
    messages = st.session_state["messages"]

    for message in messages[1:]:  # 
        if message["role"]=="user":
            st.write("質問: " + message["content"])
        if message["role"]=="answer":
            st.write("回答: " + message["content"])
        if message["role"]=="text":
            st.write("関連文書: " + message["content"])
        if message["role"]=="score":
            st.write("スコア: " + message["content"])
        st.write("====================================================================================")

