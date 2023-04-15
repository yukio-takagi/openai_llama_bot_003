### library
import streamlit as st
import openai
from llama_index import GPTSimpleVectorIndex, SimpleDirectoryReader, ServiceContext, PromptHelper
from llama_index.llm_predictor.chatgpt import ChatGPTLLMPredictor

### API Key
# Streamlit Community Cloudの「Secrets」からOpenAI API keyを取得
openai.api_key = st.secrets.OpenAIAPI.openai_api_key # secrets に後ほどAPI Keyを保存する

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
index = GPTSimpleVectorIndex.load_from_disk("/index_01-05.json", service_context = service_context)

# 検索結果を表示する関数
def show_response(response):
    response = response
    for node in response.source_nodes:
        if node.score is not None:
            node_score = node.score
        node_text = node.node.text
    return response, node_score, node_text


# チャットボットとやりとりする関数
def communicate():
    question =  "日本語で答えてください。 "  + st.session_state["user_input"]
    res = index.query(question)
    response, node_score, node_text = show_response(res)
    st.write("回答： " + response)
    st.write("=================================================================================================================")
    st.write("関連FAQ： " + node_text)
    st.write("=================================================================================================================")
    st.write("スコア： " + node_score)

    st.session_state["user_input"] = ""  # 入力欄を消去


# ユーザーインターフェイスの構築
st.title("コロナ感染症対策")
st.write("ChatGPT APIを使ったコロナ感染症対策のFAQボットです。")

user_input = st.text_input("メッセージを入力してください。", key="user_input", on_change=communicate) # 入力値はsession_stateにkey引数で指定した名前で保管

