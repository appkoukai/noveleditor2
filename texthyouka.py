import streamlit as st
import openai
from sklearn.feature_extraction.text import TfidfVectorizer
from janome.tokenizer import Tokenizer

# OpenAI APIキーを設定
openai.api_key = st.secrets["API_key"]

# Janomeトークナイザー
tokenizer = Tokenizer()

def tokenize(text):
    tokens = tokenizer.tokenize(text)
    return [token.surface for token in tokens if token.surface != 'の' and token.part_of_speech.split(',')[0] in ['名詞', '代名詞']]

def extract_keywords(text):
    vectorizer = TfidfVectorizer(tokenizer=tokenize)
    vectors = vectorizer.fit_transform([text])
    feature_names = vectorizer.get_feature_names_out()
    dense = vectors.todense()
    denselist = dense.tolist()
    keywords = sorted([(word, score) for word, score in zip(feature_names, denselist[0]) if score > 0.1], key=lambda x: x[1], reverse=True)
    return keywords

# 特徴語テーブルを定義
if "keywords_data" not in st.session_state:
    st.session_state.keywords_data = None

def extract_keywords_and_display():
    global table
    keywords = extract_keywords(user_input)
    data = pd.DataFrame(keywords, columns=['単語', 'スコア'])
    data = data.sort_values(by='スコア', ascending=False).reset_index(drop=True)
    data.insert(0, '順位', range(1, len(data) + 1))
    st.session_state.keywords_data = data # キーワードデータを保存
    # 表形式で出力
    st.write('特徴語:')
    st.session_state.table = st.table(data)

# Streamlit設定
st.title('文章評価アプリ') 

# ユーザー入力
user_input = st.text_area('文章を入力')

# 特徴語抽出ボタン
if st.button('特徴語を抽出'):
    extract_keywords_and_display()

if st.button('分析を実行'):
    # ローディングアニメーション
    with st.spinner('分析中...'):
        keywords = extract_keywords(user_input)
        
        # GPT-3.5に評価してもらう
        messages = []
        messages.append({"role":"system", "content":f"以下の文章の特徴語を分析して、文章の評価をして下さい:\n{user_input}\n\n特徴語: {', '.join([word for word, score in keywords])}"})
        messages.append({"role":"assistant", "content":"..."})

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages
        )

    # 結果表示
    st.success('分析が完了しました')
    if st.session_state.table is not None:
        st.session_state.table.empty()  # テーブルをクリアする
        st.session_state.table = st.table(st.session_state.keywords_data)  # 特徴語の表を再レンダリング
    # GPT-3.5の応答を表示
    gpt_response = response.choices[0].message.get("content")
    st.write(gpt_response)