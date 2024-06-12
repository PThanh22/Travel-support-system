import streamlit as st
from openai import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from streamlit_chat import message
from langchain.callbacks import get_openai_callback

# Load environment variables
# load_dotenv()

st.title("Chatbot 🤖")
st.info("Xin chào? chúng tôi có thể giúp gì cho bạn?")

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Initialize session state
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Load and process text files
userhotel_text = "userhotel.txt"
hotel_rating_text = "hotel_rating.txt"
hotel_with_id_text = "hotel_with_id.txt"

# Load text from files
@st.cache_data
def load_text(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

userhotel_data = load_text(userhotel_text)
hotel_rating_data = load_text(hotel_rating_text)
hotel_with_id_data = load_text(hotel_with_id_text)

# Combine text from all files
combined_text = userhotel_data + hotel_rating_data + hotel_with_id_data

# Split text into chunks
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=900,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

text_chunks = get_text_chunks(combined_text)

# Get vector store
def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings()
    knowledge_base = FAISS.from_texts(text_chunks, embeddings)
    return knowledge_base

vector_store = get_vector_store(text_chunks)

# Handle user input
if prompt := st.chat_input("Hãy để lại câu hỏi chúng tôi sẽ trả lời bạn trong thời gian sớm nhất ."):
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    with st.chat_message('user'):
        st.markdown(prompt)

    with st.chat_message('assistant'):
        full_res = ""
        holder = st.empty()

        for response in client.chat.completions.create(
            model = st.session_state["openai_model"],
            # Lấy ngữ cảnh qua từng câu hỏi
            messages = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream = True,
        ):
            full_res += (response.choices[0].delta.content or "")
            holder.markdown(full_res + "▌")
            holder.markdown(full_res)
        holder.markdown(full_res)

    st.session_state.messages.append( # mảng chứa tất cả các trao đổi đc cài đặt sẵn
        {
            "role": "assistant",
            "content": full_res
        }
    )

def main():
    pass

if __name__ == '__main__':
    main()
