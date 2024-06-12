import streamlit as st
from openai import OpenAI

st.title("ProtonX x ChatGPT")
st.info("Xin chào? chúng tôi có thể giúp gì cho bạn?")

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = []

# Load text files
@st.cache_data
def load_text(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

# Load text files
userhotel_text = load_text("userhotel.txt")
version_history_text = load_text("version_history.txt")
hotel_with_id_text = load_text("hotel_with_id.txt")

# Store data in session state
st.session_state.userhotel_data = userhotel_text
st.session_state.version_history_data = version_history_text
st.session_state.hotel_with_id_data = hotel_with_id_text

def main():
    # Lưu lại câu hỏi
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # in ra màn hình câu hỏi " Xin chào?...", nhập vào 1 cái pormpt
    if prompt := st.chat_input("Hãy để lại câu hỏi chúng tôi sẽ trả lời bạn trong thời gian sớm nhất ."):
        st.session_state.messages.append(
            {
                "role": "user",
                "content": prompt
            }
        )
        
        # in lại ra màn hình cái prompt đó.
        with st.chat_message('user'):
            st.markdown(prompt)

        with st.chat_message('assistant'):
            full_res = ""
            holder = st.empty()
            
            for response in client.chat.completions.create(
                model = st.session_state["openai_model"],
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
            
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": full_res
            }
        )

if __name__ == "__main__":
    main()
