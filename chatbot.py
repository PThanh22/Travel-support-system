# buoc 1
import streamlit as st
from openai import OpenAI
import time

st.title("ProtonX x ChatGPT")
st.info("Xin chào? chúng tôi có thể giúp gì cho bạn?")

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = []
    
# Lưu lại câu hỏi
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
    
# buoc 2: HugChat app


# in ra màn hình câu hỏi " Xin chào?...", nhập vào 1 cái pormpt
if prompt := st.chat_input("Hãy để lại câu hỏi chúng tôi sẽ trả lời bạn trong thời gian sớm nhất ."):
    st.session_state.messages.append( # mảng chứa tất cả các trao đổi đc cài đặt sẵn
        {
            "role": "user",
            "content": prompt
        }
    )
    
    # in lại ra màn hình cái prompt đó.
    with st.chat_message('user'):
        st.markdown(prompt) # markdown giúp in ra code, toán,...; gõ prompt in ra prompt


    # st.session_state.messages.append( # mảng chứa tất cả các trao đổi đc cài đặt sẵn
    #     {
    #         "role": "assistant",
    #         "content": prompt
    #     }
    # )
    
    
    
    
# tạo time chờ cho câu trả lời
    with st.chat_message('assistant'):
        full_res = ""
        holder = st.empty()

        
# thao tác với openAI
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
