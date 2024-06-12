# Step1:  Nhập thư viện
import streamlit as st
from llama_index import VectorStoreIndex, ServiceContext, Document
from llama_index.llms import OpenAI
import openai
from llama_index import SimpleDirectoryReader


# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.vectorstores import Chroma
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.chains import ConversationalRetrievalChain
# from langchain.chat_models import ChatOpenAI
# from langchain.document_loaders import UnstructuredFileLoader
# import os

# import pandas as pd
# from langchain.chat_models import ChatOpenAI

# Step2: Khởi tạo lịch sử tin nhắn
''' 
+ Đặt khóa API OpenAI của bạn từ bí mật của ứng dụng.
+ Thêm tiêu đề cho ứng dụng của bạn.
+ Sử dụng trạng thái phiên để theo dõi lịch sử tin nhắn của chatbot.
+ Khởi tạo giá trị st.session_state.messagesđể bao gồm thông báo bắt đầu của chatbot,
chẳng hạn như "Hãy hỏi tôi một câu hỏi về thư viện Python nguồn mở của Streamlit!" 
'''

openai.api_key = st.secrets.OPENAI_API_KEY
st.header("Chat with the Streamlit docs 💬 📚")

if "messages" not in st.session_state.keys(): # Khởi tạo lịch sử tin nhắn trò chuyện
    st.session_state.messages = [
        {
            "role": "assistant", 
            "content": "Xin chào? chúng tôi có thể giúp gì cho bạn? Hãy để lại câu hỏi chúng tôi sẽ trả lời bạn trong thời gian sớm nhất ."
         }
    ]

# Step3: Tải và lập chỉ mục dữ liệu
'''
+ Lưu trữ các files Cơ sở Kiến thức của bạn trong một thư mục có tên data trong ứng 
dụng. Nhưng trước khi bạn bắt đầu…

+ Tải xuống các tệp đánh dấu cho tài liệu của Streamlit từ data thư mục kho lưu trữ 
GitHub của ứng dụng demo. Thêm data folder vào cấp gốc của ứng dụng của bạn.
Ngoài ra, hãy thêm data của bạn.

+ Xác định một hàm có tên load_data(), hàm này sẽ:
    - Sử dụng LlamaIndex SimpleDirectoryReader để chuyển LlamaIndex vào folder
    nơi bạn đã lưu trữ data của mình (trong trường hợp này, nó được gọi data và 
    nằm ở cấp cơ sở của kho lưu trữ của bạn).
    - SimpleDirectoryReader sẽ chọn trình đọc files thích hợp dựa trên phần mở rộng 
    của files trong folder đó ( .md files cho ví dụ này) và sẽ tải tất cả các files 
    theo cách đệ quy từ folders đó khi chúng tôi gọi reader.load_data().
    - Xây dựng một phiên bản của LlamaIndex ServiceContext, phần tài nguyên của 
    LlamaIndex được sử dụng trong các giai đoạn lập chỉ mục và truy vấn của đường
    dẫn RAG.
    - ServiceContext cho phép chúng tôi điều chỉnh các cài đặt như LLM và model nhúng
    được sử dụng.
    - Sử dụng LlamaIndex's VectorStoreIndex cho creaLlamaIndex'sory SimpleVectorStore, 
    việc này sẽ cấu trúc dữ liệu của bạn theo cách giúp model của bạn nhanh chóng 
    truy xuất ngữ cảnh từ data của bạn. Hàm này trả về VectorStoreIndex đối tượng.
    - Chức năng này được gói trong bộ trang trí bộ nhớ đệm của Streamlit 
    st.cache_resource để giảm thiểu số lần data được tải và lập chỉ mục.

+ Cuối cùng, gọi hàm load_data, chỉ định VectorStoreIndex đối tượng trả về của nó sẽ
được gọi index.
'''

@st.cache_resource(show_spinner=False)

def load_data():
    with st.spinner(text="Đang tải và lập chỉ mục các tài liệu Streamlit – hãy chờ nhé! Quá trình này sẽ mất 1-2 phút."):
        reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
        docs = reader.load_data()
        service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo", temperature=0.5, system_prompt="Bạn là chuyên gia về thư viện Streamlit Python và công việc của bạn là trả lời các câu hỏi kỹ thuật. Giả sử rằng tất cả các câu hỏi đều liên quan đến thư viện Streamlit Python. Hãy đưa ra câu trả lời mang tính kỹ thuật và dựa trên sự thật – đừng tạo ảo giác về các tính năng."))
        index = VectorStoreIndex.from_documents(docs, service_context=service_context)
        return index

index = load_data()

# Step4: Tạo công cụ trò chuyện
'''
+ LlamaIndex cung cấp một số chế độ khác nhau của công cụ trò chuyện. Có thể hữu ích
khi kiểm tra từng chế độ bằng các câu hỏi dành riêng cho cơ sở kiến thức và 
trường hợp sử dụng của bạn, so sánh phản hồi do mô hình tạo ra trong từng chế độ.

+ LlamaIndex có bốn công cụ trò chuyện khác nhau:
    - Condense question engine_Công cụ câu hỏi cô đọng : Luôn truy vấn cơ sở kiến 
    thức. Có thể gặp rắc rối với các câu hỏi meta như “Trước đây tôi đã hỏi bạn 
    điều gì?”
    - Context chat engin_Công cụ trò chuyện theo ngữ cảnh" : Luôn truy vấn cơ sở 
    kiến thức và sử dụng văn bản được truy xuất từ cơ sở kiến thức làm ngữ cảnh 
    cho các truy vấn sau. Ngữ cảnh được truy xuất từ các truy vấn trước đó có thể 
    chiếm nhiều ngữ cảnh có sẵn cho truy vấn hiện tại.
    - ReAct agent_Tác nhân ReAct : Chọn có truy vấn cơ sở tri thức hay không. 
    Hiệu suất của nó phụ thuộc nhiều hơn vào chất lượng của LLM. Bạn có thể cần phải
    buộc công cụ trò chuyện chọn chính xác xem có truy vấn cơ sở kiến thức hay không.
    - OpenAI agent_Tác nhân OpenAI : Chọn có truy vấn cơ sở kiến thức hay không —
    tương tự như chế độ tác nhân ReAct, nhưng sử dụng khả năng truy vấn fuOpenAI's 
    tích hợp sẵn của OpenAI .
    
Bài này sử dụng chế độ câu hỏi cô đọng vì nó luôn truy vấn cơ sở kiến thức 
(các tệp từ tài liệu Streamlit) khi tạo câu trả lời. Chế độ này là tối ưu vì 
bạn muốn mô hình giữ các câu trả lời cụ thể cho các tính năng được đề cập 
trong tài liệu của Streamlit.
'''

chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)


# Step5: Nhắc người dùng nhập và hiển thị lịch sử tin nhắn
'''
+ Sử dụng tính năng Streamlit's của st.chat_inputStreamlit để user nhập câu hỏi.
+ Khi user đã nhập thông tin đầu vào, hãy thêm thông tin đầu vào đó vào lịch sử 
tin nhắn bằng cách thêm nó vào st.session_state.messages.
+ Hiển thị lịch sử tin nhắn của chatbot bằng cách duyệt qua nội dung được liên kết
với khóa “messages” ở trạng thái phiên và hiển thị từng message bằng st.chat_message.
'''

if prompt := st.chat_input("Your question"): # Nhắc user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Hiển thị tin nhắn trò chuyện trước đó
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Step6: Chuyển truy vấn tới công cụ trò chuyện và hiển thị phản hồi
'''
Nếu tin nhắn cuối cùng trong lịch sử tin nhắn không phải từ chatbot, hãy chuyển 
nội dung tin nhắn đến công cụ trò chuyện thông qua chat_engine.chat(), viết phản 
hồi cho giao diện user bằng cách sử dụng st.write và st.chat_message, đồng thời 
thêm phản hồi của công cụ trò chuyện vào lịch sử tin nhắn.
'''

# Nếu tin nhắn cuối cùng không phải từ trợ lý [assistant], hãy tạo phản hồi mới
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message) # Thêm phản hồi vào lịch sử tin nhắn
            
            
#Step7: Triển khai ứng dụng!
'''
+ Sau khi xây dựng ứng dụng, hãy triển khai nó trên Streamlit Community Cloud:
    - Tạo kho lưu trữ GitHub.
    - Điều hướng đến Streamlit Community Cloud , nhấp vào New appvà chọn kho 
    lưu trữ, nhánh và đường dẫn tệp thích hợp.
    - Đánh Deploy.
    
+ Kết quả: Bạn cũng đã xây dựng một ứng dụng chatbot sử dụng LlamaIndex để tăng c
ường GPT-3.5 trong 43 dòng mã. Tài liệu Streamlit có thể được thay thế cho bất kỳ 
nguồn dữ liệu tùy chỉnh nào. Kết quả là một ứng dụng mang lại câu trả lời chính xác
và cập nhật hơn nhiều cho các câu hỏi về thư viện Python nguồn mở Streamlit so với
ChatGPT hoặc chỉ sử dụng GPT.
'''

# Đọc dữ liệu từ tập tin CSV

def load_data(file_path):
    return pd.read_csv(file_path)

file_path = "your_dataset.csv"
data = load_data(file_path)

# Tiền xử lý dữ liệu (nếu cần)
# Ví dụ: loại bỏ các dòng trống
data.dropna(inplace=True)

# Huấn luyện chatbot
llm = ChatOpenAI(temperature=0, max_tokens=1000, model_name="gpt-3.5-turbo", streaming=True)

# Huấn luyện chatbot dựa trên dữ liệu từ tập tin CSV
for row in data.itertuples(index=False):
    user_input = row.question  # Giả sử cột "question" trong tập dữ liệu chứa các câu hỏi
    response = row.answer  # Giả sử cột "answer" trong tập dữ liệu chứa các câu trả lời

    # Huấn luyện chatbot với mỗi cặp câu hỏi - câu trả lời từ tập dữ liệu
    llm.train(user_input, response)

# Xây dựng giao diện người dùng
st.title("Simple Chatbot with Streamlit")

user_input = st.text_input("You:", "")

if user_input:
    # Gửi câu hỏi của người dùng đến chatbot
    response = llm.ask(user_input)

    st.text_area("Bot:", value=response, height=200)


