import pandas as pd
import time
import numpy as np
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from openai import OpenAI

import streamlit as st
import openai

#---------------------------------------------------------------------------
st.set_page_config(
    page_title=" Rabbit travel ",
    page_icon="🐰",
)
#---------------------------------------------------------------------------

hotel_full = pd.read_csv('hotel_full.csv')
hotel_with_id = pd.read_csv('hotel_with_id.csv')
hotels_merg = pd.read_csv('hotel_rating.csv')
hotels = pd.read_csv('userhotel.csv')

des_full = pd.read_csv('des_full.csv')
des_with_id = pd.read_csv('des_with_id.csv')
dess_merg = pd.read_csv('des_rating.csv')
dess = pd.read_csv('userdes.csv')

restaurant_full = pd.read_csv('restaurant_full.csv')
res_with_id = pd.read_csv('res_with_id.csv')
res_rating = pd.read_csv('res_rating.csv')
userres = pd.read_csv('userres.csv')
#_______________________________________________________

def About_us():
    st.markdown("# Travel support system 🧳")
    st.sidebar.markdown("# Travel support system 🧳")


#------------------------------------------------------------------------------------------
              
def Chatbot():
    st.markdown("# Chatbot 🤖")
    st.sidebar.markdown("# Chatbot 🤖")

    
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
            
            
            # for word in prompt.split():
            #     full_res += word + " "
            #     time.sleep(0.05)
            #     holder.markdown(full_res + "▌")
            # holder.markdown(full_res)   
            
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


#--------------------------------------------HOTEL--------------------------------------------------

def Recommend_Hotel():
    st.markdown("# Hotel 🏩")
    st.sidebar.markdown("# Hotel 🏩")

    def get_rating(hotel_user_id, hotel_id):
        mask = (hotel_full.Hotel_User_Id == hotel_user_id) & (hotel_full.Hotel_Id == hotel_id)
        return (hotel_full.loc[mask, 'Hotel_User_Rating'].tolist())

    def get_star(hotel_id, star):
        mask = (hotel_full.Hotel_Id == hotel_id) & (hotel_full.Hotel_Star == star)
        return (hotel_full.loc[mask, 'Hotel_Star'].tolist())

    def get_name_hotel(hotel_user_id, location):
        mask = (hotel_full.Hotel_User_Id == hotel_user_id) & (hotel_full.Hotel_City == location)
        return (hotel_full.loc[mask, 'Hotel_Name'].tolist())


    def get_hotel_ids(hotel_user_id, location):
        mask = (hotel_full.Hotel_User_Id == hotel_user_id) & (hotel_full.Hotel_City == location)
        return (hotel_full.loc[mask, 'Hotel_Id'].tolist())

    def get_hotel_price(hotel_id, max_price):
        mask = (hotel_full.Hotel_Id == hotel_id) & (hotel_full.Hotel_Price <= max_price)
        return (hotel_full.loc[mask, 'Hotel_Price'].tolist())

    def get_hotel_title(hotel_id):
        mask = (hotel_full.Hotel_Id == hotel_id)
        return (hotel_full.loc[mask, 'Hotel_Name'].iloc[0])


    def get_description(hotel_id):
        mask = (hotel_full.Hotel_Id == hotel_id)
        return (hotel_full.loc[mask, 'Hotel_Descriptions'].iloc[0])


    def get_address(hotel_id):
        mask = (hotel_full.Hotel_Id == hotel_id)
        return (hotel_full.loc[mask, 'Hotel_Address'].iloc[0])



    def pearson_correlation_score(hotel_user_1, hotel_user_2, location, max_price, star):
        both_vatch_count= []
        mask_1 = (hotel_full.Hotel_User_Id == hotel_user_1) & (hotel_full.Hotel_City == location) & (hotel_full.Hotel_Price <= max_price) & (hotel_full.Hotel_Star == star)
        mask_2 = (hotel_full.Hotel_User_Id == hotel_user_2) & (hotel_full.Hotel_City == location) & (hotel_full.Hotel_Price <= max_price) & (hotel_full.Hotel_Star == star)
        list_hotel_user_1 = hotel_full.loc[mask_1, 'Hotel_Id'].to_list()
        list_hotel_user_2 = hotel_full.loc[mask_2, 'Hotel_Id'].to_list()
        for element in list_hotel_user_1:
            if element in list_hotel_user_2:
                both_vatch_count.append(element)
            if (len(both_vatch_count)==0):
                return 0
        avg_rating_sum_1 = np.lib.function_base.average([get_rating(hotel_user_1, i) for i in both_vatch_count])# rating trung bình user1
        avg_rating_sum_2 = np.lib.function_base.average([get_rating(hotel_user_2, i) for i in both_vatch_count])# rating trung bình user2
        tu = sum([(get_rating(hotel_user_1, i)- avg_rating_sum_1)*(get_rating(hotel_user_2, i)- avg_rating_sum_2) for i in both_vatch_count])
        mau_1 = np.sqrt(sum([pow((get_rating(hotel_user_1, i) - avg_rating_sum_1), 2) for i in both_vatch_count]))
        mau_2 = np.sqrt(sum([pow((get_rating(hotel_user_2, i) - avg_rating_sum_2), 2) for i in both_vatch_count]))
        mau = mau_1 * mau_2
        if mau == 0:
            return 0
        return tu / mau


    def distance_similarity_score(hotel_user_1, hotel_user_2, location, max_price, star):
        both_watch_count = 0
        mask_1 = (hotel_full.Hotel_User_Id == hotel_user_1) & (hotel_full.Hotel_City == location) & (hotel_full.Hotel_Price <= max_price) & (hotel_full.Hotel_Star == star)
        mask_2 = (hotel_full.Hotel_User_Id == hotel_user_2) & (hotel_full.Hotel_City == location) & (hotel_full.Hotel_Price <= max_price) & (hotel_full.Hotel_Star == star)
        list_hotel_user_1 = hotel_full.loc[mask_1, 'Hotel_Id'].to_list()
        list_hotel_user_2 = hotel_full.loc[mask_2, 'Hotel_Id'].to_list()
        for element in list_hotel_user_1:
            if element in list_hotel_user_2:
                both_watch_count += 1
        if both_watch_count == 0 :
            return 0
        hotel_user_rating_1, hotel_user_rating_2 = [], []
        for element in list_hotel_user_1:
            if element in list_hotel_user_2:
                hotel_user_rating_1.append(get_rating(hotel_user_1, element))
                hotel_user_rating_2.append(get_rating(hotel_user_2, element))
        print(f"distance_similarity_score-hotel_user_rating_1: {hotel_user_rating_1}")
        print(f"distance_similarity_score-hotel_user_rating_2: {hotel_user_rating_2}")
        return np.dot(hotel_user_rating_1, hotel_user_rating_2) / (np.linalg.norm(hotel_user_rating_1) * np.linalg.norm(hotel_user_rating_2))


    def most_similar_user(hotel_user_1, number_of_user, location, max_price, star, similarity_name):
        user_ID = hotel_full.Hotel_User_Id.unique().tolist()
        print(f"most_similar_user-len: {len(user_ID)}")
        if(similarity_name == "pearson"):
            similarity_score = [(pearson_correlation_score(hotel_user_1, user_i, location, max_price, star),user_i)  for user_i in user_ID[0:1500] if user_i != hotel_user_1] #danh sách user quá nhiều nên tình chỉ tính tên dánh sách có 50 users
        if(similarity_name == "cosine"):
            similarity_score = [(distance_similarity_score(hotel_user_1, user_i, location, max_price, star),user_i)  for user_i in user_ID[0:1500] if user_i != hotel_user_1]
        similarity_score.sort() #tăng dần
        similarity_score.reverse() #tăng dần
        return similarity_score[:number_of_user] # có thể thay đổi số lượng lân cận


    #lấy ra danh sách khuyến nghị từ top populars
    def get_recommendation(hotel_user_id, number_of_user, location, similarity_name, max_price, star):
        total, similarity_sum, ranking = {}, {}, []
        list_user_popular = most_similar_user(hotel_user_id, number_of_user, location, max_price, star, similarity_name)
        
        for pearson, user in list_user_popular:
            score = pearson
            for hotel_id in get_hotel_ids(user, location):
                if hotel_id not in get_hotel_ids(hotel_user_id, location):
                    if get_star(hotel_id, star):
                        if get_hotel_price(hotel_id, max_price):
                            if hotel_id not in total:
                                total[hotel_id] = []
                                similarity_sum[hotel_id] = 0
                            total[hotel_id].extend(get_rating(user, hotel_id))  # Extend the list of ratings
                            similarity_sum[hotel_id] += score
        
        for hotel_id, ratings in total.items():
            if similarity_sum[hotel_id] == 0:
                ranking.append((8, hotel_id))
            else:
                average_rating = sum(ratings) / len(ratings)  # Calculate the average rating
                ranking.append((average_rating, hotel_id))
        
        ranking.sort()
        ranking.reverse()
        
        recommendations = [(get_hotel_title(hotel_id), score, get_address(hotel_id), get_description(hotel_id), get_hotel_price(hotel_id, max_price), get_star(hotel_id, star)) for score, hotel_id in ranking]
        
        return recommendations[:number_of_user]




    # Hàm demo content based
    def recommendations_content(hotel_user_id):
        a = hotel_with_id
        vectorizer = TfidfVectorizer(max_features= 4500)
        overview_matrix = vectorizer.fit_transform(a['Hotel_Descriptions'])
        overview_matrix_1 = vectorizer.fit_transform(hotel_full['Hotel_Descriptions'])
        cosine_sim = linear_kernel(overview_matrix_1, overview_matrix)
        for i in range(len(hotel_full['Hotel_User_Id'])):
            if (hotel_full['Hotel_User_Id'][i] == hotel_user_id):
                print(f"recommendations_content | hotel_user_id = {hotel_user_id}")
                sim_scores = list(enumerate(cosine_sim[i]))
            # Sắp xếp phim dựa trên điểm số tương tự
                sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            # Lấy điểm của 10 phim giống nhất
                sim_scores = sim_scores[1:11]
                hotel_indices = [i[0] for i in sim_scores]
                print(f"recommendations_content | hotel_indices = {hotel_indices}")
        # b = a['Hotel_'].iloc[hotel_indices]
                a['Hotel_Name'].iloc[hotel_indices].to_list()



    def recommend_hotel_based_on_description(user_description, number_of_recommendations):
        # Tạo vectorizer TF-IDF và biểu diễn văn bản người dùng
        vectorizer = TfidfVectorizer(max_features=4500)
        user_description_matrix = vectorizer.fit_transform([user_description])

        # Biểu diễn văn bản của tất cả các khách sạn
        hotel_description_matrix = vectorizer.transform(hotel_full['Hotel_Descriptions'])

        # Tính toán độ tương đồng cosine giữa mô tả người dùng và mô tả của từng khách sạn
        cosine_similarities = linear_kernel(user_description_matrix, hotel_description_matrix).flatten()

        # Sắp xếp các khách sạn theo độ tương đồng và lấy ra các khách sạn tốt nhất
        hotel_indices = cosine_similarities.argsort()[:-number_of_recommendations-1:-1]

        recommendations = []
        for index in hotel_indices:
            hotel_id = hotel_full.iloc[index]['Hotel_Id']
            hotel_title = get_hotel_title(hotel_id)
            hotel_star = get_star(hotel_id)
            hotel_address = get_address(hotel_id)
            hotel_description = get_description(hotel_id)
            hotel_price = get_hotel_price(hotel_id, float('inf'))  # Lấy giá của khách sạn, không giới hạn giá
            recommendations.append((hotel_title, hotel_star, hotel_address, hotel_description, hotel_price))

        return recommendations



    def run1():
        
        with st.sidebar:
            add_userID = st.number_input('Enter User Id:')
            print(f"add_userID: {add_userID}")
            with st.form('form1'):
                if add_userID <= 100000:
                    add_password = st.text_input('Enter password:')
                st.form_submit_button('Enter')
        time.sleep(1)
        add_selectbox = st.sidebar.selectbox(
            "How would you like to be contacted?",
            ("Email", "Home phone", "Mobile phone")
        )
        
        #########################################################
        location = st.text_input("Enter the place: ")
        if location:
            st.write('Hotel_City: ', location)
        elif add_userID:
            st.write('Hotel_User_Id: ', add_userID)
        
        max_price = st.slider("Enter maximum price:", 0, 5000000, step=100000)
        if max_price:
            st.write('Price: ', max_price)
        elif add_userID:
            st.write('Hotel_User_Id: ', add_userID)
            
        star = st.selectbox("Enter the Star: ", ('1','2','3','4','5'))
        star = int(star)
        if star:
            st.write('Hotel_Star: ', star)
        elif add_userID:
            st.write('Hotel_User_Id: ', add_userID)

        description = st.text_input("Enter your description:")
        if description:
            st.write('Hotel_Descriptions: ', description)
        elif add_userID:
            st.write('Hotel_User_Id: ', add_userID)
        

        click = st.button('Search')
        
        list_recommendations_content = []
        
        if add_userID:
            start_time = time.time()
            list_recommendations_content = recommendations_content(add_userID)
            end_time = time.time()
            elapsed_time = end_time - start_time
            # Display elapsed time
            st.write(f"Time taken for recommendations: {elapsed_time:.2f} seconds")
            if not list_recommendations_content:
                st.write(f"No Results!")
            for i in range(len(list_recommendations_content[0])):
                if location:
                    break
                col1, col2 = st.columns(2)
                with col1:
                    st.image(f'hotel-{i}.jpg', caption = '')
                with col2:
                    st.markdown(f'**Name Hotel**: {list_recommendations_content[0][i]}')
                    st.markdown(f'**Rating**: {list_recommendations_content[1][i]}')
                    st.markdown(f'**Address**: {list_recommendations_content[2][i]}')
                    st.markdown(f'**Description**: {list_recommendations_content[3][i][:200]}...')
                    st.markdown(f'**Price**: {list_recommendations_content[4][i]}')
                    st.markdown(f'**Star**: {list_recommendations_content[5][i]}')
        else:
            if click:
                start_time = time.time()
                list_recommen = get_recommendation(add_userID, 10, location, 'cosine', max_price, star)  # Thêm max_price vào đây
                end_time = time.time()
                elapsed_time = end_time - start_time
                # Display elapsed time
                st.write(f"Time taken for recommendations: {elapsed_time:.2f} seconds")
                if not list_recommen:
                    st.write(f"No Results!")
                for i in range(len(list_recommen)):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(f'hotel-{i}.jpg', caption='')
                    with col2:
                        st.markdown(f'**Name Hotel**: {list_recommen[i][0]}')
                        st.markdown(f'**Rating**: {list_recommen[i][1]}')
                        st.markdown(f'**Address**: {list_recommen[i][2]}')
                        st.markdown(f'**Description**: {list_recommen[i][3][:200]}...')
                        
                        unique_prices = set(list_recommen[i][4])  # Loại bỏ các giá trị trùng lặp
                        price_str = ", ".join(map(str, unique_prices))  # Chuyển danh sách thành chuỗi
                        st.markdown(f'**Price**: {price_str}')
                        
                        # st.markdown(f'**Star**: {list_recommen[i][5]}')
                        
                        unique_star = set(list_recommen[i][5])  # Loại bỏ các giá trị trùng lặp
                        star_str = ", ".join(map(str, unique_star))  # Chuyển danh sách thành chuỗi
                        st.markdown(f'**Star**: {star_str}')

    if __name__=="__main__":
        run1()          
#----------------------------------------------------------------------------------------------------
def Recommend_Destination():
    st.markdown("# Travel destination ✈️")
    st.sidebar.markdown("# Destination ✈️")
    
    def get_rating(des_user_id, des_id):
        mask = (des_full.Des_User_Id == des_user_id) & (des_full.Des_Id == des_id)
        return (des_full.loc[mask, 'Des_User_Rating'].tolist())

    def get_category(des_id, cate):
        mask = (des_full.Des_Id == des_id) & (des_full.Des_Category ==  cate)
        return (des_full.loc[mask, 'Des_Category'].tolist())

    def get_name_des(des_user_id, location):
        mask = (des_full.Des_User_Id == des_user_id) & (des_full.Des_City == location)
        return (des_full.loc[mask, 'Des_Name'].tolist())


    def get_Des_Id(des_user_id, location):
        mask = (des_full.Des_User_Id == des_user_id) & (des_full.Des_City == location)
        return (des_full.loc[mask, 'Des_Id'].tolist())


    def get_des_title(des_id):
        mask = (des_full.Des_Id == des_id)
        return (des_full.loc[mask, 'Des_Name'].iloc[0])


    def get_description(des_id):
        mask = (des_full.Des_Id == des_id)
        return (des_full.loc[mask, 'Des_Descriptions'].iloc[0])


    def get_address(des_id):
        mask = (des_full.Des_Id == des_id)
        return (des_full.loc[mask, 'Des_Address'].iloc[0])



    def pearson_correlation_score(des_user_1, des_user_2, location, cate):
        both_vatch_count= []
        mask_1 = (des_full.Des_User_Id == des_user_1) & (des_full.Des_City == location)  & (des_full.Des_Category== cate)
        mask_2 = (des_full.Des_User_Id == des_user_2) & (des_full.Des_City == location)  & (des_full.Des_Category== cate)
        list_des_user_1 = des_full.loc[mask_1, 'Des_Id'].to_list()
        list_des_user_2 = des_full.loc[mask_2, 'Des_Id'].to_list()
        for element in list_des_user_1:
            if element in list_des_user_2:
                both_vatch_count.append(element)
            if (len(both_vatch_count)==0):
                return 0
        avg_rating_sum_1 = np.lib.function_base.average([get_rating(des_user_1, i) for i in both_vatch_count])# rating trung bình user1
        avg_rating_sum_2 = np.lib.function_base.average([get_rating(des_user_2, i) for i in both_vatch_count])# rating trung bình user2
        tu = sum([(get_rating(des_user_1, i)- avg_rating_sum_1)*(get_rating(des_user_2, i)- avg_rating_sum_2) for i in both_vatch_count])
        mau_1 = np.sqrt(sum([pow((get_rating(des_user_1, i) - avg_rating_sum_1), 2) for i in both_vatch_count]))
        mau_2 = np.sqrt(sum([pow((get_rating(des_user_2, i) - avg_rating_sum_2), 2) for i in both_vatch_count]))
        mau = mau_1 * mau_2
        if mau == 0:
            return 0
        return tu / mau


    def distance_similarity_score(des_user_1, des_user_2, location, cate):
        both_watch_count = 0
        mask_1 = (des_full.Des_User_Id == des_user_1) & (des_full.Des_City == location) & (des_full.Des_Category== cate)
        mask_2 = (des_full.Des_User_Id == des_user_2) & (des_full.Des_City == location) & (des_full.Des_Category== cate)
        list_des_user_1 = des_full.loc[mask_1, 'Des_Id'].to_list()
        list_des_user_2 = des_full.loc[mask_2, 'Des_Id'].to_list()
        for element in list_des_user_1:
            if element in list_des_user_2:
                both_watch_count += 1
        if both_watch_count == 0 :
            return 0
        des_user_rating_1, des_user_rating_2 = [], []
        for element in list_des_user_1:
            if element in list_des_user_2:
                des_user_rating_1.append(get_rating(des_user_1, element))
                des_user_rating_2.append(get_rating(des_user_2, element))
        print(f"distance_similarity_score-des_user_rating_1: {des_user_rating_1}")
        print(f"distance_similarity_score-des_user_rating_2: {des_user_rating_2}")
        return np.dot(des_user_rating_1, des_user_rating_2) / (np.linalg.norm(des_user_rating_1) * np.linalg.norm(des_user_rating_2))


    def most_similar_user(des_user_1, number_of_user, location, cate, similarity_name):
        user_ID = des_full.Des_User_Id.unique().tolist()
        print(f"most_similar_user-len: {len(user_ID)}")
        if(similarity_name == "pearson"):
            similarity_score = [(pearson_correlation_score(des_user_1, user_i, location, cate),user_i)  for user_i in user_ID[0:1500] if user_i != des_user_1] #danh sách user quá nhiều nên tình chỉ tính tên dánh sách có 50 users
        if(similarity_name == "cosine"):
            similarity_score = [(distance_similarity_score(des_user_1, user_i, location, cate),user_i)  for user_i in user_ID[0:1500] if user_i != des_user_1]
        similarity_score.sort() #tăng dần
        similarity_score.reverse() #tăng dần
        return similarity_score[:number_of_user] # có thể thay đổi số lượng lân cận


    #lấy ra danh sách khuyến nghị từ top populars
    def get_recommendation(des_user_id, number_of_user, location, similarity_name, cate):
        total, similarity_sum, ranking = {}, {}, []
        list_user_popular = most_similar_user(des_user_id, number_of_user, location, cate, similarity_name)
        
        for pearson, user in list_user_popular:
            score = pearson
            for Des_Id in get_Des_Id(user, location):
                if Des_Id not in get_Des_Id(des_user_id, location):
                    if get_category(Des_Id, cate):
                            if Des_Id not in total:
                                total[Des_Id] = []
                                similarity_sum[Des_Id] = 0
                            total[Des_Id].extend(get_rating(user, Des_Id))  # Extend the list of ratings
                            similarity_sum[Des_Id] += score
        
        for Des_Id, ratings in total.items():
            if similarity_sum[Des_Id] == 0:
                ranking.append((8, Des_Id))
            else:
                average_rating = sum(ratings) / len(ratings)  # Calculate the average rating
                ranking.append((average_rating, Des_Id))
        
        ranking.sort()
        ranking.reverse()
        
        recommendations = [(get_des_title(des_id), score, get_address(des_id), get_description(des_id), get_category(des_id, cate)) for score, des_id in ranking]
            
        return recommendations[:number_of_user]




    # Hàm demo content based
    def recommendations_content(des_user_id):
        b = des_with_id
        vectorizer = TfidfVectorizer(max_features= 4500)
        overview_matrix = vectorizer.fit_transform(b['Des_Descriptions'])
        overview_matrix_1 = vectorizer.fit_transform(des_full['Des_Descriptions'])
        cosine_sim = linear_kernel(overview_matrix_1, overview_matrix)
        for i in range(len(des_full['Des_User_Id'])):
            if (des_full['Des_User_Id'][i] == des_user_id):
                print(f"recommendations_content | des_user_id = {des_user_id}")
                sim_scores = list(enumerate(cosine_sim[i]))
            # Sắp xếp phim dựa trên điểm số tương tự
                sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            # Lấy điểm của 10 phim giống nhất
                sim_scores = sim_scores[1:11]
                des_indices = [i[0] for i in sim_scores]
                print(f"recommendations_content | des_indices = {des_indices}")
        # b = a['des_'].iloc[des_indices]
                b['Des_Name'].iloc[des_indices].to_list()



    def recommend_des_based_on_description(user_description, number_of_recommendations):
        # Tạo vectorizer TF-IDF và biểu diễn văn bản người dùng
        vectorizer = TfidfVectorizer(max_features=4500)
        user_description_matrix = vectorizer.fit_transform([user_description])

        # Biểu diễn văn bản của tất cả các khách sạn
        des_description_matrix = vectorizer.transform(des_full['Des_Descriptions'])

        # Tính toán độ tương đồng cosine giữa mô tả người dùng và mô tả của từng khách sạn
        cosine_similarities = linear_kernel(user_description_matrix, des_description_matrix).flatten()

        # Sắp xếp các khách sạn theo độ tương đồng và lấy ra các khách sạn tốt nhất
        des_indices = cosine_similarities.argsort()[:-number_of_recommendations-1:-1]

        recommendations = []
        for index in des_indices:
            Des_Id = des_full.iloc[index]['Des_Id']
            des_title = get_des_title(Des_Id)
            Des_Category= get_category(Des_Id)
            des_address = get_address(Des_Id)
            des_description = get_description(Des_Id)
            recommendations.append((des_title, Des_Category, des_address, des_description, Des_Category))

        return recommendations



    def run2():
        
        #st.sidebar.success("Select a demo above.")
        # Using "with" notation
        with st.sidebar:
            add_userID = st.number_input('Enter User Id:')
            print(f"add_userID: {add_userID}")
            with st.form('form1'):
                if add_userID <= 100000:
                    add_password = st.text_input('Enter password:')
                st.form_submit_button('Enter')
        time.sleep(1)
        add_selectbox = st.sidebar.selectbox(
            "How would you like to be contacted?",
            ("Email", "Home phone", "Mobile phone")
        )

        
        #########################################################
        location = st.text_input("Enter the place: ")
        if location:
            st.write('Des_City: ', location)
        elif add_userID:
            st.write('Des_User_Id: ', add_userID)
        
            
        cate = st.selectbox("Enter the Category: ", 
                            ('Du lịch văn hóa',
                            'Du lịch sinh thái',
                            'Du lịch lịch sử',
                            'Du lịch tâm linh',
                            'Du lịch giải trí'))
        if cate:
            st.write('Des_cate: ', cate)
        elif add_userID:
            st.write('Des_User_Id: ', add_userID)

        
        click = st.button('Search')
        
        list_recommendations_content = []
        
        if add_userID:
            catet_time = time.time()
            list_recommendations_content = recommendations_content(add_userID)
            end_time = time.time()
            elapsed_time = end_time - catet_time
            # Display elapsed time
            st.write(f"Time taken for recommendations: {elapsed_time:.2f} seconds")
            if not list_recommendations_content:
                st.write(f"No Results!")
            for i in range(len(list_recommendations_content[0])):
                if location:
                    break
                col1, col2 = st.columns(2)
                with col1:
                    st.image(f'des-{i}.jpg', caption = '')
                with col2:
                    st.markdown(f'**Name des**: {list_recommendations_content[0][i]}')
                    st.markdown(f'**Rating**: {list_recommendations_content[1][i]}')
                    st.markdown(f'**Address**: {list_recommendations_content[2][i]}')
                    st.markdown(f'**Description**: {list_recommendations_content[3][i][:200]}...')
                    st.markdown(f'**Category**: {list_recommendations_content[4][i]}')
                    
        else:
            if click:
                catet_time = time.time()
                list_recommen = get_recommendation(add_userID, 10, location, 'cosine', cate)  # Thêm  vào đây
                end_time = time.time()
                elapsed_time = end_time - catet_time
                # Display elapsed time
                st.write(f"Time taken for recommendations: {elapsed_time:.2f} seconds")
                if not list_recommen:
                    st.write(f"No Results!")
                for i in range(len(list_recommen)):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(f'des-{i}.jpg', caption='')
                    with col2:
                        st.markdown(f'**Name des**: {list_recommen[i][0]}')
                        st.markdown(f'**Rating**: {list_recommen[i][1]}')
                        st.markdown(f'**Address**: {list_recommen[i][2]}')
                        st.markdown(f'**Description**: {list_recommen[i][3][:200]}...')
                        
                        unique_cate = set(list_recommen[i][4])  # Loại bỏ các giá trị trùng lặp
                        cate_str = ", ".join(map(str, unique_cate))  # Chuyển danh sách thành chuỗi
                        st.markdown(f'**Category**: {cate_str}')


    if __name__=="__main__":
        run2()

#-------------------------------------Restaurant-----------------------------------------
def Recommend_Restaurant():
    st.markdown("# Restaurant 🥗 ")
    st.sidebar.markdown("# Restaurant 🥗 ")

    def get_rating(res_id, rate):
        mask = (restaurant_full.Res_Id == res_id) & (restaurant_full.Res_Rating.astype(float) >= rate)
        return (restaurant_full.loc[mask, 'Res_Rating'].tolist())

    def get_name_res(res_user_id, location):
        mask = (restaurant_full.Res_User_Id == res_user_id) & (restaurant_full.Res_City == location)
        return (restaurant_full.loc[mask, 'Res_Name'].tolist())


    def get_res_ids(res_user_id, location):
        mask = (restaurant_full.Res_User_Id == res_user_id) & (restaurant_full.Res_City == location)
        return (restaurant_full.loc[mask, 'Res_Id'].tolist())

    def get_res_price(res_id, max_price):
        mask = (restaurant_full.Res_Id == res_id) & (restaurant_full.Res_Price <= max_price)
        return (restaurant_full.loc[mask, 'Res_Price'].tolist())

    def get_res_title(res_id):
        mask = (restaurant_full.Res_Id == res_id)
        return (restaurant_full.loc[mask, 'Res_Name'].iloc[0])


    def get_description(res_id):
        mask = (restaurant_full.Res_Id == res_id)
        return (restaurant_full.loc[mask, 'Res_Descriptions'].iloc[0])


    def get_address(res_id):
        mask = (restaurant_full.Res_Id == res_id)
        return (restaurant_full.loc[mask, 'Res_Address'].iloc[0])



    def pearson_correlation_score(res_user_1, res_user_2, location, max_price, rate):
        both_vatch_count= []
        mask_1 = (restaurant_full.Res_User_Id == res_user_1) & (restaurant_full.Res_City == location) & (restaurant_full.Res_Price <= max_price) & (restaurant_full.Res_Rating.astype(float) >= rate)
        mask_2 = (restaurant_full.Res_User_Id == res_user_2) & (restaurant_full.Res_City == location) & (restaurant_full.Res_Price <= max_price) & (restaurant_full.Res_Rating.astype(float) >= rate)
        list_res_user_1 = restaurant_full.loc[mask_1, 'Res_Id'].to_list()
        list_res_user_2 = restaurant_full.loc[mask_2, 'Res_Id'].to_list()
        for element in list_res_user_1:
            if element in list_res_user_2:
                both_vatch_count.append(element)
            if (len(both_vatch_count)==0):
                return 0
        avg_rating_sum_1 = np.lib.function_base.average([get_rating(res_user_1, i) for i in both_vatch_count])# rating trung bình user1
        avg_rating_sum_2 = np.lib.function_base.average([get_rating(res_user_2, i) for i in both_vatch_count])# rating trung bình user2
        tu = sum([(get_rating(res_user_1, i)- avg_rating_sum_1)*(get_rating(res_user_2, i)- avg_rating_sum_2) for i in both_vatch_count])
        mau_1 = np.sqrt(sum([pow((get_rating(res_user_1, i) - avg_rating_sum_1), 2) for i in both_vatch_count]))
        mau_2 = np.sqrt(sum([pow((get_rating(res_user_2, i) - avg_rating_sum_2), 2) for i in both_vatch_count]))
        mau = mau_1 * mau_2
        if mau == 0:
            return 0
        return tu / mau


    def distance_similarity_score(res_user_1, res_user_2, location, max_price, rate):
        both_watch_count = 0
        mask_1 = (restaurant_full.Res_User_Id == res_user_1) & (restaurant_full.Res_City == location) & (restaurant_full.Res_Price <= max_price) & (restaurant_full.Res_Rating.astype(float) >= rate)
        mask_2 = (restaurant_full.Res_User_Id == res_user_2) & (restaurant_full.Res_City == location) & (restaurant_full.Res_Price <= max_price) & (restaurant_full.Res_Rating.astype(float) >= rate)
        list_res_user_1 = restaurant_full.loc[mask_1, 'Res_Id'].to_list()
        list_res_user_2 = restaurant_full.loc[mask_2, 'Res_Id'].to_list()
        for element in list_res_user_1:
            if element in list_res_user_2:
                both_watch_count += 1
        if both_watch_count == 0 :
            return 0
        res_user_rating_1, res_user_rating_2 = [], []
        for element in list_res_user_1:
            if element in list_res_user_2:
                res_user_rating_1.append(get_rating(res_user_1, element))
                res_user_rating_2.append(get_rating(res_user_2, element))
        print(f"distance_similarity_score-res_user_rating_1: {res_user_rating_1}")
        print(f"distance_similarity_score-res_user_rating_2: {res_user_rating_2}")
        return np.dot(res_user_rating_1, res_user_rating_2) / (np.linalg.norm(res_user_rating_1) * np.linalg.norm(res_user_rating_2))


    def most_similar_user(res_user_1, number_of_user, location, max_price, rate, similarity_name):
        user_ID = restaurant_full.Res_User_Id.unique().tolist()
        print(f"most_similar_user - Value of rate: {rate}")  # Add this line to print out the value of rate
        if(similarity_name == "pearson"):
            similarity_score = [(pearson_correlation_score(res_user_1, user_i, location, max_price, rate),user_i)  for user_i in user_ID[0:1500] if user_i != res_user_1] #danh sách user quá nhiều nên tình chỉ tính tên dánh sách có 50 users
        if(similarity_name == "cosine"):
            similarity_score = [(distance_similarity_score(res_user_1, user_i, location, max_price, rate),user_i)  for user_i in user_ID[0:1500] if user_i != res_user_1]
        similarity_score.sort() #tăng dần
        similarity_score.reverse() #tăng dần
        return similarity_score[:number_of_user] # có thể thay đổi số lượng lân cận


    #lấy ra danh sách khuyến nghị từ top populars
    def get_recommendation(res_user_id, number_of_user, location, similarity_name, max_price, rate):
        total, similarity_sum, ranking = {}, {}, []
        list_user_popular = most_similar_user(res_user_id, number_of_user, location, max_price, rate, similarity_name)
        
        for pearson, user in list_user_popular:
            score = pearson
            for res_id in get_res_ids(user, location):
                if res_id not in get_res_ids(res_user_id, location):
                    if get_rating(res_id, rate):
                        if get_res_price(res_id, max_price):
                            if res_id not in total:
                                total[res_id] = []
                                similarity_sum[res_id] = 0
                            total[res_id].extend(get_rating(user, res_id))  # Extend the list of ratings
                            similarity_sum[res_id] += score
        
        for res_id, ratings in total.items():
            if similarity_sum[res_id] == 0:
                ranking.append((8, res_id))
            else:
                average_rating = sum(ratings) / len(ratings)  # Calculate the average rating
                ranking.append((average_rating, res_id))
        
        ranking.sort()
        ranking.reverse()
        
        recommendations = [(get_res_title(res_id), score, get_address(res_id), get_description(res_id), get_res_price(res_id, max_price), get_rating(res_id, rate)) for score, res_id in ranking]
        
        return recommendations[:number_of_user]




    # Hàm demo content based
    def recommendations_content(res_user_id):
        c = res_with_id
        vectorizer = TfidfVectorizer(max_features= 4500)
        overview_matrix = vectorizer.fit_transform(c['Res_Descriptions'])
        overview_matrix_1 = vectorizer.fit_transform(restaurant_full['Res_Descriptions'])
        cosine_sim = linear_kernel(overview_matrix_1, overview_matrix)
        for i in range(len(restaurant_full['Res_User_Id'])):
            if (restaurant_full['Res_User_Id'][i] == res_user_id):
                print(f"recommendations_content | res_user_id = {res_user_id}")
                sim_scores = list(enumerate(cosine_sim[i]))
            # Sắp xếp phim dựa trên điểm số tương tự
                sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            # Lấy điểm của 10 phim giống nhất
                sim_scores = sim_scores[1:11]
                res_indices = [i[0] for i in sim_scores]
                print(f"recommendations_content | res_indices = {res_indices}")
        # b = a['Res_'].iloc[res_indices]
                c['Res_Name'].iloc[res_indices].to_list()



    def recommend_res_based_on_description(user_description, number_of_recommendations):
        # Tạo vectorizer TF-IDF và biểu diễn văn bản người dùng
        vectorizer = TfidfVectorizer(max_features=4500)
        user_description_matrix = vectorizer.fit_transform([user_description])

        # Biểu diễn văn bản của tất cả các khách sạn
        res_description_matrix = vectorizer.transform(restaurant_full['Res_Descriptions'])

        # Tính toán độ tương đồng cosine giữa mô tả người dùng và mô tả của từng khách sạn
        cosine_similarities = linear_kernel(user_description_matrix, res_description_matrix).flatten()

        # Sắp xếp các khách sạn theo độ tương đồng và lấy ra các khách sạn tốt nhất
        res_indices = cosine_similarities.argsort()[:-number_of_recommendations-1:-1]

        recommendations = []
        for index in res_indices:
            res_id = restaurant_full.iloc[index]['Res_Id']
            res_title = get_res_title(res_id)
            res_rate = get_rating(res_id)
            res_address = get_address(res_id)
            res_description = get_description(res_id)
            res_price = get_res_price(res_id, float('inf'))  # Lấy giá của khách sạn, không giới hạn giá
            recommendations.append((res_title, res_rate, res_address, res_description, res_price))

        return recommendations



    def run3():

        with st.sidebar:
            add_userID = st.number_input('Enter User Id:')
            print(f"add_userID: {add_userID}")
            with st.form('form1'):
                if add_userID <= 100000:
                    add_password = st.text_input('Enter password:')
                st.form_submit_button('Enter')
        time.sleep(1)
        add_selectbox = st.sidebar.selectbox(
            "How would you like to be contacted?",
            ("Email", "Home phone", "Mobile phone")
        )
        
        #########################################################
        location = st.text_input("Enter the place: ")
        if location:
            st.write('Res_City: ', location)
        elif add_userID:
            st.write('Res_User_Id: ', add_userID)
        
        max_price = st.slider("Enter maximum price:", 0, 1000000, step=10000)
        if max_price:
            st.write('Price: ', max_price)
        elif add_userID:
            st.write('Res_User_Id: ', add_userID)
            
        rate = st.selectbox("Enter the rating: ", ('1','2','3','4','5','6','7','8','9','10'))
        rate = float(rate)
        if rate:
            st.write('Res_Rating: ', rate)
        elif add_userID:
            st.write('Res_User_Id: ', add_userID)

        

        click = st.button('Search')
        
        list_recommendations_content = []
        
        if add_userID:
            ratet_time = time.time()
            list_recommendations_content = recommendations_content(add_userID)
            end_time = time.time()
            elapsed_time = end_time - ratet_time
            # Display elapsed time
            st.write(f"Time taken for recommendations: {elapsed_time:.2f} seconds")
            if not list_recommendations_content:
                st.write(f"No Results!")
            for i in range(len(list_recommendations_content[0])):
                if location:
                    break
                col1, col2 = st.columns(2)
                with col1:
                    st.image(f'res-{i}.jpg', caption = '')
                with col2:
                    st.markdown(f'**Name Res**: {list_recommendations_content[0][i]}')
                    st.markdown(f'**Rating**: {list_recommendations_content[1][i]}')
                    st.markdown(f'**Address**: {list_recommendations_content[2][i]}')
                    st.markdown(f'**Description**: {list_recommendations_content[3][i][:200]}...')
                    st.markdown(f'**Price**: {list_recommendations_content[4][i]}')
        else:
            if click:
                ratet_time = time.time()
                list_recommen = get_recommendation(add_userID, 10, location, 'cosine', max_price, float(rate))
                end_time = time.time()
                elapsed_time = end_time - ratet_time
                # Display elapsed time
                st.write(f"Time taken for recommendations: {elapsed_time:.2f} seconds")
                if not list_recommen:
                    st.write(f"No Results!")
                for i in range(len(list_recommen)):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(f'res-{i}.jpg', caption='')
                    with col2:
                        st.markdown(f'**Name Res**: {list_recommen[i][0]}')
                        st.markdown(f'**Rating**: {list_recommen[i][1]}')
                        st.markdown(f'**Address**: {list_recommen[i][2]}')
                        st.markdown(f'**Description**: {list_recommen[i][3][:200]}...')
                        
                        unique_prices = set(list_recommen[i][4])  # Loại bỏ các giá trị trùng lặp
                        price_str = ", ".join(map(str, unique_prices))  # Chuyển danh sách thành chuỗi
                        st.markdown(f'**Price**: {price_str}')
                        
                


    if __name__=="__main__":
        run3()


#---------------------------------------------------------------------------------------------
page_names_to_funcs = {
    "About_us": About_us,
    "Chatbot": Chatbot,
    "Travel Destination": Recommend_Destination,
    "Hotel": Recommend_Hotel,
    "Restaurant": Recommend_Restaurant
}

selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()

