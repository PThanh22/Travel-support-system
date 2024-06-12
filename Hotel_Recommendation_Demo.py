import pandas as pd
import time
import numpy as np
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

ratings = pd.read_csv('hotels_users_ratings.csv')
hotels_merg = pd.read_csv('data_merge_full_6471.csv')
hotels = pd.read_csv('test.csv')


def get_rating(user_id, hotel_id):
    mask = (ratings.UserID == user_id) & (ratings.HotelID == hotel_id)
    return (ratings.loc[mask, 'Rating'].iloc[0])


def get_name_hotel(user_id, location):
   mask = (ratings.UserID == user_id) & (ratings.Location == location)
   return (ratings.loc[mask, 'Name Hotel'].tolist())


def get_hotel_ids(user_id, location):
    mask = (ratings.UserID == user_id) & (ratings.Location == location)
    return (ratings.loc[mask, 'HotelID'].tolist())


def get_hotel_title(hotel_id):
    mask = (ratings.HotelID == hotel_id)
    return (ratings.loc[mask, 'Name Hotel'].iloc[0])


def get_url(hotel_id):
    mask = (ratings.HotelID == hotel_id)
    return (ratings.loc[mask, 'URL Hotel'].iloc[0])


def get_description(hotel_id):
    mask = (ratings.HotelID == hotel_id)
    return (ratings.loc[mask, 'Descriptions'].iloc[0])


def get_address(hotel_id):
    mask = (ratings.HotelID == hotel_id)
    return (ratings.loc[mask, 'Address'].iloc[0])


def pearson_correlation_score(user_1, user_2, location):
    both_vatch_count= []
    mask_1 = (ratings.UserID == user_1) & (ratings.Location == location)
    mask_2 = (ratings.UserID == user_2) & (ratings.Location == location)
    list_hotel_user_1 = ratings.loc[mask_1, 'HotelID'].to_list()
    list_hotel_user_2 = ratings.loc[mask_2, 'HotelID'].to_list()
    for element in list_hotel_user_1:
        if element in list_hotel_user_2:
            both_vatch_count.append(element)
        if (len(both_vatch_count)==0):
            return 0
    avg_rating_sum_1 = np.lib.function_base.average([get_rating(user_1, i) for i in both_vatch_count])# rating trung bình user1
    avg_rating_sum_2 = np.lib.function_base.average([get_rating(user_2, i) for i in both_vatch_count])# rating trung bình user2
    tu = sum([(get_rating(user_1, i)- avg_rating_sum_1)*(get_rating(user_2, i)- avg_rating_sum_2) for i in both_vatch_count])
    mau_1 = np.sqrt(sum([pow((get_rating(user_1, i) - avg_rating_sum_1), 2) for i in both_vatch_count]))
    mau_2 = np.sqrt(sum([pow((get_rating(user_2, i) - avg_rating_sum_2), 2) for i in both_vatch_count]))
    mau = mau_1 * mau_2
    if mau == 0:
        return 0
    return tu / mau


def distance_similarity_score(user_1, user_2, location):
    both_watch_count = 0
    mask_1 = (ratings.UserID == user_1) & (ratings.Location == location)
    mask_2 = (ratings.UserID == user_2) & (ratings.Location == location)
    list_hotel_user_1 = ratings.loc[mask_1, 'HotelID'].to_list()
    list_hotel_user_2 = ratings.loc[mask_2, 'HotelID'].to_list()
    for element in list_hotel_user_1:
        if element in list_hotel_user_2:
            both_watch_count += 1
    if both_watch_count == 0 :
        return 0
    rating_1, rating_2 = [], []
    for element in list_hotel_user_1:
        if element in list_hotel_user_2:
            rating_1.append(get_rating(user_1, element))
            rating_2.append(get_rating(user_2, element))
    print(f"distance_similarity_score-rating_1: {rating_1}")
    print(f"distance_similarity_score-rating_2: {rating_2}")
    return np.dot(rating_1, rating_2) / (np.linalg.norm(rating_1) * np.linalg.norm(rating_2))


def most_similar_user(user_1, number_of_user, location, similarity_name):
    user_ID = ratings.UserID.unique().tolist()
    print(f"most_similar_user-len: {len(user_ID)}")
    if(similarity_name == "pearson"):
        similarity_score = [(pearson_correlation_score(user_1, user_i, location),user_i)  for user_i in user_ID[0:1500] if user_i != user_1] #danh sách user quá nhiều nên tình chỉ tính tên dánh sách có 50 users
    if(similarity_name == "cosine"):
        similarity_score = [(distance_similarity_score(user_1, user_i, location),user_i)  for user_i in user_ID[0:1500] if user_i != user_1]
    similarity_score.sort() #tăng dần
    similarity_score.reverse() #tăng dần
    return similarity_score[:number_of_user] # có thể thay đổi số lượng lân cận


#lấy ra danh sách khuyến nghị từ top populars
def get_recommendation(user_id, number_of_user, location, similarity_name):# lấy ra danh sách khuyến nghị của n người tương đồng phim có rating cao để khuyến nghị cho userid dựa vào độ đo
    # user_ids = ratings.userId.unique().tolist()
    total, similariy_sum, ranking = {}, {}, []
    list_user_popular = most_similar_user(user_id, number_of_user, location, similarity_name)
    # Iterating over subset of user ids.
    for pearson, user in list_user_popular:
        score = pearson
        for hotel_id in get_hotel_ids(user, location): #-> dánh sách các id movie đã xem bởi user khác và khởi tạo giá trị =0
            if hotel_id not in get_hotel_ids(user_id, location):
                total[hotel_id] = 0
                similariy_sum[hotel_id] = 0
        for hotel_id in get_hotel_ids(user, location): #-> dánh sách các id movie đã xem bởi user khác
            if hotel_id not in get_hotel_ids(user_id, location):
                total[hotel_id] += get_rating(user, hotel_id) * score
                similariy_sum[hotel_id] += score
    for hotel_id,tot in total.items():
        if similariy_sum[hotel_id] == 0:
            ranking.append((8,hotel_id))
        else:
            rating = tot/(similariy_sum[hotel_id])
            ranking.append((rating,hotel_id))
    ranking.sort() # sắp xếp tăng dần
    ranking.reverse() # đẩo chiều cho giảm dần
    recommendations = [(get_hotel_title(hotel_id), score, get_address(hotel_id), get_description(hotel_id), get_url(hotel_id)) for score, hotel_id in ranking]
    return recommendations[:number_of_user]


# Hàm demo content based
def recommendations_content(user_id):
    a = hotels#[(hotels.Location == 'Đà Lạt')]
    vectorizer = TfidfVectorizer(max_features= 4500)
    overview_matrix = vectorizer.fit_transform(a['Descriptions'])
    overview_matrix_1 = vectorizer.fit_transform(hotels_merg['Descriptions'])
    cosine_sim = linear_kernel(overview_matrix_1, overview_matrix)
    for i in range(len(hotels_merg['UserID'])):
        if (hotels_merg['UserID'][i] == user_id):
            print(f"recommendations_content | user_id = {user_id}")
            sim_scores = list(enumerate(cosine_sim[i]))
          # Sắp xếp phim dựa trên điểm số tương tự
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
          # Lấy điểm của 10 phim giống nhất
            sim_scores = sim_scores[1:11]
            hotel_indices = [i[0] for i in sim_scores]
            print(f"recommendations_content | hotel_indices = {hotel_indices}")
      # b = a['Name Hotel'].iloc[hotel_indices]
            a['Name Hotel'].iloc[hotel_indices].to_list()
    return [a['Name Hotel'].iloc[hotel_indices].to_list(), a['Rating'].iloc[hotel_indices].to_list(), a['Address'].iloc[hotel_indices].to_list(), a['Descriptions'].iloc[hotel_indices].to_list(), a['URL Hotel'].iloc[hotel_indices].to_list()]


def run():

    st.set_page_config(
        page_title="Hello",
        page_icon="👋",
    )
    #st.sidebar.success("Select a demo above.")
    # Using "with" notation
    with st.sidebar:
        add_userID = st.number_input('Enter User Id:')
        print(f"add_userID: {add_userID}")
        with st.form('form1'):
            if add_userID <= 6471:
                add_password = st.text_input('Enter password:')
            st.form_submit_button('Enter')
    time.sleep(1)
    add_selectbox = st.sidebar.selectbox(
        "How would you like to be contacted?",
        ("Email", "Home phone", "Mobile phone")
    )
    st.title("Hotels Recommendation System")
    st.header("Welcome to Demo")
    #ten = st.number_input("Enter your userID: ")
    #st.write('UserID: ',ten)
    
    # input đầu
    location = st.text_input("Enter the place: ")
    
    if location:
        st.write('Location: ', location)
    elif add_userID:
        st.write('UserID: ', add_userID)
    click = st.button('Search')
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
                st.markdown(f'[Go to Website]({list_recommendations_content[4][i]})')

    if click:
        start_time = time.time()
        list_recommen = get_recommendation(add_userID, 10, location, 'cosine')
        end_time = time.time()
        elapsed_time = end_time - start_time
        # Display elapsed time
        st.write(f"Time taken for recommendations: {elapsed_time:.2f} seconds")
        if not list_recommen:
            st.write(f"No Results!")
        for i in range(len(list_recommen)):
            col1,col2 = st.columns(2)
            with col1:
                st.image(f'hotel-{i}.jpg', caption = '')
            with col2:
                st.markdown(f'**Name Hotel**: {list_recommen[i][0]}')
                st.markdown(f'**Rating**: {list_recommen[i][1]}')
                st.markdown(f'**Address**: {list_recommen[i][2]}')
                st.markdown(f'**Description**: {list_recommen[i][3][:200]}...')
                st.markdown(f'[Go to Website]({list_recommen[i][4]})')
                #st.write('URL Hotel: ', list_recommen[i][1])
                # st.markdown('Streamlit is **_really_ cool**.')


if __name__=="__main__":
    run()
    
    

