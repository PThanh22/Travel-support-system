
import pandas as pd
import time
import numpy as np
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

res_full = pd.read_csv('restaurant_full.csv')
res_with_id = pd.read_csv('res_with_id.csv')
res_rating = pd.read_csv('res_rating.csv')
userres = pd.read_csv('userres.csv')


def get_res_rating(res_user_id, res_id):
    res_mask = (res_full.Res_User_Id == res_user_id) & (res_full.Res_Id == res_id)
    return (res_full.loc[res_mask, 'Res_User_Rating'].iloc[0])


def get_name_res(res_user_id, res_location):
   res_mask = (res_full.Res_User_Id == res_user_id) & (res_full.Res_City == res_location)
   return (res_full.loc[res_mask, 'Res_Name'].tolist())


def get_res_ids(res_user_id, res_location):
    res_mask = (res_full.Res_User_Id == res_user_id) & (res_full.Res_City == res_location)
    return (res_full.loc[res_mask, 'Res_Id'].tolist())


def get_res_title(res_id):
    res_mask = (res_full.Res_Id == res_id)
    return (res_full.loc[res_mask, 'Res_Name'].iloc[0])


def get_res_description(res_id):
    res_mask = (res_full.Res_Id == res_id)
    return (res_full.loc[res_mask, 'Res_Descriptions'].iloc[0])


def get_res_address(res_id):
    res_mask = (res_full.Res_Id == res_id)
    return (res_full.loc[res_mask, 'Res_Address'].iloc[0])


def res_pearson_correlation_score(res_user_1, res_user_2, res_location):
    res_both_vatch_count= []
    res_mask_1 = (res_full.Res_User_Id == res_user_1) & (res_full.Res_City == res_location)
    res_mask_2 = (res_full.Res_User_Id == res_user_2) & (res_full.Res_City == res_location)
    lis_res_user_1 = res_full.loc[res_mask_1, 'Res_Id'].to_list()
    list_res_user_2 = res_full.loc[res_mask_2, 'Res_Id'].to_list()
    for res_element in lis_res_user_1:
        if res_element in list_res_user_2:
            res_both_vatch_count.append(res_element)
        if (len(res_both_vatch_count)==0):
            return 0
    res_avg_rating_sum_1 = np.lib.function_base.average([get_res_rating(res_user_1, i) for i in res_both_vatch_count])# rating trung bình user1
    res_avg_rating_sum_2 = np.lib.function_base.average([get_res_rating(res_user_2, i) for i in res_both_vatch_count])# rating trung bình user2
    tu_res = sum([(get_res_rating(res_user_1, i)- res_avg_rating_sum_1)*(get_res_rating(res_user_2, i)- res_avg_rating_sum_2) for i in res_both_vatch_count])
    mau_res_1 = np.sqrt(sum([pow((get_res_rating(res_user_1, i) - res_avg_rating_sum_1), 2) for i in res_both_vatch_count]))
    mau_res_2 = np.sqrt(sum([pow((get_res_rating(res_user_2, i) - res_avg_rating_sum_2), 2) for i in res_both_vatch_count]))
    mau_res = mau_res_1 * mau_res_2
    if mau_res == 0:
        return 0
    return tu_res / mau_res


def res_distance_similarity_score(res_user_1, res_user_2, res_location):
    res_both_watch_count = 0
    res_mask_1 = (res_full.Res_User_Id == res_user_1) & (res_full.Res_City == res_location)
    res_mask_2 = (res_full.Res_User_Id == res_user_2) & (res_full.Res_City == res_location)
    lis_res_user_1 = res_full.loc[res_mask_1, 'Res_Id'].to_list()
    list_res_user_2 = res_full.loc[res_mask_2, 'Res_Id'].to_list()
    for res_element in lis_res_user_1:
        if res_element in list_res_user_2:
            res_both_watch_count += 1
    if res_both_watch_count == 0 :
        return 0
    res_user_rating_1, res_user_rating_2 = [], []
    for res_element in lis_res_user_1:
        if res_element in list_res_user_2:
            res_user_rating_1.append(get_res_rating(res_user_1, res_element))
            res_user_rating_2.append(get_res_rating(res_user_2, res_element))
    print(f"res_distance_similarity_score-res_user_rating_1: {res_user_rating_1}")
    print(f"res_distance_similarity_score-res_user_rating_2: {res_user_rating_2}")
    return np.dot(res_user_rating_1, res_user_rating_2) / (np.linalg.norm(res_user_rating_1) * np.linalg.norm(res_user_rating_2))


def res_most_similar_user(res_user_1, res_number_of_user, res_location, res_similarity_name):
    userres_ID = res_full.Res_User_Id.unique().tolist()
    print(f"res_most_similar_user-len: {len(userres_ID)}")
    if(res_similarity_name == "res_pearson"):
        res_similarity_score = [(res_pearson_correlation_score(res_user_1, userres_i, res_location),userres_i)  for userres_i in userres_ID[0:1500] if userres_i != res_user_1] #danh sách user quá nhiều nên tình chỉ tính tên dánh sách có 50 users
    if(res_similarity_name == "cosine"):
        res_similarity_score = [(res_distance_similarity_score(res_user_1, userres_i, res_location),userres_i)  for userres_i in userres_ID[0:1500] if userres_i != res_user_1]
    res_similarity_score.sort() #tăng dần
    res_similarity_score.reverse() #tăng dần
    return res_similarity_score[:res_number_of_user] # có thể thay đổi số lượng lân cận


#lấy ra danh sách khuyến nghị từ top populars
def res_get_recommendation(res_user_id, res_number_of_user, res_location, res_similarity_name):# lấy ra danh sách khuyến nghị của n người tương đồng phim có rating cao để khuyến nghị cho userid dựa vào độ đo
    # user_ids = res_full.userId.unique().tolist()
    res_total, res_similarity_sum, res_ranking = {}, {}, []
    res_list_user_popular = res_most_similar_user(res_user_id, res_number_of_user, res_location, res_similarity_name)
    # Iterating over subset of res_user ids.
    for res_pearson, res_user in res_list_user_popular:
        res_score = res_pearson
        for res_id in get_res_ids(res_user, res_location): #-> dánh sách các id movie đã xem bởi res_user khác và khởi tạo giá trị =0
          if res_id not in get_res_ids(res_user_id, res_location):
            res_total[res_id] = 0
            res_similarity_sum[res_id] = 0
        for res_id in get_res_ids(res_user, res_location): #-> dánh sách các id movie đã xem bởi res_user khác
          if res_id not in get_res_ids(res_user_id, res_location):
            res_total[res_id] += get_res_rating(res_user, res_id) * res_score
            res_similarity_sum[res_id] += res_score
    for res_id,res_tot in res_total.items():
        if res_similarity_sum[res_id] == 0:
            res_ranking.append((8,res_id))
        else:
            resrating = res_tot/(res_similarity_sum[res_id])
            res_ranking.append((resrating,res_id))
    res_ranking.sort() # sắp xếp tăng dần
    res_ranking.reverse() # đẩo chiều cho giảm dần
    res_recommendations = [(get_res_title(res_id), res_score, get_res_address(res_id), get_res_description(res_id)) for res_score, res_id in res_ranking]
    return res_recommendations[:res_number_of_user]



# Hàm demo content based
def res_recommendations_content(res_user_id):
    c = res_with_id
    vectorizer = TfidfVectorizer(max_features= 4500)
    overview_matrix = vectorizer.fit_transform(c['Res_Descriptions'])
    overview_matrix_1 = vectorizer.fit_transform(res_full['Res_Descriptions'])
    res_cosine_sim = linear_kernel(overview_matrix_1, overview_matrix)
    for i in range(len(res_full['Res_User_Id'])):
        if (res_full['Res_User_Id'][i] == res_user_id):
            print(f"recommendations_content | res_user_id = {res_user_id}")
            res_sim_scores = list(enumerate(res_cosine_sim[i]))
          # Sắp xếp phim dựa trên điểm số tương tự
            res_sim_scores = sorted(res_sim_scores, key=lambda x: x[1], reverse=True)
          # Lấy điểm của 10 phim giống nhất
            res_sim_scores = res_sim_scores[1:11]
            res_indices = [i[0] for i in res_sim_scores]
            print(f"res_recommendations_content | res_indices = {res_indices}")
      # b = c['Hotel_'].iloc[res_indices]
            c['Res_Name'].iloc[res_indices].to_list()
    return [c['Res_Name'].iloc[res_indices].to_list(), c['Res_Rating'].iloc[res_indices].to_list(), c['Res_Address'].iloc[res_indices].to_list(), c['Res_Descriptions'].iloc[res_indices].to_list()]


def run():
    # results = res_get_recommendation(1187, 10, 'Huế', 'cosine')
    # print(f"reasults: {results}")
    # list_recommendations_content = res_recommendations_content(1187)
    # print(f"list_recommendations_content: {list_recommendations_content}")

    st.set_page_config(
        page_title="Res Demo",
        page_icon="👋",
    )
    #st.sidebar.success("Select a demo above.")
    # Using "with" notation
    with st.sidebar:
        res_add_userID = st.number_input('Enter User Id:')
        print(f"res_add_userID: {res_add_userID}")
        with st.form('form1'):
            if res_add_userID <= 100000:
                add_password = st.text_input('Enter password:')
            st.form_submit_button('Enter')
    time.sleep(1)
    add_selectbox = st.sidebar.selectbox(
        "How would you like to be contacted?",
        ("Email", "Home phone", "Mobile phone")
    )
    st.title("Restaurant Recommendation System")
    st.header("Welcome to Res Demo")
    #ten = st.number_input("Enter your userID: ")
    #st.write('Res_User_Id: ',ten)
    res_location = st.text_input("Enter the place: ")
    if res_location:
        st.write('Res_City: ', res_location)
    elif res_add_userID:
        st.write('Res_User_Id: ', res_add_userID)

    click = st.button('Search')
    if res_add_userID:
        start_time = time.time()
        list_recommendations_content = res_recommendations_content(res_add_userID)
        end_time = time.time()
        elapsed_time = end_time - start_time
        # Display elapsed time
        st.write(f"Time taken for recommendations: {elapsed_time:.2f} seconds")
        if not list_recommendations_content:
            st.write(f"No Results!")
        for i in range(len(list_recommendations_content[0])):
            if res_location:
                break
            col1, col2 = st.columns(2)
            with col1:
                st.image(f'res-{i}.jpg', caption = '')
            with col2:
                st.markdown(f'**Name Restaurant**: {list_recommendations_content[0][i]}')
                st.markdown(f'**Rating**: {list_recommendations_content[1][i]}')
                st.markdown(f'**Address**: {list_recommendations_content[2][i]}')
                st.markdown(f'**Description**: {list_recommendations_content[3][i][:200]}...')

    if click:
        start_time = time.time()
        list_recommen = res_get_recommendation(add_userID, 10, res_location, 'cosine')
        end_time = time.time()
        elapsed_time = end_time - start_time
        # Display elapsed time
        st.write(f"Time taken for recommendations: {elapsed_time:.2f} seconds")
        if not list_recommen:
            st.write(f"No Results!")
        for i in range(len(list_recommen)):
            col1,col2 = st.columns(2)
            with col1:
                st.image(f'res-{i}.jpg', caption = '')
            with col2:
                st.markdown(f'**Name Restaurant**: {list_recommen[i][0]}')
                st.markdown(f'**Rating**: {list_recommen[i][1]}')
                st.markdown(f'**Address**: {list_recommen[i][2]}')
                st.markdown(f'**Description**: {list_recommen[i][3][:200]}...')
                
                # st.markdown('Streamlit is **_really_ cool**.')


if __name__=="__main__":
    run()

