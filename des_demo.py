import pandas as pd
import time
import numpy as np
import streamlit as st


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

des_full = pd.read_csv('des_full.csv')
des_with_id = pd.read_csv('des_with_id.csv')
dess_merg = pd.read_csv('des_rating.csv')
dess = pd.read_csv('userdes.csv')


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
    overview_matrix = vectorizer.fit_transform(b['des_Descriptions'])
    overview_matrix_1 = vectorizer.fit_transform(des_full['des_Descriptions'])
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
    des_description_matrix = vectorizer.transform(des_full['des_Descriptions'])

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



def run():
    
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
    run()