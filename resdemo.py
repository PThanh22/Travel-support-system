import pandas as pd
import time
import numpy as np
import streamlit as st


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

restaurant_full = pd.read_csv('restaurant_full.csv')
res_with_id = pd.read_csv('res_with_id.csv')
res_rating = pd.read_csv('res_rating.csv')
userres = pd.read_csv('userres.csv')


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
    a = res_with_id
    vectorizer = TfidfVectorizer(max_features= 4500)
    overview_matrix = vectorizer.fit_transform(a['Res_Descriptions'])
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
            a['Res_Name'].iloc[res_indices].to_list()



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



def run():
    
    st.set_page_config(
        page_title="Demo",
        page_icon="👋",
    )
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
    st.title("res Recommendation System")
    st.header("Welcome to Demo")
    
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
        
    rate = st.selectbox("Enter the Star: ", ('1','2','3','4','5','6','7','8','9','10'))
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
    run()