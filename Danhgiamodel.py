import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity

# Load data
hotel_full = pd.read_csv('hotel_full.csv')
hotel_with_id = pd.read_csv('hotel_with_id.csv')

# Define functions
def get_rating(hotel_user_id, hotel_id):
    mask = (hotel_full.Hotel_User_Id == hotel_user_id) & (hotel_full.Hotel_Id == hotel_id)
    return hotel_full.loc[mask, 'Hotel_User_Rating'].tolist()

def get_star(hotel_id, star):
    mask = (hotel_full.Hotel_Id == hotel_id) & (hotel_full.Hotel_Star == star)
    return hotel_full.loc[mask, 'Hotel_Star'].tolist()

def get_name_hotel(hotel_user_id, location):
    mask = (hotel_full.Hotel_User_Id == hotel_user_id) & (hotel_full.Hotel_City == location)
    return hotel_full.loc[mask, 'Hotel_Name'].tolist()

def get_hotel_ids(hotel_user_id, location):
    mask = (hotel_full.Hotel_User_Id == hotel_user_id) & (hotel_full.Hotel_City == location)
    return hotel_full.loc[mask, 'Hotel_Id'].tolist()

def get_hotel_price(hotel_id, max_price):
    mask = (hotel_full.Hotel_Id == hotel_id) & (hotel_full.Hotel_Price <= max_price)
    return hotel_full.loc[mask, 'Hotel_Price'].tolist()

def get_hotel_title(hotel_id):
    mask = (hotel_full.Hotel_Id == hotel_id)
    return hotel_full.loc[mask, 'Hotel_Name'].iloc[0]

def get_description(hotel_id):
    mask = (hotel_full.Hotel_Id == hotel_id)
    return hotel_full.loc[mask, 'Hotel_Descriptions'].iloc[0]

def get_address(hotel_id):
    mask = (hotel_full.Hotel_Id == hotel_id)
    return hotel_full.loc[mask, 'Hotel_Address'].iloc[0]

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
    avg_rating_sum_1 = np.mean([get_rating(hotel_user_1, i) for i in both_vatch_count])  # rating trung bình user1
    avg_rating_sum_2 = np.mean([get_rating(hotel_user_2, i) for i in both_vatch_count])  # rating trung bình user2
    tu = np.sum([(get_rating(hotel_user_1, i)- avg_rating_sum_1)*(get_rating(hotel_user_2, i)- avg_rating_sum_2) for i in both_vatch_count])
    mau_1 = np.sqrt(np.sum([pow((get_rating(hotel_user_1, i) - avg_rating_sum_1), 2) for i in both_vatch_count]))
    mau_2 = np.sqrt(np.sum([pow((get_rating(hotel_user_2, i) - avg_rating_sum_2), 2) for i in both_vatch_count]))
    mau = mau_1 * mau_2
    if mau == 0:
        return 0
    return tu / mau

def distance_similarity_score(hotel_user_1, hotel_user_2, location, max_price, star):
    both_watch_count = 0
    mask_1 = (hotel_full.Hotel_User_Id == hotel_user_1) & (hotel_full.Hotel_City == location) & (hotel_full.Hotel_Price <= max_price) & (hotel_full.Hotel_Star == star)
    mask_2 = (hotel_full.Hotel_User_Id == hotel_user_2) & (hotel_full.Hotel_City == location) & (hotel_full.Hotel_Price <= max_price) & (hotel_full.Hotel_Star == star)
    list_hotel_user_1 = hotel_full