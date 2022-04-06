import os
# print(os.path.abspath(__file__))
# print(os.path.dirname(os.path.realpath(__file__)))
# print(os.listdir(os.getcwd()))
# print(os.getcwd())
os.chdir("C:\\Users\\ana23\\CWNU\\reactshopkomaster\\SWcode\\python")
# print(os.getcwd())
# print(os.listdir(os.getcwd()))

import sys
from math import *
import numpy as np
import pandas as pd
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.metrics import mean_squared_error
# # import requests, json

# url = "http://localhost:5000"
# data = {}
# headers = {}
# requests.post(url, data=json.dumps(data), headers=headers)

# 사용자 데이터
user_data = pd.read_csv('./user.csv', encoding='cp949')
# 평점 데이터
score_data = pd.read_csv('./score.csv', encoding='cp949') 
# 음식점 데이터
product_data = pd.read_csv('./product.csv', encoding='cp949')
# 비선호 가게 데이터
dislike_data = pd.read_csv('./dislike.csv')



product_data.drop('title', axis = 1, inplace = True)
user_data.drop('email', axis = 1, inplace = True)
user_data.drop('name', axis = 1, inplace = True)


user_data.rename(columns={'_id':'userId'}, inplace=True) 
product_data.rename(columns={'_id':'productId'}, inplace=True)
dislike_data.rename(columns={'productId':'dProduct'}, inplace=True) 
dislike_data.rename(columns={'userId':'dUser'}, inplace=True) 


# nan값이 있는 행 제거
real =dislike_data.dropna(subset=['dProduct']) 
real2 =score_data.dropna()

user_product = pd.merge(user_data, score_data,  on = "userId")
# print(user_product)
user_product_star = pd.merge( product_data, user_product, how='outer', on = "productId")
print(user_product_star)
user_product_star = user_product_star[['userId', 'productId', 'stars']]
user_product_star.to_csv('./user_product_star_noh', index = False, header = False)

# product_user_star = user_product_star.pivot_table('stars', index = 'productId', columns = 'userId')
user_product_star = user_product_star.pivot_table('stars', index = 'userId', columns='productId')

# product_user_star.fillna(0, inplace = True)
user_product_star.fillna(0, inplace = True)
# print(user_product_star)

user_product_star_T = user_product_star.transpose()
# print(user_product_star_T)



item_sim = cosine_similarity(user_product_star_T, user_product_star_T)
item_sim_df = pd.DataFrame(data=item_sim, index=user_product_star.columns, columns=user_product_star.columns)
# print(item_sim_df.values)




def predict_rating(ratings_arr, item_sim_arr):

    # dot : 내적을 이용한 가중합 계산
    ratings_pred = ratings_arr.dot(item_sim_arr) / np.array([np.abs(item_sim_arr).sum(axis=1)])
    return ratings_pred

ratings_pred = predict_rating(user_product_star.values, item_sim_df.values)
ratings_pred_matrix = pd.DataFrame(data=ratings_pred, index=user_product_star.index, 
                                   columns = user_product_star.columns)

def get_mse(pred, actual):
    # 평점이 있는 실제 도서만 추출
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return mean_squared_error(pred, actual)

# print('MSE : ', get_mse(ratings_pred, user_product_star.values))




def predict_rating_topsim(ratings_arr, item_sim_arr, n=3):

    # 사용자-아이템 평점 행렬 크기만큼 0으로 채운 예측 행렬 초기화
    pred = np.zeros(ratings_arr.shape)
    
    # 사용자-아이템 평점 행렬의 열 크기만큼 루프 수행.
    for col in range(ratings_arr.shape[1]):
        
        # 유사도 행렬에서 유사도가 큰 순으로 n개 데이터 행렬의 인덱스 반환
        top_n_items = [np.argsort(item_sim_arr[:, col])[:-n-1:-1]]
        
        # 개인화된 예측 평점을 계산
        for row in range(ratings_arr.shape[0]):
            pred[row, col] = item_sim_arr[col, :][top_n_items].dot(ratings_arr[row,:][top_n_items].T)
            pred[row, col] /= np.sum(np.abs(item_sim_arr[col, :][top_n_items]))
            
    return pred


ratings_pred = predict_rating_topsim(user_product_star.values, item_sim_df.values, n=3)

# print('아이템 기반 최근접 Top-5 이웃 MSE : ', get_mse(ratings_pred, user_product_star.values))

# 계산된 예측 평점 데이터를 DataFrame으로 변경
ratings_pred_matrix = pd.DataFrame(data=ratings_pred, index=user_product_star.index, columns=user_product_star.columns)
print(ratings_pred_matrix.round(3))
# print(ratings_pred_matrix)

user_rating_id = user_product_star.loc[2, :]
# print(user_rating_id[ user_rating_id > 0 ].sort_values(ascending=False)[:10])


def get_unseen_books(ratings_matrix, userId):
    # userId로 입력받은 사용자의 모든 도서 정보를 추출해 Series로 반환함.
    # 반환된 user_ratings은 도서명(title)을 인덱스로 가지는 Series 객체임.
    user_rating = ratings_matrix.loc[userId, :]
    
    # user_rating이 0보다 크면 기존에 읽은 도서. 대상 인덱스를 추출해 list 객체로 만듦.
    already_seen = user_rating[ user_rating>0 ].index.tolist()
    
    # 모든 도서명을 list 객체로 만듦.
    book_list = ratings_matrix.columns.tolist()
    
    # list comprehension으로 already_seen에 해당하는 도서는 books_list에서 제외함.
    unseen_list = [book for book in book_list if book not in already_seen]
    
    return unseen_list




def recomm_product_by_userid(userId):
    pred_df = ratings_pred_matrix
    top_n = 3
    unseen_list = get_unseen_books(user_product_star, userId)
    # 예측 평점 DataFrame에서 사용자 id 인덱스와 unseen_list로 들어온 도서명 칼럼을 추출해 가장 예측 평점이 높은 순으로 정렬함.
    recomm_products = pred_df.loc[userId, unseen_list].sort_values(ascending=False)[:top_n]
    recomm_products = pd.DataFrame(data=recomm_products.values, index=recomm_products.index, columns=['pred_score'])
    first = recomm_products.index.values[0]
    print(first)
    return recomm_products

if __name__ == '__main__':
    recomm_product_by_userid(sys.argv[1])


# i = int( userId={window.localStorage.getItem('userCount')} )      # input 자리에 userId 넣기

