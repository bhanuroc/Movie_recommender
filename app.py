import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from flask import Flask, request, jsonify, render_template

movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")
final_dataset = ratings.pivot(index='movieId',columns='userId',values='rating')
final_dataset.fillna(0,inplace=True)
no_user_voted = ratings.groupby('movieId')['rating'].agg('count')
no_movies_voted = ratings.groupby('userId')['rating'].agg('count')
final_dataset=final_dataset.loc[no_user_voted[no_user_voted>10].index,:]
final_dataset=final_dataset.loc[:,no_movies_voted[no_movies_voted>50].index]
csr_data = csr_matrix(final_dataset.values)
final_dataset.reset_index(inplace=True)
knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=10, n_jobs=-1)
knn.fit(csr_data)
def get_movie_recommendation(movie_name):
    n_movies_to_reccomend = 10
    movie_list = movies[movies['title'].str.contains(movie_name)]
    if len(movie_list)>0:
        movie_idx= movie_list.iloc[0]['movieId']
        if np.sum(movie_idx)==0:
            return "No movies found. Please check your input"
        temp = final_dataset[final_dataset['movieId'] == movie_idx].index
        if temp.shape[0]<1 :
            return "Your movie was not included in our data. Try another one"
        movie_idx=temp[0]
        distances , indices = knn.kneighbors(csr_data[movie_idx],n_neighbors=n_movies_to_reccomend)    
        rec_movie_indices1=sorted(list(indices.squeeze().tolist()))
        recommend_frame = []
        i=0
        for val in rec_movie_indices1:
            movie_idx = final_dataset.iloc[val]['movieId']
            idx = movies[movies['movieId'] == movie_idx].index
            recommend_frame.append(movies.iloc[idx]['title'].values[0])
            i=i+1
        return recommend_frame
    else:
        return "No movies found. Please check your input"

app = Flask(__name__)
@app.route('/')
def home():
    return render_template('front.html')
@app.route('/recommend',methods=['POST'])
def recommend():
    s=request.form['movie']
    v=""
    f=1
    for j in range(0,len(s)):
        if s[j]=='(':
            f=0
        elif s[j]==')':
            f=1
        else:
            if f:
                v+=s[j]
    output=get_movie_recommendation(v)
    print(len(output))
    if len(output)!=10 :
        return render_template('front.html',not_found=output)
    else:
        return render_template('back.html',not_found="",predict=output[0],predict1=output[1],predict2=output[2],predict3=output[3],predict4=output[4],predict5=output[5],predict6=output[6],predict7=output[7],predict8=output[8],predict9=output[9])

if __name__== '__main__':

    app.run(debug=True)


