from flask import Flask,render_template,request
from recommender import recommend_random,recommend_nmf, recommend_cs
from utils import movies, get_poster_links
app = Flask(__name__)

@app.route('/')
def hello():
    print(movies)
    return render_template('index.html', name="Recommender", movies=movies.title.to_list())

@app.route('/movies')
def recommendation():
    print(request.args)

    if request.args.get('option') =='Random':
        recommendation_list = recommend_random()
        print(recommendation_list)

        titles = request.args.getlist('title')
        ratings = request.args.getlist('rating')
        poster_links = get_poster_links(titles)


        query = dict(zip(titles,ratings))

        for movie in query:
            query[movie] = float(query[movie])
        
        print(query)
        poster_links = get_poster_links(recommendation_list)

        return render_template('recommend.html', recommendation=recommendation_list, poster_links=poster_links)
    
    if request.args.get('option')=='NMF':

        titles = request.args.getlist('title')
        ratings = request.args.getlist('rating')
        print(titles,ratings)
        query = dict(zip(titles,ratings))

        for movie in query:
            query[movie] = float(query[movie])
        
        print(query)
        recommendation_list = recommend_nmf(query)
        poster_links = get_poster_links(recommendation_list)
        

        return render_template('recommend.html', recommendation=recommendation_list, poster_links=poster_links)
    
    # else:
    #     recommendation_list = recommend_neighborhood()
    #     return f"Not defined yet"
    
    if request.args.get('option')=='Cosine Similarity':
        titles = request.args.getlist('title')
        ratings = request.args.getlist('rating')
        print(titles,ratings)
        query = dict(zip(titles,ratings))
        print(query)
        for movie in query:
            query[movie] = float(query[movie])
        
        print(query)
        recommendation_list = recommend_cs(query, titles, ratings)
        print(recommendation_list)
        poster_links = get_poster_links(recommendation_list)
        return render_template('recommend.html', recommendation=recommendation_list, poster_links=poster_links)

if __name__=='__main__':
    app.run(port=5000,debug=True)
    
