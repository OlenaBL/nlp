from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/',methods=['POST'])
def predict():
	df = pd.read_csv('https://res.cloudinary.com/olena/raw/upload/v1621734607/csv/sentiment_2.csv')
	df['sentiment'] = df['sentiment'].replace({'positive':2, 'negative':0})
	#stop = set(stopwords.words('english')) - set(['not', 'no', 'nor', "don't", 'very', 'down', 'most', 'over', 'such'])
	#vectorizer = TfidfVectorizer(use_idf=True, lowercase=True, strip_accents='ascii', stop_words=stop)
	from sklearn.feature_extraction.text import TfidfVectorizer
	
	vectorizer = TfidfVectorizer(use_idf=True, lowercase=True, strip_accents='ascii')
	y = df['sentiment']
	X = vectorizer.fit_transform(df['text'])
	
	from sklearn.model_selection import train_test_split
	#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
	X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
	
	from sklearn.naive_bayes import MultinomialNB
	clf = MultinomialNB()
	clf.fit(X_train, y_train)
	#clf.score(X_test,y_test)

	if request.method == 'POST':
		message = request.form['text']
		data = [message]
		vect = vectorizer.transform(data).toarray()
		my_prediction = clf.predict(vect)
	return render_template('result.html',prediction = my_prediction)

if __name__ == '__main__':
	app.run(debug=True)
