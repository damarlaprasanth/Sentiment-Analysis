#!/usr/bin/env python
# coding: utf-8

from flask import Flask, render_template, url_for, redirect, request
import pickle
import re
import copy
import time
from bs4 import BeautifulSoup
from nltk.corpus import stopwords

stop = stopwords.words("english")

fname = "SentimentModel.sav"
with open(fname, 'rb') as f:
    vect, loadedModel = pickle.load(f)
    
app = Flask(__name__)

def cleaning_words(raw):
    htmlFree = BeautifulSoup(raw, "html.parser")
    letters = re.sub("[^a-zA-Z]", " ", htmlFree.get_text())
    lowCase = letters.lower()
    words = lowCase.split()
    useful = [w for w in words if not w in stop]
    return " ".join(useful)


@app.route('/', methods=['POST', 'GET'])
def hello():
    if request.method == 'GET':
        return render_template('index.html')
    elif request.method == 'POST':
        review = request.form['img']
        sentiment = loadedModel.predict(vect.transform([cleaning_words(review)]))[0]
        if sentiment == 1:
            name = "Sentiment of the given review is --> Positive"
        else:
            name = "Sentiment of the given review is --> Negative"
        kwargs = {
            'name': name,
        }
        return render_template("index.html", **kwargs)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8005)

