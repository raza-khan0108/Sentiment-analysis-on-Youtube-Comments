import os
import time
import numpy as np
import nltk
import matplotlib.pyplot as plt

from flask import Flask, render_template, request
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud, STOPWORDS
from bs4 import BeautifulSoup

from selenium.webdriver import Chrome
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# Download required NLTK data files
nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize Flask application
app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# Initialize NLTK components
wnl = WordNetLemmatizer()
sia = SentimentIntensityAnalyzer()
stop_words = stopwords.words('english')

def returnytcomments(url):
    data = []

    # Set up the Chrome WebDriver using the Service class
    service = ChromeService(ChromeDriverManager().install())
    with Chrome(service=service) as driver:
        wait = WebDriverWait(driver, 15)
        driver.get(url)

        # Scroll to load more comments
        for _ in range(5): 
            wait.until(EC.visibility_of_element_located((By.TAG_NAME, "body"))).send_keys(Keys.END)
            time.sleep(2)

        # Fetch comments from the page
        for comment in wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "#content"))):
            data.append(comment.text)

    return data

def clean(org_comments):
    y = []
    for x in org_comments:
        x = x.split()
        x = [i.lower().strip() for i in x]
        x = [i for i in x if i not in stop_words]
        x = [i for i in x if len(i) > 2]  # Keep words longer than 2 characters
        x = [wnl.lemmatize(i) for i in x]
        y.append(' '.join(x))
    return y

def create_wordcloud(clean_reviews):
    for_wc = ' '.join(clean_reviews)
    wcstops = set(STOPWORDS)
    wc = WordCloud(width=1400, height=800, stopwords=wcstops, background_color='white').generate(for_wc)
    plt.figure(figsize=(20, 10), facecolor='k', edgecolor='k')
    plt.imshow(wc, interpolation='bicubic') 
    plt.axis('off')
    plt.tight_layout()
    CleanCache(directory='static/images')
    plt.savefig('static/images/woc.png')
    plt.close()
    
def returnsentiment(x):
    score = sia.polarity_scores(x)['compound']
    
    if score > 0:
        sent = 'Positive'
    elif score == 0:
        sent = 'Neutral'  # Changed to 'Neutral' for clarity
    else:
        sent = 'Negative'
    return score, sent

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/results', methods=['GET'])
def result():    
    url = request.args.get('url')
    
    org_comments = returnytcomments(url)
    temp = []

    # Filter comments by length
    for i in org_comments:
        if len(i) > 5:  # Remove comments shorter than 5 characters
            temp.append(i)
    
    org_comments = temp
    clean_comments = clean(org_comments)

    create_wordcloud(clean_comments)
    
    np, nn, nne = 0, 0, 0
    predictions = []
    scores = []

    for i in clean_comments:
        score, sent = returnsentiment(i)
        scores.append(score)
        if sent == 'Positive':
            predictions.append('POSITIVE')
            np += 1
        elif sent == 'Negative':
            predictions.append('NEGATIVE')
            nn += 1
        else:
            predictions.append('NEUTRAL')
            nne += 1

    dic = []
    for i, cc in enumerate(clean_comments):
        x = {}
        x['sent'] = predictions[i]
        x['clean_comment'] = cc
        x['org_comment'] = org_comments[i]
        x['score'] = scores[i]  # Access the correct score
        dic.append(x)

    return render_template('result.html', n=len(clean_comments), nn=nn, np=np, nne=nne, dic=dic)

@app.route('/wc')
def wc():
    return render_template('wc.html')

class CleanCache:
    def __init__(self, directory=None):
        self.clean_path = directory
        # Check if the directory exists; if not, create it
        if not os.path.exists(self.clean_path):
            os.makedirs(self.clean_path)  # Create the directory
        # Now check and clean files
        if os.listdir(self.clean_path) != list():
            files = os.listdir(self.clean_path)
            for fileName in files:
                os.remove(os.path.join(self.clean_path, fileName))
        print("cleaned!")

if __name__ == '__main__':
    app.run(debug=True)
