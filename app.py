from flask import Flask, escape, request, render_template
import pickle
import os
import requests
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd

# Load the vectorizer and model
try:
    vector = pickle.load(open("vectorizer.pkl", 'rb'))
    model = pickle.load(open("finalized_model.pkl", 'rb'))
    print("[INFO] Model and vectorizer loaded successfully.")
except Exception as e:
    print(f"[ERROR] Failed to load model/vectorizer: {e}")
    exit(1)

app = Flask(__name__)

# Check if the URL is from a trusted source
def is_trusted_news_url(url):
    trusted_sources = [
        "timesofindia.indiatimes.com",  # Times of India
        "bbc.com",                      # BBC
        "hindustantimes.com",            # Hindustan Times
        "ndtv.com",                      # NDTV
        "cnn.com"                        # CNN
    ]
    return any(source in url for source in trusted_sources)

# Scrape title and body text from the URL using Selenium
def extract_news_content(url):
    try:
        # Set up the Selenium driver (Chrome in this case)
        chrome_options = Options()
        chrome_options.add_argument("--headless")  # Run in headless mode (no browser window)
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
        
        # Open the URL
        driver.get(url)
        
        # Give the page time to load dynamic content (you can adjust the sleep time as needed)
        driver.implicitly_wait(10)

        # Extract title (use the <title> tag)
        title = driver.title
        
        # Extract the article body text (for Hindustan Times, we may need to check for specific divs or classes)
        paragraphs = driver.find_elements(By.TAG_NAME, 'p')
        text = ' '.join([para.text for para in paragraphs])

        if not text:
            print(f"[ERROR] Failed to extract article body content for URL: {url}")
            driver.quit()
            return None, None

        driver.quit()
        return title, text
    except Exception as e:
        print(f"[ERROR] Failed to extract content: {e}")
        driver.quit()
        return None, None

# Append news article to dataset
def update_dataset(title, text, label='Real', path='fake_news_dataset.csv'):
    try:
        if os.path.exists(path):
            df = pd.read_csv(path)
        else:
            df = pd.DataFrame(columns=["title", "text", "label"])

        if ((df['title'] == title) & (df['text'] == text)).any():
            print("[INFO] Article already exists in the dataset.")
            return

        new_entry = pd.DataFrame([[title, text, label]], columns=["title", "text", "label"])
        df = pd.concat([df, new_entry], ignore_index=True)
        df.to_csv(path, index=False)
        print("[INFO] Dataset updated successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to update dataset: {e}")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == "POST":
        news = request.form.get('news', '').strip()
        if not news:
            return render_template("prediction.html", prediction_text="⚠️ Please enter a news headline or URL!")

        print(f"[INFO] Received input: {news}")

        # Check for trusted news URLs (BBC, Hindustan Times, NDTV, CNN, Times of India)
        if is_trusted_news_url(news):
            # Identify the source
            if "timesofindia.indiatimes.com" in news:
                source_name = "Times of India"
            elif "bbc.com" in news:
                source_name = "BBC"
            elif "hindustantimes.com" in news:
                source_name = "Hindustan Times"
            elif "ndtv.com" in news:
                source_name = "NDTV"
            elif "cnn.com" in news:
                source_name = "CNN"
            else:
                source_name = "Trusted News Source"

            title, text = extract_news_content(news)
            if title and text:
                update_dataset(title, text, 'Real')  # Optional
                prediction_result = f"Real (Detected from a trusted {source_name} ✅)"
            else:
                prediction_result = "Couldn't extract content from URL ❌"
        else:
            prediction = model.predict(vector.transform([news]))[0]
            prediction_result = f"{prediction}"

        return render_template("prediction.html", prediction_text=f"News headline is → {prediction_result}")

    return render_template("prediction.html")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
