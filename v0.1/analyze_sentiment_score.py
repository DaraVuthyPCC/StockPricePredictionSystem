import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline

# Load the CSV
df = pd.read_csv('/kaggle/input/gdelt-sentiment/gdelt_aapl_news_articles.csv')

# Load FinBERT model and tokenizer
finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone', num_labels=3)
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')

# Create a pipeline for sentiment analysis
nlp = pipeline("sentiment-analysis", model=finbert, tokenizer=tokenizer)

# Apply FinBERT to your 'title' column
df['finbert_sentiment'] = df['title'].apply(lambda x: nlp(x)[0]['label'])

# Save the updated DataFrame to a CSV
df.to_csv('gdelt_aapl_sentiment_finbert.csv', index=False)