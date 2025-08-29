# Add required package installation
# pip install torch transformers nltk scikit-learn pandas seaborn emot emoji

# Import only what's needed
import json
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from emot.emo_unicode import UNICODE_EMOJI, EMOTICONS_EMO
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.preprocessing import LabelEncoder
import torch
from torch.optim import AdamW
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix
import numpy as np
import os
import collections
import pickle
import emoji
import sys
sys.stdout.reconfigure(encoding='utf-8')

# Define the absolute path to the dataset
DATASET_PATH = 'C:/Users/prasn/Downloads/nlp/AMAZON_FASHION.json'  # Replace with the actual path

# Add error handling for file loading
try:
    with open(DATASET_PATH, 'r', encoding='utf-8') as file:
        # Load each line separately and process it as a JSON object
        data = []
        for line in file:
            try:
                json_object = json.loads(line)
                data.append(json_object)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
                continue  # Skip to the next line if there's an error
except FileNotFoundError:
    raise FileNotFoundError(f"{DATASET_PATH} not found. Please ensure the dataset file is in the specified directory.")
except PermissionError as e:
    raise PermissionError(f"Permission denied: {e}. Please ensure you have the necessary permissions to read the file.")
except Exception as e:
    raise Exception(f"An error occurred while loading the JSON file: {e}")

# Download necessary NLTK resources
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Initialize lemmatizer and stop words
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

TEXT_EMOJI_MAP = {
    ":)": "smile",
    ":(": "sad",
    ";)": "wink",
    ":D": "laugh",
    ":P": "playful",
    ":'(": "crying",
    ":|": "neutral"
}

# Updated function to handle emojis
def handle_emojis(text):
    # Convert Unicode emojis to text
    text = emoji.demojize(text)
    # Convert text-based emoticons to text
    for emoticon, description in TEXT_EMOJI_MAP.items():
        text = text.replace(emoticon, f" {description} ")
    return text

# Function to handle negation
def handle_negation(text):
    words = text.split()
    for i in range(len(words)):
        if words[i].lower() in ["not", "no", "never"]:
            if i + 1 < len(words):
                words[i + 1] = "NOT_" + words[i + 1]
    return " ".join(words)

# Preprocessing function
def preprocess_review(review):
    try:
        # Convert emojis and emoticons to text
        review = handle_emojis(review)
        # Lowercase the text
        review = review.lower()
        # Remove punctuation and digits
        review = re.sub(r'[^\w\s]', '', review)
        review = re.sub(r'\d+', '', review)
        # Tokenize, remove stopwords, lemmatize, and handle negation
        tokens = word_tokenize(review)
        tokens = [word for word in tokens if word not in stop_words]
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        review = handle_negation(" ".join(tokens))
        return review
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return ""

# Extract reviews and labels
reviews = [item['reviewText'] for item in data if 'reviewText' in item and isinstance(item['reviewText'], str)]
labels = [1 if item['overall'] >= 4 else 0 for item in data if 'overall' in item]  # 1 = Positive, 0 = Negative

# Limit the number of reviews to process
MAX_REVIEWS = 10000 # Reduce to 5000 for faster testing
reviews = reviews[:MAX_REVIEWS]
labels = labels[:MAX_REVIEWS]

# Example reviews for demonstration
example_review = reviews[0]

# Find examples for each preprocessing step
emoji_review = next((r for r in reviews if any(char in UNICODE_EMOJI for char in r)), example_review)
negation_review = next((r for r in reviews if "not" in r.lower() or "no" in r.lower() or "never" in r.lower()), example_review)

# Demonstrate preprocessing steps
# example_emoji = "I love it! 😍"
# example_negation = "I do not like it"

# Demonstrate preprocessing steps
example_emoji = emoji_review
example_negation = negation_review

# Perform each preprocessing step separately
emoji_handled = handle_emojis(example_emoji)
lowercased = example_emoji.lower()
no_punctuation = re.sub(r'[^\w\s]', '', example_emoji)
tokenized = word_tokenize(no_punctuation)
no_stopwords = [word for word in tokenized if word not in stop_words]
lemmatized = [lemmatizer.lemmatize(word) for word in no_stopwords]
negation_handled = handle_negation(" ".join(lemmatized))

# Create a Pandas DataFrame for the table
df = pd.DataFrame({
    'Step': ['Original Review', 'Emoji Handling', 'Lowercasing', 'Punctuation Removal', 'Tokenization', 'Stopword Removal', 'Lemmatization', 'Negation Handling'],
    'Input': [example_emoji[:50], example_emoji[:50], example_emoji[:50], example_emoji[:50], no_punctuation[:50], ' '.join(tokenized[:5])[:50], ' '.join(no_stopwords[:5])[:50], ' '.join(lemmatized[:5])[:50]],
    'Output': [example_emoji[:50], emoji_handled[:50], lowercased[:50], no_punctuation[:50], ' '.join(tokenized[:5])[:50], ' '.join(no_stopwords[:5])[:50], ' '.join(lemmatized[:5])[:50], negation_handled[:50]],
    'Purpose': ['Initial data', 'Convert emojis to text', 'Convert to lowercase', 'Remove punctuation', 'Split into tokens', 'Remove common words', 'Reduce to base form', 'Handle negation']
})

# Plot the table using matplotlib
fig, ax = plt.subplots(figsize=(15, 6))
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='left', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)
plt.title('Preprocessing Examples')
plt.show()

# Preprocess reviews
preprocessed_reviews = [preprocess_review(review) for review in reviews]

with open('preprocessed_reviews.pkl', 'wb') as f:
    pickle.dump(preprocessed_reviews, f)

with open('preprocessed_reviews.pkl', 'rb') as f:
    preprocessed_reviews = pickle.load(f)

# Convert text to feature vectors
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(ngram_range=(1, 2))  # Unigrams and bigrams
X = vectorizer.fit_transform(preprocessed_reviews)
y = labels

# Check class distribution
from collections import Counter
print("Original class distribution:", Counter(y))

# Stratified split to maintain class balance
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Train Logistic Regression model
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight

# Compute class weights for Logistic Regression
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}

# Train Logistic Regression with class weights
lr_model = LogisticRegression(max_iter=1000, class_weight=class_weight_dict, C=0.1)
lr_model.fit(X_train, y_train)

# Predict and evaluate Logistic Regression
from sklearn.metrics import classification_report
y_pred_lr = lr_model.predict(X_test)
print("Logistic Regression Performance:")
print(classification_report(y_test, y_pred_lr))

# Load BERT tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
bert_model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model.to(device)

print(f"Using device: {device}")
bert_model.to(device)  # Move the model to the CPU

# Encode labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Tokenize and encode sequences for BERT
def encode_reviews(reviews):
    return tokenizer(reviews, padding=True, truncation=True, max_length=128, return_tensors="pt") # Changed to pt

encoded_data = tokenizer(reviews, padding=True, truncation=True, max_length=128, return_tensors="pt")

# Create DataLoader
input_ids = encoded_data['input_ids']
attention_mask = encoded_data['attention_mask']
labels = torch.tensor(encoded_labels)

input_ids = input_ids.to(device)
attention_mask = attention_mask.to(device)
labels = labels.to(device)

from sklearn.utils.class_weight import compute_class_weight

# Compute class weights for BERT
labels_cpu = labels.cpu().numpy()  # Move labels to CPU if on GPU
class_weights = compute_class_weight('balanced', classes=np.unique(labels_cpu), y=labels_cpu)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

# Use class weights in the loss function for BERT
from torch.nn import CrossEntropyLoss
loss_fn = CrossEntropyLoss(weight=class_weights)

# Create DataLoader
dataset = TensorDataset(input_ids, attention_mask, labels)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)

# Optimizer
optimizer = AdamW(bert_model.parameters(), lr=2e-5)

# Update the training loop to use the custom loss function
def train_model(model, dataloader, optimizer, device, epochs=1):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels)  # Use custom loss function
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader)}")

# Call the train_model function
if __name__ == "__main__":
    # Train the model
    train_model(bert_model, dataloader, optimizer, device, epochs=6)

    # Save the model
    bert_model.save_pretrained("fine_tuned_bert")
    tokenizer.save_pretrained("fine_tuned_bert")

    # Example inference
    test_review = "This product is amazing! I love it."
    def predict_sentiment(review, model, tokenizer, device):
        model.eval()
        inputs = tokenizer(review, padding=True, truncation=True, max_length=128, return_tensors="pt")
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
    
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            prediction = torch.argmax(logits, dim=1).item()
    
        return "Positive" if prediction == 1 else "Negative"
    
        sentiment = predict_sentiment(test_review, bert_model, tokenizer, device)
        print(f"Review: {test_review}")
        print(f"Predicted Sentiment: {sentiment}")

# Save the fine-tuned model
bert_model.save_pretrained("fine_tuned_bert")
tokenizer.save_pretrained("fine_tuned_bert")

# Load the fine-tuned model
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

tokenizer = DistilBertTokenizer.from_pretrained("fine_tuned_bert")
bert_model = DistilBertForSequenceClassification.from_pretrained("fine_tuned_bert")
bert_model.to(device)

# Prediction
bert_model.eval()
all_predictions = []
all_labels = []

with torch.no_grad():
    for batch in dataloader:
        input_ids, attention_mask, labels = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        outputs = bert_model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1)

        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print("BERT Performance:")
print(classification_report(all_labels, all_predictions))

torch.cuda.empty_cache()

# Function to plot classification report as a heatmap
def plot_classification_report(y_true, y_pred, class_names, title):
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    plt.figure(figsize=(10, 6))
    sns.heatmap(report_df.iloc[:-1, :-1], annot=True, cmap='Blues', fmt='.2f')
    plt.title(title)
    plt.ylabel('Classes')
    plt.xlabel('Metrics')
    plt.show()

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, class_names, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

# Class names for visualization
class_names = ['Negative', 'Positive']

# Logistic Regression Graphs
print("Logistic Regression Performance:")
print(classification_report(y_test, y_pred_lr))
plot_classification_report(y_test, y_pred_lr, class_names, "Logistic Regression - Classification Report")
plot_confusion_matrix(y_test, y_pred_lr, class_names, "Logistic Regression - Confusion Matrix")

# BERT Graphs
print("BERT Performance:")
print(classification_report(all_labels, all_predictions))
plot_classification_report(all_labels, all_predictions, class_names, "BERT - Classification Report")
plot_confusion_matrix(all_labels, all_predictions, class_names, "BERT - Confusion Matrix")

# Extract product information
products = {}
for item in data:
    asin = item.get('asin')
    review_text = item.get('reviewText', '')
    overall_rating = item.get('overall', None)

    if asin:
        if asin not in products:
            products[asin] = {
                'reviews': [],
                'ratings': []
            }
        products[asin]['reviews'].append(review_text)
        if overall_rating is not None:
            products[asin]['ratings'].append(overall_rating)

# Analyze product reviews
def analyze_product_reviews(products):
    product_insights = {}
    for asin, product_data in products.items():
        all_reviews = ' '.join(product_data['reviews'])
        word_tokens = word_tokenize(all_reviews.lower())
        filtered_tokens = [w for w in word_tokens if not w in stop_words and w.isalnum()]
        word_counts = collections.Counter(filtered_tokens)
        most_common_words = word_counts.most_common(5)  # Reduced to top 5

        avg_rating = np.mean(product_data['ratings']) if product_data['ratings'] else None

        product_insights[asin] = {
            'most_common_words': most_common_words,
            'average_rating': avg_rating
        }
    return product_insights

product_insights = analyze_product_reviews(products)

# Create a DataFrame for product insights
product_data = []
for asin, insights in product_insights.items():
    product_data.append({
        'ASIN': asin,
        'Average Rating': insights['average_rating'],
        'Top Words': ', '.join([word for word, count in insights['most_common_words']])
    })

df_products = pd.DataFrame(product_data)
df_products = df_products.sort_values(by='Average Rating', ascending=False)

# Display the DataFrame
print("\nProduct Insights:")
print(df_products.head(10))  # Display top 10 products

# Visualization (optional)
plt.figure(figsize=(12, 6))
sns.barplot(x='ASIN', y='Average Rating', data=df_products.head(10))
plt.xticks(rotation=45, ha='right')
plt.title('Top 10 Products by Average Rating')
plt.tight_layout()
plt.show()

# Print insights for a few products
##num_products_to_display = 10
##for asin, insights in list(product_insights.items())[:num_products_to_display]:
    ##print(f"Product ASIN: {asin}")
    ##print(f"  Most Common Words: {insights['most_common_words']}")
    ##print(f"  Average Rating: {insights['average_rating']:.2f}")
    ##print("-" * 40)

def display_product_insights(product_insights, num_positive=5, num_negative=5):
    """
    Display insights for a specified number of products with positive and negative reviews.

    Args:
        product_insights (dict): Dictionary containing product insights.
        num_positive (int): Number of products with positive reviews to display.
        num_negative (int): Number of products with negative reviews to display.
    """
    print("\nProduct Insights:")
    print("-" * 40)

    # Separate products into positive and negative based on average rating
    positive_products = [
        (asin, insights) for asin, insights in product_insights.items()
        if insights['average_rating'] is not None and insights['average_rating'] >= 4
    ]
    negative_products = [
        (asin, insights) for asin, insights in product_insights.items()
        if insights['average_rating'] is not None and insights['average_rating'] < 4
    ]

    # Sort products by average rating
    positive_products = sorted(positive_products, key=lambda x: x[1]['average_rating'], reverse=True)
    negative_products = sorted(negative_products, key=lambda x: x[1]['average_rating'])

    # Display positive products
    print("Top Positive Products:")
    for i, (asin, insights) in enumerate(positive_products[:num_positive]):
        avg_rating = insights['average_rating']
        most_common_words = insights['most_common_words']
        print(f"Product {i + 1}:")
        print(f"  ASIN: {asin}")
        print(f"  Average Rating: {avg_rating:.2f}")
        print(f"  Most Common Words: {', '.join([word for word, count in most_common_words])}")
        print("-" * 40)

    # Display negative products
    print("Top Negative Products:")
    for i, (asin, insights) in enumerate(negative_products[:num_negative]):
        avg_rating = insights['average_rating']
        most_common_words = insights['most_common_words']
        print(f"Product {i + 1}:")
        print(f"  ASIN: {asin}")
        print(f"  Average Rating: {avg_rating:.2f}")
        print(f"  Most Common Words: {', '.join([word for word, count in most_common_words])}")
        print("-" * 40)

def display_product_insights_plot(product_insights, num_positive=5, num_negative=5):
    """
    Display insights for a specified number of products with positive and negative reviews
    in a DataFrame and as a bar plot.

    Args:
        product_insights (dict): Dictionary containing product insights.
        num_positive (int): Number of products with positive reviews to display.
        num_negative (int): Number of products with negative reviews to display.
    """
    # Separate products into positive and negative based on average rating
    positive_products = [
        (asin, insights) for asin, insights in product_insights.items()
        if insights['average_rating'] is not None and insights['average_rating'] >= 4
    ]
    negative_products = [
        (asin, insights) for asin, insights in product_insights.items()
        if insights['average_rating'] is not None and insights['average_rating'] < 4
    ]

    # Sort products by average rating
    positive_products = sorted(positive_products, key=lambda x: x[1]['average_rating'], reverse=True)[:num_positive]
    negative_products = sorted(negative_products, key=lambda x: x[1]['average_rating'])[:num_negative]

    # Combine positive and negative products
    selected_products = positive_products + negative_products

    # Create a DataFrame for visualization
    product_data = []
    for asin, insights in selected_products:
        product_data.append({
            'ASIN': asin,
            'Average Rating': insights['average_rating'],
            'Top Words': ', '.join([word for word, count in insights['most_common_words']]),
            'Sentiment': 'Positive' if insights['average_rating'] >= 4 else 'Negative'
        })

    df_insights = pd.DataFrame(product_data)

    # Display the DataFrame
    print("\nProduct Insights:")
    print(df_insights)

    # Plot the insights
    plt.figure(figsize=(12, 6))
    sns.barplot(
        x='ASIN', y='Average Rating', hue='Sentiment', data=df_insights,
        palette={'Positive': 'green', 'Negative': 'red'}
    )
    plt.xticks(rotation=45, ha='right')
    plt.title('Product Insights: Top Positive and Negative Products')
    plt.tight_layout()
    plt.show()

# Call the function to display product insights
display_product_insights_plot(product_insights, num_positive=5, num_negative=5)


