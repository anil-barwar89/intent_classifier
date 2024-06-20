import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

def preprocessing_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove HTML tags using regex
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'®', '', text)
    
    # Remove punctuation
    translation_table = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    text = text.translate(translation_table)
    text = ' '.join(text.split())
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # Replace any other types of numbers with the word "digits"
    text = re.sub(r'\s*\b\d+\b\s*', ' ', text)
    text = ' '.join(text.split())
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word, pos='v') for word in tokens]
    
    return " ".join(tokens)

# Load the vectorizer and model
with open('vectorizer_L1.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('naive_bayes_L1_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Set the title of the web app
st.title('Intent Classification of user queries')
# Create a text input widget
user_input = st.text_input('Enter some text for the model to process')

mapped_dict = {  0: 'customer',
                        1: 'development',
                        2: 'health',
                        3: 'hr',
                        4: 'it',
                        5: 'transport'
                        }
# Display the input text
if user_input:
    preprocessed_input = preprocessing_text(user_input)

    # Vectorize the user input
    user_input_vectorized = vectorizer.transform([preprocessed_input])
    
    # Use the model to generate a response
    response = model.predict(user_input_vectorized.toarray())[0]
    
    # Display the response
    st.write(f'L1 : {mapped_dict[response]}')
