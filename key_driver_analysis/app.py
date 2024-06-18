import pandas as pd
from scripts.data_preprocessing import load_and_preprocess_data
from scripts.keyword_extraction import extract_keywords
from scripts.sentiment_analysis import sentiment_analysis
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import os
import re
import nltk
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Download NLTK resources if not already downloaded
nltk.download('wordnet')
nltk.download('stopwords')

# Set up NLTK and sklearn stopwords
stop_words_nltk = set(stopwords.words('english'))
stop_words_sklearn = ENGLISH_STOP_WORDS
stop_words = stop_words_nltk.union(stop_words_sklearn)
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-z\s]', '', text)
    # Remove stopwords and lemmatize
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])
    return text

def analyze_data(file_path):
    try:
        # Load and preprocess data
        if file_path.endswith('.csv'):
            reviews = load_and_preprocess_data(file_path)
            if 'text' not in reviews.columns:
                return None
        elif file_path.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                text_data = f.read()
                reviews = pd.DataFrame({'text': [text_data]})
        else:
            raise ValueError("Unsupported file format. Please upload a .csv or .txt file.")

        # Extract keywords and perform sentiment analysis
        reviews['text_clean'] = reviews['text'].apply(preprocess_text)
        reviews['keywords'] = extract_keywords(reviews['text_clean'])
        reviews['sentiment_score'] = reviews['text_clean'].apply(sentiment_analysis)

        # Count frequency of each keyword
        keyword_counts = Counter([keyword for sublist in reviews['keywords'] for keyword in sublist])
        importance_df = pd.DataFrame(keyword_counts.items(), columns=['keyword', 'frequency'])
        importance_df = importance_df.sort_values(by='frequency', ascending=False)

        return importance_df, reviews

    except FileNotFoundError:
        st.error(f"File not found: {file_path}")
        return None, None
    except pd.errors.EmptyDataError:
        st.error("The uploaded file is empty or could not be read.")
        return None, None
    except pd.errors.ParserError:
        st.error("There was an error parsing the file. Please check the file format.")
        return None, None
    except ValueError as ve:
        st.error(str(ve))
        return None, None
    except Exception as e:
        return None
        return None, None

def main():
    st.title('Key Driver Analysis on IMDb Movie Reviews')

    dataset_name = st.text_input('Enter a name for your dataset:', 'imdb_reviews')
    uploaded_file = st.file_uploader("Upload a .csv or .txt file", type=['csv', 'txt'])

    if uploaded_file is not None:
        # Save the uploaded file to the data directory
        data_dir = 'data'
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        file_path = os.path.join(data_dir, uploaded_file.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())

        try:
            # Load the uploaded file to inspect its structure
            df = pd.read_csv(file_path) if file_path.endswith('.csv') else None
            if df is not None:
                st.write("Uploaded data preview:")
                st.write(df.head())

            # Perform Key Driver Analysis
            importance_df, processed_data = analyze_data(file_path)

            if importance_df is None:
                return None
            else:
                # Display the dataframe with key factors and their importance
                st.subheader("Key Factors Impacting Sentiment")
                st.dataframe(importance_df[['keyword', 'frequency']])

                # Save processed data for download
                processed_data_file = f"{dataset_name}_processed_data.csv"
                processed_data_path = os.path.join(data_dir, processed_data_file)
                processed_data.to_csv(processed_data_path, index=False)
                st.markdown(f"### Download Processed Data:")
                st.markdown(f"Download [**{processed_data_file}**](/{data_dir}/{processed_data_file})")

                # Visualization: Bar chart of keyword frequency
                st.subheader("Top Keywords by Frequency")
                plt.figure(figsize=(10, 6))
                sns.barplot(x='frequency', y='keyword', data=importance_df.head(10))
                plt.title('Top Keywords by Frequency')
                plt.xlabel('Frequency')
                plt.ylabel('Keyword')
                st.pyplot(plt)

                # Save the plot as an image
                plot_filename = f"{dataset_name}_top_keywords_frequency.png"
                plot_filepath = os.path.join(data_dir, plot_filename)
                plt.savefig(plot_filepath)
                st.markdown(f"### Download Top Keywords Plot:")
                st.markdown(f"Download [**{plot_filename}**](/{data_dir}/{plot_filename})")

        except FileNotFoundError as e:
            st.error(f"File not found: {file_path}")
        except pd.errors.EmptyDataError:
            st.error("The uploaded file is empty or could not be read.")
        except pd.errors.ParserError:
            st.error("There was an error parsing the file. Please check the file format.")
        except Exception as e:
            return None

if __name__ == "__main__":
    main()
