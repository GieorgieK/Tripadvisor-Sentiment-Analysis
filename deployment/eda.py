import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns

# Ensure set_page_config is called first
st.set_page_config(
    page_title='Sentiment Analysis of Reviews',
    layout='wide',
    initial_sidebar_state='expanded'
)

def create_wordcloud(text, title):
    wordcloud = WordCloud(background_color="white").generate(" ".join(text))
    plt.figure(figsize=(10, 7))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(title)
    st.pyplot()

def main():
    st.title('Sentiment Analysis of Reviews')
    st.subheader('Exploratory Data Analysis')

    st.write('Created by: Gieorgie Kosasih')

    st.markdown('---')

    st.write('### Sample Data')
    df = pd.read_csv('tripadvisor_hotel_reviews.csv')
    
    # Map the ratings to positive, neutral, and negative
    rating_mapping = {5: 'Positive', 4: 'Positive', 3: 'Neutral', 2: 'Negative', 1: 'Negative'}
    df['Sentiment'] = df['Rating'].map(rating_mapping)
    
    st.dataframe(df.head(5))

    st.markdown('---')

    # Calculate rating counts
    rating_counts = df['Sentiment'].value_counts()

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.pie(rating_counts, labels=rating_counts.index, autopct='%1.1f%%', shadow=True, startangle=90)
    ax.set_title('Rating Sentiment')
    st.pyplot(fig)
    
    # Word clouds for different sentiment categories
    st.subheader('Word Clouds')
    
    # All reviews
    st.subheader('All Reviews')
    all_reviews_text = df['Review'].values
    create_wordcloud(all_reviews_text, 'Word Cloud - All Reviews')
    
    # Positive reviews
    st.subheader('Positive Reviews')
    positive_reviews_text = df[df['Sentiment'] == 'Positive']['Review'].values
    create_wordcloud(positive_reviews_text, 'Word Cloud - Positive Reviews')
    
    # Neutral reviews
    st.subheader('Neutral Reviews')
    neutral_reviews_text = df[df['Sentiment'] == 'Neutral']['Review'].values
    create_wordcloud(neutral_reviews_text, 'Word Cloud - Neutral Reviews')
    
    # Negative reviews
    st.subheader('Negative Reviews')
    negative_reviews_text = df[df['Sentiment'] == 'Negative']['Review'].values
    create_wordcloud(negative_reviews_text, 'Word Cloud - Negative Reviews')

# Run the app
if __name__ == '__main__':
    main()