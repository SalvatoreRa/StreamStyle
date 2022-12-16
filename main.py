import streamlit as st
from transformers import pipeline
import requests
from bs4 import BeautifulSoup
import tensorflow
import time


# https://en.wikipedia.org/wiki/Andor_(TV_series)
def text_recovery(url):
    # Make a  request to the URL
    response = requests.get(str(url))
    
    # Parse the HTML content of the page using BeautifulSoup
    soup = BeautifulSoup(response.text, 'html.parser')
    

    # Find the table that contains the episode summary
    table = soup.find('table', {'class': 'wikiepisodetable'})
    text = []
    
    # Iterate over the rows in the table
    for row in table.find_all('tr'):
        # Find the cells in each row
        
        cells = row.find_all('td')
        
        # If the row contains episode data
        if len(cells) == 1:
            # Extract the episode number, title, and summary
            episode_summary = cells[0].text
            

            # Print the episode data
            text.append(episode_summary)
    st.write('Wiki page successfully recovered')        
    return text

@st.cache(hash_funcs={"MyUnhashableClass": lambda _: None})
def load_summarization():
    model = pipeline("summarization", model="t5-base", 
    tokenizer="t5-base", framework="tf")
    return model

def summarization(text, model):
    st.write('Making magic: please wait')
    my_bar = st.progress(0)
    for i in range(len(text)):
        percent_complete = int(100/len(text)) * (i+1)
        if percent_complete >= 100:
            percent_complete = 100
        my_bar.progress(percent_complete)
        summary_episode = model(text[i], min_length=5, max_length=512)
        summary_episode =summary[0]['summary_text']
        summary_episodes = summary_episodes + ' ' + summary_episode


        #summary_episodes = model(text[0], min_length=5, max_length=512)
    
    return summary_episodes
    


            
    

# Create the main app
def main():
    st.title("Text Summarization with Hugging Face")

    # Get the input text from the user
    url = st.text_area("Enter the link of the wikipedia page")
    result = st.button('Run on url')
    if result:
        text = text_recovery(url)
        model= load_summarization()
        summary = summarization(text, model)
        st.write(summary)
    


if __name__ == "__main__":
    main()
