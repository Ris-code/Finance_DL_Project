import streamlit as st
import requests

def fetch_news(api_key, query):
    url = "https://newsdata.io/api/1/latest"
    params = {
        "apikey": api_key,
        "q": query
    }

    response = requests.get(url, params=params)
    
    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        print("Error:", response.status_code)
        return None

def home():
    st.title('Explore The Market')

    tab1, tab2 = st.tabs(["Learn Stock Market", "Market News"])
    with tab1:
        st.markdown("### Learn About Stock Market")
        st.video("https://youtu.be/Xn7KWR9EOGQ")
    
    with tab2:
        # Replace 'YOUR_API_KEY' with your actual API key from NewsData.io
        api_key = "pub_4439555fd2dbbe7cd0cc8718c91f3faa43a43"

        # Replace 'pizza' with your desired query
        query = "Nifty"

        col1, col2 = st.columns([2, 1])

        query = st.text_input(label='Search News', placeholder='nifty')

        if query:
            # Fetch news data
            news_data = fetch_news(api_key, query)
            print(news_data)

            # Create a Streamlit web application
            st.title("Latest News Articles")

            if news_data:
                for article in news_data['results']:
                    # Split the screen into two columns
                    col1, col2 = st.columns([3, 2])

                    # Display article information on the left column
                    with col1:
                        st.write(f"**{article['title']}**")
                        st.write(article['description'])
                        # st.write(article['source_id'])
                        source_id = article.get('source_id', 'Unknown')
                        source_icon = article.get('source_icon', '')
                        if source_icon:
                            st.markdown(
                                f"""
                                <div style="display: flex; align-items: center;">
                                    <img src="{source_icon}" style="width: 20px; height: 20px; margin-right: 5px;">
                                    <span>{source_id}</span>
                                </div>
                                """, 
                                unsafe_allow_html=True
                            )
                        else:
                            st.write(f"**Source:** {source_id}")
                    # Display image or video on the right column
                    with col2:
                        if 'image_url' in article:
                            st.image(article['image_url'])
                        elif 'video_url' in article:
                            st.video(article['video_url'])
                    
                    st.write("---")
            else:
                st.error("Failed to fetch news data.")

