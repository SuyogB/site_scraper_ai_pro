import streamlit as st

# Set up the Streamlit app homepage
st.set_page_config(
    page_title="Home Page",
    page_icon="🏠",
)



# Homepage content
st.title("Site Scraper QA")

st.markdown(
    """
    ### Welcome to the Site Scraper QA! 🕸️🤖
    
    **Description**:
    This app allows you to crawl web pages, extract content, convert it into markdown format, and save it dynamically. 
    You can also use an LLM (Large Language Model) to ask questions about the content you have crawled.

    **How it Works**:
    1. **Crawl & Convert Page**:
        - Enter any website URL you wish to crawl.
        - The content of the URL will be extracted, converted to markdown, and saved dynamically.
        - You can view and download the markdown directly from this page.

    2. **Ask Questions Page**:
        - Once you’ve crawled a webpage, switch to the "Ask Questions" page.
        - Enter a question based on the content you’ve crawled.
        - The app will use a powerful LLM to provide detailed answers based on the markdown content.

    **How to Use**:
    1. Navigate to the 'Crawl & Convert' page from the sidebar.
    2. Enter a valid URL to crawl, and the content will be saved as markdown.
    3. Switch to the 'Ask Questions' page to input questions related to the crawled content.
    4. Enjoy using the LLM for answers based on the extracted data.

    #### Note:
    You must first crawl a URL and save the content before using the 'Ask Questions' feature.
    
    ---
    
    **Let’s get started by navigating to the 'Crawl & Convert' page from the sidebar!** 😊
    """
)
