import streamlit as st
import asyncio
from crawl4ai import *
import sys


st.set_page_config(
    page_title="Crawl & Convert",
    page_icon="🌐",
)

# Use ProactorEventLoop on Windows
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# Function to run the async web crawler
async def run_crawler(url):
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url)
        # Always save the output to "dynamic_output.txt" with UTF-8 encoding
        with open("dynamic_output.txt", "w", encoding="utf-8") as file:
            file.write(result.markdown)
        return result.markdown

# Streamlit UI
st.title("🔍 Crawl & Convert")

url = st.text_input("Enter the URL to crawl:", placeholder="https://example.com")

if st.button("Crawl and Convert to Markdown"):
    if url:
        with st.spinner("Crawling the website and converting to markdown..."):
            # Run the async crawler and overwrite the file each time
            markdown_content = asyncio.run(run_crawler(url))

            st.subheader("Crawled Markdown Content:")
            st.code(markdown_content)

            # Provide a download button for the markdown content
            st.download_button(
                label="Download Markdown Content",
                data=markdown_content,
                file_name="dynamic_output.txt",
                mime="text/plain",
            )

            st.success("Markdown saved as: dynamic_output.txt")
    else:
        st.warning("Please enter a valid URL.")
