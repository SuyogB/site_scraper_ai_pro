# Streamlit-based Dynamic Web Scraping & Question-Answering with Local Models

## Overview

Approach 3 introduces a **Streamlit-based web interface** that allows users to dynamically input URLs for content scraping, with the added functionality of **question-answering** using **locally running models** (DeepSeek-R1-Distill-Qwen-1.5B and DeepSeek-R1-Distill-Llama-8B). This approach significantly enhances the scraping process by providing real-time insights through local LLM models. The application allows seamless integration of **web scraping** and **QA**, making it a powerful tool for gathering and analyzing website content.

## Features

- **Dynamic URL Input:** Allows users to input URLs to scrape content from.
- **Streamlit UI:** The main page of the application is built using Streamlit.
- **Question-Answering:** Integrated with locally running DeepSeek models (DeepSeek-R1-Distill-Qwen-1.5B and DeepSeek-R1-Distill-Llama-8B) for answering questions based on the scraped content.
- **Memory Tracking:** Code is included to track the memory usage and performance of the local models.
- **Local Models:** DeepSeek-R1-Distill models run locally using Ollama.

## Setup and Installation

### Cloning the Project

To get started, first clone the repository to your local machine by running the following command:

```bash
git clone https://github.com/SuyogB/site_scraper_ai_pro.git
```

Once cloned, navigate into the project directory:
```bash
cd site_scraper_ai_pro
```

To run this project, you will need the following prerequisites:

1. **Python 3.8+**  
2. **Ollama** - For running the local DeepSeek models.
   - Download Ollama [here](https://ollama.com/).

3. **Google API Key:**  
   - Create an `.env` file with the following line:  
     ```
     GOOGLE_API_KEY=your_api_key
     ```

4. **DeepSeek Models:**  
   To download and run the local models, execute the following commands in the terminal:

   - Download **DeepSeek-R1-Distill-Qwen-1.5B**:
     ```bash
     ollama run deepseek-r1:1.5b
     ```

   - Download **DeepSeek-R1-Distill-Llama-8B**:
     ```bash
     ollama run deepseek-r1:8b
     ```

## Requirements

You can install the necessary dependencies using `pip`. A `requirements.txt` file is included, and you can install the dependencies using the following command:

```bash
pip install -r requirements.txt
```

## Running the Application

The main page of the application is 1_🏠_Homepage.py. To start the Streamlit application, navigate to the directory containing the file and run the following command:

```bash
streamlit run 1_🏠_Homepage.py
```

This will open the Streamlit app in your default web browser.

## Documentation

Approach 3 focuses on integrating local LLM models into a Streamlit web interface for dynamic web scraping and question-answering. The main enhancements from previous approaches include:

Web Scraping - Dynamic URL input allows users to specify a website, and the app scrapes content from the given URL.
Question-Answering - Local models (DeepSeek-R1-Distill-Qwen-1.5B and DeepSeek-R1-Distill-Llama-8B) provide answers to user queries based on the scraped content.
Memory Usage Tracking - The code tracks and logs memory usage during the execution of the models, providing insights into performance.

## Future Updates (Idea)
Potential addition of an image search feature that would crawl the website, extract images, provide descriptions for each image, and offer insights.

## Troubleshooting
Memory Issues: If you encounter memory-related issues, ensure that your system has enough RAM for running the models locally.
API Key Missing: If you haven’t set the GOOGLE_API_KEY in the .env file, the app might fail to function properly.



