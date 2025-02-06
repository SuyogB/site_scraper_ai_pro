import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import streamlit as st
import google.generativeai as genai
from langchain_community.vectorstores.faiss import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

st.set_page_config(
    page_title="Ask Questions Gemini",
    page_icon="❓",
)


def data_ingestion():
    loader=TextLoader("dynamic_output.txt", encoding="utf-8")
    text_documents=loader.load()
    return text_documents


def get_text_chunks(text_documents):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200,length_function=len)
    chunks=text_splitter.split_documents(text_documents)
    return chunks


# get embeddings for each chunk
def get_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001"
    )
    vector_store = FAISS.from_documents(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    prompt_template = """
    You are an intelligent assistant with access to a crawled website in markdown format. Your task is to answer user questions strictly based on the information present in the website's markdown. Your responses should only contain information that exists in the markdown. If a user's question cannot be answered based on the markdown content, respond with 'The answer is not found in the provided website content.' Do not infer or hallucinate answers.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                                   client=genai,
                                   temperature=0.1,
                                   top_k=10,
                                   )
    prompt = PromptTemplate(template=prompt_template,
                            input_variables=["context", "question"])
    chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)
    
    return chain

def clear_chat_history():
    st.session_state.gemini_messages = [
        {"role": "assistant", "content": "Ask me questions on the website."}]

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001")  # type: ignore

    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True) 
    docs = new_db.similarity_search(user_question, top_k=10)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question}, return_only_outputs=True)

    return response

# Main function to run Streamlit app
def run():
    st.title("Ask Questions Gemini")
    
    content = data_ingestion()
    
    text_chunks = get_text_chunks(content)

    if text_chunks:
        get_vector_store(text_chunks)
        st.success("Done")
    else:
        st.warning("No website content found.")


    # Clear chat history button
    st.button('Clear Chat History', on_click=clear_chat_history)

    # Chat input and display
    if "gemini_messages" not in st.session_state.keys():
        st.session_state.gemini_messages = [
            {"role": "assistant", "content": "Ask me questions on the website."}
        ]

    # Display previous chat messages
    for message in st.session_state.gemini_messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # User input for questions
    if prompt := st.chat_input("Ask a question about the crawled content:"):
        st.session_state.gemini_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # Query the embeddings and get the response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = user_input(prompt)
                if response:
                    answer = response.get('output_text', '')
                    st.write(answer)
                    st.session_state.gemini_messages.append({"role": "assistant", "content": answer})
                else:
                    st.write("No response found.")

# Run the app
if __name__ == '__main__':
    run()

