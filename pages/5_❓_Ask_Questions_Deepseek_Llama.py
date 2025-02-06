from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain_ollama.llms import OllamaLLM
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
import psutil
import os
import GPUtil


st.set_page_config(
    page_title="Ask Questions Deepseek Llama",
    page_icon="❓",
)

def get_gpu_memory_usage():
    gpus = GPUtil.getGPUs()
    if gpus:
        gpu = gpus[0]  # Assuming you're using one GPU
        gpu_memory_used = gpu.memoryUsed  # in MB
        return gpu_memory_used
    return 0  # Return 0 if no GPU is detected

def get_memory_usage():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / (1024 ** 2)  # Convert bytes to MB


def track_memory_usage():
    # Track RAM
    ram_usage = get_memory_usage()  # Your existing RAM tracking code
    # Track GPU Memory
    gpu_usage = get_gpu_memory_usage()  # GPU tracking

    st.write(f"RAM consumed: {ram_usage:.2f} MB")
    st.write(f"GPU memory consumed: {gpu_usage:.2f} MB")

    return ram_usage, gpu_usage

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
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vector_store = FAISS.from_documents(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index_local")


def get_conversational_chain():
    prompt_template = """
    You are an intelligent assistant with access to a crawled website in markdown format. Your task is to answer user questions strictly based on the information present in the website's markdown. Your responses should only contain information that exists in the markdown. If a user's question cannot be answered based on the markdown content, respond with 'The answer is not found in the provided website content.' Do not infer or hallucinate answers.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    model = OllamaLLM(model="deepseek-r1:8b",
                                   temperature=0.1,
                                   top_k=10,
                                   )
    prompt = PromptTemplate(template=prompt_template,
                            input_variables=["context", "question"])
    chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)
    
    return chain

def clear_chat_history():
    st.session_state.deepseekllama_messages = [
        {"role": "assistant", "content": "Ask me questions on the website."}]

def user_input(user_question):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    new_db = FAISS.load_local("faiss_index_local", embeddings, allow_dangerous_deserialization=True) 
    docs = new_db.similarity_search(user_question, top_k=10)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question}, return_only_outputs=True)

    return response

# Main function to run Streamlit app
def run():
    st.title("Ask Questions Deepseek Llama")

    # Track memory usage before starting the process
    st.write("Initial Memory Usage:")
    track_memory_usage()
    
    content = data_ingestion()

    # Track memory after data ingestion
    st.write("Memory Usage after data ingestion:")
    track_memory_usage()
    
    text_chunks = get_text_chunks(content)

    if text_chunks:
        # Track memory before vector store creation
        st.write("Memory Usage before vector store creation:")
        track_memory_usage()

        # Vector store creation
        get_vector_store(text_chunks)
        st.success("Done")

        # Track memory after vector store creation
        st.write("Memory Usage after vector store creation:")
        track_memory_usage()
    else:
        st.warning("No website content found.")


    # Clear chat history button
    st.button('Clear Chat History', on_click=clear_chat_history)

    # Chat input and display
    if "deepseekllama_messages" not in st.session_state.keys():
        st.session_state.deepseekllama_messages = [
            {"role": "assistant", "content": "Ask me questions on the website."}
        ]

    # Display previous chat messages
    for message in st.session_state.deepseekllama_messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # User input for questions
    if prompt := st.chat_input("Ask a question about the crawled content:"):
        st.session_state.deepseekllama_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # Query the embeddings and get the response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                
                # Track memory before generating the response
                st.write("Memory Usage before generating response:")
                track_memory_usage()

                # Generate response
                response = user_input(prompt)

                # Track memory after generating the response
                st.write("Memory Usage after generating response:")
                track_memory_usage()

                if response:
                    answer = response.get('output_text', '')
                    st.write(answer)
                    st.session_state.deepseekllama_messages.append({"role": "assistant", "content": answer})
                else:
                    st.write("No response found.")

# Run the app
if __name__ == '__main__':
    run()

