import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI

OPENAI_API_KEY = "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"
st.header('First Chatbot')

with st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader("Upload a PDF file and start asking questions", type="pdf")

# Extract the text
if file is not None:
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
        # st.write(text)

    # Break it into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n"],
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    # st.write(chunks)

    # generating embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    # creating vector store - FAISS
    vector_store = FAISS.from_texts(chunks, embeddings)

    # get user's question
    user_question = st.text_input("Type your question")

    # do similarity search
    if user_question:
        match = vector_store.similarity_search(user_question)
        # st.write(match)

        # define the llm
        llm = ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            temperature=0.7,
            max_tokens=1000,
            model='gpt-4-turbo'
        )

        # output results
        # chain ->
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents =match, question=user_question)
        st.write(response)