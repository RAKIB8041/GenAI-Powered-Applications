import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
import tempfile

# Streamlit App Title
st.title("RAG-based MCQ & Short Answer Generator")

# Upload PDF
uploaded_file = st.file_uploader("Upload a GenAI-related PDF", type=["pdf"])

if uploaded_file is not None:
    # Temporary save
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # Load PDF using LangChain
    loader = PyPDFLoader(tmp_path)
    documents = loader.load()

    if not documents:
        st.error("No content extracted from PDF. Check the file.")
    else:
        st.success("PDF Loaded Successfully!")

        # Split text into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = splitter.split_documents(documents)

        if not texts:
            st.error("Text splitting resulted in zero chunks.")
        else:
            # Embeddings
            embeddings = AzureOpenAIEmbeddings(
                openai_api_base=os.getenv("EMBEDDING_AZURE_OPENAI_API_BASE"),
                azure_endpoint=os.getenv("EMBEDDING_AZURE_OPENAI_API_ENDPOINT"),
                openai_api_version=os.getenv("EMBEDDING_AZURE_OPENAI_API_VERSION"),
                openai_api_key=os.getenv("EMBEDDING_AZURE_OPENAI_API_KEY"),
                deployment=os.getenv("EMBEDDING_AZURE_OPENAI_DEPLOYMENT_NAME"),
                model="text-embedding-3-large",
                chunk_size=10
            )

            # FAISS Vector DB
            db = FAISS.from_documents(texts, embeddings)
            retriever = db.as_retriever()

            # LLM (Azure GPT)
            llm = AzureChatOpenAI(
                openai_api_base=os.getenv("AZURE_OPENAI_API_BASE"),
                openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
                openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
                model_name="gpt-4o",
                temperature=0.5,
                model_kwargs={"top_p": 0.9, "max_tokens": 1500}
            )

            # Retrieval QA Chain
            qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
                llm=llm,
                retriever=retriever,
                chain_type="stuff"
            )

            # PROMPTS for MCQ and Short Answer
            mcq_prompt = """
            Based on the provided content, generate 5 multiple choice questions.
            For each question, provide:
            - The question
            - 4 options (A, B, C, D)
            - The correct answer with explanation.
            """
            
            short_ans_prompt = """
            Based on the provided content, generate 5 short answer questions and their correct answers.
            For each, provide:
            - The question
            - The answer
            """

            # Generate MCQs
            mcq_result = qa_chain.invoke(mcq_prompt)
            st.subheader("Generated 5 MCQs with Options and Answers:")
            st.write(mcq_result)

            # Generate Short Answers
            short_ans_result = qa_chain.invoke(short_ans_prompt)
            st.subheader("Generated 5 Short Answer Questions with Answers:")
            st.write(short_ans_result)

            st.success("MCQ & Short Answer Generation Completed!")