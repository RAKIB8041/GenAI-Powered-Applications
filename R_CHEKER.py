import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import AzureChatOpenAI
import tempfile

st.title("AI-based Resume Ranker")

# Upload JD PDF
jd_file = st.file_uploader("Upload Job Description (PDF)",z type=["pdf"])

# Upload Resume PDFs
resume_files = st.file_uploader("Upload 10 Resumes (PDF)", type=["pdf"], accept_multiple_files=True)

if jd_file is not None and resume_files:
    if len(resume_files) != 10:
        st.warning("⚠️ Please upload exactly 10 resumes.")
    else:
        # Read JD PDF content
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_jd:
            tmp_jd.write(jd_file.read())
            jd_path = tmp_jd.name

        jd_loader = PyPDFLoader(jd_path)
        jd_docs = jd_loader.load()
        jd_text = " ".join([doc.page_content for doc in jd_docs])

        # Read all Resume PDF contents
        resume_texts = []
        for resume_file in resume_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_resume:
                tmp_resume.write(resume_file.read())
                resume_path = tmp_resume.name

            resume_loader = PyPDFLoader(resume_path)
            resume_docs = resume_loader.load()
            resume_text = " ".join([doc.page_content for doc in resume_docs])
            resume_texts.append(resume_text)

        # Azure OpenAI LLM setup
        llm = AzureChatOpenAI(
            openai_api_base=os.getenv("AZURE_OPENAI_API_BASE"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            model_name="gpt-4o",
            temperature=0.3,
            max_tokens=500
        )

        # Score calculation for each resume
        scores = []
        for idx, resume_text in enumerate(resume_texts):
            prompt = f"""
            You are a Resume Screening Assistant.

            Compare the following Resume to the given Job Description.
            Score the resume on a scale of 0 (poor fit) to 100 (perfect fit) ONLY based on relevance.

            Return only the score as an integer without any explanation or extra text.

            Job Description:
            {jd_text}

            Resume:
            {resume_text}
            """

            result = llm.invoke(prompt)
            score_str = result.content.strip()

            # Ensure the returned value is an integer
            try:
                score = int(score_str)
            except ValueError:
                score = 0  # Default if LLM fails to return an integer

            scores.append((f"Resume {idx+1}", score))

        # Sort resumes by score descending
        ranked_resumes = sorted(scores, key=lambda x: x[1], reverse=True)

        st.subheader("🏆 Ranked Resumes:")
        for resume_name, score in ranked_resumes:
            st.write(f"{resume_name}: Score {score}")

        st.success("✅ Resume Ranking Completed Successfully!")