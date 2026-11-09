
import streamlit as st
import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

from rag import create_vector_store

# Load API key
load_dotenv(Path(__file__).resolve().parent / ".env")

st.set_page_config(
    page_title="RAG Q&A",
    page_icon="📚"
)

st.title("📚 RAG Q&A System")
st.write("Ask questions about the uploaded document.")

# Check API key
if not os.getenv("GOOGLE_API_KEY"):
    st.error("Gemini API key not found. Please check your .env file.")
    st.stop()


# Initialize Gemini only once
@st.cache_resource
def load_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-3.8-flash",
        temperature=0
    )


# Create vector database only once per session
if "vector_store" not in st.session_state:
    with st.spinner("Reading document and creating embeddings..."):
        st.session_state.vector_store = create_vector_store()

    st.success("Document loaded successfully!")


# Initialize session cache
if "answer_cache" not in st.session_state:
    st.session_state.answer_cache = {}


# Prompt template
prompt = PromptTemplate(
    template="""
You are a helpful assistant.

Answer the question using ONLY the information
provided in the context below.

If the answer is not present in the context,
say "I couldn't find the answer in the document."

Context:
{context}

Question:
{question}

Answer:
""",
    input_variables=["context", "question"]
)


# Ask a question
question = st.text_input("Ask your question:")

if st.button("Ask"):

    if not question.strip():
        st.warning("Please enter a question.")

    else:
        try:
            with st.spinner("Finding the answer..."):

                # Retrieve relevant chunks
                docs = st.session_state.vector_store.similarity_search(
                    question,
                    k=3
                )

                context = "\n\n".join(
                    doc.page_content for doc in docs
                )

                # Create cache key
                cache_key = (question.strip().lower(), context)

                # Check whether answer already exists
                if cache_key in st.session_state.answer_cache:
                    answer = st.session_state.answer_cache[cache_key]
                    st.info("Retrieved from cache (no API call).")

                else:
                    llm = load_llm()

                    final_prompt = prompt.format(
                        context=context,
                        question=question
                    )

                    response = llm.invoke(final_prompt)

                    # Handle different response formats
                    if isinstance(response.content, list):
                        answer = "\n".join(
                            item["text"]
                            for item in response.content
                            if isinstance(item, dict)
                            and item.get("type") == "text"
                        )
                    else:
                        answer = response.content

                    # Save answer in session cache
                    st.session_state.answer_cache[cache_key] = answer

            st.subheader("Answer")
            st.write(answer)

        except Exception as e:
            error_message = str(e)

            if (
                "429" in error_message
                or "RESOURCE_EXHAUSTED" in error_message
                or "quota" in error_message.lower()
            ):
                st.error(
                    "Gemini API quota exceeded. "
                    "Please wait for the quota reset or check your API billing."
                )
            else:
                st.error(f"Something went wrong: {error_message}")
