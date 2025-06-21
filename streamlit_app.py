__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from openai import OpenAI
from langchain.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
CHROMA_PATH = "chroma_db"

PROMPT_TEMPLATE = """
Use the following context to answer the user's question.
Do not include any source references in your answer — those will be added separately.
Context:
{context}

---

User: {question}
Assistant:"""

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Page config and title
st.set_page_config(page_title="AI Chat (Docs)", layout="centered")
st.title("AI Chat Assistant (SharePoint Docs)")

# Display existing messages
for msg, role in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(msg)

# Input box
user_input = st.chat_input("Ask your question...")
if user_input:
    st.chat_message("user").markdown(user_input)
    st.session_state.chat_history.append((user_input, "user"))

    # Search vector store
    embedding = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding)

    with st.status("Searching the vector store...", expanded=False):
        results = db.similarity_search_with_relevance_scores(user_input, k=5)

    if not results or results[0][1] < 0.7:
        fallback_prompt = f"""
        Answer the following question as best as you can, even if no supporting documents are available:
        Question: {user_input}
        """
        with st.status("No strong matches found, generating general answer...", expanded=False):
            if LLM_PROVIDER == "openrouter":
                client = OpenAI(api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")
                response = client.chat.completions.create(
                    model="openai/gpt-4o-mini",
                    messages=[{"role": "user", "content": fallback_prompt}]
                )
                bot_response = response.choices[0].message.content
            else:
                llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-4o-mini-2024-07-18", temperature=0.7)
                bot_response = llm.invoke(fallback_prompt).content

    else:
        context = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
        sources = list({doc.metadata.get("source", "unknown") for doc, _ in results})
        prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE).format(
            context=context, question=user_input
        )

        # LLM response with loading indicator
        with st.status("Generating response...", expanded=False) as status:
            if LLM_PROVIDER == "openrouter":
                client = OpenAI(api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")
                response = client.chat.completions.create(
                    model="openai/gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}]
                )
                bot_response = response.choices[0].message.content
            else:
                llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-4o-mini-2024-07-18", temperature=0)
                response = llm.invoke(prompt)
                bot_response = response.content

            refiner_prompt = f"""
            Take the following assistant response and improve it by summarizing or generalizing where appropriate.
            Keep the meaning intact and do NOT add any sources or references — those will be added later.

            Response:
            {bot_response}

            Refined Response:"""

            if LLM_PROVIDER == "openrouter":
                refined_response_raw = client.chat.completions.create(
                    model="openai/gpt-4o-mini",  # or use a more concise model here
                    messages=[{"role": "user", "content": refiner_prompt}]
                )
                bot_response = refined_response_raw.choices[0].message.content
            else:
                llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-4o-mini-2024-07-18", temperature=0.3)
                bot_response = llm.invoke(refiner_prompt).content

            # Append sources to the assistant's message
            if sources:
                unique_sources = list(dict.fromkeys(sources))
                formatted_sources = "\n\n**Sources:**\n" + "\n".join(
                    [f"- [{src}]({src})" if src.startswith("http") else f"- `{src}`" for src in unique_sources]
                )
                
                bot_response += formatted_sources
            status.update(label="Done", state="complete")

    # Display assistant message
    with st.chat_message("assistant"):
        st.markdown(bot_response)
    st.session_state.chat_history.append((bot_response, "assistant"))
