import pandas as pd
import streamlit as st
import re

import streamlit as st


# # # ----------------------FETCH DATA--------------------------------------------------------
# # =======DATA PROCESSING / CLEANING=================================
# First Run web_scrapping file then run this file


# ---------------------------EMBEDDING------------------------------------------------------------
# fetct new cleaned csv
cleaned_data=pd.read_csv("flipkart_laptops_cleaned.csv")
# print(cleaned_data)

from langchain_core.documents import Document
import json
# 4. Create LangChain Documents
documents = [
        Document(
            page_content=json.dumps({"name": row["Name"], 
                    "RAM": row["RAM"],
                    "OS": row["OS"],
                    "Rating": row["Rating"],
                    "Storage": row["Storage"],
                    "Price": row["Price"],
                    "specifications":row['NormalizedSpecs']}),
        
            metadata=({
                    "name": row["Name"],
                    "RAM": row["RAM"],
                    "OS": row["OS"],
                    "Rating": row["Rating"],
                    "Storage": row["Storage"],
                    "Price": row["Price"]})
                )
        for _, row in cleaned_data.iterrows()
    ]

# print(documents)

# Create embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from  langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS

# embedding_model = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(documents, embedding_model)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})


# -------------------------------------RAG-----------------------------------------------------------
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

# Load Groq LLM
groq_key="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"
groq_llm = ChatGroq(
    groq_api_key=groq_key,  # ✅ Your Groq API Key
    model_name="llama3-8b-8192"
)

qa_chain = RetrievalQA.from_chain_type(
    llm=groq_llm,
    retriever=retriever
)

# ------------------------------------AGENT-------------------------------------------------

import re

def extract_laptop_options(answer):
    # Split answer into lines and keep numbered items
    options = re.findall(r"\d\.\s+(.*)", answer)
    return options

# **************************************************************
def user_interaction():
    # Streamlit UI
    st.title("🎓 Laptop Recommender - GenAI Powered")

        
    with st.form(key='laptop_form', clear_on_submit=True):
         # Ask your question
        query = st.text_input("Enter your laptop preference (e.g., budget-friendly SSD under 40000):")
        submit_button = st.form_submit_button(label='Laptops Recommendations are :')
    
    # If query submitted
    if submit_button and query:

        # Only reaches here if both conditions are met
        prompt = "if have price show price of three laptop details, always should be in list view and "
        response = qa_chain.invoke(prompt + query)
                     
        # Save response in session
        st.session_state.response = response
        st.session_state.options = extract_laptop_options(response['result'])

    if "response" in st.session_state:    
        # Display options
        st.markdown("### 🔍 Recommendations :")
        st.markdown(st.session_state.response['result'])
        # st.text(response) 
        if "options" in st.session_state and st.session_state.options:
            selected = st.selectbox("Pick your favorite:", st.session_state.options)
            st.session_state.selected = selected
               
            if st.button("Explain My Choice"):
                followup_prompt = f"Explain why {selected} is a good choice for a student."
                explanation = qa_chain.invoke(followup_prompt)
                st.markdown("### 💡 Reasoning")
                st.markdown(explanation if isinstance(explanation, str) else explanation.get("result", str(explanation)))
            
    else:
        st.error("No laptop options found .")
            
import asyncio

async def main():
    user_interaction()
    # ... your logic ...

if __name__ == "__main__":
    import torch
    torch.classes.__path__ = []
    asyncio.run(main())

