import pandas as pd
import re
import requests
import streamlit as st
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import time
import csv

# --------------WEB SCRAPPING-----------------------------------------------

# Setup headless Chrome
options = Options()
options.add_argument("--headless")
options.add_argument("--disable-gpu")
driver = webdriver.Chrome(options=options)

all_data = []

# Loop through multiple pages
for page in range(1, 31):  # 8 pages * ~25 products ≈ 200
    url = f"https://www.flipkart.com/search?q=laptop&page={page}"
    driver.get(url)
    time.sleep(3)

    soup = BeautifulSoup(driver.page_source, "html.parser")
    product_containers = soup.find_all("div", {"data-id": True})

    for product in product_containers:
        # 1) Product Name
        name_tag = product.find("div", class_="KzDlHZ")
        # 2) Specifications
        specs_tag = product.find("div", class_="_6NESgJ")
        specs_list = specs_tag.find_all("li") if specs_tag else []

        # Each Line in specification come attached prev line last word get concat with next lines first word
        # Seperate each <li> text by comma
        specs = []
        for li in specs_list:
            text = li.get_text(separator=" ", strip=True)
            specs.append(text)

        # Join with space or newline (as needed)
        cleaned_specs = ",".join(specs)          
        
        # 3) Rating
        rating_tag = product.find("div", class_="XQDdHH")

        # 4) Price   
        price_tag = product.find("div", class_="Nx9bqj _4b5DiR") 
    

        product_name = name_tag.get_text().strip() if name_tag else "Not found"
        specifications = cleaned_specs.strip() if cleaned_specs else "Not found"
        rating = rating_tag.get_text().strip() if rating_tag else "Not found"
        price = price_tag.get_text().strip() if price_tag else "Not found"

        all_data.append([product_name, specifications, rating, price])

    print(f"✅ Page {page} scraped — Total products so far: {len(all_data)}")

driver.quit()

# Save to CSV
with open("flipkart_laptops.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Product Name", "Specifications", "Rating", "Price"])
    writer.writerows(all_data)

print("✅ Saved data to 'flipkart_laptops.csv'")


# # ----------------------FETCH DATA--------------------------------------------------------
# =======DATA PROCESSING / CLEANING=================================
#1.Load data and clean
laptop_data=pd.read_csv("flipkart_laptops.csv")
# print(laptop_data)

# 2. Clean & convert the Price column to integer
laptop_data['Price']=laptop_data['Price'].str.replace("₹","").str.replace(",","").astype(int)

# 3. Extract Brand from the Name (assumes first word is brand)
def extract_laptop_name(name):
    match = re.match(r"^(.*?)\s*\(", name)
    return match.group(1).strip() if match else name.strip()

laptop_data['Name'] = laptop_data['Product Name'].apply(extract_laptop_name)

# # 4. Pull out RAM and Storage from the Specifications column
def parse_specs(name):
    """
    Extracts RAM, Storage, and OS from the product name string.
    Returns a pandas Series with three elements.
    """
    m = re.search(r"\(([^)]+)\)", name)
    if not m:
        return pd.Series([None, None, None])
    parts = [p.strip() for p in m.group(1).split("/")]

    ram = storage = os_ = None
    for p in parts:
        
        if re.search(r"\b(SSD|EMMC|HDD)\b", p, re.IGNORECASE):
            storage = p
        elif re.search(r"\b\d+\s*GB\b", p, re.IGNORECASE) :
            ram = p
        else:
            os_ = p
    return pd.Series([ram, storage, os_])

laptop_data[["RAM", "Storage", "OS"]] = laptop_data["Product Name"].apply(parse_specs)

# 5. Convert Rating to float (if present)
def clean_rating(r):
    try:
        return float(r)
    except:
        return None

laptop_data["Rating"] = laptop_data["Rating"].apply(clean_rating)

# print(laptop_data)

# 6. Drop rows missing the core data
laptop_data = laptop_data.dropna(subset=["Name", "Price","RAM","Storage"])

# 7. Normalize Specifications text for RAG / text‐search
laptop_data["NormalizedSpecs"] = laptop_data["Specifications"].str.lower()

# 8. (Optional) Reorder columns for readability
cols = ["Name","OS","RAM", "Storage","Price","Rating", "NormalizedSpecs"]
laptop_data = laptop_data[[c for c in cols if c in laptop_data.columns]]


# 9. Save the cleaned DataFrame back to CSV
laptop_data.to_csv("flipkart_laptops_cleaned.csv", index=False)

st.text("✅ Data processing complete. Here’s a preview:")
print("✅ Data processing complete. Here’s a preview:")
# print(laptop_data)



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

    if not submit_button or not query:
        st.error("Please enter a query and submit the form.")
        return  # Prevent further execution

    # Only reaches here if both conditions are met
    prompt = "if have price show price of three laptop details, always should be in list view and "
    response = qa_chain.invoke(prompt + query)
                 
    # Display options
    st.markdown("### 🔍 Recommendations :")
    st.markdown(response['result'])
    # st.text(response) 

    options = extract_laptop_options(response['result'])
    st.text(options)
    if options:
        selected = st.selectbox("Pick your favorite:", options)
        st.text(f"You selected: {selected}")
        if st.button("Explain My Choice"):
            followup_prompt = f"Explain why {selected} is a good choice for a student."
            explanation = groq_llm.invoke(followup_prompt)
            st.markdown("### 💡 Reasoning")
            st.write(explanation.content)
           
    else:
        st.error("No laptop options found in the response.")
            

import asyncio

async def main():
    user_interaction()
    # ... your logic ...

if __name__ == "__main__":
    import torch
    torch.classes.__path__ = []
    asyncio.run(main())

