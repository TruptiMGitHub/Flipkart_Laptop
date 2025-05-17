1. Web Scraping ✅
  Scrape Flipkart data and save it as flipkart_laptops.csv.

2. Data Processing
Data Preprocessing: Raw laptop data is cleaned and normalized for consistency.

  Structure and clean:
  Extract RAM, storage, price, GPU, etc. from specs.

3.Embedding Generation: Each laptop's specifications are converted into vector embeddings using sentence-transformers.
  Optional: Create embeddings for product descriptions.


4. Build a Vector Store (RAG)
  Indexing: Embeddings are stored in a FAISS index to facilitate efficient similarity searches.
  Use FAISS or Chroma to enable similarity search.

5.User Query Handling: User inputs are embedded and compared against the FAISS index to retrieve the most relevant laptops

6.LLM Response Generation: For each recommended laptop, the LLM provides a detailed explanation highlighting its suitability based on the user's preferences.

7.User Interaction - Streamlit
