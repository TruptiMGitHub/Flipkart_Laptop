import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
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

