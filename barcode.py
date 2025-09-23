import requests

# Lấy thông tin sản phẩm bằng barcode
barcode = "6931008488382" 
url = f"https://world.openfoodfacts.org/api/v0/product/{barcode}.json"
response = requests.get(url)
data = response.json()

# In tên sản phẩm và danh sách thành phần
product_name = data.get("product", {}).get("product_name", "N/A")
ingredients = data.get("product", {}).get("ingredients_text", "N/A")

print(f"Product: {product_name}")
print(f"Ingredients: {ingredients}")
