import requests
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- 1. EXTRACT ---------------- #

# Отримуємо дані про телефони з DummyJSON API
url = "https://dummyjson.com/products/category/smartphones"
response = requests.get(url)
data = response.json()

products = data["products"]

df = pd.DataFrame(products)[["title", "price"]]
df.rename(columns={"title": "name", "price": "price_usd"}, inplace=True)

print("Дані отримані ✅")
print(df.head())

# API НБУ — курс валют
rates = requests.get("https://bank.gov.ua/NBUStatService/v1/statdirectory/exchange?json").json()
currency = {item["cc"]: item["rate"] for item in rates if item["cc"] in ["USD", "EUR"]}

usd_to_uah = currency["USD"]
eur_to_uah = currency["EUR"]


# ---------------- 2. TRANSFORM ---------------- #

df["price_uah"] = round(df["price_usd"] * usd_to_uah, 2)
df["price_eur"] = round(df["price_uah"] / eur_to_uah, 2)


# ---------------- 3. LOAD ---------------- #

df.to_csv("phones_prices.csv", index=False, encoding="utf-8")
df.to_json("phones_prices.json", orient="records", force_ascii=False)

print("✅ Дані збережено у CSV та JSON")


# ---------------- 4. BI-Аналіз + Графіки ---------------- #

plt.figure()
df["price_usd"].hist(bins=10)
plt.title("Розподіл цін у доларах")
plt.xlabel("USD")
plt.ylabel("Кількість товарів")
plt.tight_layout()
plt.show()

plt.figure()
df[["price_uah", "price_usd", "price_eur"]].mean().plot(kind="bar")
plt.title("Середні ціни у валютах")
plt.ylabel("Ціна")
plt.tight_layout()
plt.show()

plt.figure()
df.nlargest(5, "price_uah").set_index("name")["price_uah"].plot(kind="barh")
plt.title("ТОП-5 найдорожчих телефонів")
plt.xlabel("Ціна, UAH")
plt.tight_layout()
plt.show()

print("✅ BI-аналітика виконана!")
