# ✅ Manually create small mock datasets (products, customers, purchases)

# Product data
products = [
    ("B001", "iPhone 15", "Electronics", 79999, 4.5, "Apple"),
    ("B002", "Galaxy S24", "Electronics", 69999, 4.3, "Samsung"),
    ("B003", "Nike Air Max", "Fashion", 8999, 4.7, "Nike"),
    ("B004", "Echo Dot", "Electronics", 4499, 4.4, "Amazon")
]

# Customer purchase data
purchases = [
    ("C001", "B001", 1, "2024-01-01", 79999),
    ("C001", "B003", 2, "2024-01-03", 17998),
    ("C002", "B002", 1, "2024-01-02", 69999),
    ("C002", "B003", 1, "2024-01-04", 8999),
    ("C003", "B004", 1, "2024-01-05", 4499),
    ("C003", "B001", 1, "2024-01-06", 79999),
]

# Product schema
product_schema = StructType([
    StructField("product_id", StringType(), True),
    StructField("title", StringType(), True),
    StructField("category", StringType(), True),
    StructField("price", IntegerType(), True),
    StructField("rating", FloatType(), True),
    StructField("brand", StringType(), True),
])

# Purchase schema
purchase_schema = StructType([
    StructField("customer_id", StringType(), True),
    StructField("product_id", StringType(), True),
    StructField("quantity", IntegerType(), True),
    StructField("purchase_date", StringType(), True),
    StructField("price_paid", IntegerType(), True),
])

# ✅ Create DataFrames
df_products = spark.createDataFrame(products, product_schema)
df_purchases = spark.createDataFrame(purchases, purchase_schema)

logger.info("Sample data loaded into DataFrames.")


# frequently bought together
# ✅ Step 2.1: Group purchases by customer
customer_products = df_purchases.groupBy("customer_id").agg(
    collect_set("product_id").alias("products_bought")
)

# ✅ Step 2.2: Generate product pairs (co-purchase)
product_pairs = customer_products.selectExpr("explode(products_bought) as prod1", "products_bought")

# ✅ Step 2.3: Remove self-pairs and flatten
recommendations = product_pairs.withColumn("prod2", explode("products_bought")) \
                               .filter("prod1 != prod2") \
                               .groupBy("prod1", "prod2").count() \
                               .orderBy(desc("count"))

logger.info("Recommendation pairs computed.")
recommendations.show()


#product ranking

# ✅ Compute product metrics: sales count & average selling price
product_stats = df_purchases.groupBy("product_id").agg(
    count("customer_id").alias("total_sales"),
    sum("quantity").alias("units_sold"),
    avg("price_paid").alias("avg_selling_price")
)

# ✅ Join with product ratings
ranked_products = df_products.join(product_stats, "product_id")

# ✅ Create a simple ranking formula (sales + rating)
ranked_products = ranked_products.withColumn(
    "rank_score",
    (col("total_sales") * 0.5) + (col("rating") * 10)
).orderBy(desc("rank_score"))

logger.info("Product ranking computed.")
ranked_products.select("product_id", "title", "rank_score").show()



# prime customer identification

# ✅ Identify high-value customers
customer_stats = df_purchases.groupBy("customer_id").agg(
    count("product_id").alias("total_purchases"),
    sum("price_paid").alias("total_spent")
)

# ✅ Filter: spend > 10,000 and more than 2 purchases
prime_customers = customer_stats.filter(
    (col("total_purchases") >= 2) & (col("total_spent") > 10000)
)

logger.info("Prime candidate customers identified.")
prime_customers.show()
