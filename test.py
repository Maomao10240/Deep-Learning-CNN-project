from pyspark.sql import SparkSession

# Create SparkSession
spark = SparkSession.builder \
    .appName("Concatenate Strings Between Discharge") \
    .getOrCreate()

# Sample data
data = [
    (1, 11, "discharge"),
    ("a", "b", "c"),
    ("ccc", "happy", "merge"),
    (2, 22, "discharge"),
    ("dd", "ee", "gg"),
    ("happy", "lesson", "lib"),
    (3, 33, "discharge"),
    ("sheet", None, None),
    ("book", None, None),
    ("ee", "kalar", None)
]

# Create DataFrame
df = spark.createDataFrame(data, ["index", "ID", "category"])
df.show(truncate=False)

# Register DataFrame as a temporary view
df.createOrReplaceTempView("data")

# Define SQL query to calculate index_group
sql_query = """
  WITH data2 as(
    SELECT row_number() OVER (ORDER BY monotonically_increasing_id()) - 1 AS idx, index, ID, category
    FROM data),
  data3 as(
   SELECT *,
          SUM(CASE WHEN category = 'discharge' THEN 1 ELSE 0 END) OVER (ORDER BY idx) AS group_id
    FROM data2),
    data4 as(
  SELECT group_id,
    CONCAT_WS(' ', COLLECT_LIST(CONCAT_WS(' ', index, ID, category))) AS merged
    FROM data3
    WHERE data3.category != 'discharge' or data3.category is null
  GROUP BY group_id),
  firstline as(
    SELECT *
    FROM data3
    WHERE data3.category = 'discharge')

 SELECT firstline.*, data4.merged
 from firstline
 left join data4
 on data4.group_id = firstline.group_id
"""

# Execute SQL query
result = spark.sql(sql_query)

# Show final result
result.show(truncate=False)

# Stop SparkSession
spark.stop()
