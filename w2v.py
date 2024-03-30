from pyspark.sql import SparkSession
from pyspark.sql.types import *
import pyspark.sql.functions as F
import re
from gensim.models import Word2Vec
from time import time
import random


spark = SparkSession.builder.master("local").appName("preprocess").getOrCreate()
ne_struct = StructType([StructField("row_id", StringType(), True),
                      StructField("subject_id", StringType(), True),
                      StructField("hadm_id", IntegerType(), True),
                      StructField("chartdate", IntegerType(), True),
                      StructField("charttime", IntegerType(), True),

                      StructField("storetime", StringType(), True),
                      StructField("category", StringType(), True),
                      StructField("description", StringType(), True),
                      StructField("cgid", IntegerType(), True),
                      StructField("iserror", IntegerType(), True),
                      StructField("text", StringType(), True)])
df = spark.read.csv("./NOTEEVENTS.csv", header=True, schema=ne_struct)
print(df.count())

df = df.limit(10000000)

df.createOrReplaceTempView("noteevents")
# 1) add indx; 2) add group_id; 3) first lines are lines with patient informatin ie. id, category; 4) merge all strings between; 5) join first lines and merged string (note)
df_revise = spark.sql("""
	WITH data2 as(
    SELECT row_number() OVER (ORDER BY monotonically_increasing_id()) - 1 AS idx, *
    FROM noteevents),
  data3 as(
   SELECT *,
          SUM(CASE WHEN category = 'Discharge summary' THEN 1 ELSE 0 END) OVER (ORDER BY idx) AS group_id
    FROM data2),
    FirstLine AS(select *
    from data3 
    WHERE data3.category = 'Discharge summary'),
  data4 as (SELECT group_id,
    CONCAT_WS(' ', COLLECT_LIST(CONCAT_WS(' ', row_id, subject_id, hadm_id))) AS merged
    FROM data3
    WHERE data3.category != 'discharge' or data3.category is null
  GROUP BY group_id)

 SELECT Firstline.subject_id,Firstline.hadm_id, data4.merged
 from Firstline
 left join data4
 on data4.group_id = Firstline.group_id

	""")
df_revise.createOrReplaceTempView("noteevents_2")
# df_revise.show(10, truncate=False)

df_annot = spark.read.csv('./annotations.csv', header = True).withColumnRenamed('subject.id', 'subject_id')
df_annot = df_annot.withColumnRenamed('Hospital.Admission.ID', 'hadm_id')
df_annot = df_annot.drop('chart.time')
df_annot = df_annot.toDF(*(column.replace('.', '_') for column in df_annot.columns))

#Check if there are rows that have the duplicate subject ids and hospital admission ids (identical ids may have different annotations).
df_annot = df_annot.dropDuplicates(subset = ["subject_id", "hadm_id"])


def pre_cleaning(text):
    pattern = re.compile('<[^>]*>|\W(\w\.)+\w?')
    text = re.sub(pattern, ' ', text.lower())
    text = re.sub(r"[^A-Za-z0-9()!?,'`\"]", ' ', text)
    text = re.sub(r"\'s", " \'s", text)
    text = re.sub(r"\'ve", " \'ve", text)
    text = re.sub(r"n\'t", " n\'t", text)
    text = re.sub(r"\'re", " \'re", text)
    text = re.sub(r"\'d", " \'d", text)
    text = re.sub(r"\'ll", " \'ll", text)
    text = re.sub(r",", " , ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\(", " ( ", text)
    text = re.sub(r"\)", " ) ", text)
    text = re.sub(r"\?", " ? ", text)
    return [i for i in text.split()]
     


# df_annot.show(10, truncate=False)

print(df_annot.count())

df_annot.createOrReplaceTempView('annotations')
#Use spark sql to select rows with distinct ids. The ids should be identical for both df and df_annot.
df_annot_text = spark.sql("""SELECT annotations.*, merged FROM noteevents_2 LEFT JOIN annotations 
                             ON noteevents_2.subject_id = annotations.subject_id 
                             AND noteevents_2.hadm_id = annotations.hadm_id""").na.drop()

print(df_annot_text.count())

textUDF = F.udf(lambda x: pre_cleaning(x), ArrayType(StringType()))

df_annot_text2 = df_annot_text.withColumn('merged', textUDF(F.col('merged')))
df_annot_text2.show(20, truncate= 30)

#Convert the text column to a list. Note that rdd.map().collect() will just collect the lists in each cell
#into a list. There is no need to add list() for the code below.
token_list = df_annot_text2.rdd.map(lambda x: x.merged).collect()
df_annot_text3 = df_annot_text2.toPandas()
df_annot_text3.to_pickle('df_annot_text_split.pkl')

vector_size = 32
model_w2v = Word2Vec(min_count = 1, window = 5, vector_size = vector_size, sample = 1e-5, negative = 5, workers = 8, sg = 0)
model_w2v.build_vocab(token_list)
indices = list(range(len(token_list)))
t0 = time()

for i in range(20):
  random.shuffle(indices)
  texts_samples = [token_list[j] for j in indices]
  model_w2v.train(texts_samples, total_examples = model_w2v.corpus_count, epochs = 5)

print('time to train the model:{0:.2f} mins.'.format((time() - t0)/60))
similar_words = model_w2v.wv.most_similar("alcohol")
print(similar_words)
model_w2v.wv.save_word2vec_format('./w2v.txt', binary = False)


#print(df.count())
spark.stop()