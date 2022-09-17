# Databricks notebook source
import mlflow

# COMMAND ----------

df = spark.table("sandbox_apoiadores.abt_dota_pre_match_new").toPandas()


# COMMAND ----------

model = mlflow.sklearn.load_model("models:/fatec_dota/production")

predict = model.predict(df[df.columns[2:]])

predict

# COMMAND ----------

from sklearn import metrics
metrics.accuracy_score(df['radiant_win'],predict)

# COMMAND ----------


