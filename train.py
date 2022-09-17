# Databricks notebook source
# DBTITLE 1,Imports
from sklearn import tree
from sklearn import model_selection
from sklearn import linear_model
from sklearn import metrics
from sklearn import ensemble

import mlflow

# COMMAND ----------

# DBTITLE 1,Dados
print("obtendo dados...")
df = spark.table("sandbox_apoiadores.abt_dota_pre_match").toPandas()

df

# COMMAND ----------

# DBTITLE 1,Setup do Experimento
exp_name = "/Users/douglhorvath@gmail.com/fatec_dota_douglashorvath"
mlflow.set_experiment(exp_name)

# COMMAND ----------

target = "radiant_win"
id_column = "match_id"
features = list(set(df.columns) - set([target, id_column]))

# COMMAND ----------

X_train, X_test, y_train, y_test = model_selection.train_test_split(df[features],df[target],test_size = 0.2, random_state = 42)

# COMMAND ----------

print("Geral: ", df[target].mean())
print("Treino: ", y_train.mean())
print("Teste: ", y_test.mean())

# COMMAND ----------


