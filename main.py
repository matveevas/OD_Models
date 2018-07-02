
from __future__ import division
from __future__ import print_function
from pyspark.sql import SparkSession
import exampleOSV



import os
import sys

from sklearn.utils import check_X_y
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pyod.models.ocsvm import OCSVM
from pyod.utils.data import generate_data
from pyod.utils.data import get_color_codes
from pyod.utils.data import evaluate_print

import re
from pyspark.sql.functions import UserDefinedFunction
from pyspark.sql.functions import *



spark = SparkSession\
    .builder\
    .appName("Python Spark SQL basic example")\
    .config("spark.some.config.option", "some-value")\
    .getOrCreate()

df3 = spark.read.csv("/Users/svetlana.matveeva/Documents/MasterThesis/Dataset/joinresult/part-00000-056da02e-ec76-43d8-873f-10a6a8ee8029-c000.csv")
df3.printSchema()
df3.select("_c2").show()
df1 = df3.select("_c1", regexp_replace("_c0", "POLYGON [(][(]", "").alias("polygon"), regexp_replace("_c2", "POINT [(]", "").alias("point"), "_c3", "_c4", "_c5")
df2 = df1.select("_c1", regexp_replace("polygon", "[)][)]", "").alias("polygon"), regexp_replace("point", "[)]", "").alias("point"), "_c3", "_c4", "_c5")
df = df2.withColumnRenamed("_c1", "polygonID").withColumnRenamed("_c3", "pointID").withColumnRenamed("_c4", "addresstext").withColumnRenamed("_c5", "creteddatetime")
df.createTempView("df")
# dfCNT = spark.sql("select polygonID, creteddatetime, count(pointID) as count from df groupby polygonID, cretaeddatetime")
# "polygonID", "creteddatetime"


print("here")
exampleOSV.SVM()
print("here2")

