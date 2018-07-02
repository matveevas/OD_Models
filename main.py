
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

df =spark.read.csv("/Users/svetlana.matveeva/Documents/MasterThesis/Dataset/joinresult/part-00000-647e9d46-6169-41d6-8dab-948c865be136-c000.csv")
df.printSchema()
df.select("_c1").show()
df.select(regexp_replace("_c1","POINT (", "").alias("point")).collect()
# df.withColumn('_c1', regexp_replace(df("_c1"),"POINT (", ""))
#df.withColumn('address', regexp_replace('address', 'lane', 'ln'))
df.select("_c1").show()


print("here")
exampleOSV.SVM()
print("here2")

