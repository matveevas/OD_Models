
from __future__ import division
from __future__ import print_function
from pyspark.sql import SparkSession




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
import statsmodels.api as smt
import pandas as pd
from pandas import DataFrame
import pyarrow
import matplotlib.pyplot as plt


def tsplot(y, lags=None, figsize=(12, 7), style='bmh'):
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    with plt.style.context(style):
        fig = plt.figure(figsize=figsize)
        layout = (2, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))

        y.plot(ax=ts_ax)
        ts_ax.set_title('Time Series Analysis Plots')
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax, alpha=0.5)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax, alpha=0.5)

        print("Критерий Дики-Фуллера: p=%f" % sm.tsa.stattools.adfuller(y)[1])

        plt.tight_layout()
    return


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
dfCNT = spark.sql("select creteddatetime , count(pointID) as count from df group by  creteddatetime order by creteddatetime")
# "polygonID", "creteddatetime"
# dfCNT = spark.sql("select cast(count(pointID) as Int) as count from df")
# dfCNT.collect()
dfCNT.show(50)
dfP = dfCNT.toPandas()
# spark.conf.set("spark.sql.execution.arrow.enabled", "true")
c = dfP.count()
print(c)


res = dfP.describe()
dfP.hist()

dfP.creteddatetime = pd.to_datetime(dfP['creteddatetime'], format='%Y-%m-%d')
dfP.set_index(['creteddatetime'],inplace=True)
dfP.plot(figsize=(12, 6))


plt.show()

tsplot(dfP, lags=30)




print("here")



