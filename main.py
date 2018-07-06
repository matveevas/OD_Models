
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
from statsmodels.graphics.tsaplots import *
from sklearn.metrics import r2_score
import ml_metrics as metrics


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
        # smt.graphics.tsa.plot_acf(y, lags=lags, ax=acf_ax, alpha=0.5)
        # smt.graphics.tsa.plot_pacf(y, lags=lags, ax=pacf_ax, alpha=0.5)
        # smt.graphics.plot_

        print("Критерий Дики-Фуллера: p=%f" % smt.tsa.stattools.adfuller(y,)[1])

        plt.tight_layout()
    return


spark = SparkSession\
    .builder\
    .appName("Python Spark SQL basic example")\
    .config("spark.some.config.option", "some-value")\
    .getOrCreate()


df3 = spark.read.csv("/Users/svetlana.matveeva/Documents/MasterThesis/Dataset/joinresult/part-00000-24e98f2a-72a0-4feb-9ba2-240d3d2e818a-c000.csv")
df3.printSchema()
df3.select("_c2").show()
df1 = df3.select("_c1", regexp_replace("_c0", "POLYGON [(][(]", "").alias("polygon"), regexp_replace("_c2", "POINT [(]", "").alias("point"), "_c3", "_c4", "_c5")
df2 = df1.select("_c1", regexp_replace("polygon", "[)][)]", "").alias("polygon"), regexp_replace("point", "[)]", "").alias("point"), "_c3", "_c4", "_c5")
df = df2.withColumnRenamed("_c1", "polygonID").withColumnRenamed("_c3", "pointID").withColumnRenamed("_c4", "addresstext").withColumnRenamed("_c5", "creteddatetime")
df.createTempView("df")
dfCNT = spark.sql("select count(pointID) as count, creteddatetime from df group by  creteddatetime order by creteddatetime")
# "polygonID", "creteddatetime"
# dfCNT = spark.sql("select cast(count(pointID) as Int) as count from df")
# dfCNT.collect()
dfCNT.show(50)
dfP = dfCNT.toPandas()
# spark.conf.set("spark.sql.execution.arrow.enabled", "true")
c = dfP.count()
print(c)
print(dfP)

res = dfP.describe()
dfP.hist()

dfP.creteddatetime = pd.to_datetime(dfP['creteddatetime'], format='%Y-%m-%d')
dfP.set_index(['creteddatetime'],inplace=True)
dfP.plot(figsize=(12, 6))

# dfPdiff = dfP.diff(periods=1).dropna()
# print(dfPdiff)
# series1 = pd.Series(dfPdiff['count'])

plt.show()

# s = pd.Series()
# for row in dfP.iterrows():
#     s.append(row)
# series = dfP.iloc[1, :]
series= pd.Series(dfP['count'])
# series = dfP.iloc[1, :]
# type(series)
# pd.core.series.Series
    # s.append(series)

# test = smt.tsa.adfuller(series1)
# print('adf: ', test[0])
# print('p-value: ', test[1])
# print('Critical values: ', test[4])
# if test[0]> test[4]['5%']:
#     print('есть единичные корни, ряд не стационарен')
# else:
#     print('единичных корней нет, ряд стационарен')

tsplot(series, lags=2)

fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = smt.graphics.tsa.plot_acf(dfP.values.squeeze(), lags=25, ax=ax1)
ax2 = fig.add_subplot(212)
fig = smt.graphics.tsa.plot_pacf(dfP, lags=25, ax=ax2)
plt.show()

src_data_model = dfP[:'2017-11-30 02:00:00']
model = smt.tsa.ARIMA(src_data_model['count'], order=(1, 0, 1)).fit(disp=-1)
print(model.summary())
plt.plot(dfP['count'])

tsplot(model.resid[24:], lags=30)
q_test = smt.tsa.stattools.acf(model.resid, qstat=True) #свойство resid, хранит остатки модели, qstat=True, означает что применяем указынный тест к коэф-ам
print(DataFrame({'Q-stat': q_test[1], 'p-value': q_test[2]}))

# prediction
pred = model.predict(start=2000, end=2150)
trn = dfP['2017-11-01 00:00:00':'2017-12-01 00:00:00']
print(pred)
pred.plot(figsize=(12, 8), color='red')
plt.show()
# r2 = r2_score(trn, pred[1:32])
# print('R_2= %1.2f' % r2)

# RMSE for ARIMA
rmse = metrics.rmse(trn, pred)
print(rmse)

# MAE for ARIMA
mae = metrics.mae(trn,pred)
print(mae)

fig = plt.figure(figsize=(17, 6))
plt.plot(dfP['count']['2017-11-20 00:00:00':])
plt.plot(pred, color='green')
# plt.plot(trn,color='red')
plt.show()
print("here")



