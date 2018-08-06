from __future__ import division
from __future__ import print_function
from pyspark.sql import SparkSession

from pyspark.sql.functions import *
import statsmodels.api as smt
import pandas as pd
from pandas import DataFrame
import pyarrow
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import *
from sklearn.metrics import r2_score
import ml_metrics as metrics
import itertools
import scipy.stats as scs


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
        smt.graphics.tsa.plot_acf(y, lags=lags, ax=acf_ax, alpha=0.5)
        smt.graphics.tsa.plot_pacf(y, lags=lags, ax=pacf_ax, alpha=0.5)


        print("Критерий Дики-Фуллера: p=%f" % smt.tsa.stattools.adfuller(y,)[1])
        # print("Критерий Дики-Фуллера: p=%f" % smt.tsa.stattools.adfuller(y, )[2])
        plt.tight_layout()
    return

spark = SparkSession\
    .builder\
    .appName("Python Spark SQL basic example")\
    .config("spark.some.config.option", "some-value")\
    .getOrCreate()

df3 = spark.read.csv("/Users/svetlana.matveeva/Documents/MasterThesis/Dataset/joinresult/part-00000-3a94d824-4491-4a1a-b650-e61bd752ed5a-c000.csv")
df3.printSchema()
df1 = df3.select("_c1", regexp_replace("_c0", "POLYGON [(][(]", "").alias("polygon"), regexp_replace("_c2", "POINT [(]", "").alias("point"), "_c3", "_c4", "_c5")
df2 = df1.select("_c1", regexp_replace("polygon", "[)][)]", "").alias("polygon"), regexp_replace("point", "[)]", "").alias("point"), "_c3", "_c4", "_c5")
df = df2.withColumnRenamed("_c1", "polygonID").withColumnRenamed("_c3", "pointID").withColumnRenamed("_c4", "addresstext").withColumnRenamed("_c5", "createddatetime")
df.createTempView("df")
dfCNT = spark.sql("select count(pointID) as count, createddatetime from df  group by  createddatetime  order by createddatetime")
dfPCNT = spark.sql("select count(pointID) as count, polygonID from df group by  polygonID ")
dfP = dfCNT.toPandas()
print(dfP)

# histograma and description
res = dfP.describe()
dfP.hist()

dfP.createddatetime = pd.to_datetime(dfP['createddatetime'], format='%Y-%m-%d')
dfP.set_index(['createddatetime'], inplace=True)
dfP = dfP.resample('D').mean().bfill()
dfP.plot(figsize=(12, 6))
plt.show()

# dfPdiff= dfP.diff(periods=30).dropna()
series = pd.Series(dfP['count'])
# tsplot(series, lags=30)

def double_exponential_smoothing(series, alpha, beta):
    result = [series[0]]
    for n in range(1, len(series)+1):
        if n == 1:
            level, trend = series[0], series[1] - series[0]
        if n >= len(series): # прогнозируем
            value = result[-1]
        else:
            value = series[n]
        last_level, level = level, alpha*value + (1-alpha)*(level+trend)
        trend = beta*(level-last_level) + (1-beta)*trend
        result.append(level+trend)
    return result

tsplot(series, lags=30)
print(type(series))

def invboxcox(y,lmbda):
    # обратное преобразование Бокса-Кокса
    if lmbda == 0:
        return(np.exp(y))
    else:
        return(np.exp(np.log(lmbda*y+1)/lmbda))

data = dfP.copy()
print(type(data))
data["Box"], lmbda = scs.boxcox(data) # прибавляем единицу, так как в исходном ряде есть нули
tsplot(data.Box, lags=30)
print("Оптимальный параметр преобразования Бокса-Кокса: %f" % lmbda)
print(data)
print(type(data))
# data["Shift"] = data.Box - data.Box.shift
dfPdiff= data.diff(periods=1).dropna()
series1 = pd.Series(data.Box)
tsplot(series1, lags=2)
# alpha = 0.09
# beta = 0.9
# exp = double_exponential_smoothing(series, alpha, beta)
plt.figure(figsize=(20, 8))
plt.plot(series1)
plt.plot(series, label='Actual')
plt.show()
tsplot(series1, lags=30)




fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
# fig = smt.graphics.tsa.plot_acf(series.values.squeeze(), lags=30, ax=ax1)
fig = smt.graphics.tsa.plot_acf(series1, lags=30, ax=ax1)
ax2 = fig.add_subplot(212)
fig = smt.graphics.tsa.plot_pacf(series1, lags=25, ax=ax2)
plt.show()

p = data.drop('count', axis=1)
# p.createddatetime = pd.to_datetime(p['createddatetime'], format='%Y-%m-%d')
# p.set_index(['createddatetime'], inplace=True)
print(p)
print(type(p))

src_data_model = p[:'2017-11-01 00:00:00']
print(src_data_model)
# src_data_model.index = pd.to_datetime(src_data_model.index)
model = smt.tsa.ARIMA(src_data_model['Box'], order=(1, 0, 1), freq='D').fit(disp=-1)
print(model.summary())
plt.plot(p['Box'])
# plt.plot(dfP['count'])

tsplot(model.resid[24:], lags=30)
q_test = smt.tsa.stattools.acf(model.resid, qstat=True) #свойство resid, хранит остатки модели, qstat=True, означает что применяем указынный тест к коэф-ам
print(DataFrame({'Q-stat': q_test[1], 'p-value': q_test[2]}))

# prediction
# pred = model.predict(start=src_data_model.shape[0], end=src_data_model.shape[0]+100)
pred = model.predict(start='2017-10-01 00:00:00', end='2017-11-02 00:00:00')
trn = p['2017-10-01 00:00:00':'2017-11-02 00:00:00']
print(pred)
# pred.plot(figsize=(12, 8), color='red')
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
plt.plot(p['Box']['2017-10-01 00:00:00':'2017-11-02 00:00:00'])
plt.plot(pred, color='green')
# plt.plot(trn,color='red')
plt.show()
print("here")

arima(dfP)
