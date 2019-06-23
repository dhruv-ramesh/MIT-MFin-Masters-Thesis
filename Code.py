# -*- coding: utf-8 -*-
"""
Code for thesis
"""

#Importing the necessary modules

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, HuberRegressor, ElasticNet
import matplotlib.pyplot as plt
import cvxopt as opt
from cvxopt import solvers, matrix
from statsmodels.regression.quantile_regression import QuantReg
from scipy.optimize import least_squares
import warnings

warnings.filterwarnings("ignore")
solvers.options['show_progress'] = False

#Loading in the data and processsing it
FFloc = '/Users/dramesh/Desktop/MIT_Coursework/Thesis/Raw_Data/FamaFrenchFactors.CSV'
ThreeF = pd.read_csv(FFloc)
ThreeF.columns = ['Date', 'MktPremium', 'SMB', 'HML', 'RF']
Momloc = '/Users/dramesh/Desktop/MIT_Coursework/Thesis/Raw_Data/FamaFrenchMom.CSV'
MomF = pd.read_csv(Momloc)
MomF.columns = ['Date', 'Mom']
factorData = ThreeF.merge(MomF, how = 'inner', on = 'Date')
factorData = factorData[factorData.Date > 199612]
factorData.Date = factorData.Date.astype('str')
factorData.set_index('Date', inplace = True)
factorData = factorData.loc['199801':'201612', :]
factorData = factorData[['MktPremium', 'HML', 'RF', 'Mom']]
factorData = factorData/100

stockDataLocation = '/Users/dramesh/Desktop/MIT_Coursework/Thesis/Raw_Data/RawSecurityData.csv'
stockData = pd.read_csv(stockDataLocation)
stockData = stockData[['Date', 'CUSIP', 'Ret1M']]
stockData.Date = pd.Series([d[0:4] + d[5:7] for d in stockData.Date])

stockData2 = stockData.pivot_table(index = 'Date', columns = 'CUSIP', values = 'Ret1M')
stockData2.columns.name = None
stockData2 = stockData2.loc['199801':, :]

finalStockData = stockData2.dropna(axis = 'columns')
finalStockData = finalStockData / 100

#Function that trains a model to minimize a choice of error and 
#predicts future returns. We train initially from 1998 to 2002 and then expand
#the window month by month
def forecaster(returns, ff, loss = 'MSE'):
            
        output = []
        dates = sorted(list(ff.index))
        dataset = ff.merge(returns, left_index = True, right_index = True)
        columnNames = ['MktPremium', 'HML', 'Mom']
        name = returns.columns.tolist()[0]
        
        i = dates.index('200201')
        
        for j in range(i,(len(dates))):
            trainData = dataset.loc['199801':dates[j], :]
            trainX = trainData[columnNames]
            trainY = trainData[[name]]
            model = LinearRegression()
            if loss == 'MSE':
                model = LinearRegression()
            if loss == 'Ridge':
                model = Ridge()
            if loss == 'Lasso':
                model = Lasso()
            if loss == 'Hub':
                model = HuberRegressor()
            if loss == 'ElasticNet':
                model = ElasticNet()
            model.fit(trainX, trainY)
            testData = pd.DataFrame(dataset.loc[dates[j], :]).T
            testX = testData[columnNames]
            prediction = model.predict(testX)
            if loss == 'LAD':
                model = QuantReg(endog = trainY, exog = trainX)
                res = model.fit(q = 0.5)
                prediction = model.predict(res.params, exog = testX)
            if loss == '1Q':
                model = QuantReg(endog = trainY, exog = trainX)
                res = model.fit(q = 0.25)
                prediction = model.predict(res.params, exog = testX)
            if loss == '3Q':
                model = QuantReg(endog = trainY, exog = trainX)
                res = model.fit(q = 0.75)
                prediction = model.predict(res.params, exog = testX)
                
            if loss in ['Lasso', 'Hub', 'ElasticNet', 'LAD', '1Q', '3Q']:
                output.append(prediction[0])
            else:
                output.append(prediction[0][0])
         
        return (name, output)

    
forecastedReturns = {}
securities = finalStockData.columns.tolist()
dates = sorted(list(finalStockData[[securities[0]]].index))
for s in securities:
    ans = forecaster(finalStockData[[s]], factorData)
    forecastedReturns[ans[0]] = ans[1]
ind = [d for d in dates if d > '200112' and d < '201701']


#Final dataframe of estimated returns from 2002 to end of 2016. 
#This will be fed into the mean varianc eoptimizer
estimates = pd.DataFrame.from_dict(forecastedReturns)
estimates.index = ind

ind.insert(0, '200112')

#Covariance matrix calculator/estimator
def covcalculator(returns):
    return np.cov(returns, rowvar = False)

#Basic MV optimizer with all assets fully invested
def basicMVO(returns, covmat, rp = 10):
    numOfAssets = covmat.shape[0]
    P = rp * covmat
    P = matrix(P, tc = 'd')
    q = np.array(returns)
    q = -1 * q
    q.shape = (numOfAssets, 1)
    q = matrix(q, tc = 'd')
    A = np.ones(numOfAssets)
    A.shape = (1,numOfAssets)
    G = -opt.matrix(np.eye(numOfAssets))   
    h = opt.matrix(0.0, (numOfAssets ,1))
    A = matrix(A, tc = 'd')
    b = np.array([1])
    b.shape = (1,1)
    b = matrix(b, tc = 'd')
    sol = solvers.qp(P = P, q = q, G = G, h = h, A = A, b = b)
    sol['x'] = sol['x'] / sum(sol['x'])
    return sol['x']

#Run MVO and compute returns
def retExtract(ans, ests):
    for i in range(len(ind) - 1):
        info = finalStockData.loc['199801':ind[i], :]
        covmat = covcalculator(info)
        returns = ests.loc[ind[i+1], :]
        wgts = basicMVO(returns, covmat)
        wgts = wgts.T
        ret = np.dot(wgts, finalStockData.loc[ind[i+1], :])
        ans.append(ret[0])
    return (ans)

mseReturns = []
mseReturns = retExtract(mseReturns, estimates)

def optReturns(loss):
    
    forecast = {}
    for s in securities:
        ans = forecaster(finalStockData[[s]], factorData, loss = loss)
        forecast[ans[0]] = ans[1]
    estimates = pd.DataFrame.from_dict(forecast)
    estimates.index = ind[1:]
    returns = []
    returns = retExtract(returns, estimates)
    return returns 
    

ridgeReturns = optReturns('Ridge')
lassoReturns = optReturns('Lasso')
elReturns = optReturns('ElasticNet')
ladReturns = optReturns('LAD')
q1Returns = optReturns('1Q')
q3Returns = optReturns('3Q')


def minF(x, regressors, targets):
    ans = np.dot(regressors, x) - np.array((targets.iloc[:,0]))
    ans.shape = (len(ans),)
    return (ans)

def scipyForecaster(returns, ff, loss = 'cauchy'):
    
        output = []
        dates = sorted(list(ff.index))
        dataset = ff.merge(returns, left_index = True, right_index = True)
        columnNames = ['MktPremium', 'HML', 'Mom']
        name = returns.columns.tolist()[0]
        #series = finalStockData.loc['199801':'200112', name]
        #series = list(series)
        #output = output + series
        
        i = dates.index('200201')
        
        for j in range(i,(len(dates))):
            trainData = dataset.loc['199801':dates[j], :]
            trainX = trainData[columnNames]
            trainY = trainData[[name]]
            x0 = np.array([1.0, 1.0, 1.0])
            model = ''
            if loss == 'cauchy':
                model = least_squares(minF, x0, loss = 'cauchy', f_scale = 0.1, args = (trainX, trainY))
            if loss == 'atan':
                model = least_squares(minF, x0, loss = 'arctan', f_scale = 0.1, args = (trainX, trainY))
            if loss == 'softl1':
                model = least_squares(minF, x0, loss = 'soft_l1', f_scale = 0.1, args = (trainX, trainY))
            if loss == 'huber':
                model = least_squares(minF, x0, loss = 'huber', f_scale = 0.1, args = (trainX, trainY))
                
            testData = pd.DataFrame(dataset.loc[dates[j], :]).T
            testX = testData[columnNames]
            prediction = np.dot(testX, model.x)
            output.append(prediction[0])
         
        return (name, output)

def scipyReturns(loss):
    
    forecast = {}
    for s in securities:
        ans = scipyForecaster(finalStockData[[s]], factorData, loss = loss)
        forecast[ans[0]] = ans[1]
    estimates = pd.DataFrame.from_dict(forecast)
    estimates.index = ind[1:]
    returns = []
    returns = retExtract(returns, estimates)
    return returns
    
cauchyReturns = scipyReturns('cauchy')
softl1Returns = scipyReturns('softl1')
arctanReturns = scipyReturns('atan')
huberReturns = scipyReturns('huber')

    
xaxis = pd.date_range('2002-01-01','2016-12-31', freq='MS').strftime("%Y-%b").tolist()

def summarizer(ls):
    print (np.mean(ls), np.std(ls), np.mean(ls) / np.std(ls), np.median(ls), np.percentile(ls, 25), np.percentile(ls, 75))
    
plt.plot(mseReturns, label = 'MSE')
plt.plot(ridgeReturns, label = 'Ridge')
plt.plot(lassoReturns, label = 'Lasso')
plt.plot(ladReturns, label = 'LAD')
plt.plot(cauchyReturns, label = 'Cauchy')
plt.plot(huberReturns, label = 'Huber')
plt.legend(loc='upper right')
plt.show()
    
    
    
    
    
    