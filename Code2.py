0#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 09:02:18 2018

@author: dramesh
"""

"""
Code for thesis
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, HuberRegressor
import cvxopt as opt
from cvxopt import solvers, matrix
from statsmodels.regression.quantile_regression import QuantReg
from scipy.optimize import least_squares
import warnings
from scipy.optimize import minimize

warnings.filterwarnings("ignore")
solvers.options['show_progress'] = False

#Loading in the data and processsing it
FFloc = '/Users/dramesh/Desktop/MIT_Coursework/Thesis/Raw_Data/FamaFrenchFactors.CSV'
ThreeF = pd.read_csv(FFloc)
ThreeF.columns = ['Date', 'Mkt.Rf', 'SMB', 'HML', 'RMW', 'CMA', 'RF']
Momloc = '/Users/dramesh/Desktop/MIT_Coursework/Thesis/Raw_Data/FamaFrenchMom.CSV'
MomF = pd.read_csv(Momloc)
MomF.columns = ['Date', 'Mom']
factorData = ThreeF.merge(MomF, how = 'inner', on = 'Date')
factorData.Date = factorData.Date.astype('str')
factorData.set_index('Date', inplace = True)
factorData = factorData[['Mkt.Rf', 'HML', 'RMW', 'CMA', 'RF', 'Mom']]
factorData = factorData/100

stockDataLocation = '/Users/dramesh/Desktop/MIT_Coursework/Thesis/Raw_Data/Raw.csv'
stockData = pd.read_csv(stockDataLocation)
stockData.RET = pd.to_numeric(stockData.RET, errors = 'coerce')
stockData.date = stockData.date.astype('str')
stockData.date = stockData.date.str.slice(0,6)
stockData['Ret'] = stockData.DLRET.fillna(0) + stockData.RET.fillna(0)
stockData = stockData[['PERMNO', 'date', 'Ret']]

processedData = stockData.pivot_table(index = 'date', columns = 'PERMNO', values = 'Ret')
processedData.columns.name = None


securities = processedData.columns.tolist()


#Forecasting returns from 1990 to 2018 for each security

def forecaster(returns, ff, loss = 'MSE'):
    
    output = []
    factorLoadings = []
    varianceOfErrors = []
    df = ff.merge(returns, left_index = True, right_index = True)
    name = returns.columns.tolist()[0]
    df[name] = df[name] - df['RF']
    regressors = ['Mkt.Rf', 'HML', 'Mom', 'RMW', 'CMA']
    
    for j in range(120, len(df.index.tolist())):
        trainData = df.iloc[(j - 120):j, :]
        trainX = trainData[regressors]
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
            
        if True == trainY.isnull().values.any():
            output.append(np.nan)
            factorLoadings.append(np.zeros((1,5)))
            varianceOfErrors.append(np.nan)
            continue 
        
        
        model.fit(trainX, trainY)
        
        res = ''
        
        if loss == 'LAD':
            model = QuantReg(endog = trainY, exog = trainX)
            res = model.fit(q = 0.5)
            
        if loss == '1Q':
            model = QuantReg(endog = trainY, exog = trainX)
            res = model.fit(q = 0.25)

        if loss == '3Q':
            model = QuantReg(endog = trainY, exog = trainX)
            res = model.fit(q = 0.75)
        
        if loss in ['LAD', '1Q', '3Q']:
            factorLoadings.append(np.array(res.params))
        else:
            factorLoadings.append(model.coef_)
            
        if loss not in ['Lasso', 'Hub', 'LAD', '1Q', '3Q']:
            varianceOfErrors.append(np.var(trainY - model.predict(trainX)).tolist()[0])
        if loss in ['Lasso', 'Hub']:
            varianceOfErrors.append(np.var(np.array(trainY) - model.predict(trainX)))
        if loss in ['LAD', '1Q', '3Q']:
            varianceOfErrors.append(np.var(model.predict(res.params, exog = trainX) - np.array(trainY)))
        
        testData = pd.DataFrame(df.iloc[j, :]).T
        testX = testData[regressors]
        
        if loss in ['LAD', '1Q', '3Q']:
            prediction = model.predict(res.params, exog = testX)
        else:
            prediction = model.predict(testX)
        
        if loss in ['Lasso', 'Hub', 'LAD', '1Q', '3Q']:
            output.append(prediction[0])
        else:
            output.append(prediction[0][0])
        
    return (name, output, factorLoadings, varianceOfErrors)

#Computing variance-covariance matrix of returns using factors 

def varCov(betas, variances, names, count):
    numofAssets = len(names)
    arrays = []
    sigmas = []
    for n in range(numofAssets):
        arrays.append(betas[names[n]][count - 1])
        sigmas.append(variances[names[n]][count - 1])
    facs = ['Mkt.Rf', 'HML', 'Mom', 'RMW', 'CMA']
    ff = factorData[facs]
    ff = ff.iloc[(318+count - 1):(438+count - 1),:]
    iii = np.eye(numofAssets)
    for k in range(len(sigmas)):
        iii[k,k] = sigmas[k]
    factorLoadings = np.vstack(arrays)  
    vc = np.matmul(factorLoadings, ff.cov())
    vc = np.matmul(vc, factorLoadings.T)   
    vc = vc + iii
    return vc

def calculatePorVol(w, Sigma):
    return ((np.matmul(np.nan_to_num(np.matmul(w, Sigma)), w.T))[0,0])

def minVarOpt(covmat):
    numOfAssets = covmat.shape[0]
    w0 = (1 / numOfAssets) * np.ones((1, numOfAssets))
    cons = ({'type': 'eq', 'fun': lambda x:  np.sum(x)-1.0})
    res= minimize(calculatePorVol, w0, args=np.matrix(covmat), method='SLSQP',constraints=cons)
    return res.x

def basicMVO(returns, covmat, rp = 1):
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



def estimateCreator(loss):
    forecastedReturns = {}
    factorLoadings = {}
    idiosyncraticVariance = {}
    dates = sorted(list(processedData[[securities[0]]].index))
    for s in securities:
        ans = forecaster(processedData[[s]], factorData, loss)
        forecastedReturns[ans[0]] = ans[1]
        factorLoadings[ans[0]] = ans[2]
        idiosyncraticVariance[ans[0]] = ans[3]
    
    ind = [d for d in dates if d > '199412' and d < '201802']
    estimates = pd.DataFrame.from_dict(forecastedReturns)
    estimates.index = ind
    
    return (estimates, factorLoadings, idiosyncraticVariance)

def ewMinVarReturns(estimates):
    
    dates = sorted(list(processedData[[securities[0]]].index))
    ind = [d for d in dates if d > '199412' and d < '201802']
    count = 1
    ewReturns = []
    minvarReturns = []
    
    for i in range(len(ind)):
        ests = estimates[0].loc[ind[i]:ind[i], :]
        ests = ests.dropna(axis = 1)
        names = ests.columns.tolist()
        assert len(names) != 0
        covmat = varCov(estimates[1], estimates[2], names, count)
        count += 1
        minvarWgts = minVarOpt(covmat)
        actualReturns = processedData[names].loc[ind[i]:ind[i], :]
        actualReturns = np.array(actualReturns)
        actualReturns = np.nan_to_num(actualReturns)
        actualReturns.shape = (len(names), 1)
        ewReturns.append(np.mean(actualReturns))
        minvarReturns.append(np.dot(minvarWgts, actualReturns)[0])
    
    return (ewReturns, minvarReturns)

def MVOreturns(estimates, rp):

    dates = sorted(list(processedData[[securities[0]]].index))
    ind = [d for d in dates if d > '199412' and d < '201802']
    count = 1
    mvoReturns = []
    
    for i in range(len(ind)):        
        ests = estimates[0].loc[ind[i]:ind[i], :]
        ests = ests.dropna(axis = 1)
        names = ests.columns.tolist()
        assert len(names) != 0
        covmat = varCov(estimates[1], estimates[2], names, count)
        count += 1
        wgts = basicMVO(ests, covmat, rp)
        actualReturns = processedData[names].loc[ind[i]:ind[i], :]
        actualReturns = np.array(actualReturns)
        actualReturns = np.nan_to_num(actualReturns)
        actualReturns.shape = (len(names), 1) 
        mvoReturns.append(np.dot(wgts.T, actualReturns)[0, 0])
    
    return mvoReturns


mseEstimates = estimateCreator('MSE') 

mseMinVarReturns = ewMinVarReturns(mseEstimates)[1]

msemvoRets = MVOreturns(mseEstimates, 10)       
    
ridgeEstimates = estimateCreator('Ridge')

ridgeEWReturns, ridgeMinVarReturns = ewMinVarReturns(ridgeEstimates)

ridge1mvoRets = MVOreturns(ridgeEstimates, 10)

lassoEstimates = estimateCreator('Lasso')

lassoEWReturns, lassoMinVarReturns = ewMinVarReturns(lassoEstimates)

lassomvoRets = MVOreturns(lassoEstimates, 10)

hubEstimates = estimateCreator('Hub')

hubEWReturns, hubMinVarReturns = ewMinVarReturns(hubEstimates)

hubmvoRets = MVOreturns(hubEstimates, 1)

hubDMR = doubleMVOreturns(hubEstimates, 10)

ladEstimates = estimateCreator('LAD')

ladEWReturns, ladMinVarReturns = ewMinVarReturns(ladEstimates)

ladmvoRets = MVOreturns(ladEstimates, 1)

q1Estimates = estimateCreator('1Q')

q1EWReturns, q1MinVarReturns = ewMinVarReturns(q1Estimates)

q1mvoRets = MVOreturns(q1Estimates, 1)

q3Estimates = estimateCreator('3Q')

q3EWReturns, q3MinVarReturns = ewMinVarReturns(q3Estimates)

q3mvoRets = MVOreturns(q3Estimates, 1)


def minF(x, regressors, targets):
    ans = np.dot(regressors, x) - np.array((targets.iloc[:,0]))
    ans.shape = (len(ans),)
    return (ans)


def scipyForecaster(returns, ff, loss = 'cauchy'):
    
    output = []
    factorLoadings = []
    varianceOfErrors = []
    df = ff.merge(returns, left_index = True, right_index = True)
    name = returns.columns.tolist()[0]
    df[name] = df[name] - df['RF']
    regressors = ['Mkt.Rf', 'HML', 'Mom', 'RMW', 'CMA']
    
    for j in range(120,len(df.index.tolist())):
        trainData = df.iloc[(j - 120):j, :]
        trainX = trainData[regressors]
        trainY = trainData[[name]]
        x0 = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        
        if True == trainY.isnull().values.any():
            output.append(np.nan)
            factorLoadings.append(np.zeros((1,5)))
            varianceOfErrors.append(np.nan)
            continue
               
        model = ''
        if loss == 'cauchy':
            model = least_squares(minF, x0, loss = 'cauchy', f_scale = 0.1, args = (trainX, trainY))
        if loss == 'atan':
            model = least_squares(minF, x0, loss = 'arctan', f_scale = 0.1, args = (trainX, trainY))
        if loss == 'softl1':
            model = least_squares(minF, x0, loss = 'soft_l1', f_scale = 0.1, args = (trainX, trainY))
        if loss == 'huber':
            model = least_squares(minF, x0, loss = 'huber', f_scale = 0.1, args = (trainX, trainY))
            
        factorLoadings.append(np.array(model.x))
        model.x.shape = (5,1)
        varianceOfErrors.append(np.var(trainY - np.dot(trainX, model.x)).tolist()[0])
                
        testData = pd.DataFrame(df.iloc[j, :]).T
        testX = testData[regressors]
        prediction = np.dot(testX, model.x)
        output.append(prediction[0][0])
         
    return (name, output, factorLoadings, varianceOfErrors)

def scipyEstimateCreator(loss):
    forecastedReturns = {}
    factorLoadings = {}
    idiosyncraticVariance = {}
    dates = sorted(list(processedData[[securities[0]]].index))
    for s in securities:
        ans = scipyForecaster(processedData[[s]], factorData, loss)
        forecastedReturns[ans[0]] = ans[1]
        factorLoadings[ans[0]] = ans[2]
        idiosyncraticVariance[ans[0]] = ans[3]
    
    ind = [d for d in dates if d > '199412' and d < '201802']
    estimates = pd.DataFrame.from_dict(forecastedReturns)
    estimates.index = ind
    
    return (estimates, factorLoadings, idiosyncraticVariance)
    

cauchyEstimates = scipyEstimateCreator('cauchy')
 
cauchyEWReturns, cauchyMinVarReturns = ewMinVarReturns(cauchyEstimates)

cauchymvoRets = MVOreturns(cauchyEstimates, 1)

atanEstimates = scipyEstimateCreator('atan')

atanEWReturns, atanMinVarReturns = ewMinVarReturns(atanEstimates)

atanmvoRets = MVOreturns(atanEstimates, 1)

softl1Estimates = scipyEstimateCreator('softl1')

softl1EWReturns, softl1MinVarReturns = ewMinVarReturns(softl1Estimates)

softl1mvoRets = MVOreturns(softl1Estimates, 1)









