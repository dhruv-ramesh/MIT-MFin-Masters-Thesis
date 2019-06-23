#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 11:11:40 2018

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
from statsmodels.tsa.ar_model import AR
from scipy.stats import norm

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
factorData = factorData[['Mkt.Rf', 'SMB', 'HML', 'RMW', 'CMA', 'RF', 'Mom']]
factorData = factorData/100

stockDataLocation = '/Users/dramesh/Desktop/MIT_Coursework/Thesis/Raw_Data/Raw.csv'
stockData = pd.read_csv(stockDataLocation)
stockData.RET = pd.to_numeric(stockData.RET, errors = 'coerce')
stockData.date = stockData.date.astype('str')
stockData.date = stockData.date.str.slice(0,6)
stockData['Ret'] = pd.to_numeric(stockData.DLRETX, errors = 'coerce').fillna(0.0) + stockData.RET.fillna(0.0)
stockData = stockData[['PERMNO', 'date', 'Ret']]

processedData = stockData.pivot_table(index = 'date', columns = 'PERMNO', values = 'Ret')
processedData.columns.name = None

securities = processedData.columns.tolist()

#Fitting AR(p) processes for factors. 
def factorEstimator(series):
    ans = []
    
    for i in range(0,(len(series) - 258)):
        arModel = AR(series[i:(i + 258)])
        res = arModel.fit(ic = 'bic')
        ans.append(arModel.predict(res.params)[-1])
    
    return pd.Series(list(series[:258]) + list(ans))

factors = ['Mkt.Rf', 'SMB', 'HML', 'RMW', 'CMA', 'Mom']
factorEstimateList = [factorEstimator(factorData[x]) for x in factors]
factorEstimates = pd.concat(factorEstimateList, axis = 1)
factorEstimates.columns = factors
factorEstimates.index = factorData.index

factorEstimates['MS'] = factorEstimates['Mkt.Rf']*factorEstimates['SMB']
factorEstimates['MH'] = factorEstimates['Mkt.Rf']*factorEstimates['HML']
factorEstimates['MR'] = factorEstimates['Mkt.Rf']*factorEstimates['RMW']
factorEstimates['MC'] = factorEstimates['Mkt.Rf']*factorEstimates['CMA']
factorEstimates['MM'] = factorEstimates['Mkt.Rf']*factorEstimates['Mom']
factorEstimates['SH'] = factorEstimates['HML']*factorEstimates['SMB']
factorEstimates['SR'] = factorEstimates['RMW']*factorEstimates['SMB']
factorEstimates['SC'] = factorEstimates['CMA']*factorEstimates['SMB']
factorEstimates['SM'] = factorEstimates['Mom']*factorEstimates['SMB']
factorEstimates['HR'] = factorEstimates['HML']*factorEstimates['RMW']
factorEstimates['HC'] = factorEstimates['HML']*factorEstimates['CMA']
factorEstimates['HM'] = factorEstimates['HML']*factorEstimates['Mom']
factorEstimates['RC'] = factorEstimates['CMA']*factorEstimates['RMW']
factorEstimates['RM'] = factorEstimates['Mom']*factorEstimates['RMW']
factorEstimates['CM'] = factorEstimates['CMA']*factorEstimates['Mom']
factorEstimates['RF'] = factorData['RF']

#Forecasting returns from for each security

def forecaster(returns, ff, nf = 6, length = 60, loss = 'MSE'):
    
    output = []
    factorLoadings = []
    varianceOfErrors = []
    df = ff.merge(returns, left_index = True, right_index = True)
    name = returns.columns.tolist()[0]
    df[name] = df[name] - df['RF']
    regressors = ['Mkt.Rf', 'SMB', 'HML', 'Mom', 'RMW', 'CMA']
    
    if nf == 3:
        regressors = ['Mkt.Rf', 'SMB', 'HML']
    
    if nf == 21:
        regressors = ff.columns.tolist()
        regressors.remove('RF')
          
    for j in range(length, len(df.index.tolist())):
        trainData = df.iloc[(j - length):j, :]
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
            factorLoadings.append(np.zeros((1,nf)))
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

def varCov(betas, variances, names, count, nf = 6, length = 60):
    numofAssets = len(names)
    arrays = []
    sigmas = []
    for n in range(numofAssets):
        arrays.append(betas[names[n]][count - 1])
        sigmas.append(variances[names[n]][count - 1])
    facs = ['Mkt.Rf', 'SMB', 'HML', 'Mom', 'RMW', 'CMA']
    if nf == 3:
        facs = ['Mkt.Rf', 'SMB', 'HML']
    if nf == 21:
        facs = factorEstimates.columns.tolist()
        facs.remove('RF')
        
    ff = factorEstimates[facs]
    ff = ff.iloc[(257 - length +count - 1):(257+count - 1),:]
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

def estimateCreator(loss, nf = 6, length = 60):
    forecastedReturns = {}
    factorLoadings = {}
    idiosyncraticVariance = {}
    dates = sorted(list(processedData[[securities[0]]].index))
    for s in securities:
        ans = forecaster(processedData[[s]], factorEstimates, nf, length, loss)
        forecastedReturns[ans[0]] = ans[1]
        factorLoadings[ans[0]] = ans[2]
        idiosyncraticVariance[ans[0]] = ans[3]
    
    if length == 60:
        ind = [d for d in dates if d > '198412' and d < '201807']
    if length == 30:
        ind = [d for d in dates if d > '198206' and d < '201807']
    if length == 90:
        ind = [d for d in dates if d > '198706' and d < '201807']
    if length == 120:
        ind = [d for d in dates if d > '198912' and d < '201807']
    estimates = pd.DataFrame.from_dict(forecastedReturns)
    estimates.index = ind
    
    return (estimates, factorLoadings, idiosyncraticVariance, ind)

def ewMinVarReturns(estimates, nf = 6, length = 60):
    
    ind = estimates[3]
    count = 1
    ewReturns = []
    minvarReturns = []
    
    for i in range(len(ind)):
        ests = estimates[0].loc[ind[i]:ind[i], :]
        ests = ests.dropna(axis = 1)
        names = ests.columns.tolist()
        assert len(names) != 0
        covmat = varCov(estimates[1], estimates[2], names, count, nf, length)
        count += 1
        minvarWgts = minVarOpt(covmat)
        actualReturns = processedData[names].loc[ind[i]:ind[i], :]
        actualReturns = np.array(actualReturns)
        actualReturns = np.nan_to_num(actualReturns)
        actualReturns.shape = (len(names), 1)
        ewReturns.append(np.mean(actualReturns))
        minvarReturns.append(np.dot(minvarWgts, actualReturns)[0])
    
    return (ewReturns, minvarReturns)

def MVOreturns(estimates, rp, nf = 6, length = 60):

    ind = estimates[3]
    count = 1
    mvoReturns = []
    
    for i in range(len(ind)):        
        ests = estimates[0].loc[ind[i]:ind[i], :]
        ests = ests.dropna(axis = 1)
        names = ests.columns.tolist()
        assert len(names) != 0
        covmat = varCov(estimates[1], estimates[2], names, count, nf, length)
        count += 1
        wgts = basicMVO(ests, covmat, rp)
        actualReturns = processedData[names].loc[ind[i]:ind[i], :]
        actualReturns = np.array(actualReturns)
        actualReturns = np.nan_to_num(actualReturns)
        actualReturns.shape = (len(names), 1) 
        mvoReturns.append(np.dot(wgts.T, actualReturns)[0, 0])
    
    return mvoReturns

def max_drawdown(returns):
    lr = np.log(1 + np.array(returns))
    cum_lr = np.cumsum(lr)
    cum_lr = list(cum_lr)
    maxd = 0
    for i in range(0, (len(cum_lr) - 1)):
        for j in range(i + 1, len(cum_lr)):
            if cum_lr[j] - cum_lr[i] < maxd:
                maxd = cum_lr[j] - cum_lr[i]
    return np.exp(maxd) - 1

def performanceStats (returns, EWR = None, t = 1):
    returns = np.array(returns)
    
    print ('Mean: ', np.mean(returns))
    print ('Std.Dev: ', np.std(returns))
    print ('Sharpe: ', np.mean(returns) / np.std(returns))
    negR = returns[returns < 0]    
    print('Sortino: ', np.mean(returns) / np.std(negR))
    if t != 1:
        EWR = np.array(EWR)
        print ('IR: ', np.mean(returns - EWR) / np.std(returns - EWR))
    print ('Fraction of Returns below 5%: ' ,
           len(returns[returns < -0.05]) / len(returns))
    print ('Fraction of Returns below 10%: ' ,
           len(returns[returns < -0.1]) / len(returns))
    print ('Fraction of Returns below 20%: ' ,
           len(returns[returns < -0.2]) / len(returns))
    print ('Max Drawdown: ', max_drawdown(list(returns)))
    print ('99% Var: ', norm.ppf(1-0.01)*np.std(returns)- np.mean(returns))
    print ('99% ES: ', 0.01**-1 * norm.pdf(norm.ppf(0.01))*np.std(returns) - np.mean(returns))
    return None

#mseEstimates330 = estimateCreator('MSE', 3, 30) 
#EW30Returns, mse330MinVarReturns = ewMinVarReturns(mseEstimates330, 3, 30)
#mse330mvoRets = MVOreturns(mseEstimates330, 1, 3, 30)   
#mseEstimates360 = estimateCreator('MSE', 3, 60) 
#EW60Returns, mse360MinVarReturns = ewMinVarReturns(mseEstimates360, 3, 60)
#mse360mvoRets = MVOreturns(mseEstimates360, 1, 3, 60) 
#mseEstimates390 = estimateCreator('MSE', 3, 90) 
#EW90Returns, mse390MinVarReturns = ewMinVarReturns(mseEstimates390, 3, 90)
#mse390mvoRets = MVOreturns(mseEstimates390, 1, 3, 90) 
#mseEstimates3120 = estimateCreator('MSE', 3, 120) 
#EW120Returns, mse3120MinVarReturns = ewMinVarReturns(mseEstimates3120, 3, 120)
#mse3120mvoRets = MVOreturns(mseEstimates3120, 1, 3, 120) 

#mseEstimates630 = estimateCreator('MSE', 6, 30) 
#EW30Returns, mse630MinVarReturns = ewMinVarReturns(mseEstimates630, 6, 30)
#mse630mvoRets = MVOreturns(mseEstimates630, 1, 6, 30)   
#mseEstimates660 = estimateCreator('MSE', 6, 60) 
#EW60Returns, mse660MinVarReturns = ewMinVarReturns(mseEstimates660, 6, 60)
#mse660mvoRets = MVOreturns(mseEstimates660, 1, 6, 60) 
#mseEstimates690 = estimateCreator('MSE', 6, 90) 
#EW90Returns, mse690MinVarReturns = ewMinVarReturns(mseEstimates690, 6, 90)
#mse690mvoRets = MVOreturns(mseEstimates690, 1, 6, 90) 
#mseEstimates6120 = estimateCreator('MSE', 6, 120) 
#EW120Returns, mse6120MinVarReturns = ewMinVarReturns(mseEstimates6120, 6, 120)
#mse6120mvoRets = MVOreturns(mseEstimates6120, 1, 6, 120) 

#mseEstimates2130 = estimateCreator('MSE', 21, 30) 
#EW30Returns, mse2130MinVarReturns = ewMinVarReturns(mseEstimates2130, 21, 30)
#mse2130mvoRets = MVOreturns(mseEstimates2130, 1, 21, 30)   
#mseEstimates2160 = estimateCreator('MSE', 21, 60) 
#EW60Returns, mse2160MinVarReturns = ewMinVarReturns(mseEstimates2160, 21, 60)
#mse2160mvoRets = MVOreturns(mseEstimates2160, 1, 21, 60) 
#mseEstimates2190 = estimateCreator('MSE', 21, 90) 
#EW90Returns, mse2190MinVarReturns = ewMinVarReturns(mseEstimates2190, 21, 90)
#mse2190mvoRets = MVOreturns(mseEstimates2190, 1, 21, 90) 
#mseEstimates21120 = estimateCreator('MSE', 21, 120) 
#EW120Returns, mse21120MinVarReturns = ewMinVarReturns(mseEstimates21120, 21, 120)
#mse21120mvoRets = MVOreturns(mseEstimates21120, 1, 21, 120) 

#ridgeEstimates330 = estimateCreator('Ridge', 3, 30) 
#EW30Returns, ridge330MinVarReturns = ewMinVarReturns(ridgeEstimates330, 3, 30)
#ridge330mvoRets = MVOreturns(ridgeEstimates330, 1, 3, 30)   
#ridgeEstimates360 = estimateCreator('Ridge', 3, 60) 
#EW60Returns, ridge360MinVarReturns = ewMinVarReturns(ridgeEstimates360, 3, 60)
#ridge360mvoRets = MVOreturns(ridgeEstimates360, 1, 3, 60) 
#ridgeEstimates390 = estimateCreator('Ridge', 3, 90) 
#EW90Returns, ridge390MinVarReturns = ewMinVarReturns(ridgeEstimates390, 3, 90)
#ridge390mvoRets = MVOreturns(ridgeEstimates390, 1, 3, 90) 
#ridgeEstimates3120 = estimateCreator('Ridge', 3, 120) 
#EW120Returns, ridge3120MinVarReturns = ewMinVarReturns(ridgeEstimates3120, 3, 120)
#ridge3120mvoRets = MVOreturns(ridgeEstimates3120, 1, 3, 120) 
#
#ridgeEstimates630 = estimateCreator('Ridge', 6, 30) 
#EW30Returns, ridge630MinVarReturns = ewMinVarReturns(ridgeEstimates630, 6, 30)
#ridge630mvoRets = MVOreturns(ridgeEstimates630, 1, 6, 30)   
#ridgeEstimates660 = estimateCreator('Ridge', 6, 60) 
#EW60Returns, ridge660MinVarReturns = ewMinVarReturns(ridgeEstimates660, 6, 60)
#ridge660mvoRets = MVOreturns(ridgeEstimates660, 1, 6, 60) 
#ridgeEstimates690 = estimateCreator('Ridge', 6, 90) 
#EW90Returns, ridge690MinVarReturns = ewMinVarReturns(ridgeEstimates690, 6, 90)
#ridge690mvoRets = MVOreturns(ridgeEstimates690, 1, 6, 90) 
#ridgeEstimates6120 = estimateCreator('Ridge', 6, 120) 
#EW120Returns, ridge6120MinVarReturns = ewMinVarReturns(ridgeEstimates6120, 6, 120)
#ridge6120mvoRets = MVOreturns(ridgeEstimates6120, 1, 6, 120) 
#
#ridgeEstimates2130 = estimateCreator('Ridge', 21, 30) 
#EW30Returns, ridge2130MinVarReturns = ewMinVarReturns(ridgeEstimates2130, 21, 30)
#ridge2130mvoRets = MVOreturns(ridgeEstimates2130, 1, 21, 30)   
#ridgeEstimates2160 = estimateCreator('Ridge', 21, 60) 
#EW60Returns, ridge2160MinVarReturns = ewMinVarReturns(ridgeEstimates2160, 21, 60)
#ridge2160mvoRets = MVOreturns(ridgeEstimates2160, 1, 21, 60) 
#ridgeEstimates2190 = estimateCreator('Ridge', 21, 90) 
#EW90Returns, ridge2190MinVarReturns = ewMinVarReturns(ridgeEstimates2190, 21, 90)
#ridge2190mvoRets = MVOreturns(ridgeEstimates2190, 1, 21, 90) 
#ridgeEstimates21120 = estimateCreator('Ridge', 21, 120) 
#EW120Returns, ridge21120MinVarReturns = ewMinVarReturns(ridgeEstimates21120, 21, 120)
#ridge21120mvoRets = MVOreturns(ridgeEstimates21120, 1, 21, 120) 

#lassoEstimates330 = estimateCreator('Lasso', 3, 30) 
#EW30Returns, lasso330MinVarReturns = ewMinVarReturns(lassoEstimates330, 3, 30)
#lasso330mvoRets = MVOreturns(lassoEstimates330, 1, 3, 30)   
#lassoEstimates360 = estimateCreator('Lasso', 3, 60) 
#EW60Returns, lasso360MinVarReturns = ewMinVarReturns(lassoEstimates360, 3, 60)
#lasso360mvoRets = MVOreturns(lassoEstimates360, 1, 3, 60) 
#lassoEstimates390 = estimateCreator('Lasso', 3, 90) 
#EW90Returns, lasso390MinVarReturns = ewMinVarReturns(lassoEstimates390, 3, 90)
#lasso390mvoRets = MVOreturns(lassoEstimates390, 1, 3, 90) 
#lassoEstimates3120 = estimateCreator('Lasso', 3, 120) 
#EW120Returns, lasso3120MinVarReturns = ewMinVarReturns(lassoEstimates3120, 3, 120)
#lasso3120mvoRets = MVOreturns(lassoEstimates3120, 1, 3, 120) 
#
#lassoEstimates630 = estimateCreator('Lasso', 6, 30) 
#EW30Returns, lasso630MinVarReturns = ewMinVarReturns(lassoEstimates630, 6, 30)
#lasso630mvoRets = MVOreturns(lassoEstimates630, 1, 6, 30)   
#lassoEstimates660 = estimateCreator('Lasso', 6, 60) 
#EW60Returns, lasso660MinVarReturns = ewMinVarReturns(lassoEstimates660, 6, 60)
#lasso660mvoRets = MVOreturns(lassoEstimates660, 1, 6, 60) 
#lassoEstimates690 = estimateCreator('Lasso', 6, 90) 
#EW90Returns, lasso690MinVarReturns = ewMinVarReturns(lassoEstimates690, 6, 90)
#lasso690mvoRets = MVOreturns(lassoEstimates690, 1, 6, 90) 
#lassoEstimates6120 = estimateCreator('Lasso', 6, 120) 
#EW120Returns, lasso6120MinVarReturns = ewMinVarReturns(lassoEstimates6120, 6, 120)
#lasso6120mvoRets = MVOreturns(lassoEstimates6120, 1, 6, 120) 
#
#lassoEstimates2130 = estimateCreator('Lasso', 21, 30) 
#EW30Returns, lasso2130MinVarReturns = ewMinVarReturns(lassoEstimates2130, 21, 30)
#lasso2130mvoRets = MVOreturns(lassoEstimates2130, 1, 21, 30)   
#lassoEstimates2160 = estimateCreator('Lasso', 21, 60) 
#EW60Returns, lasso2160MinVarReturns = ewMinVarReturns(lassoEstimates2160, 21, 60)
##lasso2160mvoRets = MVOreturns(lassoEstimates2160, 1, 21, 60) 
#ladEstimates2190 = estimateCreator('LAD', 21, 90) 
#EW90Returns, lad2190MinVarReturns = ewMinVarReturns(ladEstimates2190, 21, 90)
#lad2190mvoRets = MVOreturns(ladEstimates2190, 1, 21, 90) 
#ladEstimates21120 = estimateCreator('LAD', 21, 120) 
#EW120Returns, lad21120MinVarReturns = ewMinVarReturns(ladEstimates21120, 21, 120)
#lad21120mvoRets = MVOreturns(ladEstimates21120, 1, 21, 120) 

#q1Estimates330 = estimateCreator('1Q', 3, 30) 
#print('330 Done')
#EW30Returns, q1330MinVarReturns = ewMinVarReturns(q1Estimates330, 3, 30)
#q1330mvoRets = MVOreturns(q1Estimates330, 1, 3, 30)   
#q1Estimates360 = estimateCreator('1Q', 3, 60) 
#print('360 Done')
#EW60Returns, q1360MinVarReturns = ewMinVarReturns(q1Estimates360, 3, 60)
#q1360mvoRets = MVOreturns(q1Estimates360, 1, 3, 60) 
#q1Estimates390 = estimateCreator('1Q', 3, 90) 
#print('390 Done')
#EW90Returns, q1390MinVarReturns = ewMinVarReturns(q1Estimates390, 3, 90)
#q1390mvoRets = MVOreturns(q1Estimates390, 1, 3, 90) 
#q1Estimates3120 = estimateCreator('1Q', 3, 120) 
#print('3120 Done')
#EW120Returns, q13120MinVarReturns = ewMinVarReturns(q1Estimates3120, 3, 120)
#q13120mvoRets = MVOreturns(q1Estimates3120, 1, 3, 120) 
#
#q1Estimates630 = estimateCreator('1Q', 6, 30) 
#print('630 Done')
#EW30Returns, q1630MinVarReturns = ewMinVarReturns(q1Estimates630, 6, 30)
#q1630mvoRets = MVOreturns(q1Estimates630, 1, 6, 30)   
#q1Estimates660 = estimateCreator('1Q', 6, 60) 
#print('660 Done')
#EW60Returns, q1660MinVarReturns = ewMinVarReturns(q1Estimates660, 6, 60)
#q1660mvoRets = MVOreturns(q1Estimates660, 1, 6, 60) 
#q1Estimates690 = estimateCreator('1Q', 6, 90) 
#print('690 Done')
#EW90Returns, q1690MinVarReturns = ewMinVarReturns(q1Estimates690, 6, 90)
#q1690mvoRets = MVOreturns(q1Estimates690, 1, 6, 90) 
#q1Estimates6120 = estimateCreator('1Q', 6, 120) 
#print('6120 Done')
#EW120Returns, q16120MinVarReturns = ewMinVarReturns(q1Estimates6120, 6, 120)
#q16120mvoRets = MVOreturns(q1Estimates6120, 1, 6, 120) 
#
#q1Estimates2130 = estimateCreator('1Q', 21, 30) 
#print('2130 Done')
#EW30Returns, q12130MinVarReturns = ewMinVarReturns(q1Estimates2130, 21, 30)
#q12130mvoRets = MVOreturns(q1Estimates2130, 1, 21, 30)   
#q1Estimates2160 = estimateCreator('1Q', 21, 60) 
#print('2160 Done')
#EW60Returns, q12160MinVarReturns = ewMinVarReturns(q1Estimates2160, 21, 60)
#q12160mvoRets = MVOreturns(q1Estimates2160, 1, 21, 60) 
#q1Estimates2190 = estimateCreator('1Q', 21, 90) 
#print('2190 Done')
#EW90Returns, q12190MinVarReturns = ewMinVarReturns(q1Estimates2190, 21, 90)
#q12190mvoRets = MVOreturns(q1Estimates2190, 1, 21, 90) 
#q1Estimates21120 = estimateCreator('1Q', 21, 120) 
#print('2120 Done')
#EW120Returns, q121120MinVarReturns = ewMinVarReturns(q1Estimates21120, 21, 120)
#q121120mvoRets = MVOreturns(q1Estimates21120, 1, 21, 120) 
    

#q3Estimates330 = estimateCreator('3Q', 3, 30) 
#print('330 Done')
#EW30Returns, q3330MinVarReturns = ewMinVarReturns(q3Estimates330, 3, 30)
#q3330mvoRets = MVOreturns(q3Estimates330, 1, 3, 30)   
#q3Estimates360 = estimateCreator('3Q', 3, 60) 
#print('360 Done')
#EW60Returns, q3360MinVarReturns = ewMinVarReturns(q3Estimates360, 3, 60)
#q3360mvoRets = MVOreturns(q3Estimates360, 1, 3, 60) 
#q3Estimates390 = estimateCreator('3Q', 3, 90) 
#print('390 Done')
#EW90Returns, q3390MinVarReturns = ewMinVarReturns(q3Estimates390, 3, 90)
#q3390mvoRets = MVOreturns(q3Estimates390, 1, 3, 90) 
#q3Estimates3120 = estimateCreator('3Q', 3, 120) 
#print('3120 Done')
#EW120Returns, q33120MinVarReturns = ewMinVarReturns(q3Estimates3120, 3, 120)
#q33120mvoRets = MVOreturns(q3Estimates3120, 1, 3, 120) 
#
#q3Estimates630 = estimateCreator('3Q', 6, 30) 
#print('630 Done')
#EW30Returns, q3630MinVarReturns = ewMinVarReturns(q3Estimates630, 6, 30)
#q3630mvoRets = MVOreturns(q3Estimates630, 1, 6, 30)   
#q3Estimates660 = estimateCreator('3Q', 6, 60) 
#print('660 Done')
#EW60Returns, q3660MinVarReturns = ewMinVarReturns(q3Estimates660, 6, 60)
#q3660mvoRets = MVOreturns(q3Estimates660, 1, 6, 60) 
#q3Estimates690 = estimateCreator('3Q', 6, 90) 
#print('690 Done')
#EW90Returns, q3690MinVarReturns = ewMinVarReturns(q3Estimates690, 6, 90)
#q3690mvoRets = MVOreturns(q3Estimates690, 1, 6, 90) 
#q3Estimates6120 = estimateCreator('3Q', 6, 120) 
#print('6120 Done')
#EW120Returns, q36120MinVarReturns = ewMinVarReturns(q3Estimates6120, 6, 120)
#q36120mvoRets = MVOreturns(q3Estimates6120, 1, 6, 120) 
#
#q3Estimates2130 = estimateCreator('3Q', 21, 30) 
#print('2130 Done')
#EW30Returns, q32130MinVarReturns = ewMinVarReturns(q3Estimates2130, 21, 30)
#q32130mvoRets = MVOreturns(q3Estimates2130, 1, 21, 30)   
#q3Estimates2160 = estimateCreator('3Q', 21, 60) 
#print('2160 Done')
#EW60Returns, q32160MinVarReturns = ewMinVarReturns(q3Estimates2160, 21, 60)
#q32160mvoRets = MVOreturns(q3Estimates2160, 1, 21, 60) 
#q3Estimates2190 = estimateCreator('3Q', 21, 90) 
#print('2190 Done')
#EW90Returns, q32190MinVarReturns = ewMinVarReturns(q3Estimates2190, 21, 90)
#q32190mvoRets = MVOreturns(q3Estimates2190, 1, 21, 90) 
#q3Estimates21120 = estimateCreator('3Q', 21, 120) 
#print('2120 Done')
#EW120Returns, q321120MinVarReturns = ewMinVarReturns(q3Estimates21120, 21, 120)
#q321120mvoRets = MVOreturns(q3Estimates21120, 1, 21, 120) 
#

def minF(x, regressors, targets):
    ans = np.dot(regressors, x) - np.array((targets.iloc[:,0]))
    ans.shape = (len(ans),)
    return (ans)


def scipyForecaster(returns, ff, nf = 6, length = 60, loss = 'cauchy'):
    
    output = []
    factorLoadings = []
    varianceOfErrors = []
    df = ff.merge(returns, left_index = True, right_index = True)
    name = returns.columns.tolist()[0]
    df[name] = df[name] - df['RF']
    regressors = ['Mkt.Rf', 'SMB', 'HML', 'Mom', 'RMW', 'CMA']
    
    if nf ==3:
        regressors = ['Mkt.Rf', 'SMB', 'HML']
    
    if nf == 21:
        regressors = ff.columns.tolist()
        regressors.remove('RF')
    
    for j in range(length,len(df.index.tolist())):
        trainData = df.iloc[(j - length):j, :]
        trainX = trainData[regressors]
        trainY = trainData[[name]]
        x0 = np.array([1.0 for _ in range(0, nf)])
        
        if True == trainY.isnull().values.any():
            output.append(np.nan)
            factorLoadings.append(np.zeros((1,nf)))
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
        model.x.shape = (nf,1)
        varianceOfErrors.append(np.var(trainY - np.dot(trainX, model.x)).tolist()[0])
                
        testData = pd.DataFrame(df.iloc[j, :]).T
        testX = testData[regressors]
        prediction = np.dot(testX, model.x)
        output.append(prediction[0][0])
         
    return (name, output, factorLoadings, varianceOfErrors)

def scipyEstimateCreator(loss, nf = 6, length = 60):
    forecastedReturns = {}
    factorLoadings = {}
    idiosyncraticVariance = {}
    dates = sorted(list(processedData[[securities[0]]].index))
    for s in securities:
        ans = scipyForecaster(processedData[[s]], factorEstimates, nf, length, loss)
        forecastedReturns[ans[0]] = ans[1]
        factorLoadings[ans[0]] = ans[2]
        idiosyncraticVariance[ans[0]] = ans[3]
    
    if length == 60:
        ind = [d for d in dates if d > '198412' and d < '201807']
    if length == 30:
        ind = [d for d in dates if d > '198206' and d < '201807']
    if length == 90:
        ind = [d for d in dates if d > '198706' and d < '201807']
    if length == 120:
        ind = [d for d in dates if d > '198912' and d < '201807']
    estimates = pd.DataFrame.from_dict(forecastedReturns)
    estimates.index = ind
    
    return (estimates, factorLoadings, idiosyncraticVariance, ind)
    
#huberEstimates330 = scipyEstimateCreator('huber', 3, 30) 
#print('330 Done')
#EW30Returns, huber330MinVarReturns = ewMinVarReturns(huberEstimates330, 3, 30)
#huber330mvoRets = MVOreturns(huberEstimates330, 1, 3, 30)   
#huberEstimates360 = scipyEstimateCreator('huber', 3, 60) 
#print('360 Done')
#EW60Returns, huber360MinVarReturns = ewMinVarReturns(huberEstimates360, 3, 60)
#huber360mvoRets = MVOreturns(huberEstimates360, 1, 3, 60) 
#huberEstimates390 = scipyEstimateCreator('huber', 3, 90) 
#print('390 Done')
#EW90Returns, huber390MinVarReturns = ewMinVarReturns(huberEstimates390, 3, 90)
#huber390mvoRets = MVOreturns(huberEstimates390, 1, 3, 90) 
#huberEstimates3120 = scipyEstimateCreator('huber', 3, 120) 
#print('3120 Done')
#EW120Returns, huber3120MinVarReturns = ewMinVarReturns(huberEstimates3120, 3, 120)
#huber3120mvoRets = MVOreturns(huberEstimates3120, 1, 3, 120) 
#
#huberEstimates630 = scipyEstimateCreator('huber', 6, 30) 
#print('630 Done')
#EW30Returns, huber630MinVarReturns = ewMinVarReturns(huberEstimates630, 6, 30)
#huber630mvoRets = MVOreturns(huberEstimates630, 1, 6, 30)   
#huberEstimates660 = scipyEstimateCreator('huber', 6, 60) 
#print('660 Done')
#EW60Returns, huber660MinVarReturns = ewMinVarReturns(huberEstimates660, 6, 60)
#huber660mvoRets = MVOreturns(huberEstimates660, 1, 6, 60) 
#huberEstimates690 = scipyEstimateCreator('huber', 6, 90) 
#print('690 Done')
#EW90Returns, huber690MinVarReturns = ewMinVarReturns(huberEstimates690, 6, 90)
#huber690mvoRets = MVOreturns(huberEstimates690, 1, 6, 90) 
#huberEstimates6120 = scipyEstimateCreator('huber', 6, 120) 
#print('6120 Done')
#EW120Returns, huber6120MinVarReturns = ewMinVarReturns(huberEstimates6120, 6, 120)
#huber6120mvoRets = MVOreturns(huberEstimates6120, 1, 6, 120) 
#
#huberEstimates2130 = scipyEstimateCreator('huber', 21, 30) 
#print('2130 Done')
#EW30Returns, huber2130MinVarReturns = ewMinVarReturns(huberEstimates2130, 21, 30)
#huber2130mvoRets = MVOreturns(huberEstimates2130, 1, 21, 30)   
#huberEstimates2160 = scipyEstimateCreator('huber', 21, 60) 
#print('2160 Done')
#EW60Returns, huber2160MinVarReturns = ewMinVarReturns(huberEstimates2160, 21, 60)
#huber2160mvoRets = MVOreturns(huberEstimates2160, 1, 21, 60) 
#huberEstimates2190 = scipyEstimateCreator('huber', 21, 90) 
#print('2190 Done')
#EW90Returns, huber2190MinVarReturns = ewMinVarReturns(huberEstimates2190, 21, 90)
#huber2190mvoRets = MVOreturns(huberEstimates2190, 1, 21, 90) 
#huberEstimates21120 = scipyEstimateCreator('huber', 21, 120) 
#print('2120 Done')
#EW120Returns, huber21120MinVarReturns = ewMinVarReturns(huberEstimates21120, 21, 120)
#huber21120mvoRets = MVOreturns(huberEstimates21120, 1, 21, 120) 


#cauchyEstimates330 = scipyEstimateCreator('cauchy', 3, 30) 
#print('330 Done')
#EW30Returns, cauchy330MinVarReturns = ewMinVarReturns(cauchyEstimates330, 3, 30)
#cauchy330mvoRets = MVOreturns(cauchyEstimates330, 1, 3, 30)   
#cauchyEstimates360 = scipyEstimateCreator('cauchy', 3, 60) 
#print('360 Done')
#EW60Returns, cauchy360MinVarReturns = ewMinVarReturns(cauchyEstimates360, 3, 60)
#cauchy360mvoRets = MVOreturns(cauchyEstimates360, 1, 3, 60) 
#cauchyEstimates390 = scipyEstimateCreator('cauchy', 3, 90) 
#print('390 Done')
#EW90Returns, cauchy390MinVarReturns = ewMinVarReturns(cauchyEstimates390, 3, 90)
#cauchy390mvoRets = MVOreturns(cauchyEstimates390, 1, 3, 90) 
#cauchyEstimates3120 = scipyEstimateCreator('cauchy', 3, 120) 
#print('3120 Done')
#EW120Returns, cauchy3120MinVarReturns = ewMinVarReturns(cauchyEstimates3120, 3, 120)
#cauchy3120mvoRets = MVOreturns(cauchyEstimates3120, 1, 3, 120) 
#
#cauchyEstimates630 = scipyEstimateCreator('cauchy', 6, 30) 
#print('630 Done')
#EW30Returns, cauchy630MinVarReturns = ewMinVarReturns(cauchyEstimates630, 6, 30)
#cauchy630mvoRets = MVOreturns(cauchyEstimates630, 1, 6, 30)   
#cauchyEstimates660 = scipyEstimateCreator('cauchy', 6, 60) 
#print('660 Done')
#EW60Returns, cauchy660MinVarReturns = ewMinVarReturns(cauchyEstimates660, 6, 60)
#cauchy660mvoRets = MVOreturns(cauchyEstimates660, 1, 6, 60) 
#cauchyEstimates690 = scipyEstimateCreator('cauchy', 6, 90) 
#print('690 Done')
#EW90Returns, cauchy690MinVarReturns = ewMinVarReturns(cauchyEstimates690, 6, 90)
#cauchy690mvoRets = MVOreturns(cauchyEstimates690, 1, 6, 90) 
#cauchyEstimates6120 = scipyEstimateCreator('cauchy', 6, 120) 
#print('6120 Done')
#EW120Returns, cauchy6120MinVarReturns = ewMinVarReturns(cauchyEstimates6120, 6, 120)
#cauchy6120mvoRets = MVOreturns(cauchyEstimates6120, 1, 6, 120) 
#
#cauchyEstimates2130 = scipyEstimateCreator('cauchy', 21, 30) 
#print('2130 Done')
#EW30Returns, cauchy2130MinVarReturns = ewMinVarReturns(cauchyEstimates2130, 21, 30)
#cauchy2130mvoRets = MVOreturns(cauchyEstimates2130, 1, 21, 30)   
#cauchyEstimates2160 = scipyEstimateCreator('cauchy', 21, 60) 
#print('2160 Done')
#EW60Returns, cauchy2160MinVarReturns = ewMinVarReturns(cauchyEstimates2160, 21, 60)
#cauchy2160mvoRets = MVOreturns(cauchyEstimates2160, 1, 21, 60) 
#cauchyEstimates2190 = scipyEstimateCreator('cauchy', 21, 90) 
#print('2190 Done')
#EW90Returns, cauchy2190MinVarReturns = ewMinVarReturns(cauchyEstimates2190, 21, 90)
#cauchy2190mvoRets = MVOreturns(cauchyEstimates2190, 1, 21, 90) 
#cauchyEstimates21120 = scipyEstimateCreator('cauchy', 21, 120) 
#print('21120 Done')
#EW120Returns, cauchy21120MinVarReturns = ewMinVarReturns(cauchyEstimates21120, 21, 120)
#cauchy21120mvoRets = MVOreturns(cauchyEstimates21120, 1, 21, 120) 
#


#atanEstimates330 = scipyEstimateCreator('atan', 3, 30) 
#print('330 Done')
#EW30Returns, atan330MinVarReturns = ewMinVarReturns(atanEstimates330, 3, 30)
#atan330mvoRets = MVOreturns(atanEstimates330, 1, 3, 30)   
#atanEstimates360 = scipyEstimateCreator('atan', 3, 60) 
#print('360 Done')
#EW60Returns, atan360MinVarReturns = ewMinVarReturns(atanEstimates360, 3, 60)
#atan360mvoRets = MVOreturns(atanEstimates360, 1, 3, 60) 
#atanEstimates390 = scipyEstimateCreator('atan', 3, 90) 
#print('390 Done')
#EW90Returns, atan390MinVarReturns = ewMinVarReturns(atanEstimates390, 3, 90)
#atan390mvoRets = MVOreturns(atanEstimates390, 1, 3, 90) 
#atanEstimates3120 = scipyEstimateCreator('atan', 3, 120) 
#print('3120 Done')
#EW120Returns, atan3120MinVarReturns = ewMinVarReturns(atanEstimates3120, 3, 120)
#atan3120mvoRets = MVOreturns(atanEstimates3120, 1, 3, 120) 
#
#atanEstimates630 = scipyEstimateCreator('atan', 6, 30) 
#print('630 Done')
#EW30Returns, atan630MinVarReturns = ewMinVarReturns(atanEstimates630, 6, 30)
#atan630mvoRets = MVOreturns(atanEstimates630, 1, 6, 30)   
#atanEstimates660 = scipyEstimateCreator('atan', 6, 60) 
#print('660 Done')
#EW60Returns, atan660MinVarReturns = ewMinVarReturns(atanEstimates660, 6, 60)
#atan660mvoRets = MVOreturns(atanEstimates660, 1, 6, 60) 
#atanEstimates690 = scipyEstimateCreator('atan', 6, 90) 
#print('690 Done')
#EW90Returns, atan690MinVarReturns = ewMinVarReturns(atanEstimates690, 6, 90)
#atan690mvoRets = MVOreturns(atanEstimates690, 1, 6, 90) 
#atanEstimates6120 = scipyEstimateCreator('atan', 6, 120) 
#print('6120 Done')
#EW120Returns, atan6120MinVarReturns = ewMinVarReturns(atanEstimates6120, 6, 120)
#atan6120mvoRets = MVOreturns(atanEstimates6120, 1, 6, 120) 
#
#atanEstimates2130 = scipyEstimateCreator('atan', 21, 30) 
#print('2130 Done')
#EW30Returns, atan2130MinVarReturns = ewMinVarReturns(atanEstimates2130, 21, 30)
#atan2130mvoRets = MVOreturns(atanEstimates2130, 1, 21, 30)   
#atanEstimates2160 = scipyEstimateCreator('atan', 21, 60) 
#print('2160 Done')
#EW60Returns, atan2160MinVarReturns = ewMinVarReturns(atanEstimates2160, 21, 60)
#atan2160mvoRets = MVOreturns(atanEstimates2160, 1, 21, 60) 
#atanEstimates2190 = scipyEstimateCreator('atan', 21, 90) 
#print('2190 Done')
#EW90Returns, atan2190MinVarReturns = ewMinVarReturns(atanEstimates2190, 21, 90)
#atan2190mvoRets = MVOreturns(atanEstimates2190, 1, 21, 90) 
#atanEstimates21120 = scipyEstimateCreator('atan', 21, 120) 
#print('21120 Done')
#EW120Returns, atan21120MinVarReturns = ewMinVarReturns(atanEstimates21120, 21, 120)
#atan21120mvoRets = MVOreturns(atanEstimates21120, 1, 21, 120) 
    

#softl1Estimates330 = scipyEstimateCreator('softl1', 3, 30) 
#print('330 Done')
#EW30Returns, softl1330MinVarReturns = ewMinVarReturns(softl1Estimates330, 3, 30)
#softl1330mvoRets = MVOreturns(softl1Estimates330, 1, 3, 30)   
#softl1Estimates360 = scipyEstimateCreator('softl1', 3, 60) 
#print('360 Done')
#EW60Returns, softl1360MinVarReturns = ewMinVarReturns(softl1Estimates360, 3, 60)
#softl1360mvoRets = MVOreturns(softl1Estimates360, 1, 3, 60) 
#softl1Estimates390 = scipyEstimateCreator('softl1', 3, 90) 
#print('390 Done')
#EW90Returns, softl1390MinVarReturns = ewMinVarReturns(softl1Estimates390, 3, 90)
#softl1390mvoRets = MVOreturns(softl1Estimates390, 1, 3, 90) 
#softl1Estimates3120 = scipyEstimateCreator('softl1', 3, 120) 
#print('3120 Done')
#EW120Returns, softl13120MinVarReturns = ewMinVarReturns(softl1Estimates3120, 3, 120)
#softl13120mvoRets = MVOreturns(softl1Estimates3120, 1, 3, 120) 
#
#softl1Estimates630 = scipyEstimateCreator('softl1', 6, 30) 
#print('630 Done')
#EW30Returns, softl1630MinVarReturns = ewMinVarReturns(softl1Estimates630, 6, 30)
#softl1630mvoRets = MVOreturns(softl1Estimates630, 1, 6, 30)   
#softl1Estimates660 = scipyEstimateCreator('softl1', 6, 60) 
#print('660 Done')
#EW60Returns, softl1660MinVarReturns = ewMinVarReturns(softl1Estimates660, 6, 60)
#softl1660mvoRets = MVOreturns(softl1Estimates660, 1, 6, 60) 
#softl1Estimates690 = scipyEstimateCreator('softl1', 6, 90) 
#print('690 Done')
#EW90Returns, softl1690MinVarReturns = ewMinVarReturns(softl1Estimates690, 6, 90)
#softl1690mvoRets = MVOreturns(softl1Estimates690, 1, 6, 90) 
#softl1Estimates6120 = scipyEstimateCreator('softl1', 6, 120) 
#print('6120 Done')
#EW120Returns, softl16120MinVarReturns = ewMinVarReturns(softl1Estimates6120, 6, 120)
#softl16120mvoRets = MVOreturns(softl1Estimates6120, 1, 6, 120) 
#
#softl1Estimates2130 = scipyEstimateCreator('softl1', 21, 30) 
#print('2130 Done')
#EW30Returns, softl12130MinVarReturns = ewMinVarReturns(softl1Estimates2130, 21, 30)
#softl12130mvoRets = MVOreturns(softl1Estimates2130, 1, 21, 30)   
#softl1Estimates2160 = scipyEstimateCreator('softl1', 21, 60) 
#print('2160 Done')
#EW60Returns, softl12160MinVarReturns = ewMinVarReturns(softl1Estimates2160, 21, 60)
#softl12160mvoRets = MVOreturns(softl1Estimates2160, 1, 21, 60) 
#softl1Estimates2190 = scipyEstimateCreator('softl1', 21, 90) 
#print('2190 Done')
#EW90Returns, softl12190MinVarReturns = ewMinVarReturns(softl1Estimates2190, 21, 90)
#softl12190mvoRets = MVOreturns(softl1Estimates2190, 1, 21, 90) 
#softl1Estimates21120 = scipyEstimateCreator('softl1', 21, 120) 
#print('21120 Done')
#EW120Returns, softl121120MinVarReturns = ewMinVarReturns(softl1Estimates21120, 21, 120)
#softl121120mvoRets = MVOreturns(softl1Estimates21120, 1, 21, 120) 







