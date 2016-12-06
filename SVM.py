# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 13:58:41 2016

@author: ZHULI
"""
from pprint import pprint
from yahoo_finance import Share
import numpy as np, sys, random, math

#yahoo = Share('^IXIC') # NASDAQ Composite
#yahoo = Share('^GSPC') # S&P 500
#yahoo = Share('^FTSE') # FTSE 100
yahoo = Share('AAPL') # APPLE

def preprocess(data_raw, myPriceType):
    data_raw = sorted(data_raw, key=lambda k: k['Date']) # sort by date
    recentDays = 10
    periods = [25, 50, 100, 150, 200, 250]
    for prd in periods:
        histClose = []
        histHigh = []
        histLow = []
        histVolume = []
        myPriceType.append('Close_Avg_'+str(prd))
        myPriceType.append('Volume_Avg_'+str(prd))
        myPriceType.append('Volume_Rat_'+str(prd))
        myPriceType.append('Volume_Rat_recent_'+str(prd))
        myPriceType.append('Highest_'+str(prd))
        myPriceType.append('Highest_Rat_'+str(prd))
        myPriceType.append('Highest_Rat_recent_'+str(prd))
        myPriceType.append('Lowest_'+str(prd))
        myPriceType.append('Lowest_Rat_'+str(prd))
        myPriceType.append('Lowest_Rat_recent_'+str(prd))
        for day in data_raw:
            index = data_raw.index(day)
            averageClose = None
            averageVolme = None 
            volumeRatio = None # today's volume : past average volume
            volumeRatio_recent = None # recent (recentDays) days' average volume : past average volume
            highest = None
            highestRatio = None # today's high : past highest
            highestRatio_recent = None # recent (recentDays) days' high : past highest
            lowest = None
            lowestRatio = None # today's low : past lowest
            lowestRatio_recent = None # recent (recentDays) days' low : past lowest
            if len(histClose) == prd:
                averageClose = float(np.mean(histClose))
                averageVolme = float(np.mean(histVolume))
                volumeRatio = float(int(day['Volume'])/averageVolme)
                recentVol = []
                for i in range(recentDays):
                    recentVol.append(int(data_raw[index-i]['Volume']))
                volumeRatio_recent = float(np.mean(recentVol)/averageVolme)
                highest =  float(np.max(histHigh))
                highestRatio = float(float(day['High'])/highest)
                recentHigh = []
                for i in range(recentDays):
                    recentHigh.append(float(data_raw[index-i]['High']))
                highestRatio_recent = float(np.max(recentHigh)/highest)
                lowest = float(np.min(histLow))
                lowestRatio = float(float(day['Low'])/lowest)
                recentLow = []
                for i in range(recentDays):
                    recentLow.append(float(data_raw[index-i]['Low']))
                lowestRatio_recent = float(np.min(recentLow)/lowest)
            day['Close_Avg_'+str(prd)] = averageClose
            day['Volume_Avg_'+str(prd)] = averageVolme
            day['Volume_Rat_'+str(prd)] = volumeRatio
            day['Volume_Rat_recent_'+str(prd)] = volumeRatio_recent
            day['Highest_'+str(prd)] = highest
            day['Highest_Rat_'+str(prd)] = highestRatio
            day['Highest_Rat_recent_'+str(prd)] = highestRatio_recent
            day['Lowest_'+str(prd)] = lowest
            day['Lowest_Rat_'+str(prd)] = lowestRatio
            day['Lowest_Rat_recent_'+str(prd)] = lowestRatio_recent
            histClose.append(float(day['Close']))
            histHigh.append(float(day['High']))
            histLow.append(float(day['Low']))
            histVolume.append(int(day['Volume']))
            while len(histClose) > prd:
                del histClose[0]
                del histHigh[0]
                del histLow[0]
                del histVolume[0]
    
def get_data(data_all, priceType, startDate, endDate):
    predictDays = 30
    res = []
    lastClose = None
    for day in data_all:
        if lastClose is not None:
            if day['Date'] >= startDate and day['Date'] <= endDate:
                if data_raw.index(day) > len(data_raw) - predictDays:
                    break
                ans = data_raw[data_raw.index(day)+predictDays-1]['Adj_Close']
                if float(ans) > lastClose:
                    label = 1
                else:
                    label = -1
                value = [1]
                for pt in priceType:
                    value.append(float(day[pt]))
                res.append((label, value))
        lastClose = float(day['Adj_Close'])
    with open(startDate + '.txt', 'w') as file_out:
        for pair in res:
            print(pair,file = file_out)
    return res
	
def svm_train(data, dim, W):
    learningRate = 0.0001
    X = [0 for v in range(dim + 1)] # inputs
    grad = [0 for v in range(dim + 1)] # gradient
    num_train = len(data)
    for i in range(10):
        index = random.randint(0, num_train - 1)
        y = data[index][0]
        for j in range(dim + 1):
            X[j] = data[index][1][j]
        WX = 0.0
        for j in range(dim + 1):
            WX += W[j] * X[j]
        if 1 - WX *y > 0:
            for j in range(dim + 1):
                grad[j] = 0.0001 * W[j] - X[j] * y
        else:
            for j in range(dim + 1):
                grad[j] = 0.0001 * W[j] - 0
        # update 
        for j in range(dim + 1): 
            W[j] = W[j] - learningRate * grad[j]

def svm_predict(data, dim , W):
    num_test = len(data)
    num_correct = 0
    for i in range(num_test):
        target = data[i][0]
        X = data[i][1]
        sum = 0.0
        for j in range(dim + 1):
            sum += X[j] * W[j]
        predict = -1
        if sum > 0:
            predict = 1
        if predict * target > 0:
            num_correct += 1
    return num_correct * 1.0 / num_test	

if __name__ == "__main__":
    data_raw = yahoo.get_historical('2007-01-01', '2016-12-4')   
    myPriceType = []
    preprocess(data_raw, myPriceType)
    myPriceType = myPriceType + ['Close', 'Adj_Close', 'High', 'Open', 'Low', 'Volume']
    W = [0.0 for v in range(len(myPriceType) + 1)]
    data_train = get_data(data_raw, myPriceType, '2009-01-01', '2015-12-31') # taining data list
    data_test = get_data(data_raw, myPriceType, '2016-01-01', '2016-12-04') # test data list
    times = 10
    for i in range(times):
        svm_train(data_train, len(myPriceType), W)
        accuracy = svm_predict(data_test, len(myPriceType), W)
        print ("accuracy:%f"%(accuracy))
    sys.exit(0)