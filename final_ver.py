# -*- coding: utf-8 -*-
"""
Created on Tue May  3 13:12:14 2022

@author: George
"""

import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt

def net_value(ret):
    log_ret = np.log(1+ret)
    net_ret = np.exp( log_ret.cumsum() )
    return net_ret

os.chdir('D:\hkust\job_code\derivative_china')

data  = pd.read_csv('factor_ret.csv')
data['date'] = pd.to_datetime(data['date'].astype(str))

data_net = data.copy()
data.dropna(inplace=True)
data.reset_index(inplace=True,drop=True)

data_net = data_net[data_net['date'] >= '2008-12-24']
data_net.reset_index(inplace=True,drop=True)
data_net.iloc[0, 1:] =0
data_net.iloc[:, 1:] = data_net.iloc[:, 1:].apply(lambda x: net_value(x))


class Trader():
    
    def __init__(self, value = 1, cost = 0, factors = 40):
        
        self.value = value
        self.cost = cost
        self.factor_positions = np.zeros(factors)
        
        
    def allocate_position(self, positions):
        
        temp = self.factor_positions
        self.factor_positions = positions
        turnover = sum( np.abs(self.factor_positions - temp) )
        if self.validate_position():
            return turnover*self.value
    
    def validate_position(self):
        if sum(np.abs(self.factor_positions))>1.005:
            raise NameError('Sum of positions exceeds 1 !!!')
        if max(np.abs(self.factor_positions[:10])) > 0.2:
            raise NameError('style factors exceed risk cap!!!')
        if max(np.abs(self.factor_positions[11:])) > 0.05:
            raise NameError('industry factors exceed risk cap!!!')
        
        return True
    
    def calculate_value(self, factor_ret):
        self.value = self.value * ( 1 + np.sum(self.factor_positions * factor_ret) )
        ret = np.sum(self.factor_positions * factor_ret)
        return self.value, ret
    
class Brain():
    
    def __init__(self, data, data_net):
        self.ret_data = data
        self.ret_data_net = data_net
        self.market_view = 'normal'
        
    def view_market(self, short_days, long_days, date):
        date_index = int(np.where([self.ret_data['date'] == date])[1])
        short_ma = self.ret_data_net['country'].rolling(short_days).mean()[date_index]
        long_ma = self.ret_data_net['country'].rolling(long_days).mean()[date_index]
        if short_ma > long_ma:
            self.market_view = 'bull'
        else:
            self.market_view = 'bear'
        
    def mom(self, days, factors, start_date, end_date):
        factor_index = np.where(self.ret_data.columns.isin(factors))[0]
        start_index = int(np.where([self.ret_data['date'] == start_date])[1])
        end_index =   int(np.where([self.ret_data['date'] == end_date])[1])      
        ret = self.ret_data.iloc[start_index: end_index+1, factor_index]
        for factor in ret.columns:
            ret[factor] = ret[factor].rolling(window = days).mean() #- ret[factor].rolling(window = days).std()
        return ret



for short_day in [10]:
    for long_day in [10]:
    
        brain = Brain(data, data_net)  
        factors = list(data.columns[1 :11])  ## no country factor
        style_mom = brain.mom(150, factors, start_date, end_date).shift(1)
        factors = list(data.columns[12:]) 
        industry_mom = brain.mom(150, factors, start_date, end_date).shift(1)
        
        
        trader = Trader()
        porfolio = []
        porfolio_return = []
        style_positions = np.zeros(11)
        industry_positions = np.zeros(29)
        
        bull_factor_index = [0, 1, 5, 7, 4]
        bear_factor_index = [3, 6, 9, 0, 4]
        
        for i in brain.ret_data['date'].index:
            if i < 150:
                continue
            temp_style = list(style_mom.loc[i, :])
            temp_industry = list(industry_mom.loc[i, :])
            
            style_positions = np.zeros(11)
            industry_positions = np.zeros(29)
            if i % 20 == 0:
                if brain.market_view == 'bull':
                    temp_style = sorted(range(len(temp_style)), key=lambda i: temp_style[i])
                    
                    long_factors = []
                    short_factors = []
                    for j in range(len(temp_style)):
                        if (temp_style[len(temp_style) - 1 - j] in bull_factor_index) & (len(long_factors)<3) :
                            long_factors.append(temp_style[len(temp_style) - 1 - j])
                        if (temp_style[j] in bear_factor_index) & (len(short_factors)<2):
                            short_factors.append(temp_style[j])
                            
                    style_positions[long_factors] = [0.2, 0.1, 0.1]
                    style_positions[short_factors] = [-0.2, -0.1]
                    
                    long_factors = sorted(range(len(temp_industry)), key=lambda i: temp_industry[i])[-4:]
                    short_factors = sorted(range(len(temp_industry)), key=lambda i: temp_industry[i])[:2]
                    industry_positions[long_factors] = [0.05, 0.05, 0.025, 0.025]
                    industry_positions[short_factors] = [-0.05, -0.05]
                    
                elif brain.market_view == 'bear':
                    temp_style = sorted(range(len(temp_style)), key=lambda i: temp_style[i])
                    
                    long_factors = []
                    short_factors = []
                    for j in range(len(temp_style)):
                        if (temp_style[len(temp_style) - 1 - j] in bear_factor_index) & (len(long_factors)<2) :
                            long_factors.append(temp_style[len(temp_style) - 1 - j])
                        if (temp_style[j] in bull_factor_index) & (len(short_factors)<3):
                            short_factors.append(temp_style[j])
                            
                    style_positions[long_factors] = [0.2, 0.1]
                    style_positions[short_factors] = [-0.2, -0.1, -0.1]
                    
                    long_factors = sorted(range(len(temp_industry)), key=lambda i: temp_industry[i])[-2:]
                    short_factors = sorted(range(len(temp_industry)), key=lambda i: temp_industry[i])[:4]
                    industry_positions[long_factors] = [0.05, 0.05]
                    industry_positions[short_factors] = [-0.05, -0.05, -0.025, -0.025]
                
                else:
                    long_factors = sorted(range(len(temp_style)), key=lambda i: temp_style[i])[-2:]
                    short_factors = sorted(range(len(temp_style)), key=lambda i: temp_style[i])[:2]
                    style_positions[long_factors] = [0.2, 0.1]
                    style_positions[short_factors] = [-0.2, -0.1]
                    
                    long_factors = sorted(range(len(temp_industry)), key=lambda i: temp_industry[i])[-4:]
                    short_factors = sorted(range(len(temp_industry)), key=lambda i: temp_industry[i])[:4]
                    industry_positions[long_factors] = [0.05, 0.05, 0.05, 0.05]
                    industry_positions[short_factors] = [-0.05, -0.05, -0.05, -0.05]
                        
                positions = np.array(list(style_positions) + list(industry_positions))      
                trader.allocate_position(positions)
                
                print(brain.market_view)

            if i > 50:
                brain.view_market(20, 50, brain.ret_data['date'][i])  
                print(trader.factor_positions)

            p_value, p_return = trader.calculate_value(brain.ret_data.iloc[i,1:])
            porfolio.append(p_value)
            porfolio_return.append(p_return)
        
        sharp.append((np.array(porfolio_return).mean() ) / np.array(porfolio_return).std())
    
plt.plot(porfolio)
plt.plot(porfolio[start_index:end_index])
plt.plot(data_net['liquidity'])
plt.plot(data_net['country'][start_index:end_index])
plt.plot(data_net['liquidity'][start_index:end_index])
market_ret = data['country'][2681:3168]
(np.array(porfolio_return).mean() ) / (np.array(porfolio_return)).std() *np.sqrt(252)
data['beta'].mean() / data['beta'].std()



brain = Brain(data, data_net)  
factors = list(data.columns[1 :11])  ## no country factor
style_mom = brain.mom(150, factors, start_date, end_date).shift(1)
factors = list(data.columns[12:]) 
industry_mom = brain.mom(150, factors, start_date, end_date).shift(1)


trader = Trader()
porfolio = []
porfolio_return = []
style_positions = np.zeros(11)
industry_positions = np.zeros(29)

bull_factor_index = [0, 1, 5, 7, 4]
bear_factor_index = [3, 6, 9, 0, 4]

brain.view_market(20, 50, brain.ret_data['date'][10])  

for i in range(150, len(brain.ret_data)):
    
    temp_style = list(style_mom.loc[i, :])
    temp_industry = list(industry_mom.loc[i, :])
    
    style_positions = np.zeros(11)
    industry_positions = np.zeros(29)
    if i % 20 == 0:
        if brain.market_view == 'bull':
            temp_style = sorted(range(len(temp_style)), key=lambda i: temp_style[i])
            
            long_factors = []
            short_factors = []
            for j in range(len(temp_style)):
                if (temp_style[len(temp_style) - 1 - j] in bull_factor_index) & (len(long_factors)<3) :
                    long_factors.append(temp_style[len(temp_style) - 1 - j])
                if (temp_style[j] in bear_factor_index) & (len(short_factors)<2):
                    short_factors.append(temp_style[j])
                    
            style_positions[long_factors] = [0.2, 0.1, 0.1]
            style_positions[short_factors] = [-0.2, -0.1]
            
            long_factors = sorted(range(len(temp_industry)), key=lambda i: temp_industry[i])[-4:]
            short_factors = sorted(range(len(temp_industry)), key=lambda i: temp_industry[i])[:2]
            industry_positions[long_factors] = [0.05, 0.05, 0.025, 0.025]
            industry_positions[short_factors] = [-0.05, -0.05]
            
        elif brain.market_view == 'bear':
            temp_style = sorted(range(len(temp_style)), key=lambda i: temp_style[i])
            
            long_factors = []
            short_factors = []
            for j in range(len(temp_style)):
                if (temp_style[len(temp_style) - 1 - j] in bear_factor_index) & (len(long_factors)<2) :
                    long_factors.append(temp_style[len(temp_style) - 1 - j])
                if (temp_style[j] in bull_factor_index) & (len(short_factors)<3):
                    short_factors.append(temp_style[j])
                    
            style_positions[long_factors] = [0.2, 0.1]
            style_positions[short_factors] = [-0.2, -0.1, -0.1]
            
            long_factors = sorted(range(len(temp_industry)), key=lambda i: temp_industry[i])[-2:]
            short_factors = sorted(range(len(temp_industry)), key=lambda i: temp_industry[i])[:4]
            industry_positions[long_factors] = [0.05, 0.05]
            industry_positions[short_factors] = [-0.05, -0.05, -0.025, -0.025]
        
        
        positions = np.array(list(style_positions) + list(industry_positions))      
        trader.allocate_position(positions)
        
        print(brain.market_view)

        brain.view_market(20, 50, brain.ret_data['date'][i])  
        print(trader.factor_positions)

    p_value, p_return = trader.calculate_value(brain.ret_data.iloc[i,1:])
    porfolio.append(p_value)
    porfolio_return.append(p_return)


start_date, end_date = '2009-01-05', '2022-01-05'
start_date, end_date = '2021-01-05', '2022-01-05'
start_date, end_date = '2015-01-05', '2017-01-05'
start_index = int(np.where([data['date'] == start_date])[1])
end_index = int(np.where([data['date'] == end_date])[1])
plt.plot(data_net['country'][start_index: end_index])
x = list(brain.MA(20, 'country', start_date, end_date).index)
y1 = list(brain.MA(20, 'country', start_date, end_date))
y2 = list(brain.MA(50, 'country', start_date, end_date))
y3 = list(data_net['earnings_yield'][start_index:end_index])
y3.insert(0, y3[0])

plt.plot(x, y1, label='20 days MA',alpha=0.3)
plt.plot(x,y2, label='50 days MA')
plt.tick_params(axis = 'y', labelcolor = 'b')
plt.legend(loc = 'upper left')
ax2 = plt.twinx()
ax2.plot(x, y3, label = 'momentum')
plt.tick_params(axis = 'y')
plt.legend(loc = 'upper right')
plt.show()

plt.plot(data_net['country'][start_index:end_index].index,data_net['country'][start_index:end_index],label='Country')
plt.plot(data_net['momentum'][start_index:end_index].index,data_net['momentum'][start_index:end_index],label='momentum')
plt.legend(loc = 'upper left')
data['country'].rolling(window = 25).mean()
    
trader = Trader()

positions = np.zeros(40)
factors = [1, 2, 3, 8, 9, 6, 7]
positions[factors] = [0.2, 0.2, 0.1, 0.2, 0.1, 0.1, 0.1] 

factors = [i for i in range(10)]
positions[factors] = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1] 

trader.allocate_position(positions)

trader.factor_positions
porfolio = []

start_index = int(np.where([data['date'] == '2020-01-03'])[1])
for i in range(start_index, start_index+252):
    porfolio.append(trader.calculate_value(data.iloc[i,1:]))


plt.plot(data['date'][start_index:start_index+252], porfolio)
plt.title(data.columns[1:][factor])

start_index = int(np.where([data['date'] == '2008-12-25'])[1])
for i in range(len(data)):
    porfolio.append(trader.calculate_value(data.iloc[i,1:]))


plt.plot(data['date'][start_index:], porfolio)
plt.title(data.columns[1:][factor])

