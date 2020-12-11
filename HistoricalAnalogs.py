# -*- coding: utf-8 -*-
"""
Small script to illustrate Historical Analogs aka Time Series Clustering

RX data collected from Investing.com
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def similarity_test(target,test,criteria = 0.3):
    # simple measure = 1 - Pearson correlation
    
    # Simple improvement here would be to run correlation on relative changes rather than levels
    # i.e. run correlation on log(prices)
    measure = 1 - np.corrcoef(np.log(target),np.log(test))[0][1]
    
    if measure < criteria:
        return (True, measure)
    else:
        return (False, measure)

df = pd.read_csv("./RXData.csv")
df['Dates'] = pd.to_datetime(df['Date'])
df.set_index('Dates', inplace=True)
df_price = df['Price']


series_addon = 25
chunk_size = 100
target = df_price[:chunk_size]
chunks = [df_price[i-series_addon:i+chunk_size] for i in range(chunk_size -series_addon,len(df_price)-chunk_size)]



# Space Complexity O(2n)
fits = {} # Dict to store actual data chunk, used for plotting later
fit_measures = {} # Dict to store value from similarity test - We need this to get the best fit for overlapping periods


for (i,chunk) in enumerate(chunks):
    sim = similarity_test(target.to_numpy(),chunk[series_addon:chunk_size+series_addon].to_numpy())
    if sim[0]:
        # if chunk passes similarity test we need to check if overlapping period with existing fit
        # i.e. if any index +- chunk_size exists, then we need to compare and keep the best fit   
        
        fits[i] = chunk
        fit_measures[i] = sim[1]
        
        for n in range(i-chunk_size,i+chunk_size):
            x = fit_measures.get(n)
            
            if n == i:break
            
            if x != None and sim[1] < x:
                # Remove old fit 
                fits.pop(n) # Can use del here, key is guaranteed to exist, if not we would'nt be in this part of the code
                fit_measures.pop(n) # pop also works fine though
                
            elif x!= None and sim[1] >= x:
                fits.pop(i)
                fit_measures.pop(i)
                break
            
            else:
                #No conflict -> Add new fit
                fits[i] = chunk
                fit_measures[i] = sim[1]

  
with PdfPages('./HistAnalogsCharts.pdf') as export_pdf:
        
    #Sort by best fit - i.e. lowest similarity measure
    #And plot in that order
    for w in sorted(fit_measures.items(), key = lambda x: x[1]):
        
        plt.figure()
        fig,ax = plt.subplots()
        
        #Workaound to align x-axis in plot - Strictly speaking not very good practice...
        target.index = fits[w[0]][series_addon:chunk_size+series_addon].index  
        
        plt.xticks(rotation=70)
        ax.plot(np.flip(target), label="'Now'", color="blue")
        plt.legend(loc = "upper center")
        ax2 = ax.twinx()
        ax2.plot(np.flip(fits[w[0]]), label ="Historic fit (RHS)", color="red")
        plt.grid(True)
        plt.legend(loc = "upper left")
        plt.title("RX Distance: " + str(round(w[1],4)))
        fig.tight_layout()
        export_pdf.savefig()
        plt.close()
    