# -*- coding: utf-8 -*-
"""
Small script to illustrate Historical Analogs aka Time Series Clustering

RX data collected from Investing.com
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def similarity_test(target,test,criteria = 0.2):
    # simple measure = 1 - Pearson correlation
    measure = 1 - np.corrcoef(target,chunk)[0][1]
    
    if measure < criteria:
        return (True, measure)
    else:
        return (False, measure)

df = pd.read_csv("./RXData.csv")
dates = df['Date'].to_numpy()
df_price = df['Price']


chunk_size = 25
target = df_price[:chunk_size].to_numpy()
chunks = [df_price[i:i+chunk_size].to_numpy() for i in range(chunk_size,len(df_price)-chunk_size)]

# Space Complexity O(2n)
fits = {} # Dict to store actual data chunk, used for plotting later
fit_measures = {} # Dict to store value from similarity test - We need this to get the best fit for overlapping periods


for (i,chunk) in enumerate(chunks):
    sim = similarity_test(target,chunk)
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
                fit_measures.pop(n)
                
            elif x!= None and sim[1] >= x:
                fits.pop(i)
                fit_measures.pop(i)
                break
            
            else:
                #No conflict -> Add new fit
                fits[i] = chunk
                fit_measures[i] = sim[1]
                
# Sort Dictionaries by best fit
sorted_measures = {}
sorted_chunks = {}

for w in sorted(fit_measures.items(), key = lambda x: x[1]):
    sorted_measures[w[0]] = w[1]
    sorted_chunks[w[0]] = fits[w[0]]


with PdfPages('./TestCharts.pdf') as export_pdf:
    
    for chunk in sorted_chunks:
        plt.figure()
        fig,ax = plt.subplots()
        ax.plot(np.flip(target), label="Current", color="blue")
        ax2 = ax.twinx()
        ax2.plot(np.flip(sorted_chunks[chunk]), label="Historic Fit", color="red")
        #plt.plot(sorted_chunks[chunk],label="Historic Fit")
        plt.grid(True)
        export_pdf.savefig()
        plt.close()
