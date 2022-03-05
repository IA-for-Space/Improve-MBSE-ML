#AQUÍ TENEMOS QUE VER UN HISTOGRAMA DE CON QUÉ UMBRALES SE CONSIGUEN MÁS Y MENOS RELACIONES EN LOS NPWD
import numpy as np
from matplotlib import pyplot as plt
import math
import seaborn as sns

# function to create the expected distribution for one feature
def create_expected_distribution_for_one_feature(cols, bin_ratio=0.001):

    bins = create_expected_bin(cols.min(), cols.max(), bin_ratio)

    hist, bins, ppf, cdf = gen_histogram(cols, True, bins)

    print("sum ppf: ", sum(ppf))

    return hist, bins, ppf, cdf



# function to create the bins considering a ratio of separation
def create_expected_bin(min, max, bin_ratio=0.001):

    data_ratio = max / min
    
    n_bins = math.ceil( math.log(data_ratio) / math.log(bin_ratio))
    n_bins = n_bins + 1               # bin ranges are defined as [min, max)

    bins = np.full(n_bins, bin_ratio) # initialise the ratios for the bins limits
    bins[0] = min                     # initialise the lower limit for the 1st bin
    bins = np.cumprod(bins)           # generate bins

    print("max : ", max)
    print("min : ", min)
    print("bins length: ", len(bins))

    return bins


# Generate from an array:
# - the histogram
# - bins: range or placement on the number line
# - ppf: percent point function
# - cdf: cumulative distribution function. Not yet used.
def gen_histogram(dataset, bins = ''):
    draw = True
    if bins == '':
        hist, bins = np.histogram(dataset)
    else:
        hist, bins = np.histogram(dataset, bins)

    ppf = hist / sum(hist)

    cdf = np.cumsum(ppf)

    # printing histogram
    print()
    print("H:", hist) 
    print("ppf:", ppf) 
    print("bins:", bins) 

    if draw:
        # Creating plot
        fig = plt.figure(figsize =(10, 7))
        
        plt.hist(dataset, bins) #, weights=[1/len(dataset)] *len(dataset)) 

        # plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
        
        plt.title("Numpy Histogram") 

        # plotting PPF and CDF
        plt.plot(bins[1:], ppf, color="red", label="PPF")
        # plt.plot(bins[1:], cdf, label="CDF")
        plt.legend()
        
        # show plot
        plt.show()
		
    return hist, bins, ppf, cdf