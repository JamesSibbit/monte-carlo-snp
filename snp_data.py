from pandas_datareader import data
import pandas as pd
import pandas_datareader.data as web
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import datetime, math

def snp_data(index):
    today = datetime.date.today() - datetime.timedelta(days=1)
    last_three_years = today - datetime.timedelta(days=3*365)

    #Pull S&P data over last three years
    try:
        snp = web.DataReader(index, 'yahoo', last_three_years, today)
        print("Pulled SNP data")
        return snp
    except Exception as e:
        raise e
