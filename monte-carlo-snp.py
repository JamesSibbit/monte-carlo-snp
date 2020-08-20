from pandas_datareader import data
import pandas as pd
import pandas_datareader.data as web
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import datetime, math, seaborn
from snp_data import snp_data

snp=snp_data("^GSPC")

def monte_carlo_sim(data):
    #Calculate log Returns
    log_returns = np.log(data['Close'].pct_change() + 1).dropna()

    #Calc some summary stats
    mu = log_returns.mean()
    sigma = log_returns.std()

    """We are going to model our stock process using lognormal random walk. Do so
    using dS = mu*S*dt + sigma*S*dX, where S is underlying, X(t) is normal Brownian
    motion. Let F=log(S), then solution is dF = (mu-0.5*sigma**2)dt + sigma*dX."""

    drift = mu - 0.5*(sigma**2)

    """We now need to create array of monte carlo trials for returns. Formula for
    solution of PDE is S = exp(drift*t + sigma*X), where X will hold an n x m array
    of random inverse std normal values of random floats between 0 and 1. We want to
    forecast prices for the next year (365 days) and compute 100 trials."""

    n_trials = 1000
    n_days = 10*365

    x = norm.ppf(np.random.rand(n_days, n_trials))

    returns_forecast = np.exp(drift + sigma*x)

    """Now, we need to get our most recent S&P500 value."""

    s_0 = data['Close'].iloc[-1]

    """Create array of zeros, same shape as returns_forecast, and set first row to s_0
    (as this will be the starting point for our simulations)."""

    stock_prices = np.zeros(returns_forecast.shape)
    stock_prices[0] = s_0

    for t in range(1,n_days):
        stock_prices[t] = stock_prices[t-1]*returns_forecast[t]

    return stock_prices

today = datetime.date.today() - datetime.timedelta(days=1)
ten_year = today + datetime.timedelta(days=10*365)
snp_sim = pd.DataFrame(monte_carlo_sim(snp), index = pd.date_range(start=today, end=ten_year, periods=10*365))
snp_sim['year'] = snp_sim.index.year

print(snp_sim.head())

value_vars = []
for i in range(1000):
    value_vars.append(i)

"""Plot yearly boxplots and time series of sims"""

df_plot = snp_sim.melt(id_vars='year', value_vars=value_vars)
fig, ax= plt.subplots(figsize=(10,7))
seaborn.boxplot(x='year', y='value', hue='year', data=df_plot, width=5)
plt.show()
fig, ax= plt.subplots(figsize=(10,7))
plt.plot(snp_sim)
plt.show()
