import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

# Black-Scholes Option Pricing
def bs_price(S, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S/K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    else:
        return K * np.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def bs_delta(S, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S/K) + (r + 0.5 * sigma**2)*T)/(sigma*np.sqrt(T))
    if option_type == "call":
        return norm.cdf(d1)
    else:
        return -norm.cdf(-d1)

def bs_vega(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5 * sigma**2)*T)/(sigma*np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T)

def bs_gamma(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5 * sigma**2)*T)/(sigma*np.sqrt(T))
    return norm.pdf(d1)/(S * sigma * np.sqrt(T))


# Bond Pricing and DV01
def bond_price(face, rate, T, ytm):
    return sum([face * rate * np.exp(-ytm*t) for t in range(1, T+1)]) + face * np.exp(-ytm*T)

def dv01(face, rate, T, ytm):
    price = bond_price(face, rate, T, ytm)
    price_up = bond_price(face, rate, T, ytm + 0.0001)
    return price - price_up

# FX Forward Pricing
def fx_forward(S, r_d, r_f, T):
    return S * np.exp((r_d - r_f) * T)

# Define Portfolio Instruments
portfolio = {
    'equity_option': {'S': 100, 'K': 105, 'T': 0.5, 'r': 0.05, 'sigma': 0.25, 'option_type': 'call'},
    'bond': {'face': 1000, 'rate': 0.06, 'T': 5, 'ytm': 0.05},
    'fx_forward': {'S': 1.2, 'r_d': 0.03, 'r_f': 0.01, 'T': 1.0}
}

# Compute Risk Metrics
call = portfolio['equity_option']
call_price = bs_price(**call)
call_delta = bs_delta(**call)
call_vega = bs_vega(call['S'], call['K'], call['T'], call['r'], call['sigma'])
call_gamma = bs_gamma(call['S'], call['K'], call['T'], call['r'], call['sigma'])

bond = portfolio['bond']
bond_val = bond_price(**bond)
bond_dv01 = dv01(**bond)

fx = portfolio['fx_forward']
fx_val = fx_forward(**fx)

# Summary Table
summary = pd.DataFrame({
    'Instrument': ['Equity Option', 'Bond', 'FX Forward'],
    'Price': [call_price, bond_val, fx_val],
    'Delta': [call_delta, np.nan, np.nan],
    'Gamma': [call_gamma, np.nan, np.nan],
    'Vega': [call_vega, np.nan, np.nan],
    'DV01': [np.nan, bond_dv01, np.nan]
})

print("\nPortfolio Risk Metrics:\n")
print(summary.round(2))

# Visualization
S_range = np.linspace(80, 120, 100)
prices = [bs_price(S, call['K'], call['T'], call['r'], call['sigma'], call['option_type']) for S in S_range]

plt.figure(figsize=(8, 5))
plt.plot(S_range, prices, label='Call Option Price')
plt.title('Equity Option Price vs Stock Price')
plt.xlabel('Stock Price')
plt.ylabel('Option Price')
plt.grid(True)
plt.legend()
plt.savefig("option_price_plot.png")
plt.show()
