import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

class GBM:
    def __init__(self, ticker, start_date, end_date):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        self.log_returns = None
        self.mu = None
        self.sigma = None
    
    def download_data(self):
        self.data = yf.download(self.ticker, start=self.start_date, end=self.end_date)['Adj Close']
    
    # Calculate daily logarithmic returns
    def calculate_log_returns(self):
        if self.data is None:
            self.download_data()
        self.log_returns = np.log(1 + self.data.pct_change())
    
    # Calculate parameters of the GBM model
    def calculate_gbm_parameters(self):
        if self.log_returns is None:
            self.calculate_log_returns()
        self.mu = self.log_returns.mean()
        self.sigma = self.log_returns.std()
    
    # Simulate the GBM model
    def simulate_gbm(self, n_simulations=1000, simulation_days=50):
        if self.data is None:
            self.download_data()
        if self.mu is None or self.sigma is None:
            self.calculate_gbm_parameters()
            
        starting_price = self.data.iloc[-1]  # Current price
        simulations = []
        for _ in range(n_simulations):
            simulated_prices = [starting_price]
            for _ in range(1, simulation_days + 1):
                drift = self.mu - (0.5 * self.sigma**2)
                shock = self.sigma * np.random.normal()
                price = simulated_prices[-1] * np.exp(drift + shock)
                simulated_prices.append(price)
            simulations.append(simulated_prices)
        return simulations

    def visualize(self, simulations):
        if simulations == None:
            simulations = self.simulate_gbm()

        for i in range(len(simulations)):
            plt.plot(simulations[i])
        plt.xlabel('Days')
        plt.ylabel('Price')
        plt.title(f'Price Simulation using the GBM Model: {self.ticker}')
        plt.show()