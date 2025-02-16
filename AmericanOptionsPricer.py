import numpy as np
import pandas as pd
from scipy.stats import norm
from GeometricBrownianMotion import GBM

def Value_American_Put_Option(future_prices, strike_price, Nsteps, num_paths, rate, time):
    time_step = time/Nsteps
    discount_factor = np.exp(-rate * time_step)   
    
    exercise_payoff = np.maximum(strike_price - future_prices, 0)             
    
    cash_flow = exercise_payoff.copy()
    cash_flow[:] = 0
    cash_flow.iloc[:,Nsteps] = exercise_payoff.iloc[:, Nsteps]
    
    for t in range(Nsteps-1,0,-1):
        table_t = pd.DataFrame({"Y":cash_flow.iloc[:,t+1]*discount_factor, "X":future_prices.iloc[:,t]})
        id_money_t = future_prices[future_prices.iloc[:, t] < strike_price].index
        
        table_t_inmoney=table_t.loc[id_money_t]
        rg_t = np.polyfit(table_t_inmoney["X"], table_t_inmoney["Y"], 2)
        C_t = np.polyval(rg_t, future_prices.loc[id_money_t].iloc[:,t])
    
        cash_flow.loc[id_money_t,t] = np.where(exercise_payoff.loc[id_money_t,t] > C_t, 
            exercise_payoff.loc[id_money_t,t], 0)

        for tt in range(t, Nsteps):
            cash_flow.loc[id_money_t,tt+1] = np.where(cash_flow.loc[id_money_t,t] > 0, 
                0, cash_flow.loc[id_money_t,tt+1])

    Sum_DCF = 0
    for t in range(Nsteps,0,-1):
        Sum_DCF += sum(cash_flow.loc[:,t])*np.exp(-time_step*rate*t)

    Option_Value = Sum_DCF/num_paths

    # return both cashflow and the price of the put option
    return cash_flow, Option_Value

n_simulations = 100
simulation_days = 5

gbm = GBM(ticker='AAPL', start_date='2024-06-01', end_date='2024-11-25')
simulations = gbm.simulate_gbm(n_simulations, simulation_days)
gbm.visualize(simulations)

price_paths = np.array(simulations)
paths_df = pd.DataFrame(data=price_paths, index=np.arange(1,n_simulations + 1))

future_prices = paths_df            # Prices predicted by GBM
strike_price = 232.5                # Strike price 
Nsteps = simulation_days            # Number of exercise times until end of horizon
time = simulation_days              # End of the horizon
num_paths = n_simulations           # Number of paths
r_val = 0.0443/252                  # Risk-free rate as of 11/25/24

CF,Value = Value_American_Put_Option(future_prices=future_prices, strike_price=strike_price, num_paths=num_paths, rate=r_val, time=time, Nsteps=Nsteps)

print("The Cash Flow matrix resulting from LSM method")
print(CF)
print("The Value of American Put Option is:  %.4f" % Value)