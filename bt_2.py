import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def black_scholes(S, K, T, r, sigma, option_type='call'):
    if T <= 0:
        if option_type == 'call':
            return 1.0 if S > K else 0.0 # max(S - K, 0)
        else:
            return -1.0 if S < K else 0.0 # max(K - S, 0)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    if option_type == 'call':
        delta = norm.cdf(d1)
    else:
        delta = -norm.cdf(-d1)
    print(delta,"-----")
    return delta

def calculate_real_pnl(data, K, r, sigma, option_type, use_custom_T=False, T_input=None):
    dates = data.index
    spot_prices = data['Spot'].values
    dt = (dates[-1] - dates[0]).days / 365.25
    
    cash = np.zeros(len(data))
    stock_qty = np.zeros(len(data))
    pnl = np.zeros(len(data))
    deltas = np.zeros(len(data))
    
    # calculate the initial time to expiration
    if use_custom_T:
        T_initial = T_input 
    else:
        T_initial = (dates[-1] - dates[0]).days / 365.25

    # Initial position
    # T_initial = (dates[-1] - dates[0]).days / 365.25

    deltas[0] = black_scholes(spot_prices[0], K, T_initial, r, sigma, option_type)
    cash[0] = 0  # assume the initial amount of cash is 0 

    for i in range(1, len(data)):
        days_to_expiry = (dates[-1] - dates[i]).days
        T = days_to_expiry / 365.25
        
        # calculate Delta
        deltas[i] = black_scholes(spot_prices[i], K, T, r, sigma, option_type)
        
        # calculate the change in assets
        delta_change = deltas[i] - deltas[i-1]
        cash[i] = cash[i-1] * np.exp(r*(dates[i]-dates[i-1]).days/365.25) - delta_change * spot_prices[i]
        
        # calculate PnL
        price_change = spot_prices[i] - spot_prices[i-1]
        pnl[i] = deltas[i-1] * price_change + cash[i-1] * (np.exp(r*(dates[i]-dates[i-1]).days/365.25) - 1)
        
    # calculate the payoff at expiration
    if option_type == 'call':
        payoff = max(spot_prices[-1] - K, 0)
    else:
        payoff = max(K - spot_prices[-1], 0)
    
    final_pnl = deltas[-1]*(spot_prices[-1] - spot_prices[-2]) + cash[-1] - payoff
    return pnl, deltas, final_pnl

# Streamlit frontend
st.title('Interactive Option Delta Hedging Backtest')

# upload file
uploaded_file = st.file_uploader("Upload Market Data (CSV)", type="csv")
if uploaded_file:
    data = pd.read_csv(uploaded_file, parse_dates=['Date'], index_col='Date')
    data.sort_index(inplace=True)
    
    st.subheader("Data Preview")
    st.write(data.head())
    
    # input parameters
    st.sidebar.header("Option Parameters")
    K = st.sidebar.number_input("Strike Price", value=data['Spot'].iloc[0])
    r = st.sidebar.number_input("Risk-free Rate", value=0.05)
    T = st.sidebar.number_input("Maturity", value=30)
    sigma = st.sidebar.number_input("Volatility", value=0.2)
    use_custom_T = st.sidebar.checkbox("Manually enter the expiration time (years)", False)
    option_type = st.sidebar.selectbox("Option Type", ['call', 'put'])

    if use_custom_T:
        T_input = st.sidebar.number_input("he expiration time (years)", 
                                        min_value=0.01, 
                                        max_value=10.0,
                                        value=0.25,
                                        step=0.01)
    else:
        st.sidebar.write("Automatically calculate expiration time based on data range")
    
    if st.button("Run Backtest"):
        if 'Spot' not in data.columns:
            st.error("Data must contain 'Spot' column")
        else:
            pnl, deltas, final_pnl = calculate_real_pnl(data, K, r, sigma, option_type, use_custom_T=use_custom_T, T_input=T_input if use_custom_T else None)
            cumulative_pnl = np.cumsum(pnl)
            
            # create interactive plot
            fig = make_subplots(rows=3, cols=1, 
                             shared_xaxes=True,
                             vertical_spacing=0.05,
                             subplot_titles=('Spot Price', 'Delta Position', 'Cumulative PnL'))
            
            # spot price of the underlying asset
            fig.add_trace(
                go.Scatter(x=data.index, y=data['Spot'], 
                         mode='lines', name='Spot Price',
                         line=dict(color='royalblue')),
                row=1, col=1
            )
            
            # the change of Delta 
            fig.add_trace(
                go.Scatter(x=data.index, y=deltas,
                         mode='lines', name='Delta',
                         line=dict(color='orange')),
                row=2, col=1
            )
            
            # accumulated PnL
            fig.add_trace(
                go.Scatter(x=data.index, y=cumulative_pnl,
                         mode='lines', name='Cumulative PnL',
                         line=dict(color='green')),
                row=3, col=1
            )
            
            # 添加最终损益标记
            fig.add_annotation(
                x=data.index[-1],
                y=final_pnl,
                text=f'Final PnL: {final_pnl:.2f}',
                showarrow=True,
                arrowhead=4,
                ax=-30,
                ay=-30,
                row=3,
                col=1
            )
            
            # update layout
            fig.update_layout(
                height=800,
                hovermode="x unified",
                template="plotly_white",
                margin=dict(t=40)
            )
            
            # set y axis
            fig.update_yaxes(title_text="Price", row=1, col=1)
            fig.update_yaxes(title_text="Delta", row=2, col=1)
            fig.update_yaxes(title_text="PnL", row=3, col=1)
            
            # show plot
            st.plotly_chart(fig, use_container_width=True)
            
            # show the statistics
            st.subheader("Performance Statistics")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Final PnL", f"{final_pnl:.2f}")
            col2.metric("Max Drawdown", f"{cumulative_pnl.min():.2f}")
            col3.metric("Volatility", f"{np.std(pnl):.2f}")
            col4.metric("Sharpe Ratio", 
                      f"{np.mean(pnl)/np.std(pnl):.2f}" if np.std(pnl)!=0 else "N/A")
            
            # interactive data table
            st.subheader("Interactive Data Table")
            result_df = pd.DataFrame({
                'Date': data.index,
                'Spot': data['Spot'],
                'Delta': deltas,
                'Daily PnL': pnl,
                'Cumulative PnL': cumulative_pnl
            })
            st.dataframe(
                result_df.style.format({
                    'Spot': '{:.2f}',
                    'Delta': '{:.4f}',
                    'Daily PnL': '{:.2f}',
                    'Cumulative PnL': '{:.2f}'
                }).background_gradient(cmap='RdYlGn', subset=['Daily PnL']),
                height=300
            )

else:
    st.info("""
    Please upload CSV file with columns: ['Date', 'Spot'] \n
    **Example**: \n
    Date,Spot  \n
    2023-01-03,100.0
    """, icon="ℹ️")