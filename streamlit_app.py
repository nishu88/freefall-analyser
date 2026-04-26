# Date	     PnL		Instrument	Buy/Sell
# 1/31/2024   0			BANKNIFTY	Sell

import pandas as pd
import numpy as np
import plotly.graph_objs as go
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap
from thirdparty import calplot, quantstats_reports


# Flexible date parser to support multiple date formats
def parse_date_flexible(date_str):
    """Parse date string supporting DD-MM-YYYY and M/D/YYYY formats"""
    if pd.isna(date_str):
        return pd.NaT
    
    date_str = str(date_str).strip()
    
    # Try formats in order: DD-MM-YYYY, M/D/YYYY
    formats = ['%d-%m-%Y', '%m/%d/%Y']
    for fmt in formats:
        try:
            return pd.to_datetime(date_str, format=fmt)
        except (ValueError, TypeError):
            continue
    
    # If all formats fail, let pandas try to infer
    try:
        return pd.to_datetime(date_str)
    except:
        return pd.NaT


# Calculate Winning and Losing Streaks


def GetStreaks(tradesData):
    win_streak = 0
    loss_streak = 0
    max_win_streak = 0
    max_loss_streak = 0
    prev_trade_profit = 0
    for trade_profit in tradesData['PnL']:
        if trade_profit > 0:
            win_streak += 1
            loss_streak = 0
            if win_streak > max_win_streak:
                max_win_streak = win_streak
        elif trade_profit < 0:
            loss_streak += 1
            win_streak = 0
            if loss_streak > max_loss_streak:
                max_loss_streak = loss_streak
        else:
            win_streak = 0
            loss_streak = 0
        prev_trade_profit = trade_profit

    return max_win_streak, max_loss_streak


# Calculate Expectancy
def GetExpectancy(tradesData):
    winningTrades = tradesData[tradesData['PnL'] > 0]['PnL']
    losingTrades = tradesData[tradesData['PnL'] < 0]['PnL']

    winningTradesAvg = winningTrades.mean()
    losingTradesAvg = losingTrades.mean()

    winningTradesCount = len(winningTrades)
    losingTradesCount = len(losingTrades)

    totalTrades = winningTradesCount + losingTradesCount
    winningTradesPct = winningTradesCount / totalTrades
    losingTradesPct = losingTradesCount / totalTrades

    if losingTradesAvg == 0:
        return 0
    
    expectancy = (abs(winningTradesAvg / losingTradesAvg)
                  * winningTradesPct) - losingTradesPct

    return expectancy


def box(col, key, value, percentage=None, color='green'):
    with col:
        style = """
            <style>
            .metric__card {
                position: relative;
                padding: 0;
                height: 100%;
                border-radius: 12px;
                text-align: center;
                overflow: hidden;
                background: white;
                box-shadow: 0 2px 8px rgba(0,0,0,0.06);
                transition: all 0.2s ease;
            }
            .metric__card:hover {
                box-shadow: 0 4px 16px rgba(0,0,0,0.12);
                transform: translateY(-2px);
            }
            .metric__card .__title {
                padding: 8px 12px;
                font-size: 0.75rem;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }
            .metric__card .__value {
                padding: 12px 8px 14px 8px;
                font-size: 1.1rem;
                font-weight: 700;
                color: #1F2937;
            }
            .metric__green {
                border: 1px solid #10B981;
            }
            .metric__green .__title {
                color: #059669;
                background: linear-gradient(135deg, #D1FAE5 0%, #A7F3D0 100%);
                border-bottom: 1px solid #10B981;
            }
            .metric__yellow {
                border: 1px solid #F59E0B;
            }
            .metric__yellow .__title {
                color: #D97706;
                background: linear-gradient(135deg, #FEF3C7 0%, #FDE68A 100%);
                border-bottom: 1px solid #F59E0B;
            }
            .metric__red {
                border: 1px solid #EF4444;
            }
            .metric__red .__title {
                color: #DC2626;
                background: linear-gradient(135deg, #FEE2E2 0%, #FECACA 100%);
                border-bottom: 1px solid #EF4444;
            }
            .stats_percent {
                font-size: 0.8rem;
                font-weight: 500;
                opacity: 0.7;
                display: block;
                margin-top: 2px;
            }
            </style>
        """

        color_class = 'metric__yellow' if color == 'yellow' else ('metric__red' if color == 'red' else 'metric__green')
        percent = f'<span class="stats_percent">{percentage}</span>' if percentage is not None else ''
        st.markdown(
            f'<div class="metric__card {color_class}"><div class="__title">{key}</div><div class="__value">{value}{percent}</div></div>{style}',
            unsafe_allow_html=True
        )


def showStats(initialCapital: int, numOfFiles: int, tradesData: pd.DataFrame):
    if tradesData.empty:
        return

    overallPnL = tradesData['PnL'].sum()
    averageProfit = tradesData['PnL'].mean()
    maxProfit = tradesData['PnL'].max()
    maxLoss = tradesData['PnL'].min()
    
    # Advanced metrics
    wins = tradesData['PnL'][tradesData['PnL'] > 0].count()
    losses = tradesData['PnL'][tradesData['PnL'] < 0].count()
    breaks = tradesData['PnL'][tradesData['PnL'] == 0].count()
    totalCount = tradesData['PnL'].count()
    winPercentage = (wins / totalCount) * 100 if totalCount > 0 else 0
    lossPercentage = (losses / totalCount) * 100 if totalCount > 0 else 0
    
    # Profit factor calculation
    totalWins = tradesData[tradesData['PnL'] > 0]['PnL'].sum()
    totalLosses = abs(tradesData[tradesData['PnL'] < 0]['PnL'].sum())
    profitFactor = totalWins / totalLosses if totalLosses != 0 else 0
    
    # Average values
    averageProfitOnWins = tradesData['PnL'][tradesData['PnL'] > 0].mean()
    averageLossOnLosses = tradesData['PnL'][tradesData['PnL'] < 0].mean()
    
    # Monthly analysis
    monthlyProfit = tradesData.resample('ME', on='Date').sum(numeric_only=True)['PnL'].mean()
    monthlyCount = len(tradesData.groupby(tradesData['Date'].dt.to_period('M')))
    profitableMonths = len(tradesData.resample('ME', on='Date').sum(numeric_only=True)[tradesData.resample('ME', on='Date').sum(numeric_only=True)['PnL'] > 0])
    
    # ROI and CAGR
    roi = (overallPnL / initialCapital) * 100
    days_traded = (tradesData['Date'].max() - tradesData['Date'].min()).days
    years_traded = days_traded / 365.25 if days_traded > 0 else 0
    cagr = ((initialCapital + overallPnL) / initialCapital) ** (1 / years_traded) - 1 if years_traded > 0 else 0

    # Key metrics row 1
    col1, col2, col3, col4, col5 = st.columns(5, gap='small')
    box(col1, 'Initial Capital', f'{formatINR(initialCapital)}')
    box(col2, 'Overall Profit/Loss',
        f'{formatINR(overallPnL)}', f'{roi:.2f}%')
    box(col3, 'CAGR', f'{cagr*100:.2f}%',
        color='yellow')
    box(col4, 'Profit Factor', f'{profitFactor:.2f}')
    box(col5, 'Total Trades', f'{totalCount}')
    st.write('')

    # Key metrics row 2
    col1, col2, col3, col4, col5 = st.columns(5, gap='small')
    box(col1, 'Win Rate', f'{winPercentage:.2f}% ({wins})')
    box(col2, 'Loss Rate',
        f'{lossPercentage:.2f}% ({losses})', color='red')
    box(col3, 'Break Even', f'{(breaks/totalCount)*100:.2f}% ({breaks})',
        color='yellow')
    box(col4, 'Avg Win', f'{formatINR(averageProfitOnWins)}')
    box(col5, 'Avg Loss', f'{formatINR(averageLossOnLosses)}', color='red')
    st.write('')

    # Drawdown analysis
    cumulativePnL = tradesData['PnL'].cumsum()
    runningMaxPnL = cumulativePnL.cummax()
    drawdown = runningMaxPnL - cumulativePnL
    mdd = drawdown.max()
    mddPercent = (mdd / initialCapital) * 100

    # Calculate drawdown durations and keep track of start and end dates
    drawdown_durations = []
    drawdown_start_dates = []
    drawdown_end_dates = []

    prev_drawdown_idx = None
    for idx, pnl in enumerate(drawdown):
        if pnl > 0:
            if prev_drawdown_idx is None:
                prev_drawdown_idx = idx
        elif prev_drawdown_idx is not None:
            drawdown_start_date = tradesData['Date'].iloc[prev_drawdown_idx]
            drawdown_end_date = tradesData['Date'].iloc[idx - 1]
            drawdown_duration = (drawdown_end_date -
                                 drawdown_start_date).days + 1
            drawdown_durations.append(drawdown_duration)
            drawdown_start_dates.append(drawdown_start_date)
            drawdown_end_dates.append(drawdown_end_date)
            prev_drawdown_idx = None

    # Check if the last trade is still in a drawdown
    if prev_drawdown_idx is not None:
        drawdown_start_date = tradesData['Date'].iloc[prev_drawdown_idx]
        drawdown_end_date = tradesData['Date'].iloc[len(drawdown) - 1]
        drawdown_duration = (drawdown_end_date -
                            drawdown_start_date).days + 1
        drawdown_durations.append(drawdown_duration)
        drawdown_start_dates.append(drawdown_start_date)
        drawdown_end_dates.append(drawdown_end_date)

    # Filter out None values from drawdown_start_dates and drawdown_end_dates
    drawdown_start_dates = [
        date for date in drawdown_start_dates if date is not None]
    drawdown_end_dates = [
        date for date in drawdown_end_dates if date is not None]

    if not drawdown_start_dates or not drawdown_end_dates:  # Check if any drawdowns are found
        longest_drawdown_duration = 0
        longest_drawdown_start_date = None
        longest_drawdown_end_date = None
    else:
        # Find the index of the longest drawdown duration
        longest_drawdown_index = drawdown_durations.index(
            max(drawdown_durations))

        # Retrieve the longest drawdown duration and its corresponding start and end dates
        longest_drawdown_duration = drawdown_durations[longest_drawdown_index]
        longest_drawdown_start_date = drawdown_start_dates[longest_drawdown_index]
        longest_drawdown_end_date = drawdown_end_dates[longest_drawdown_index]

    # Retrieve the longest drawdown duration and its corresponding start and end dates
    mddDays = longest_drawdown_duration
    mddStartDate = longest_drawdown_start_date
    mddEndDate = longest_drawdown_end_date
    mddDateRange = f"{mddStartDate.strftime('%d %b %Y') if mddStartDate is not None else ''} - {mddEndDate.strftime('%d %b %Y') if mddEndDate is not None else ''}"

    # Calculate the Return to MDD ratio
    averageYearlyProfit = overallPnL / years_traded if years_traded > 0 else 0
    returnToMddRatio = abs(averageYearlyProfit / mdd) if mdd != 0 else None

    # Recovery factor
    recoveryFactor = overallPnL / mdd if mdd > 0 else 0

    # Risk-Adjusted Metrics row
    col1, col2, col3, col4 = st.columns(4, gap='small')
    box(col1, 'Max Drawdown', f'{formatINR(mdd)}',
        f'{mddPercent:.2f}%', color='red')
    box(col2, 'MDD Days', f'{mddDays}',
        mddDateRange, color='red')
    box(col3, 'Recovery Factor', f'{recoveryFactor:.2f}')
    box(col4, 'Return/MDD', 
        'Requires 1Y+' if returnToMddRatio is None else f'{returnToMddRatio:.2f}')
    st.write('')

    # Trading Statistics
    maxWinningStreak, maxLosingStreak = GetStreaks(tradesData)
    expectancy = GetExpectancy(tradesData)

    col1, col2, col3, col4 = st.columns(4, gap='small')
    box(col1, 'Max Win Streak', f'{maxWinningStreak}')
    box(col2, 'Max Loss Streak',
        f'{maxLosingStreak}', color='red')
    box(col3, 'Expectancy', f'{expectancy:.2f}')
    box(col4, 'Strategies', f'{numOfFiles}')
    st.write('')
    
    # Monthly Performance Table - Enhanced
    with st.expander('📊 Monthly Performance Details', expanded=False):
        monthly_data = tradesData.set_index('Date').resample('ME')['PnL'].agg(['sum', 'count', 'mean', 'std'])
        monthly_data.columns = ['Total PnL', 'Trades', 'Avg Trade', 'Std Dev']
        
        # Add ROI%, Win Rate, and Cumulative PnL
        monthly_wins = tradesData.set_index('Date').resample('ME')['PnL'].apply(lambda x: (x > 0).sum())
        monthly_data['Win Rate'] = (monthly_wins / monthly_data['Trades'] * 100).round(1)
        monthly_data['ROI %'] = (monthly_data['Total PnL'] / initialCapital * 100).round(2)
        monthly_data['Cumulative PnL'] = monthly_data['Total PnL'].cumsum()
        monthly_data.index = monthly_data.index.strftime('%b %Y')
        
        # Reorder columns
        monthly_data = monthly_data[['Total PnL', 'ROI %', 'Trades', 'Win Rate', 'Avg Trade', 'Std Dev', 'Cumulative PnL']]
        
        # Create styled display
        def color_pnl(val):
            if isinstance(val, (int, float)):
                color = '#10B981' if val > 0 else '#EF4444' if val < 0 else '#6B7280'
                return f'color: {color}; font-weight: 600'
            return ''
        
        display_monthly = monthly_data.copy()
        
        # Format columns
        for col in ['Total PnL', 'Avg Trade', 'Std Dev', 'Cumulative PnL']:
            display_monthly[col] = monthly_data[col].apply(lambda x: formatINR(x) if pd.notna(x) else '-')
        display_monthly['Trades'] = monthly_data['Trades'].astype(int)
        display_monthly['Win Rate'] = monthly_data['Win Rate'].apply(lambda x: f'{x:.1f}%' if pd.notna(x) else '-')
        display_monthly['ROI %'] = monthly_data['ROI %'].apply(lambda x: f'{x:+.2f}%' if pd.notna(x) else '-')
        
        st.markdown("""
        <style>
        .monthly-table th { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; }
        </style>
        """, unsafe_allow_html=True)
        
        st.dataframe(
            display_monthly,
            use_container_width=True,
            height=min(400, 35 * len(display_monthly) + 38)
        )
        
        # Summary stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            best_month = monthly_data['Total PnL'].idxmax()
            st.metric("Best Month", best_month, f"{formatINR(monthly_data['Total PnL'].max())}")
        with col2:
            worst_month = monthly_data['Total PnL'].idxmin()
            st.metric("Worst Month", worst_month, f"{formatINR(monthly_data['Total PnL'].min())}")
        with col3:
            avg_monthly_roi = monthly_data['ROI %'].mean()
            st.metric("Avg Monthly ROI", f"{avg_monthly_roi:.2f}%")
        with col4:
            profitable_pct = (monthly_data['Total PnL'] > 0).sum() / len(monthly_data) * 100
            st.metric("Profitable Months", f"{profitable_pct:.1f}%")
    
    # Advanced Metrics - Enhanced with Quant Metrics
    with st.expander('📈 Advanced Metrics & Quant Analysis', expanded=False):
        # Calculate additional quant metrics
        returns = tradesData['PnL'] / initialCapital
        std_dev = tradesData['PnL'].std()
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else 0
        
        # Sharpe Ratio (annualized, assuming daily returns)
        sharpe = (averageProfit / std_dev) * np.sqrt(252) if std_dev > 0 else 0
        
        # Sortino Ratio
        sortino = (averageProfit / (downside_std * initialCapital)) * np.sqrt(252) if downside_std > 0 else 0
        
        # Calmar Ratio
        calmar = (cagr / (mddPercent / 100)) if mddPercent > 0 else 0
        
        # Omega Ratio (threshold = 0)
        gains = tradesData['PnL'][tradesData['PnL'] > 0].sum()
        losses = abs(tradesData['PnL'][tradesData['PnL'] < 0].sum())
        omega = gains / losses if losses > 0 else 0
        
        # Tail Ratio
        percentile_95 = tradesData['PnL'].quantile(0.95)
        percentile_05 = tradesData['PnL'].quantile(0.05)
        tail_ratio = abs(percentile_95 / percentile_05) if percentile_05 != 0 else 0
        
        # Value at Risk (95%)
        var_95 = tradesData['PnL'].quantile(0.05)
        
        # Expected Shortfall / CVaR
        cvar = tradesData['PnL'][tradesData['PnL'] <= var_95].mean() if len(tradesData['PnL'][tradesData['PnL'] <= var_95]) > 0 else 0
        
        # Skewness and Kurtosis
        skewness = tradesData['PnL'].skew()
        kurtosis = tradesData['PnL'].kurtosis()
        
        # Kelly Criterion
        win_prob = wins / totalCount if totalCount > 0 else 0
        avg_win = averageProfitOnWins if pd.notna(averageProfitOnWins) else 0
        avg_loss = abs(averageLossOnLosses) if pd.notna(averageLossOnLosses) else 0
        kelly = (win_prob - ((1 - win_prob) / (avg_win / avg_loss))) if avg_loss > 0 and avg_win > 0 else 0
        
        # Create three columns for metrics
        st.markdown("#### 📊 Performance Metrics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div style="background: #F0FDF4; padding: 1rem; border-radius: 8px; border-left: 4px solid #10B981;">
                <h5 style="margin:0; color: #065F46;">Return Metrics</h5>
            </div>
            """, unsafe_allow_html=True)
            metrics1 = {
                'ROI': f'{roi:.2f}%',
                'CAGR': f'{cagr*100:.2f}%',
                'Avg Daily P&L': formatINR(averageProfit),
                'Avg Monthly P&L': formatINR(monthlyProfit),
                'Total Return': formatINR(overallPnL),
            }
            for k, v in metrics1.items():
                st.markdown(f"**{k}:** {v}")
        
        with col2:
            st.markdown("""
            <div style="background: #FEF3C7; padding: 1rem; border-radius: 8px; border-left: 4px solid #F59E0B;">
                <h5 style="margin:0; color: #92400E;">Risk-Adjusted Metrics</h5>
            </div>
            """, unsafe_allow_html=True)
            metrics2 = {
                'Sharpe Ratio': f'{sharpe:.3f}',
                'Sortino Ratio': f'{sortino:.3f}',
                'Calmar Ratio': f'{calmar:.3f}',
                'Omega Ratio': f'{omega:.3f}',
                'Profit Factor': f'{profitFactor:.2f}',
            }
            for k, v in metrics2.items():
                st.markdown(f"**{k}:** {v}")
        
        with col3:
            st.markdown("""
            <div style="background: #FEE2E2; padding: 1rem; border-radius: 8px; border-left: 4px solid #EF4444;">
                <h5 style="margin:0; color: #991B1B;">Risk Metrics</h5>
            </div>
            """, unsafe_allow_html=True)
            metrics3 = {
                'Max Drawdown': f'{mddPercent:.2f}%',
                'VaR (95%)': formatINR(var_95),
                'CVaR / ES': formatINR(cvar),
                'Std Deviation': formatINR(std_dev),
                'Recovery Factor': f'{recoveryFactor:.2f}',
            }
            for k, v in metrics3.items():
                st.markdown(f"**{k}:** {v}")
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 📉 Statistical & Trading Metrics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div style="background: #EEF2FF; padding: 1rem; border-radius: 8px; border-left: 4px solid #667eea;">
                <h5 style="margin:0; color: #4338CA;">Distribution Stats</h5>
            </div>
            """, unsafe_allow_html=True)
            metrics4 = {
                'Skewness': f'{skewness:.3f}',
                'Kurtosis': f'{kurtosis:.3f}',
                'Tail Ratio': f'{tail_ratio:.3f}',
                '95th Percentile': formatINR(percentile_95),
                '5th Percentile': formatINR(percentile_05),
            }
            for k, v in metrics4.items():
                st.markdown(f"**{k}:** {v}")
        
        with col2:
            st.markdown("""
            <div style="background: #F3E8FF; padding: 1rem; border-radius: 8px; border-left: 4px solid #9333EA;">
                <h5 style="margin:0; color: #6B21A8;">Trading Stats</h5>
            </div>
            """, unsafe_allow_html=True)
            metrics5 = {
                'Win Rate': f'{winPercentage:.2f}%',
                'Loss Rate': f'{lossPercentage:.2f}%',
                'Win Streak (Max)': f'{maxWinningStreak}',
                'Loss Streak (Max)': f'{maxLosingStreak}',
                'Expectancy': f'{expectancy:.3f}',
            }
            for k, v in metrics5.items():
                st.markdown(f"**{k}:** {v}")
        
        with col3:
            st.markdown("""
            <div style="background: #ECFDF5; padding: 1rem; border-radius: 8px; border-left: 4px solid #059669;">
                <h5 style="margin:0; color: #065F46;">Position Sizing</h5>
            </div>
            """, unsafe_allow_html=True)
            metrics6 = {
                'Kelly Criterion': f'{kelly*100:.2f}%',
                'Profit/Loss Ratio': f'{abs(avg_win/avg_loss):.2f}x' if avg_loss > 0 else 'N/A',
                'Avg Win': formatINR(avg_win),
                'Avg Loss': formatINR(-avg_loss),
                'Days Traded': f'{days_traded}',
            }
            for k, v in metrics6.items():
                st.markdown(f"**{k}:** {v}")
    
    st.write('')


def plotScatterMAE(tradesData):
    tradesData[['MAE', 'MFE', 'PnL']] = tradesData[[
        'MAE', 'MFE', 'PnL']].fillna(0)

    mae = tradesData['MAE']
    pnl = tradesData['PnL']

    # Check for collinearity
    correlation_matrix = tradesData[['MAE', 'MFE', 'PnL']].corr()
    if correlation_matrix.iloc[0, 1] > 0.9:
        st.error(
            "High collinearity detected between MAE and MFE. Consider removing one of the variables.")
        return

    # Handle missing or invalid data
    tradesData[['MAE', 'MFE', 'PnL']] = tradesData[[
        'MAE', 'MFE', 'PnL']].fillna(0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=abs(mae),
        y=abs(pnl),
        mode='markers',
        name='MAE',
        marker=dict(
            color=np.where(pnl < 0, '#EF4444', '#10B981'),
            size=8,
            opacity=0.7,
            line=dict(width=0.5, color='rgba(100,100,100,0.3)')
        ),
        hovertemplate='<b>MAE: ₹%{x:,.2f}</b><br>PnL: ₹%{y:,.2f}<extra></extra>'
    ))
    fig.update_layout(
        title=dict(text='<b>MAE vs PnL</b>', x=0.5, xanchor='center', font=dict(size=16)),
        xaxis_title='MAE (₹)',
        yaxis_title='PnL (₹)',
        hovermode='closest',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='white',
        font=dict(family='Arial, sans-serif', size=11),
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(200,200,200,0.2)', zeroline=False),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(200,200,200,0.2)', zeroline=False),
        margin=dict(l=60, r=40, t=70, b=60),
        height=450,
        showlegend=False
    )
    st.plotly_chart(fig, width='stretch')


def plotScatterMFE(tradesData):
    tradesData[['MAE', 'MFE', 'PnL']] = tradesData[[
        'MAE', 'MFE', 'PnL']].fillna(0)

    mfe = tradesData['MFE']
    pnl = tradesData['PnL']

    # Check for collinearity
    correlation_matrix = tradesData[['MAE', 'MFE', 'PnL']].corr()
    if correlation_matrix.iloc[0, 1] > 0.9:
        st.error(
            "High collinearity detected between MAE and MFE. Consider removing one of the variables.")
        return

    # Handle missing or invalid data
    tradesData[['MAE', 'MFE', 'PnL']] = tradesData[[
        'MAE', 'MFE', 'PnL']].fillna(0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=mfe,
        y=abs(pnl),
        mode='markers',
        name='MFE',
        marker=dict(
            color=np.where(pnl < 0, '#EF4444', '#10B981'),
            size=8,
            opacity=0.7,
            line=dict(width=0.5, color='rgba(100,100,100,0.3)')
        ),
        hovertemplate='<b>MFE: ₹%{x:,.2f}</b><br>PnL: ₹%{y:,.2f}<extra></extra>'
    ))
    fig.update_layout(
        title=dict(text='<b>MFE vs PnL</b>', x=0.5, xanchor='center', font=dict(size=16)),
        xaxis_title='MFE (₹)',
        yaxis_title='PnL (₹)',
        hovermode='closest',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='white',
        font=dict(family='Arial, sans-serif', size=11),
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(200,200,200,0.2)', zeroline=False),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(200,200,200,0.2)', zeroline=False),
        margin=dict(l=60, r=40, t=70, b=60),
        height=450,
        showlegend=False
    )
    st.plotly_chart(fig, width='stretch')


def customCmap():
    # Define the colors for the custom colormap
    red = (0.86, 0.08, 0.24)   # RGB values for extreme red
    white = (1.0, 1.0, 1.0)         # RGB values for white
    green = (0.0, 1.0, 0.0) # RGB values for extreme green

    # Create a custom color map using LinearSegmentedColormap
    return LinearSegmentedColormap.from_list('custom_map', [red, white, green], N=256)

def formatINR(number):
    number = float(number)
    number = round(number,2)
    is_negative = number < 0
    number = abs(number)
    s, *d = str(number).partition(".")
    r = ",".join([s[x-2:x] for x in range(-3, -len(s), -2)][::-1] + [s[-3:]])
    value = "".join([r] + d)
    if is_negative:
       value = '-' + value
    return '₹'+ value

def inject_custom_css():
    """Inject modern custom CSS for better UI appearance."""
    st.markdown("""
    <style>
    /* Modern Typography and Spacing */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Header styling */
    .app-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    .app-header h1 {
        color: white;
        margin: 0;
        font-weight: 700;
        font-size: 1.8rem;
    }
    .app-header p {
        color: rgba(255,255,255,0.85);
        margin: 0.3rem 0 0 0;
        font-size: 0.95rem;
    }
    
    /* Modern card styling */
    .stExpander {
        border: 1px solid #E5E7EB !important;
        border-radius: 12px !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important;
    }
    .stExpander > div:first-child {
        border-radius: 12px !important;
    }
    
    /* Better file uploader */
    .stFileUploader > div {
        border: 2px dashed #D1D5DB !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
        background: #FAFAFA !important;
        transition: all 0.2s ease;
    }
    .stFileUploader > div:hover {
        border-color: #667eea !important;
        background: #F5F3FF !important;
    }
    
    /* Improved buttons */
    .stButton > button {
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 0.5rem 1.5rem !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* Modern select boxes */
    .stSelectbox > div > div {
        border-radius: 8px !important;
    }
    
    /* Improved metrics cards */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        border: 1px solid #E5E7EB;
        transition: all 0.2s ease;
    }
    .metric-card:hover {
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
        transform: translateY(-2px);
    }
    
    /* Divider styling */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(to right, transparent, #E5E7EB, transparent);
        margin: 2rem 0;
    }
    
    /* Info/Warning boxes */
    .stAlert {
        border-radius: 10px !important;
    }
    
    /* DataFrame styling */
    .stDataFrame {
        border-radius: 10px !important;
        overflow: hidden;
    }
    
    /* Hide Streamlit footer only (keep hamburger menu for Print) */
    footer {visibility: hidden;}
    
    /* Section headers */
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #1F2937;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #667eea;
        display: inline-block;
    }
    </style>
    """, unsafe_allow_html=True)


def inject_print_styles():
    """Inject print-optimized styles."""
    st.markdown("""
    <style>
    @media print {
        .no-print, [data-testid="stSidebar"], header, footer, 
        .stDeployButton, [data-testid="stToolbar"], [data-testid="stHeader"],
        .stFileUploader, button { 
            display: none !important; 
        }
        .stApp { background: white !important; }
        .main .block-container { padding-top: 1rem !important; max-width: 100% !important; }
        .element-container { break-inside: avoid; }
    }
    </style>
    """, unsafe_allow_html=True)


def render_screenshot_button():
    """Render save as PDF button with popover instructions."""
    with st.popover("🖨️ Save as PDF"):
        st.markdown("""
        **To save this report as PDF:**
        
        1. Press **Ctrl+P** (Windows/Linux) or **⌘+P** (Mac)
        2. Set Destination to **"Save as PDF"**
        3. Click **Save**
        
        💡 *Tip: Expand all sections you want to include before printing*
        """)
        st.info("The page is print-optimized - buttons and sidebars will be hidden automatically.")


def main():
    st.set_page_config(
        page_title="Trades Analyzer", 
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    inject_custom_css()
    inject_print_styles()
    
    # Modern header with screenshot button
    col_header, col_btn = st.columns([6, 1])
    with col_header:
        st.markdown("""
        <div class="app-header">
            <h1>📊 Trades Analyzer</h1>
            <p>Comprehensive trading performance analytics and visualization</p>
        </div>
        """, unsafe_allow_html=True)
    with col_btn:
        st.markdown("<div style='padding-top: 1rem;'></div>", unsafe_allow_html=True)
        render_screenshot_button()
    
    col1, col, col2 = st.columns([1, 8, 1])

    with st.expander('📚 How to Use', expanded=False):
        st.info("Upload .csv in the following format: Date, PnL, Instrument, Buy/Sell")
        
        # Date	     PnL		Instrument	Buy/Sell
        # 1/31/2024   0			BANKNIFTY	Sell

        data = {
                "Date": ["31-12-2023","2/26/2024","3/27/2024" ],
                "PnL": [5000.63, -4500.85, 5045],
                "Instrument": ["BANKNIFTY","BANKNIFTY","FINNIFTY"],
                "Buy/Sell": ["Sell","Buy","Sell"]
            }
        df = pd.DataFrame(data)
        st.table(df)

        try:
            with open('sample.csv', 'r') as f:
                csv_content = f.read()
            st.download_button('Download SAMPLE CSV', csv_content, "sample.csv")
        except FileNotFoundError:
            st.warning("Sample CSV file not found") 

        st.warning("You can Group by these SCRIPS: (BANKNIFTY, NIFTY, FINNIFTY, MIDCPNIFTY, SENSEX, BANKEX)") 
        st.success("📄 **Print to PDF:** Click ☰ menu (top-right) → Print → Save as PDF")
        st.info("You can add multiple CSV files to aggregate strategies")


    with col:
        uploadedFiles = st.file_uploader(
            "📁 Upload Trade CSV Files",
            key="1",
            help="Upload one or more CSV files with your trading data. Multiple files will be aggregated.",
            accept_multiple_files=True
        )
        if len(uploadedFiles) == 0:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #F0F9FF 0%, #E0F2FE 100%); 
                        border: 2px dashed #0EA5E9;
                        padding: 2rem;
                        border-radius: 12px;
                        text-align: center;
                        margin: 1rem 0;">
                <p style="margin: 0; color: #0369A1; font-size: 1.1rem; font-weight: 600;">
                    👆 Upload your trades CSV file to get started
                </p>
                <p style="margin: 0.5rem 0 0 0; color: #0284C7; font-size: 0.9rem;">
                    Supports multiple files for strategy aggregation
                </p>
            </div>
            """, unsafe_allow_html=True)
            st.stop()

        # List to store the dataframes
        dataframes = []

        for uploaded_file in uploadedFiles:
            # Read each CSV file as a dataframe
            df = pd.read_csv(uploaded_file)

            df = df[df['PnL'] != 0]
            df['Entry Date/Time'] = df['Date']
            df['Exit Date/Time'] = df['Date']
            # Append the dataframe to the list
            dataframes.append(df)

        tradesData = pd.concat(dataframes)
        
        # Filter out zero PnL trades completely
        tradesData = tradesData[tradesData['PnL'] != 0].reset_index(drop=True)
        
        if tradesData.empty:
            st.warning("No trades with non-zero PnL found in uploaded files.")
            st.stop()
        
        tradesData.sort_values(by="Entry Date/Time", inplace=True)
        tradesData.reset_index(drop=True, inplace=True)

        tradesData["Entry Date/Time"] = tradesData["Entry Date/Time"].astype(str).apply(parse_date_flexible)
        tradesData["Exit Date/Time"] = tradesData["Exit Date/Time"].astype(str).apply(parse_date_flexible)
        tradesData["Date"] = tradesData["Date"].astype(str).apply(parse_date_flexible)

        groupCol, initialCapitalCol, fromDateCol, toDateCol, slippageCol = st.columns([
            1, 1, 2, 2, 2])

        with groupCol:
            selectedGroupCriteria = st.selectbox(
                'Group by', ('Date', 'Day'))

        with initialCapitalCol:
            initialCapital = st.text_input(
                'Initial Capital', placeholder='200000')
            if initialCapital is not None and initialCapital != '':
                initialCapital = int(float(initialCapital))
            else:
                return

        min_date = tradesData["Entry Date/Time"].min().date()
        max_date = tradesData["Exit Date/Time"].max().date()
        with fromDateCol:
            selected_from_date = st.date_input("From Date", min_value=min_date, max_value=max_date,
                                               value=min_date)

        with toDateCol:
            selected_to_date = st.date_input("To Date", min_value=min_date, max_value=max_date,
                                             value=max_date)

        tradesData = tradesData[(tradesData['Entry Date/Time'].dt.date >= selected_from_date) &
                                (tradesData['Exit Date/Time'].dt.date <= selected_to_date)]

        # with slippageCol:
        #     slippage = st.slider('Slippage %', 0.0, 5.0, 0.0, 0.1)

        #     if slippage:
        #         tradesData['PnL_WithoutSlippage'] = tradesData['PnL']
        #         tradesData['PnL'] = tradesData['PnL'] - \
        #             (((tradesData['Entry Price'] * slippage) /
        #              100.0) * tradesData['Quantity'])

        if 'DTE' in tradesData.columns and len(tradesData['DTE'].unique()):
            dtesMapping = dict()
            dtes = sorted(tradesData['DTE'].unique())
            dteColumns = st.columns(len(dtes))
            for index, dte in enumerate(dtes):
                with dteColumns[index]:
                    dtesMapping[dte] = st.checkbox(
                        label=str(f'{dte} DTE'), value=True, key=f'{dte}DTE')

            selectedDtes = [dte for dte,
                            isSelected in dtesMapping.items() if isSelected]

            tradesData = tradesData[tradesData['DTE'].isin(selectedDtes)]

        col1, col2 = st.columns([3, 1])
        with col1:
            instruments = st.multiselect(
                'Instrument',
                ['BANKNIFTY', 'NIFTY', 'FINNIFTY',
                    'MIDCPNIFTY', 'SENSEX', 'BANKEX'],
                ['BANKNIFTY', 'NIFTY', 'FINNIFTY', 'MIDCPNIFTY', 'SENSEX', 'BANKEX'])

            if instruments:
                tradesData = tradesData[tradesData['Instrument'].str.startswith(
                    tuple(instruments))]

        with col2:
            buySell = st.multiselect(
                'Buy/Sell',
                ['Buy', 'Sell'],
                ['Buy', 'Sell'])

            if buySell:
                tradesData = tradesData[tradesData['Buy/Sell']
                                        .str.startswith(tuple(buySell))]

        if selectedGroupCriteria is not None:
            if selectedGroupCriteria == 'Date':
                groupBy = tradesData.groupby('Date')
                xAxisTitle = 'Date'
                showStats(initialCapital, len(uploadedFiles),
                          groupBy['PnL'].sum().reset_index())
            elif selectedGroupCriteria == 'Day':
                groupBy = tradesData.groupby(
                    tradesData['Date'].dt.day_name())
                xAxisTitle = 'Day'
                showStats(initialCapital, len(uploadedFiles), tradesData)
            else:
                groupBy = None
                xAxisTitle = selectedGroupCriteria
                showStats(initialCapital, len(uploadedFiles), tradesData)

            if groupBy is not None:
                pnl = groupBy['PnL'].sum()
                # Sort by day of week if grouping by Day
                if selectedGroupCriteria == 'Day':
                    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                    pnl = pnl.reindex([d for d in day_order if d in pnl.index])
                groupByDate = tradesData.groupby(
                    'Date')['PnL'].sum().reset_index()
                cumulativePnL = groupByDate['PnL'].cumsum()
                cumulativePnL.index = groupByDate['Date']
            else:
                pnl = tradesData['PnL']
                pnl.index = tradesData['Entry Date/Time']
                cumulativePnL = tradesData['PnL'].cumsum()
                cumulativePnL.index = tradesData['Entry Date/Time']
                runningMaxPnL = cumulativePnL.cummax()

            st.markdown("<br>", unsafe_allow_html=True)
            st.divider()
            st.markdown("<br>", unsafe_allow_html=True)
            
            with st.expander('🔬 Generate Quantstats Report', expanded=False):
                result = st.button('Run')
                if result:
                    try:
                        pnls = tradesData['PnL']
                        returns = pnls / initialCapital
                        returns.index = pd.to_datetime(tradesData['Entry Date/Time'])
                        returns = returns.sort_index()
                        print(returns)
                        st.components.v1.html(quantstats_reports.html(returns, title='Tearsheet', compounded=False, output='tearsheet1.html'),
                                            height=1000,
                                            scrolling=True)
                        try:
                            st.components.v1.html(quantstats_reports.html(returns, "^NSEI", title='Strategy vs Benchmark', compounded=False, output='tearsheet2.html'),
                                                height=1000,
                                                scrolling=True)
                        except Exception as e:
                            st.warning("Could not generate benchmark comparison: " + str(e))
                    except Exception as e:
                        st.error(f"Error generating Quantstats report: {str(e)}")   
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.divider()

            runningMaxPnL = cumulativePnL.cummax()
            drawdown = runningMaxPnL - cumulativePnL
            drawdown.index = cumulativePnL.index

            color = ['#10B981' if x > 0 else '#EF4444' for x in pnl]
            fig = go.Figure(
                data=[go.Bar(x=pnl.index, y=pnl.values, marker=dict(color=color, line=dict(width=0)), hovertemplate='<b>%{x}</b><br>PnL: ₹%{y:,.2f}<extra></extra>')])
            fig.update_layout(
                title=dict(text='<b>PnL over Time</b>', x=0.5, xanchor='center', font=dict(size=18)),
                xaxis_title=xAxisTitle,
                yaxis_title='PnL (₹)',
                hovermode='x unified',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='white',
                font=dict(family='Arial, sans-serif', size=11),
                xaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(200,200,200,0.2)', zeroline=False),
                yaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(200,200,200,0.2)', zeroline=True, zerolinewidth=1, zerolinecolor='rgba(100,100,100,0.3)'),
                margin=dict(l=60, r=40, t=80, b=60),
                height=450
            )
            st.plotly_chart(fig, width='stretch')

            col1, col2 = st.columns([1, 1])
            with col1:
                fig = go.Figure(
                    data=[go.Scatter(x=cumulativePnL.index, y=cumulativePnL, fill='tozeroy', mode='lines', name='Cumulative PnL', line=dict(color='#3B82F6', width=3), fillcolor='rgba(59, 130, 246, 0.1)', hovertemplate='<b>%{x}</b><br>Cumulative PnL: ₹%{y:,.2f}<extra></extra>')])
                fig.update_layout(
                    title=dict(text='<b>Cumulative PnL over Time</b>', x=0.5, xanchor='center', font=dict(size=16)),
                    xaxis_title='Date',
                    yaxis_title='Cumulative PnL (₹)',
                    hovermode='x unified',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='white',
                    font=dict(family='Arial, sans-serif', size=10),
                    xaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(200,200,200,0.2)', zeroline=False),
                    yaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(200,200,200,0.2)', zeroline=True, zerolinewidth=1, zerolinecolor='rgba(100,100,100,0.3)'),
                    margin=dict(l=60, r=40, t=70, b=50),
                    height=420,
                    showlegend=False
                )
                st.plotly_chart(fig, width='stretch')

            with col2:
                fig = go.Figure(
                    data=[go.Scatter(x=drawdown.index, y=drawdown, fill='tozeroy', mode='lines', name='Drawdown', line=dict(color='#F59E0B', width=3), fillcolor='rgba(245, 158, 11, 0.1)', hovertemplate='<b>%{x}</b><br>Drawdown: ₹%{y:,.2f}<extra></extra>')])
                fig.update_layout(
                    title=dict(text='<b>Drawdown</b>', x=0.5, xanchor='center', font=dict(size=16)),
                    xaxis_title='Date',
                    yaxis_title='Drawdown (₹)',
                    hovermode='x unified',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='white',
                    font=dict(family='Arial, sans-serif', size=10),
                    xaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(200,200,200,0.2)', zeroline=False),
                    yaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(200,200,200,0.2)', zeroline=True, zerolinewidth=1, zerolinecolor='rgba(100,100,100,0.3)'),
                    margin=dict(l=60, r=40, t=70, b=50),
                    height=420,
                    showlegend=False
                )
                st.plotly_chart(fig, width='stretch')

            # Split pnl data by year
            print(pnl)
            try:
                yearlyPnl = pnl.groupby(pnl.index.year)
            except AttributeError:
                st.warning("Cannot create yearly heatmap: Index is not datetime type")
                yearlyPnl = None
            
            if yearlyPnl is not None:
                # Iterate over each year and plot the heatmap
                for year, data in yearlyPnl:
                    fig, _ = calplot.calplot(data, textfiller='-',
                                             cmap=customCmap(),
                                             vmin=-max(data.max(), abs(data.min())),
                                             vmax=max(data.max(), abs(data.min())))
                    st.pyplot(fig=fig)

            col1, col2 = st.columns([1, 1])
            # with col1:
            #     plotScatterMAE(tradesData)

            # with col2:
            #     plotScatterMFE(tradesData)

        # Spacer
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Callout for Quantstats
        st.markdown("""
        <div style="background: linear-gradient(135deg, #EEF2FF 0%, #E0E7FF 100%); 
                    border-left: 4px solid #667eea; 
                    padding: 1rem 1.5rem; 
                    border-radius: 0 10px 10px 0;
                    margin: 1rem 0;">
            <p style="margin: 0; color: #4338CA; font-weight: 600;">
                📈 Pro Tip: Check out the Quantstats Report above for detailed analytics!
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.toast('Check out the Quantstats Report', icon='📊') 

        with st.expander('📖 What do these metrics mean?'):
            st.info("""***Disclaimer: ChatGPT copy-paste xD***

**ROI (Return on Investment):** The percentage gain or loss relative to your initial capital. Calculated as (Total Profit / Initial Capital) × 100.

**Average Day Profit:** The average profit earned per trading day. Calculated by summing profits and dividing by the total number of trading days.

**Max Drawdown:** The maximum peak-to-trough decline in trading capital. It quantifies the worst loss experienced from any peak.

**Expectancy:** Average profit/loss expected per trade. Calculated as (Win Rate × Avg Win) - (Loss Rate × Avg Loss). Positive values indicate a profitable strategy.

**CAGR (Compound Annual Growth Rate):** The annualized growth rate of your investment, accounting for compounding effects over time.

**Sharpe Ratio:** Risk-adjusted return metric. Higher values (>1) indicate better returns relative to volatility. Calculated as (Return - Risk-free Rate) / Standard Deviation.

**Sortino Ratio:** Like Sharpe but only considers downside volatility. Better for strategies where upside variance isn't a concern.

**Calmar Ratio:** Annualized return divided by max drawdown. Higher values indicate better risk-adjusted performance relative to drawdown risk.

**Omega Ratio:** Ratio of gains to losses above/below a threshold (usually 0). Values >1 indicate more upside than downside.

**Profit Factor:** Gross profits divided by gross losses. Values >1 are profitable; >2 is considered excellent.

**Recovery Factor:** Total profit divided by max drawdown. Measures how well the strategy recovers from losses.

**VaR (Value at Risk - 95%):** The maximum expected loss at 95% confidence. Only 5% of days should have losses worse than this.

**CVaR / Expected Shortfall:** The average loss when losses exceed VaR. Shows the expected loss in the worst 5% of scenarios.

**Standard Deviation:** Measures the dispersion of returns. Higher values indicate more volatile/risky performance.

**Skewness:** Measures asymmetry of returns. Positive skew = more extreme gains; Negative skew = more extreme losses.

**Kurtosis:** Measures "fat tails" in returns. High kurtosis means more extreme events (both gains and losses) than a normal distribution.

**Tail Ratio:** Ratio of the 95th percentile to the 5th percentile. Values >1 indicate larger gains than losses at the extremes.

**Win Rate:** Percentage of trades that were profitable. Higher is better, but must be considered alongside avg win/loss size.

**Kelly Criterion:** Optimal position size to maximize long-term growth. Calculated from win rate and profit/loss ratio. Use half-Kelly for safety.

**Profit/Loss Ratio:** Average winning trade divided by average losing trade. Combined with win rate, determines overall profitability.

**Win/Loss Streak:** Maximum consecutive wins and losses. Useful for understanding psychological and drawdown risk.

**Risk of Ruin:** Probability of losing a significant portion of capital. Critical for assessing long-term survival of a strategy.""")
        
        # Modern footer
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("""
        <div style="background: #F9FAFB; 
                    border-top: 1px solid #E5E7EB;
                    padding: 1.5rem;
                    border-radius: 12px;
                    text-align: center;
                    margin-top: 2rem;">
            <p style="margin: 0; color: #6B7280; font-size: 0.9rem;">
                Built with ❤️ for traders | 
                <span style="color: #667eea; font-weight: 600;">Trades Analyzer</span>
            </p>
            <p style="margin: 0.5rem 0 0 0; color: #9CA3AF; font-size: 0.8rem;">
                Analyze • Optimize • Profit
            </p>
        </div>
        """, unsafe_allow_html=True)
        
if __name__ == "__main__":
    main()
