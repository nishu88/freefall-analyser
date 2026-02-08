# Date	     PnL		Instrument	Buy/Sell
# 1/31/2024   0			BANKNIFTY	Sell

import pandas as pd
import numpy as np
import plotly.graph_objs as go
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap
from thirdparty import calplot, quantstats_reports


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
            .average__card {
                position: relative;
                padding: 0;
                height: 100%;
                border: 1px solid green;
                border-radius: 4px;
                text-align: center;
                overflow: hidden;
            }
            .average__card .__title {
                padding: 5px 0;
                border-bottom: 1px solid;
                border-radius: 4px 4px 0 0;
                color: green;
                background-color: rgba(0, 128, 0, 0.15);
            }
            .average__card .__value {
                padding: 10px 0;
            }
            .__average__price {
                border-color: #f19f15;
            }
            .__average__price .__title {
                color: #f19f15;
                background-color: #fef8e1;
            }
            .__loss__price {
                border-color: #ef9a99;
            }
            .__loss__price .__title {
                color: red;
                border-color: #ef9a99;
                background-color: #fbebee;
            }
            .stats_percent {
                font-size: 13px;
                opacity: 0.8;
            }
            </style>
        """

        subclass = '__average__price' if color == 'yellow' else (
            '__loss__price' if color == 'red' else '')
        percent = f'<span class="stats_percent">({percentage})</span>' if percentage is not None else ''
        st.markdown(
            f'<div class="average__card {subclass}"><div class="__title">{key}</div><div class="__value">{value} {percent}</div></div>{style}',
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
    monthlyProfit = tradesData.resample('M', on='Date').sum(numeric_only=True)['PnL'].mean()
    monthlyCount = len(tradesData.groupby(tradesData['Date'].dt.to_period('M')))
    profitableMonths = len(tradesData.resample('M', on='Date').sum(numeric_only=True)[tradesData.resample('M', on='Date').sum(numeric_only=True)['PnL'] > 0])
    
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
    averageYearlyProfit = tradesData.set_index(
        'Date')['PnL'].cumsum().resample('Y').last().diff().mean()
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
    
    # Monthly Performance Table
    with st.expander('📊 Monthly Performance Details', expanded=False):
        monthly_data = tradesData.set_index('Date').resample('M')['PnL'].agg(['sum', 'count', 'mean'])
        monthly_data.columns = ['Total PnL', 'Trades', 'Avg Trade']
        monthly_data.index = monthly_data.index.strftime('%Y-%m')
        
        # Add formatting
        display_monthly = monthly_data.copy()
        display_monthly['Total PnL'] = display_monthly['Total PnL'].apply(lambda x: f'{formatINR(x)}')
        display_monthly['Avg Trade'] = display_monthly['Avg Trade'].apply(lambda x: f'{formatINR(x)}')
        display_monthly['Trades'] = display_monthly['Trades'].astype(int)
        
        st.dataframe(display_monthly.style.highlight_max(axis=0, color='#90EE90').highlight_min(axis=0, color='#FFB6C6'), use_container_width=True)
    
    # Additional Metrics Table
    with st.expander('📈 Advanced Metrics', expanded=False):
        metrics_dict = {
            'Metric': [
                'ROI',
                'Days Traded',
                'Avg Daily Profit',
                'Std Deviation',
                'Sharpe Ratio (approx)',
                'Profit/Win Ratio',
                'Loss/Win Ratio',
                'Consecutive Wins/Losses',
                'Profitable Months',
                'Average Monthly Profit'
            ],
            'Value': [
                f'{roi:.2f}%',
                f'{days_traded} days',
                f'{formatINR(averageProfit)}',
                f'{formatINR(tradesData["PnL"].std())}',
                f'{(averageProfit / tradesData["PnL"].std()) * np.sqrt(252):.2f}' if tradesData["PnL"].std() > 0 else 'N/A',
                f'{abs(averageProfitOnWins / averageLossOnLosses):.2f}x' if averageLossOnLosses != 0 else 'N/A',
                f'{abs(averageLossOnLosses / averageProfitOnWins):.2f}x' if averageProfitOnWins != 0 else 'N/A',
                f'{maxWinningStreak} / {maxLosingStreak}',
                f'{profitableMonths}/{monthlyCount} months',
                f'{formatINR(monthlyProfit)}'
            ]
        }
        metrics_df = pd.DataFrame(metrics_dict)
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    
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
    st.plotly_chart(fig, use_container_width=True)


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
    st.plotly_chart(fig, use_container_width=True)


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

def main():
    st.set_page_config(page_title="Trades Analyzer", layout="wide")
    col1, col, col2 = st.columns([1, 8, 1])

    with st.expander('How to USE??'):
        st.info("Upload .csv in the following format: Date, PnL, Instrument, Buy/Sell")
        
        # Date	     PnL		Instrument	Buy/Sell
        # 1/31/2024   0			BANKNIFTY	Sell

        data = {
                "Date": ["12/31/2023","2/26/2024","3/27/2024" ],
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
        st.success("You can print to PDF from top right hamburger icon")
        st.info("You can add multiple CSV files to aggregate strategies")


    with col:
        uploadedFiles = st.file_uploader(
            "",
            key="1",
            help="To activate 'wide mode', go to the hamburger menu > Settings > turn on 'wide mode'",
            accept_multiple_files=True
        )
        if len(uploadedFiles) == 0:
            st.info("👆 Upload a trades csv file first.")
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
        tradesData.sort_values(by="Entry Date/Time", inplace=True)
        tradesData.reset_index(drop=True, inplace=True)

        tradesData["Entry Date/Time"] = pd.to_datetime(
            tradesData["Entry Date/Time"])
        tradesData["Exit Date/Time"] = pd.to_datetime(
            tradesData["Exit Date/Time"])
        tradesData["Date"] = pd.to_datetime(tradesData["Date"])

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

            st.divider()
            st.title(" ")    
            with st.expander('Get Quantstats Report'):
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
            st.title(" ")
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
            st.plotly_chart(fig, use_container_width=True)

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
                st.plotly_chart(fig, use_container_width=True)

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
                st.plotly_chart(fig, use_container_width=True)

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

        # with st.expander("Check dataframe"):
        #     st.dataframe(tradesData, use_container_width=True)
        st.title(" ")
        st.title(" ")
        st.title(" ")
        
        st.info('Check out the Quantstats Report (Scroll ☝️☝️☝️)', icon='🧐') 
        st.title(" ")
        st.title(" ")
        st.title(" ")
        st.toast('Check out the Quantstats Report', icon='🧐') 

        with st.expander('What do these metrics mean??'):
            st.info("***Disclaimer: ChatGPT copy-paste xD*** \n\n \nAverage Day Profit: The average profit earned per trading day during the period under consideration. It's calculated by summing up the profits of each trading day and dividing by the total number of trading days.\n\n\nMax Drawdown: The maximum peak-to-trough decline in the trading capital during the trading period. It quantifies the maximum loss experienced from the peak capital value.\n\n\nExpectancy: A statistical measure that quantifies the average amount of profit or loss expected per unit of risk. It's calculated as (Probability of Win * Average Win) - (Probability of Loss * Average Loss). It provides insight into the profitability of the trading strategy over the long term.\n\n\n Compound Annual Growth Rate (CAGR): CAGR is a measure of the annual growth rate of an investment over a specified time period, considering the effect of compounding. It provides a smooth representation of the average annual growth rate of an investment, regardless of any fluctuations in the market.\n\n\nSharpe Ratio: The Sharpe Ratio measures the risk-adjusted return of an investment or trading strategy. It's calculated by dividing the excess return of the investment (return above the risk-free rate) by the standard deviation of the investment's returns. A higher Sharpe Ratio indicates better risk-adjusted returns.\n\n\nSortino Ratio: Similar to the Sharpe Ratio, the Sortino Ratio measures the risk-adjusted return of an investment. However, it focuses only on the downside risk, considering only the standard deviation of negative returns. It is often preferred over the Sharpe Ratio for strategies where downside risk is a significant concern.\n\n\nOmega Ratio: The Omega Ratio evaluates the risk-return profile of an investment by comparing the probability-weighted average return of the investment to a target return or threshold. It provides a measure of the likelihood of achieving returns above a specified threshold.\n\n\nCalmar Ratio: The Calmar Ratio measures the risk-adjusted performance of an investment or trading strategy by comparing the average annual rate of return to the maximum drawdown experienced during a specified period. A higher Calmar Ratio indicates better risk-adjusted returns relative to drawdown.\n\n\nKurtosis: Kurtosis is a statistical measure that describes the distribution of returns around its mean in a probability distribution. In the context of trading, it indicates the degree of peakedness or flatness of a return distribution compared to a normal distribution. High kurtosis implies fat tails, meaning higher probabilities of extreme returns.\n\n\nKelly Criterion: The Kelly Criterion is a mathematical formula used to determine the optimal size of a series of bets or investments to maximize long-term wealth growth. It takes into account the probability of success and the payoff ratio of each bet or investment.\n\n\nRisk of Ruin: Risk of ruin is the probability of losing a significant portion of or the entire trading capital due to consecutive losses or a prolonged drawdown. It's an important metric for assessing the probability of financial ruin under a given trading strategy.\n\n\nValue at Risk (VaR): Value at Risk is a statistical measure that quantifies the potential loss of an investment or portfolio over a specified time horizon and at a given confidence level. It provides an estimate of the maximum potential loss that a portfolio may incur within a certain probability.")
        
if __name__ == "__main__":
    main()
