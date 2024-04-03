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

    col1, col2, col3, col4, col5 = st.columns(5, gap='small')
    box(col1, 'Initial Capital', f'{formatINR(initialCapital)}')
    box(col2, 'Overall Profit/Loss',
        f'{formatINR(overallPnL)}', f'{round((overallPnL/initialCapital)*100, 2)}%')
    box(col3, 'Average Day Profit', f'{formatINR(averageProfit)}',
        f'{round((averageProfit/initialCapital)*100, 2)}%', color='yellow')
    box(col4, 'Max Profit', f'{formatINR(maxProfit)}',
        f'{round((maxProfit/initialCapital)*100, 2)}%')
    box(col5, 'Max Loss', f'{formatINR(maxLoss)}',
        f'{round((maxLoss/initialCapital)*100, 2)}%', color='red')
    st.write('')

    wins = tradesData['PnL'][tradesData['PnL'] > 0].count()
    losses = tradesData['PnL'][tradesData['PnL'] < 0].count()
    totalCount = tradesData['PnL'].count()
    winPercentage = (wins / totalCount) * 100
    lossPercentage = (losses / totalCount) * 100
    monthlyProfit = tradesData.resample(
        'M', on='Date').sum(numeric_only=True)['PnL'].mean()
    averageProfitOnWins = tradesData['PnL'][tradesData['PnL'] > 0].mean()
    averageLossOnLosses = tradesData['PnL'][tradesData['PnL'] < 0].mean()

    col1, col2, col3, col4, col5 = st.columns(5, gap='small')
    box(col1, 'Win% (Days)', f'{round(winPercentage, 2)} ({wins})')
    box(col2, 'Loss% (Days)',
        f'{round(lossPercentage, 2)} ({losses})', color='red')
    box(col3, 'Avg Monthly Profit', '{}'.format(
        formatINR(monthlyProfit)), f'{round((monthlyProfit/initialCapital)*100, 2)}%', color='yellow')
    box(col4, 'Avg Profit On Win Days', '{}'.format(
        formatINR(averageProfitOnWins)), f'{round((averageProfitOnWins/initialCapital)*100, 2)}%', initialCapital)
    box(col5, 'Avg Loss On Loss Days', '{}'.format(
        formatINR(averageLossOnLosses)), f'{round((averageLossOnLosses/initialCapital)*100, 2)}%', color='red')
    st.write('')

    cumulativePnL = tradesData['PnL'].cumsum()
    runningMaxPnL = cumulativePnL.cummax()
    drawdown = cumulativePnL - runningMaxPnL
    mdd = drawdown.min()

    # Calculate drawdown durations and keep track of start and end dates
    drawdown_durations = []
    drawdown_start_dates = []
    drawdown_end_dates = []

    prev_drawdown_idx = None
    for idx, pnl in enumerate(drawdown):
        if pnl < 0:
            if prev_drawdown_idx is None:
                prev_drawdown_idx = idx
        elif prev_drawdown_idx is not None:
            drawdown_start_date = tradesData['Date'][prev_drawdown_idx]
            drawdown_end_date = tradesData['Date'][idx - 1]
            drawdown_duration = (drawdown_end_date -
                                 drawdown_start_date).days + 1
            drawdown_durations.append(drawdown_duration)
            drawdown_start_dates.append(drawdown_start_date)
            drawdown_end_dates.append(drawdown_end_date)
            prev_drawdown_idx = None

    # Check if the last trade is still in a drawdown
    if prev_drawdown_idx is not None:
        drawdown_start_date = tradesData['Date'][prev_drawdown_idx]
        drawdown_end_date = tradesData['Date'][len(drawdown) - 1]
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
    returnToMddRatio = abs(averageYearlyProfit / mdd)

    col1, col2, col3 = st.columns(3, gap='small')
    box(col1, 'Max Drawdown (MDD)',
        f'{formatINR(mdd)}', f'{(mdd/initialCapital)*100:.2f}%', color='red')
    box(col2, 'MDD Days (Recovery Days)',
        f'{mddDays}', mddDateRange, color='red')
    box(col3, 'Return to MDD Ratio',
        'Requires minimum 1Yr data' if returnToMddRatio is not None else f'{returnToMddRatio:.2f}')
    st.write('')

    maxWinningStreak, maxLosingStreak = GetStreaks(tradesData)
    expectancy = GetExpectancy(tradesData)

    col1, col2, col3, col4 = st.columns(4, gap='small')
    box(col1, 'Number of Strategies', f'{numOfFiles}')
    box(col2, 'Max Winning Streak', f'{maxWinningStreak}')
    box(col3, 'Max Losing Streak',
        f'{maxLosingStreak}', color='red')
    box(col4, 'Expectancy', f'{expectancy:.2f}')
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
        marker=dict(color=np.where(pnl < 0, 'red', 'green'))
    ))
    fig.update_layout(
        title='MAE vs PnL',
        xaxis_title='MAE',
        yaxis_title='PnL'
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
        marker=dict(color=np.where(pnl < 0, 'red', 'green'))
    ))
    fig.update_layout(
        title='MFE vs PnL',
        xaxis_title='MFE',
        yaxis_title='PnL'
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

        with open('sample.csv') as f:
            st.download_button('Download SAMPLE CSV', f, "sample.csv") 

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
                    tradesData['Date'].dt.strftime('%A'))
                xAxisTitle = 'Day'
                showStats(initialCapital, len(uploadedFiles), tradesData)
            else:
                groupBy = None
                xAxisTitle = selectedGroupCriteria
                showStats(initialCapital, len(uploadedFiles), tradesData)

            if groupBy is not None:
                pnl = groupBy['PnL'].sum()
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
                    pnls = tradesData['PnL']
                    returns = pnls / initialCapital
                    returns.index = pd.to_datetime(tradesData['Entry Date/Time'])
                    returns = returns.sort_index()
                    print(returns)
                    st.components.v1.html(quantstats_reports.html(returns, title='Tearsheet', compounded=False, output='tearsheet1.html'),
                                        height=1000,
                                        scrolling=True)
                    st.components.v1.html(quantstats_reports.html(returns, "^NSEI", title='Strategy vs Benchmark', compounded=False, output='tearsheet2.html'),
                                        height=1000,
                                        scrolling=True)   
            st.title(" ")
            st.divider()

            runningMaxPnL = cumulativePnL.cummax()
            drawdown = cumulativePnL - runningMaxPnL
            drawdown.index = cumulativePnL.index

            color = ['green' if x > 0 else 'red' for x in pnl]
            fig = go.Figure(
                data=[go.Bar(x=pnl.index, y=pnl.values, marker={'color': color})])
            fig.update_layout(title='PnL over time',
                              xaxis_title=xAxisTitle, yaxis_title='PnL')
            st.plotly_chart(fig, use_container_width=True)

            col1, col2 = st.columns([1, 1])
            with col1:
                fig = go.Figure(
                    data=[go.Scatter(x=cumulativePnL.index, y=cumulativePnL)])
                fig.update_layout(title='Cumulative PnL over time',
                                  xaxis_title='Date', yaxis_title='Cumulative PnL')
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                fig = go.Figure(
                    data=[go.Scatter(x=drawdown.index, y=drawdown)])
                fig.update_layout(
                    title='Drawdown', xaxis_title='Date', yaxis_title='Drawdown')
                st.plotly_chart(fig, use_container_width=True)

            # Split pnl data by year
            print(pnl)
            yearlyPnl = pnl.groupby(pnl.index.year)

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
