"""
Calendar heatmaps from Pandas time series data.

Plot Pandas time series data sampled by day in a heatmap per calendar year.
Modern, beautified version with professional styling.
"""

import calendar
import datetime
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd

from matplotlib.colors import ColorConverter, ListedColormap, LinearSegmentedColormap
from matplotlib.patches import Polygon, FancyBboxPatch
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects


def create_profit_loss_cmap():
    """Create a modern diverging colormap for profit/loss visualization."""
    colors = [
        '#DC2626',  # Deep red (large loss)
        '#EF4444',  # Red
        '#F87171',  # Light red
        '#FCA5A5',  # Very light red
        '#FEE2E2',  # Pale red
        '#F9FAFB',  # Neutral (near zero)
        '#D1FAE5',  # Pale green
        '#6EE7B7',  # Very light green
        '#34D399',  # Light green
        '#10B981',  # Green
        '#059669',  # Deep green (large profit)
    ]
    return LinearSegmentedColormap.from_list('profit_loss', colors, N=256)

def yearplot(data, year=None, how='sum',
             vmin=None, vmax=None,
             cmap=None, fillcolor='#F3F4F6',
             linewidth=0.5, linecolor='white', edgecolor='#E5E7EB',
             daylabels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], dayticks=True,
             dropzero=None,
             textformat=None, textfiller='', textcolor='#6B7280',
             monthlabels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                         'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], monthlabeloffset=15,
             monthticks=True, highlight_weekends=True,
             ax=None, **kwargs):
    """
    Plot one year from a timeseries as a calendar heatmap.

    Parameters
    ----------
    data : Series
        Data for the plot. Must be indexed by a DatetimeIndex.
    year : integer
        Only data indexed by this year will be plotted. If `None`, the first
        year for which there is data will be plotted.
    how : string
        Method for resampling data by day. If `None`, assume data is already
        sampled by day and don't resample. Otherwise, this is passed to Pandas
        `Series.resample`.
    vmin, vmax : floats
        Values to anchor the colormap. If `None`, min and max are used after
        resampling data by day.
    cmap : matplotlib colormap name or object
        The mapping from data values to color space.
    fillcolor : matplotlib color
        Color to use for days without data.
    linewidth : float
        Width of the lines that will divide each day.
    linecolor : color
        Color of the lines that will divide each day. If `None`, the axes
        background color is used, or 'white' if it is transparent.
    daylabels : list
        Strings to use as labels for days, must be of length 7.
    dayticks : list or int or bool
        If `True`, label all days. If `False`, don't label days. If a list,
        only label days with these indices. If an integer, label every n day.
    dropzero : bool
        If `True`, don't fill a color for days with a zero value.
    monthlabels : list
        Strings to use as labels for months, must be of length 12.
    monthlabeloffset : integer
        Day offset for labels for months to adjust horizontal alignment.
    monthticks : list or int or bool
        If `True`, label all months. If `False`, don't label months. If a
        list, only label months with these indices. If an integer, label every
        n month.
    edgecolor : color
        Color of the lines that will divide months.
    textformat : string
        Text format string for grid cell text
    textfiller : string
        Fallback text for grid cell text for cells with no data
    textcolor : color
        Color of the grid cell text
    ax : matplotlib Axes
        Axes in which to draw the plot, otherwise use the currently-active
        Axes.
    kwargs : other keyword arguments
        All other keyword arguments are passed to matplotlib `ax.pcolormesh`.

    Returns
    -------
    ax : matplotlib Axes
        Axes object with the calendar heatmap.

    """

    if year is None:
        year = data.index.sort_values()[0].year

    if how is None:
        by_day = data
    else:
        by_day = data.resample('D').agg(how)

    # Default to dropping zero values for a series with over 50% of rows being zero.
    if not (dropzero is False) and (by_day[by_day == 0].count() > 0.5 * by_day.count()):
        dropzero = True

    if dropzero:
        by_day = by_day.replace({0: np.nan}).dropna()

    # Min and max per day.
    if vmin is None:
        vmin = by_day.min()
    if vmax is None:
        vmax = by_day.max()

    # Use custom profit/loss colormap if not specified
    if cmap is None:
        cmap = create_profit_loss_cmap()

    if ax is None:
        ax = plt.gca()

    if linecolor is None:
        linecolor = ax.get_facecolor()
        if ColorConverter().to_rgba(linecolor)[-1] == 0:
            linecolor = 'white'

    # Filter on year.
    try:
      # could be empty due to `dropzero`
      by_day = by_day[str(year)]
    except KeyError:
      pass

    # Add missing days.
    by_day = by_day.reindex(
        pd.date_range(start=str(year), end=str(year + 1),
                      freq='D', tz=by_day.index.tzinfo)[:-1])

    # Create data frame we can pivot later.
    by_day = pd.DataFrame({'data': by_day,
                           'fill': 1,
                           'day': by_day.index.dayofweek,
                           'week': by_day.index.isocalendar().week})

    # There may be some days assigned to previous year's last week or
    # next year's first week. We create new week numbers for them so
    # the ordering stays intact and week/day pairs unique.
    by_day.loc[(by_day.index.month == 1) & (by_day.week > 50), 'week'] = 0
    by_day.loc[(by_day.index.month == 12) & (by_day.week < 10), 'week'] \
        = by_day.week.max() + 1

    # Pivot data on day and week and mask NaN days.
    plot_data = by_day.pivot(index='day', columns='week', values='data').values[::-1]
    plot_data = np.ma.masked_where(np.isnan(plot_data), plot_data)

    # Do the same for all days of the year, not just those we have data for.
    fill_data = by_day.pivot(index='day', columns='week', values='fill').values[::-1]
    fill_data = np.ma.masked_where(np.isnan(fill_data), fill_data)

    # Draw heatmap for all days of the year with fill color.
    ax.pcolormesh(fill_data, vmin=0, vmax=1, cmap=ListedColormap([fillcolor]))

    # Draw heatmap.
    kwargs['linewidth'] = linewidth
    kwargs['edgecolors'] = linecolor
    ax.pcolormesh(plot_data, vmin=vmin, vmax=vmax, cmap=cmap, **kwargs)

    # Limit heatmap to our data.
    ax.set(xlim=(0, plot_data.shape[1]), ylim=(0, plot_data.shape[0]))

    # Square cells.
    ax.set_aspect('equal')

    # Remove spines and ticks.
    for side in ('top', 'right', 'left', 'bottom'):
        ax.spines[side].set_visible(False)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_tick_params(which='both', length=0)

    # Get indices for monthlabels.
    if monthticks is True:
        monthticks = range(len(monthlabels))
    elif monthticks is False:
        monthticks = []

    # Get indices for daylabels.
    if dayticks is True:
        dayticks = range(len(daylabels))
    elif dayticks is False:
        dayticks = []

    ax.set_xlabel('')
    ax.set_xticks([by_day.loc[pd.Timestamp(
                   datetime.date(year, i + 1, monthlabeloffset))].week
                   for i in monthticks])
    ax.set_xticklabels([monthlabels[i] for i in monthticks], 
                       fontsize=10, fontweight='600', color='#374151',
                       fontfamily='sans-serif')

    ax.set_ylabel('')
    ax.yaxis.set_ticks_position('right')
    ax.set_yticks([6 - i + 0.5 for i in dayticks])
    
    # Style day labels with weekend highlighting
    day_colors = ['#6B7280'] * 5 + ['#9CA3AF', '#9CA3AF']  # Lighter for weekends
    ax.set_yticklabels([daylabels[i] for i in dayticks], rotation='horizontal',
                       va='center', fontsize=9, fontweight='500', 
                       fontfamily='sans-serif')
    
    # Color weekend labels differently
    for i, label in enumerate(ax.get_yticklabels()):
        if i < 2:  # Saturday and Sunday (reversed order)
            label.set_color('#9CA3AF')
        else:
            label.set_color('#6B7280')

    # Enhanced monthly summary display with pill-style badges
    monthly_sum = by_day.groupby(pd.Grouper(freq='ME')).sum()['data']
    for month, week_start in zip(range(1, 13), ax.get_xticks()):
        try:
            profit = round(monthly_sum.iloc[month - 1], 2)
        except (IndexError, KeyError):
            continue
        
        # Modern color scheme with gradients
        if profit >= 0:
            bg_color = '#D1FAE5'
            text_color = '#065F46'
            prefix = '+'
        else:
            bg_color = '#FEE2E2'
            text_color = '#991B1B'
            prefix = ''
        
        # Format number with K/L suffix for readability
        if abs(profit) >= 100000:
            display_text = f'{prefix}{profit/100000:.1f}L'
        elif abs(profit) >= 1000:
            display_text = f'{prefix}{profit/1000:.1f}K'
        else:
            display_text = f'{prefix}{profit:,.0f}'
        
        # Create pill-style badge
        text = ax.text(week_start, -1.8, display_text,
                       color=text_color, ha='center', va='top', 
                       fontsize=8, fontfamily='sans-serif',
                       fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3,rounding_size=0.5', 
                                facecolor=bg_color, edgecolor='none', alpha=0.9))

    # Text in mesh grid if format is specified.
    if textformat is not None:
        for y in range(plot_data.shape[0]):
            for x in range(plot_data.shape[1]):
                content = ''
                masked = plot_data[y, x]
                if masked is np.ma.masked:
                    if fill_data[y, x] == 1:
                        content = textfiller
                else:
                    content = textformat.format(masked)
                if content:
                    ax.text(x + 0.5, y + 0.5, content, color=textcolor,
                            ha='center', va='center', fontsize=8, fontweight='500')

    # Month borders - modern subtle styling
    xticks = []
    start = datetime.datetime(year, 1, 1).weekday()
    for month in range(1, 13):
        first = datetime.datetime(year, month, 1)
        last = first + relativedelta(months=1, days=-1)
        y0 = 7 - first.weekday()
        y1 = 7 - last.weekday()
        x0 = (int(first.strftime('%j'))+start-1)//7
        x1 = (int(last.strftime('%j'))+start-1)//7
        P = [(x0, y0),
             (x0+1, y0),
             (x0+1, 7),
             (x1+1, 7),
             (x1+1, y1-1),
             (x1, y1-1),
             (x1, 0),
             (x0, 0)]
        xticks.append(x0 + (x1-x0+1)/2)
        # Elegant month borders with subtle shadow effect
        poly = Polygon(P, edgecolor='#D1D5DB', facecolor='None',
                       linewidth=1.2, zorder=20, clip_on=False, 
                       linestyle='-', capstyle='round', joinstyle='round')
        ax.add_artist(poly)

    return ax


def calplot(data, how='sum',
            yearlabels=True, yearascending=True,
            yearlabel_kws=None, subplot_kws=None, gridspec_kws=None,
            figsize=None, fig_kws=None, colorbar=None,
            suptitle=None, suptitle_kws=None,
            tight_layout=True, **kwargs):
    """
    Plot a timeseries as a calendar heatmap.

    Parameters
    ----------
    data : Series
        Data for the plot. Must be indexed by a DatetimeIndex.
    how : string
        Method for resampling data by day. If `None`, assume data is already
        sampled by day and don't resample. Otherwise, this is passed to Pandas
        `Series.resample`.
    figsize : (float, float)
        Size of figure for the plot.
    suptitle : string
        Title for the plot.
    yearlabels : bool
       Whether or not to draw the year label for each subplot.
    yearascending : bool
       Sort the calendar in ascending or descending order.
    yearlabel_kws : dict
       Keyword arguments passed to the matplotlib `set_ylabel` call which is
       used to draw the year for each subplot.
    subplot_kws : dict
        Keyword arguments passed to the matplotlib `subplots` call.
    gridspec_kws : dict
        Keyword arguments passed to the matplotlib `GridSpec` constructor used
        to create the grid the subplots are placed on.
    fig_kws : dict
        Keyword arguments passed to the matplotlib `subplots` call.
    suptitle_kws : dict
        Keyword arguments passed to the matplotlib `suptitle` call.
    kwargs : other keyword arguments
        All other keyword arguments are passed to `yearplot`.

    Returns
    -------
    fig, axes : matplotlib Figure and Axes
        Tuple where `fig` is the matplotlib Figure object `axes` is an array
        of matplotlib Axes objects with the calendar heatmaps, one per year.

    """

    if yearlabel_kws is None:
        yearlabel_kws = dict()
    if subplot_kws is None:
        subplot_kws = dict()
    if gridspec_kws is None:
        gridspec_kws = dict()
    if fig_kws is None:
        fig_kws = dict()
    if suptitle_kws is None:
        suptitle_kws = dict()

    years = np.unique(data.index.year)
    if not yearascending:
        years = years[::-1]

    if colorbar is None:
        colorbar = data.nunique() > 1

    if figsize is None:
        figsize = (16+(colorbar*2), 2.8*len(years))

    fig, axes = plt.subplots(nrows=len(years), ncols=1, squeeze=False,
                             figsize=figsize,
                             subplot_kw=subplot_kws,
                             gridspec_kw=gridspec_kws, **fig_kws)
    axes = axes.T[0]
    
    # Modern figure styling with clean background
    fig.patch.set_facecolor('#FFFFFF')
    for ax in axes:
        ax.set_facecolor('#FAFAFA')

    # We explicitely resample by day only once. This is an optimization.
    by_day = data
    if how is not None:
        by_day = by_day.resample('D').agg(how)

    ylabel_kws = dict(
        fontsize=24,
        color='#1F2937',
        fontfamily='sans-serif',
        fontweight='bold',
        ha='center')
    ylabel_kws.update(yearlabel_kws)

    max_weeks = 0

    for year, ax in zip(years, axes):
        yearplot(by_day, year=year, how=None, ax=ax, **kwargs)
        max_weeks = max(max_weeks, ax.get_xlim()[1])

        if yearlabels:
            ax.set_ylabel(str(year), **ylabel_kws)
            # Modern pill-style year label
            ax.yaxis.label.set_bbox(dict(
                boxstyle='round,pad=0.4,rounding_size=0.3', 
                facecolor='#F3F4F6', 
                alpha=0.9, 
                edgecolor='#E5E7EB',
                linewidth=0.5
            ))

    # In a leap year it might happen that we have 54 weeks (e.g., 2012).
    # Here we make sure the width is consistent over all years.
    for ax in axes:
        ax.set_xlim(0, max_weeks)

    stitle_kws = dict(fontsize=18, fontweight='bold', color='#111827', fontfamily='sans-serif')

    if tight_layout:
        plt.tight_layout()
        stitle_kws.update({'y': 0.98})
        # Add extra space for monthly summary badges
        plt.subplots_adjust(bottom=0.12, hspace=0.4)

    if colorbar:
        if tight_layout:
            stitle_kws.update({'x': 0.425, 'y': 1.02})

        if len(years) == 1:
            cbar = fig.colorbar(axes[0].get_children()[1], ax=axes.ravel().tolist(),
                         orientation='vertical', pad=0.03, aspect=30)
            cbar.ax.tick_params(labelsize=9, colors='#6B7280')
            cbar.outline.set_edgecolor('#E5E7EB')
            cbar.outline.set_linewidth(0.5)
            # Add labels for colorbar
            cbar.set_label('Daily P&L', fontsize=10, color='#374151', fontfamily='sans-serif')
        else:
            fig.subplots_adjust(right=0.82)
            cax = fig.add_axes([0.86, 0.08, 0.015, 0.84])
            cbar = fig.colorbar(axes[0].get_children()[1], cax=cax, orientation='vertical')
            cbar.ax.tick_params(labelsize=9, colors='#6B7280')
            cbar.outline.set_edgecolor('#E5E7EB')
            cbar.outline.set_linewidth(0.5)
            cbar.set_label('Daily P&L', fontsize=10, color='#374151', fontfamily='sans-serif')

    stitle_kws.update(suptitle_kws)
    if suptitle:
        plt.suptitle(suptitle, **stitle_kws)

    return fig, axes
