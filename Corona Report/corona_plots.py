import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator, MultipleLocator, ScalarFormatter, AutoLocator
from matplotlib.path import Path
from matplotlib.patches import BoxStyle
import seaborn as sns
import datetime
import numpy as np
import os
from corona_data_scandinavia import load_data, get_scb_deaths, get_swe_death_stats

os.chdir(os.path.dirname(__file__))

# Set Seaborn style
sns.set_style(
    "whitegrid", {"axes.facecolor": "1.0", "grid.color": ".95", "grid.linestyle": "--"}
)

# Background color
bg_col = "white"  # "floralwhite"

# Set line width
l_w = 3

# Load data
scandic_df, scandic_deaths_df, scandi_outbreak, mortality_rate = load_data()
scb_deaths = get_scb_deaths()
swe_death_stats = get_swe_death_stats()

num_countries = len(scandic_df["Country"].unique())

# Get date of latest data
latest_data = scandic_df.index[-1].date()

# Set date intervals for plots
from_date = "2020-02-25"
to_date = ""
if to_date == "":
    to_date = str(scandic_df.index[-1].date())

# Class for extending text boxes
class ExtendedTextBox(BoxStyle._Base):
    """
    An Extended Text Box that expands to the axes limits 
                        if set in the middle of the axes
    """

    def __init__(self, pad=0.3, width=500.0):
        """
        width: 
            width of the textbox. 
            Use `ax.get_window_extent().width` 
                   to get the width of the axes.
        pad: 
            amount of padding (in vertical direction only)
        """
        self.width = width
        self.pad = pad
        super(ExtendedTextBox, self).__init__()

    def transmute(self, x0, y0, width, height, mutation_size):
        """
        x0 and y0 are the lower left corner of original text box
        They are set automatically by matplotlib
        """
        # padding
        pad = mutation_size * self.pad

        # we add the padding only to the box height
        height = height + 2.0 * pad
        # boundary of the padded box
        y0 = y0 - pad
        y1 = y0 + height
        _x0 = x0
        x0 = _x0 + width / 2.0 - self.width / 2.0
        x1 = _x0 + width / 2.0 + self.width / 2.0

        cp = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]

        com = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]

        path = Path(cp, com)

        return path


# register the custom textbox style
BoxStyle._style_list["ext"] = ExtendedTextBox

# Plot confirmed cases
def plot_confirmed_cases(scandic_df, from_date, to_date, num_countries, save=False):
    fig, ax = plt.subplots(figsize=(20, 12))

    # Set background color
    fig.patch.set_facecolor(bg_col)
    ax.patch.set_facecolor(bg_col)

    # Plot per country
    sns.lineplot(
        x=scandic_df[from_date:to_date].index,
        y="Confirmed",
        data=scandic_df[from_date:to_date],
        palette=sns.color_palette("Set1", n_colors=num_countries, desat=0.8),
        hue="Country",
    )

    # Plot Sweden rolling mean
    # sns.lineplot(x=scandic_df[scandic_df["Country"] == "Sweden"][from_date:to_date].index, y="Mean7", data=scandic_df[scandic_df["Country"] == "Sweden"][from_date:to_date], color="salmon", alpha=.6)

    # Plot Scandinavian mean
    sns.lineplot(
        x=scandic_df[from_date:to_date].index,
        y="Confirmed",
        data=scandic_df[from_date:to_date],
        color="purple",
        label="Scandinavia (mean)",
        alpha=0.6,
        ci="sd",
    )

    # Set xticks location and format
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    # ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=range(0,7,2)))
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))

    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'));

    # Format confidence interval
    ax.get_children()[0].set_alpha(0.1)

    # Set the linewidth
    for l in range(len(ax.lines)):
        ax.lines[l].set_linewidth(l_w)
        # print(l, ax.lines[l].get_color())

    # Create and display infotext
    textstr = "ci = standard deviation"
    # bbcol = ax.lines[7].get_color()
    props = dict(boxstyle="round", facecolor="steelblue", alpha=0.9)
    ax.text(
        0.05,
        0.2,
        textstr,
        transform=ax.transAxes,
        fontsize=14,
        color="#f0f0f0",
        verticalalignment="top",
        bbox=props,
    )

    # Set title and axis parameters
    plt.title("Confirmed cases of Coronavirus\nScandinavia", size=30)

    # plt.xlabel('Date', size=20)
    ax.xaxis.label.set_visible(False)
    plt.ylabel("Cases", size=20)

    # Format xticks
    plt.xticks(rotation=90, size=15)

    # Plot
    offset = 7
    data_source = "CSSE at Johns Hopkins University"
    name_string = "CUMULATIVE CASES"
    show_plot(fig, ax, offset, name_string, data_source, save=save)


def plot_daily_new_cases(scandic_df, from_date, to_date, save=False):
    # Plot daily new cases
    fig, ax = plt.subplots(figsize=(20, 12))

    # Set background color
    fig.patch.set_facecolor(bg_col)
    ax.patch.set_facecolor(bg_col)

    # Plot per country
    # sns.lineplot(x=scandic_df[from_date:].index, y="Increase", data=scandic_df[from_date:], palette=sns.color_palette("Set1", n_colors=num_countries, desat=.8), hue="Country")

    # Plot Scandinavian mean
    sns.lineplot(
        x=scandic_df[from_date:].index,
        y="Increase",
        data=scandic_df[from_date:],
        label="Scandinavia (mean)",
        alpha=0.6,
        ci="sd",
    )

    # Set xticks location and format
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    # ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=[0,1,2,3,4,5,6]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))

    # Format confidence interval
    ax.get_children()[0].set_alpha(0.1)

    # Set the linewidth
    for l in range(len(ax.lines)):
        ax.lines[l].set_linewidth(l_w)

    # Create and display infotext
    textstr = "ci = standard deviation"
    props = dict(boxstyle="round", facecolor="steelblue", alpha=0.9)
    ax.text(
        0.05,
        0.95,
        textstr,
        transform=ax.transAxes,
        fontsize=14,
        color="#f0f0f0",
        verticalalignment="top",
        bbox=props,
    )

    # Set title and axis parameters
    plt.title("Daily new cases of Coronavirus\nScandinavia (mean)", size=30)

    # plt.xlabel('Date', size=20)
    ax.xaxis.label.set_visible(False)
    plt.ylabel("Cases", size=20)

    # Format xticks
    plt.xticks(rotation=90, size=15)

    # Plot
    offset = 6
    data_source = "CSSE at Johns Hopkins University"
    name_string = "DAILY INCREASE"
    show_plot(fig, ax, offset, name_string, data_source, save=save)


def plot_confirmed_cases_from_day_x(
    scandi_outbreak, from_date, to_date, num_countries, save=False
):
    # Plot confirmed cases from day with X cases
    fig, ax = plt.subplots(figsize=(20, 12))
    sns.lineplot(
        data=scandi_outbreak,
        palette=sns.color_palette("Set1", n_colors=num_countries, desat=0.8),
    )

    # Set background color
    fig.patch.set_facecolor(bg_col)
    ax.patch.set_facecolor(bg_col)

    # Set title and axis parameters
    plt.title(
        "Confirmed cases of Coronavirus per day\nfrom day with 5 cases or more", size=30
    )

    plt.xlabel("Days from confirmed cases > 5", size=20)
    plt.ylabel("Cases", size=20)

    # Format xticks
    plt.xticks(range(0, len(scandi_outbreak), 5), size=15, rotation=45)

    # Set the linewidth
    for l in range(len(ax.lines)):
        ax.lines[l].set_linewidth(l_w)

    # Position legend
    plt.legend(loc="upper left")

    # Plot
    offset = 3.8
    data_source = "CSSE at Johns Hopkins University"
    name_string = "CUMULATIVE CASES FROM DAY n > 5"
    show_plot(fig, ax, offset, name_string, data_source, save=save, plot_type=2)


def plot_deaths(
    scandic_deaths_df, from_date, to_date, num_countries, plot_mean=False, save=False
):
    # Plot deaths
    fig, ax = plt.subplots(figsize=(20, 12))
    textstr = ""

    # Set background color
    fig.patch.set_facecolor(bg_col)
    ax.patch.set_facecolor(bg_col)

    # Plot per country
    sns.lineplot(
        x=scandic_deaths_df[from_date:to_date].index,
        y="Deaths",
        data=scandic_deaths_df[from_date:to_date],
        palette=sns.color_palette("Set1", n_colors=num_countries, desat=0.8),
        hue="Country",
    )

    # Plot mean if plot_mean set to true
    if plot_mean:
        # Plot Scandinavian mean
        sns.lineplot(
            x=scandic_deaths_df[from_date:to_date].index,
            y="Deaths",
            data=scandic_deaths_df[from_date:to_date],
            color="purple",
            label="Scandinavia (mean)",
            alpha=0.6,
            ci="sd",
        )

        # Format confidence interval
        ax.get_children()[0].set_alpha(0.1)

        # Create and display infotext
        textstr = "ci = standard deviation"
        props = dict(boxstyle="round", facecolor="steelblue", alpha=0.9)
        ax.text(
            0.05,
            0.2,
            textstr,
            transform=ax.transAxes,
            fontsize=14,
            color="#f0f0f0",
            verticalalignment="top",
            bbox=props,
        )

    # Set xticks location and format
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    # ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=[0,1,2,3,4,5,6]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))

    # Set the linewidth
    for l in range(len(ax.lines)):
        ax.lines[l].set_linewidth(l_w)

    # Set title and axis parameters
    plt.title("Deaths with Coronavirus\nScandinavia", size=30)

    # plt.xlabel('Date', size=20)
    ax.xaxis.label.set_visible(False)
    plt.ylabel("Deaths", size=20)

    # Format xticks
    plt.xticks(rotation=90, size=15)

    # Plot
    offset = 4
    data_source = "CSSE at Johns Hopkins University"
    name_string = "CUMULATIVE DEATHS"
    show_plot(fig, ax, offset, name_string, data_source, save=save, plot_type=3)


def plot_case_fatality_rate(
    mortality_rate, from_date, to_date, num_countries, save=False
):
    ## Den här är letalitet (case fatality rate) och inte mortalitet (mortality).
    ## CFR är andelen av insjuknade som dör medans mortalitet är andel av populationen

    # Plot Case Fatality Rate
    fig, ax = plt.subplots(figsize=(20, 12))

    # Set background color
    fig.patch.set_facecolor(bg_col)
    ax.patch.set_facecolor(bg_col)

    f_d = 47
    genomsnitt = np.mean(mortality_rate[["Sweden", "Norway", "Denmark"]][f_d:])
    print(genomsnitt)
    sns.lineplot(
        data=mortality_rate[from_date:],
        palette=sns.color_palette("Set1", n_colors=num_countries, desat=0.8),
    )
    # sns.lmplot(x=scandic_deaths_df.index.unique()[f_d:], y=mortality_rate_swe[f_d:])
    ax.axhline(genomsnitt.mean(), linestyle="-.", color="grey", alpha=0.4)

    ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    # ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=range(0,7)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))

    # ax.lines[0].set_linestyle("--")
    # ax.lines[0].set_alpha(.5)

    # Set the linewidth
    for l in range(len(ax.lines)):
        ax.lines[l].set_linewidth(l_w)

    # Format x-ticks
    plt.xticks(rotation=90, size=15)

    # Set title and axis parameters
    plt.title("Case Fatality Rate (deaths/confirmed)", size=30)
    # plt.xlabel("Date", size=20)
    ax.xaxis.label.set_visible(False)
    plt.ylabel("CFR", size=20)

    # Plot
    offset = 4
    data_source = "CSSE at Johns Hopkins University"
    name_string = "CASE FATALITY RATE"
    show_plot(fig, ax, offset, name_string, data_source, save=save, plot_type=3)


def plot_mortality_rate(
    scandic_deaths_df, from_date, to_date, num_countries, plot_mean=True, save=False
):
    ## Denna är också mortalitet men per 100k istället för per 1k

    # Plot deaths per capita
    fig, ax = plt.subplots(figsize=(20, 12))
    textstr = ""

    # Set background color
    fig.patch.set_facecolor(bg_col)
    ax.patch.set_facecolor(bg_col)

    # Plot per country
    sns.lineplot(
        x=scandic_deaths_df[from_date:to_date].index,
        y="Deaths per 100k",
        data=scandic_deaths_df[from_date:to_date],
        palette=sns.color_palette("Set1", n_colors=num_countries, desat=0.8),
        hue="Country",
    )

    # Plot mean if plot_mean set to true
    if plot_mean:
        # Plot Scandinavian mean
        sns.lineplot(
            x=scandic_deaths_df[from_date:to_date].index,
            y="Deaths per 100k",
            data=scandic_deaths_df[from_date:to_date],
            color="purple",
            label="Scandinavia (mean)",
            alpha=0.6,
            ci="sd",
        )

        # Format confidence interval
        ax.get_children()[0].set_alpha(0.1)
        textstr = "ci = standard deviation\n"

    # Set xticks location and format
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    # ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=[0,1,2,3,4,5,6]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))

    # Create and display infotext
    textstr = textstr + "Population data:\n2018 from from www.nordicstatistics.org"
    props = dict(boxstyle="round", facecolor="steelblue", alpha=0.9)
    ax.text(
        0.05,
        0.2,
        textstr,
        transform=ax.transAxes,
        fontsize=14,
        color="#f0f0f0",
        verticalalignment="top",
        bbox=props,
    )

    # Set title and axis parameters
    plt.title("Deaths per 100 000 citizens with Coronavirus\nScandinavia", size=30)

    # plt.xlabel("Date", size=20)
    ax.xaxis.label.set_visible(False)
    plt.ylabel("Deaths per 100k", size=20)

    # Set the linewidth
    for l in range(len(ax.lines)):
        ax.lines[l].set_linewidth(l_w)

    # Format xticks
    plt.xticks(rotation=90, size=15)

    # Find last values
    swe_last_dpc = scandic_deaths_df[scandic_deaths_df["Country"] == "Sweden"][
        "Deaths per 100k"
    ][-1]
    nor_last_dpc = scandic_deaths_df[scandic_deaths_df["Country"] == "Norway"][
        "Deaths per 100k"
    ][-1]
    den_last_dpc = scandic_deaths_df[scandic_deaths_df["Country"] == "Denmark"][
        "Deaths per 100k"
    ][-1]

    # Add annotations
    ann_col = "salmon"
    txt_col = "black"
    alpha = 0.4
    ax.annotate(
        str(round(swe_last_dpc, 3)),
        xy=(scandic_deaths_df.index[-1], swe_last_dpc),
        xycoords="data",
        xytext=(-30, 0),
        textcoords="offset points",
        size=13,
        ha="right",
        va="center",
        bbox=dict(boxstyle="round", alpha=alpha, color=ax.lines[0].get_color()),
        arrowprops=dict(
            arrowstyle="wedge,tail_width=0.5",
            alpha=alpha,
            color=ax.lines[0].get_color(),
        ),
        color=txt_col,
    )
    ax.annotate(
        str(round(nor_last_dpc, 3)),
        xy=(scandic_deaths_df.index[-1], nor_last_dpc),
        xycoords="data",
        xytext=(-30, 0),
        textcoords="offset points",
        size=13,
        ha="right",
        va="center",
        bbox=dict(boxstyle="round", alpha=alpha, color=ax.lines[1].get_color()),
        arrowprops=dict(
            arrowstyle="wedge,tail_width=0.5",
            alpha=alpha,
            color=ax.lines[1].get_color(),
        ),
        color=txt_col,
    )
    ax.annotate(
        str(round(den_last_dpc, 3)),
        xy=(scandic_deaths_df.index[-1], den_last_dpc),
        xycoords="data",
        xytext=(-30, 0),
        textcoords="offset points",
        size=13,
        ha="right",
        va="center",
        bbox=dict(boxstyle="round", alpha=alpha, color=ax.lines[2].get_color()),
        arrowprops=dict(
            arrowstyle="wedge,tail_width=0.5",
            alpha=alpha,
            color=ax.lines[2].get_color(),
        ),
        color=txt_col,
    )

    # Plot
    offset = 2.3
    data_source = "CSSE at Johns Hopkins University"
    name_string = "MORTALITY RATE"
    show_plot(fig, ax, offset, name_string, data_source, save=save, plot_type=3)


def plot_log_deaths(scandic_deaths_df, from_date, to_date, num_countries, save=False):
    # Plot log of deaths
    fig, ax = plt.subplots(figsize=(20, 12))
    textstr = ""
    from_date = "2020-03-14"

    # Set background color
    fig.patch.set_facecolor(bg_col)
    ax.patch.set_facecolor(bg_col)

    # Plot per country
    sns.lineplot(
        x=scandic_deaths_df[from_date:to_date].index,
        y=scandic_deaths_df[from_date:to_date]["Deaths"].apply(lambda x: np.log(x)),
        data=scandic_deaths_df[from_date:to_date],
        palette=sns.color_palette("Set1", n_colors=num_countries, desat=0.8),
        hue="Country",
    )

    # Plot mean if set to true
    plot_mean = False
    if plot_mean:
        # Plot Scandinavian mean
        sns.lineplot(
            x=scandic_deaths_df[from_date:to_date].index,
            y="Deaths",
            data=scandic_deaths_df[from_date:to_date],
            color="purple",
            label="Scandinavia (mean)",
            alpha=0.6,
            ci="sd",
        )

        # Format confidence interval
        ax.get_children()[0].set_alpha(0.1)

        # Create and display infotext
        textstr = "ci = standard deviation"
        props = dict(boxstyle="round", facecolor="steelblue", alpha=0.9)
        ax.text(
            0.05,
            0.2,
            textstr,
            transform=ax.transAxes,
            fontsize=14,
            color="#f0f0f0",
            verticalalignment="top",
            bbox=props,
        )

    # Set xticks location and format
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    # ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=[0,1,2,3,4,5,6]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))

    # Set the linewidth
    for l in range(len(ax.lines)):
        ax.lines[l].set_linewidth(l_w)
        ax.lines[l].set_linestyle("--")

    # Set title and axis parameters
    plt.title("Log Deaths with Coronavirus\nScandinavia", size=30)
    ax.set(yscale="log")
    ax.set_ylim(1e0,)
    # plt.xlabel('Date', size=20)
    ax.xaxis.label.set_visible(False)
    plt.ylabel("Log Deaths", size=20)

    # Format xticks
    plt.xticks(rotation=90, size=15)

    # Plot
    offset = 0.73
    data_source = "CSSE at Johns Hopkins University"
    name_string = "LOG OF CUMULATIVE DEATHS"
    show_plot(fig, ax, offset, name_string, data_source, save=save, plot_type=3)


def plot_swedish_deaths(swe_death_stats, save=False):
    # Plot deaths in 2020 and mean deaths in Sweden per day
    fig, ax = plt.subplots(figsize=(20, 12))
    textstr = ""

    # Set background color
    fig.patch.set_facecolor(bg_col)
    ax.patch.set_facecolor(bg_col)

    # Plot per country
    sns.lineplot(
        data=swe_death_stats["2020-02-24":],
        palette=sns.color_palette("Set1", n_colors=2, desat=0.8),
    )

    # Set xticks location and format
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    # ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=[0,1,2,3,4,5,6]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))

    # Set the linewidth
    for l in range(len(ax.lines)):
        ax.lines[l].set_linewidth(l_w)

    # Set title and axis parameters
    plt.title(
        "Cumulative Deaths in 2020 and mean of 2015-2019\nSweden\n(not normalized for population)",
        size=30,
    )

    # plt.xlabel('Date', size=20)
    ax.xaxis.label.set_visible(False)
    plt.ylabel("Deaths", size=20)

    # Format xticks
    plt.xticks(rotation=90, size=12)

    # Plot
    offset = 6500
    data_source = "Statistiska Centralbyrån, www.scb.se"
    name_string = "CUMULATIVE DEATHS"
    show_plot(fig, ax, offset, name_string, data_source, save=save, plot_type=4)


def plot_daily_new_deaths(scandic_deaths_df, from_date, to_date, save=False):
    # Plot daily new deaths and rolling average
    fig = plt.figure(figsize=(20, 12), dpi=300)

    tmp_date = from_date

    # Create dates for x-axis
    x_dates = scandic_deaths_df[from_date:].index.strftime("%d %b")  # .unique()

    # Add subplot
    ax = fig.add_subplot(111)
    ax2 = fig.add_subplot(111)
    # ax3 = fig.add_subplot(111)

    # Set background color
    fig.patch.set_facecolor(bg_col)
    ax.patch.set_facecolor(bg_col)

    # Plot per country
    ax = sns.lineplot(
        x=x_dates,
        y="Rolling7",
        data=scandic_deaths_df[from_date:],
        sort=False,
        palette=sns.color_palette("Set1", n_colors=3, desat=0.8),
        hue="Country",
        linestyle="--",
    )
    ax.set_xlabel("Date", fontsize=16)
    ax.set_ylabel("Increase", fontsize=16)

    # ax3 = sns.lineplot(x=x_dates, y="Rolling30", data=scandic_deaths_df[from_date:], sort=False, palette=sns.color_palette("Set1", n_colors=3, desat=.8), hue="Country", linestyle="-.")

    ax2 = sns.barplot(
        x=x_dates,
        y="Increase",
        data=scandic_deaths_df[from_date:],
        palette=sns.color_palette("Set1", n_colors=3, desat=0.8),
        hue="Country",
        linewidth=0,
    )  # edgecolor="none")

    # Set the linestyle
    for l in range(len(ax.lines)):
        ax.lines[l].set_linestyle("--")
        ax.lines[l].set_linewidth(l_w)

    # Fix legend
    handles, labels = ax.get_legend_handles_labels()
    # print(*handles)
    handles.pop(1)
    handles.pop(1)
    handles.append(handles[0])
    handles.pop(0)
    labels = labels[3:]
    labels.append("7 day rolling average")
    handles[3].set_color("black")
    ax.legend(handles=handles, labels=labels)

    # Fix the list of dates so it works with MultipleLocator
    date_jump = 5
    x_date_ticks = []
    for i in range(date_jump):
        x_date_ticks.append(0)

    x_date_ticks.extend(x_dates)
    # print(x_date_ticks)

    # Set xticks location and format
    ax.xaxis.set_major_locator(MultipleLocator(base=5))
    ax.set_xticklabels(
        labels=x_date_ticks[0::date_jump], rotation=90, ha="center", size=10
    )

    # Set title and axis parameters
    plt.title("Daily deaths with Coronavirus\nScandinavia", size=30)

    ax.xaxis.label.set_visible(False)
    plt.ylim(bottom=0)

    # Plot
    offset = 9
    data_source = "CSSE at Johns Hopkins University"
    name_string = "DAILY DEATHS"
    show_plot(fig, ax, offset, name_string, data_source, save=save, plot_type=1)


def plot_yearly_deaths_sweden(scb_deaths, save=False):
    # Plot deaths in 2020 and mean deaths in Sweden per day
    fig, ax = plt.subplots(figsize=(20, 12))
    textstr = ""

    to_date = str(datetime.date.today() - datetime.timedelta(days=14))
    to_date = "1900" + to_date[4:]

    # Set color palette and background color
    # color_pal = ["grey", "grey", "grey", "grey", "grey", "red"]
    color_pal = ["red", "grey", "steelblue"]

    fig.patch.set_facecolor(bg_col)
    ax.patch.set_facecolor(bg_col)

    # Plot per country
    # sns.lineplot(data=swe_death_compare, palette=sns.color_palette("Set1", n_colors=6, desat=.8), alpha=.7)
    # print(scb_deaths.loc["1900-10-01":"1900-10-15", :].iloc[:, 5:])
    sns.lineplot(
        data=scb_deaths.loc[:to_date, :].iloc[:, 5:],
        palette=sns.color_palette(color_pal, n_colors=3),
        alpha=0.7,
    )

    # Set xticks location and format
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    ##ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=[0,1,2,3,4,5,6]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))

    # Set the linewidth
    for l in range(len(ax.lines)):
        ax.lines[l].set_linewidth(l_w / 2)
        ax.lines[l].set_linestyle("-")
    ax.lines[0].set_linewidth((l_w / 2) + 1)
    # ax.lines[5].set_color("black")

    # Fix legend
    handles, labels = ax.get_legend_handles_labels()
    # handles[5].set_color("black")
    ax.legend(handles=handles, labels=labels)

    # Set title and axis parameters
    plt.title("Daily Deaths 2015-2020\nSweden", size=30)

    # plt.xlabel('Date', size=20)
    ax.xaxis.label.set_visible(False)
    plt.ylabel("Deaths", size=20)

    # Format xticks
    plt.xticks(rotation=90, size=12)

    # Plot
    offset = 30
    data_source = "Statistiska Centralbyrån, www.scb.se"
    name_string = "DEATHS PER DAY"
    show_plot(fig, ax, offset, name_string, data_source, save=save, plot_type=4)


def save_plot(fname, fig):
    # fname = str(datetime.date.today()) + f"_{fname}.png"
    fname = f"./tmp/{fname}.png"
    plt.savefig(
        fname,
        format="png",
        dpi="figure",
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )


def show_plot(fig, ax, offset, name_string, data_source, save=False, plot_type=1):
    # Control the y value depending on the type of plot
    if plot_type == 1:
        x_value = ax.get_xlim()[0]
        y_value = ax.get_ylim()[0] - ax.get_ylim()[1] / offset
        bb_width = ax.get_xlim()[1]
    elif plot_type == 2:
        x_value = ax.get_xlim()[0] - 2
        y_value = ax.get_ylim()[0] * offset
        bb_width = ax.get_window_extent().width * 5
    elif plot_type == 3:
        x_value = ax.get_xlim()[0]
        y_value = ax.get_ylim()[0] * offset
        bb_width = ax.get_xlim()[1]
    elif plot_type == 4:
        x_value = ax.get_xlim()[0]
        y_value = ax.get_ylim()[0] - offset
        bb_width = ax.get_xlim()[1]

    # The signature bar
    str1 = f"Latest data: {latest_data}"
    str2 = f"Source: {data_source}"  # Statistiska centralbyrån, www.scb.se"
    signature = ax.text(
        x=x_value,
        y=y_value,
        s=str1,
        fontsize=14,
        color="#f0f0f0",
        backgroundcolor="steelblue",
    )
    bb = signature.get_bbox_patch()
    bb.set_boxstyle("ext", pad=0.6, width=bb_width)

    middle = (ax.get_xlim()[0] + ax.get_xlim()[1]) / 2
    ax.text(
        x=middle,
        y=y_value,
        s=name_string,  # "DEATHS PER DAY",
        fontsize=14,
        color="#f0f0f0",
        ha="center",
        backgroundcolor="steelblue",
    )
    ax.text(
        x=ax.get_xlim()[1],
        y=y_value,
        s=str2,
        fontsize=14,
        color="#f0f0f0",
        ha="right",
        backgroundcolor="steelblue",
    )

    sns.despine()
    if save:
        save_plot(name_string.replace(" ", "_"), fig)
    else:
        plt.show()


def make_plot_images():
    plot_confirmed_cases(scandic_df, from_date, to_date, num_countries, save=True)
    plot_yearly_deaths_sweden(scb_deaths, save=True)
    plot_daily_new_cases(scandic_df, from_date, to_date, save=True)
    plot_deaths(scandic_deaths_df, from_date, to_date, num_countries, save=True)


if __name__ == "__main__":
    # plot_confirmed_cases(scandic_df, from_date, to_date, num_countries)
    # plot_yearly_deaths_sweden(scb_deaths)
    # plot_daily_new_cases(scandic_df, from_date, to_date)
    # plot_confirmed_cases_from_day_x(scandi_outbreak, from_date, to_date, num_countries)
    # plot_deaths(scandic_deaths_df, from_date, to_date, num_countries)
    # plot_case_fatality_rate(mortality_rate, from_date, to_date, num_countries)
    # plot_mortality_rate(scandic_deaths_df, from_date, to_date, num_countries)
    # plot_log_deaths(scandic_deaths_df, from_date, to_date, num_countries)
    # plot_swedish_deaths(swe_death_stats)

    ## NOT WORKING
    # plot_daily_new_deaths(scandic_deaths_df, from_date, to_date)

    pass
