import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator, MultipleLocator, ScalarFormatter, AutoLocator
from matplotlib.path import Path
from matplotlib.patches import BoxStyle
import seaborn as sns
import datetime
import locale


# Read the data from Johns Hopkins
conf_url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
confirmed_df = pd.read_csv(conf_url)

death_url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv"
deaths_df = pd.read_csv(death_url)


# Population per country 2018 (data from https://www.nordicstatistics.org/population/)

swe_pop = 10120242
nor_pop = 5295619
den_pop = 5781190


# Extract the Scandinavian countries and combine to a dataframe
scandic = ["Sweden", "Norway", "Denmark"]
scandic_conf_df = confirmed_df[confirmed_df["Country/Region"].isin(scandic)]
scandic_deaths_df = deaths_df[deaths_df["Country/Region"].isin(scandic)]


# Drop not used columns
scandic_conf_df = scandic_conf_df.drop(["Province/State","Lat", "Long"], axis=1).groupby("Country/Region").sum()
scandic_deaths_df = scandic_deaths_df.drop(["Province/State","Lat", "Long"], axis=1).groupby("Country/Region").sum()


scandic_conf_df.head()


scandic_deaths_df.head()


# Create lists from the dataframe to use in creation of long dataframe
dates = scandic_conf_df.keys()

mortality_rate_swe = []
mortality_rate_nor = []
mortality_rate_den = []

sweden_cases = []
sweden_deaths = []

norway_cases = []
norway_deaths = []

denmark_cases = []
denmark_deaths = []

uk_cases = []
uk_deaths = []

for i in dates:
    sweden_sum = scandic_conf_df[i]["Sweden"]
    norway_sum = scandic_conf_df[i]["Norway"]
    denmark_sum = scandic_conf_df[i]["Denmark"]
    #death_sum = deaths[i].sum()
    sweden_death_sum = scandic_deaths_df[i]["Sweden"]
    norway_death_sum = scandic_deaths_df[i]["Norway"]
    denmark_death_sum = scandic_deaths_df[i]["Denmark"]
                                               
    sweden_cases.append(sweden_sum)
    norway_cases.append(norway_sum)
    denmark_cases.append(denmark_sum)

    #total_deaths.append(death_sum)
    sweden_deaths.append(sweden_death_sum)
    norway_deaths.append(norway_death_sum)
    denmark_deaths.append(denmark_death_sum)
    
    mortality_rate_swe.append(sweden_death_sum/sweden_sum)
    mortality_rate_nor.append(norway_death_sum/norway_sum)
    mortality_rate_den.append(denmark_death_sum/denmark_sum)





print(len(scandic_deaths_df.keys()))
print(len(mortality_rate_swe))
print(len(mortality_rate_nor))
print(len(mortality_rate_den))


mortality_rate = pd.DataFrame({"Date": scandic_deaths_df.keys(),"Sweden": mortality_rate_swe, "Norway": mortality_rate_nor, "Denmark": mortality_rate_den})
mortality_rate["Date"] = pd.to_datetime(mortality_rate["Date"])
mortality_rate.set_index("Date", inplace=True)
mortality_rate.tail()


# Make dataframe that starts from a value of confirmed cases

#above5 S: 36, N: 37, D: 41
#above100 S: 44, N: 45, D: 48

s5 = 36
s100 = 44
n5 = 37
n100 = 44
d5 = 41
d100 = 48

scandi_outbreak = pd.DataFrame({"Sweden": sweden_cases[s5:], "Norway": norway_cases[n5:] + (len(sweden_cases[s5:]) - len(norway_cases[n5:])) * [None],
                                "Denmark": denmark_cases[d5:] + (len(sweden_cases[s5:]) - len(denmark_cases[d5:])) * [None],})
scandi_outbreak.head()


# Rearrange the confirmed dataframe to long format with date as index
d = {"Date": scandic_conf_df.keys(), "Sweden": sweden_cases, "Norway": norway_cases, "Denmark": denmark_cases}
scandic_df = pd.DataFrame(data=d)
scandic_df = scandic_df.melt(id_vars=["Date"], var_name="Country", value_name="Confirmed")
scandic_df["Date"] = pd.to_datetime(scandic_df["Date"], format="get_ipython().run_line_magic("m/%d/%y")", "")
scandic_df.set_index("Date", inplace=True)

#scandic_df["2020-03-25":]





# Rearrange the deaths dataframe to long format with date as index
d = {"Date": scandic_deaths_df.keys(), "Sweden": sweden_deaths, "Norway": norway_deaths, "Denmark": denmark_deaths}
scandic_deaths_df = pd.DataFrame(data=d)
scandic_deaths_df = scandic_deaths_df.melt(id_vars=["Date"], var_name="Country", value_name="Deaths")
scandic_deaths_df["Date"] = pd.to_datetime(scandic_deaths_df["Date"])
scandic_deaths_df.set_index("Date", inplace=True)

# Calculate deaths per capita (per mille)
country_normalize = {"Sweden": swe_pop, "Norway": nor_pop, "Denmark": den_pop}
scandic_deaths_df["Pop"] = [country_normalize[x] for x in scandic_deaths_df["Country"]]
scandic_deaths_df["Deaths per capita"] = (scandic_deaths_df["Deaths"] / scandic_deaths_df["Pop"]) * 1000
scandic_deaths_df["Deaths per 100k"] = (scandic_deaths_df["Deaths"] / scandic_deaths_df["Pop"]) * 100000
scandic_deaths_df.drop("Pop", axis=1, inplace=True)

#scandic_deaths_df["2020-04-01":]


scandic_deaths_df["2020-03-29":].tail(20)


# Calculate the daily increase in cases and deaths and add them to the dataframes
diff_list = []
diff_death = []
rolling_7 = []
rolling_30 = []
for c in scandic_df["Country"].unique():
    diff_list.extend(scandic_df[scandic_df["Country"] == c]["Confirmed"].diff().values)
    diff_death.extend(scandic_deaths_df[scandic_deaths_df["Country"] == c]["Deaths"].diff().values)

len(diff_list)
scandic_df["Increase"] = diff_list
scandic_deaths_df["Increase"] = diff_death

for c in scandic_deaths_df["Country"].unique():
    rolling_7.extend(scandic_deaths_df[scandic_deaths_df["Country"] == c]["Increase"].rolling(7).mean())
    rolling_30.extend(scandic_deaths_df[scandic_deaths_df["Country"] == c]["Increase"].rolling(30).mean())
    
scandic_deaths_df["Rolling7"] = rolling_7
scandic_deaths_df["Rolling30"] = rolling_30


scandic_deaths_df["2020-03-01":].head(20)


# Calculate 7 day rolling average for deaths
mean_list = []
for c in scandic_df["Country"].unique():
    mean_list.extend(scandic_df[scandic_df["Country"] == c]["Confirmed"].rolling(7).mean())

print(len(mean_list))
scandic_df["Mean7"] = mean_list

#scandic_df["2020-04-05":]


class ExtendedTextBox(BoxStyle._Base):
    """
    An Extended Text Box that expands to the axes limits 
                        if set in the middle of the axes
    """

    def __init__(self, pad=0.3, width=500.):
        """
        width: 
            width of the textbox. 
            Use `ax.get_window_extent().width` 
                   to get the width of the axes.
        pad: 
            amount of padding (in vertical direction only)
        """
        self.width=width
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
        height = height + 2.*pad
        # boundary of the padded box
        y0 = y0 - pad
        y1 = y0 + height
        _x0 = x0
        x0 = _x0 +width /2. - self.width/2.
        x1 = _x0 +width /2. + self.width/2.

        cp = [(x0, y0),
              (x1, y0), (x1, y1), (x0, y1),
              (x0, y0)]

        com = [Path.MOVETO,
               Path.LINETO, Path.LINETO, Path.LINETO,
               Path.CLOSEPOLY]

        path = Path(cp, com)

        return path


# Set date intervals for plots

from_date = "2020-02-25"
to_date = ""
if to_date == "":
    to_date = str(scandic_df.index[-1].date())
    
# Set Seaborn style
sns.set_style("whitegrid", {"axes.facecolor": "1.0", "grid.color": ".95", "grid.linestyle": "--"})

# Background color
bg_col = "floralwhite"

# Set line width
l_w = 3

# Calculate number of countries in dataframe
num_countries = len(scandic_df["Country"].unique())

# register the custom textbox style
BoxStyle._style_list["ext"] = ExtendedTextBox


# Old textbox
"""
# Create and display infotext
textstr = '\n'.join((
    "ci = standard deviation",
    f"Latest data: {scandic_df.index[-1].date()}",
    "Source: CSSE at Johns Hopkins University"))
props = dict(boxstyle='round', facecolor='salmon', alpha=0.5)
ax.text(0.05, 0.2, textstr, transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
"""


scandic_df[scandic_df["Country"] == "Sweden"][from_date:to_date].shape


# Plot confirmed cases
fig, ax = plt.subplots(figsize=(20,12))

# Set background color
fig.patch.set_facecolor(bg_col)
ax.patch.set_facecolor(bg_col)

# Plot per country
sns.lineplot(x=scandic_df[from_date:to_date].index, y="Confirmed", data=scandic_df[from_date:to_date], palette=sns.color_palette("Set1", n_colors=num_countries, desat=.8), hue="Country")

# Plot Sweden rolling mean
#sns.lineplot(x=scandic_df[scandic_df["Country"] == "Sweden"][from_date:to_date].index, y="Mean7", data=scandic_df[scandic_df["Country"] == "Sweden"][from_date:to_date], color="salmon", alpha=.6)

# Plot Scandinavian mean
sns.lineplot(x=scandic_df[from_date:to_date].index, y="Confirmed", data=scandic_df[from_date:to_date], color="purple", label="Scandinavia (mean)", alpha=.6, ci="sd")



# Set xticks location and format
ax.xaxis.set_major_locator(mdates.DayLocator(interval=14))
#ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=range(0,7,2)))
ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('get_ipython().run_line_magic("d", " %b'));")


#ax.xaxis.set_major_formatter(mdates.DateFormatter('get_ipython().run_line_magic("d", " %b'));")

# Format confidence interval
ax.get_children()[0].set_alpha(.1)

# Set the linewidth
for l in range(len(ax.lines)):
    ax.lines[l].set_linewidth(l_w)
    #print(l, ax.lines[l].get_color())

# Create and display infotext
textstr = "ci = standard deviation"
# bbcol = ax.lines[7].get_color()
props = dict(boxstyle='round', facecolor="steelblue", alpha=.9)
ax.text(0.05, 0.2, textstr, transform=ax.transAxes, fontsize=14,
        color="#f0f0f0", verticalalignment='top', bbox=props)

# Set title and axis parameters
plt.title('Confirmed cases of Coronavirus\nScandinavia', size=30)

#plt.xlabel('Date', size=20)
ax.xaxis.label.set_visible(False)
plt.ylabel('Cases', size=20)

# Format xticks
plt.xticks(rotation=90, size=15)


# The signature bar
offset = 7
#str1 = f"Latest data: {scandic_df.index[-1].date()}. Fetched at {datetime.datetime.utcnow():get_ipython().run_line_magic("Y-%m-%d", " %H:%M} UTZ\"")
str1 = f"Latest data: {scandic_df.index[-1].date()}"
str2 = "Source: CSSE at Johns Hopkins University"
signature = ax.text(x = ax.get_xlim()[0], y = ax.get_ylim()[0]-ax.get_ylim()[1]/offset,
                    s = str1, fontsize = 14, color = "#f0f0f0",
                    backgroundcolor = "steelblue")
bb = signature.get_bbox_patch()
bb.set_boxstyle("ext", pad=0.6, width=ax.get_xlim()[1])

middle = (ax.get_xlim()[0] + ax.get_xlim()[1]) / 2
ax.text(x = middle, y = ax.get_ylim()[0]-ax.get_ylim()[1]/offset,
                    s = "CUMULATIVE CASES", fontsize = 14, color = "#f0f0f0",
                    ha="center", backgroundcolor = "steelblue")
ax.text(x = ax.get_xlim()[1], y = ax.get_ylim()[0]-ax.get_ylim()[1]/offset,
                    s = str2, fontsize = 14, color = "#f0f0f0",
                    ha="right", backgroundcolor = "steelblue")


# Save as jpg and show plot
fname = str(datetime.date.today()) + "_scandinavia_conf" + ".png"
plt.savefig(fname, format="png", dpi="figure", bbox_inches="tight", facecolor=fig.get_facecolor())
sns.despine()
plt.show()


# Plot daily new cases
fig, ax = plt.subplots(figsize=(20,12))

# Set background color
fig.patch.set_facecolor(bg_col)
ax.patch.set_facecolor(bg_col)

# Plot per country
#sns.lineplot(x=scandic_df[from_date:].index, y="Increase", data=scandic_df[from_date:], palette=sns.color_palette("Set1", n_colors=num_countries, desat=.8), hue="Country")

# Plot Scandinavian mean
sns.lineplot(x=scandic_df[from_date:].index, y="Increase", data=scandic_df[from_date:], label="Scandinavia (mean)", alpha=.6, ci="sd")

# Set xticks location and format
ax.xaxis.set_major_locator(mdates.DayLocator(interval=14))
#ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=[0,1,2,3,4,5,6]))
ax.xaxis.set_major_formatter(mdates.DateFormatter('get_ipython().run_line_magic("d", " %b'));")

# Format confidence interval
ax.get_children()[0].set_alpha(.1)

# Set the linewidth
for l in range(len(ax.lines)):
    ax.lines[l].set_linewidth(l_w)

# Create and display infotext
textstr = "ci = standard deviation"
props = dict(boxstyle='round', facecolor='steelblue', alpha=.9)
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
        color="#f0f0f0",verticalalignment='top', bbox=props)

# Set title and axis parameters
plt.title('Daily new cases of Coronavirus\nScandinavia (mean)', size=30)

#plt.xlabel('Date', size=20)
ax.xaxis.label.set_visible(False)
plt.ylabel('Cases', size=20)

# Format xticks
plt.xticks(rotation=90, size=15)

# The signature bar
offset = 6
str1 = f"Latest data: {scandic_df.index[-1].date()}"
str2 = "Source: CSSE at Johns Hopkins University"
#print(ax.get_ylim())
signature = ax.text(x = ax.get_xlim()[0], y = ax.get_ylim()[0]-ax.get_ylim()[1]/offset,
                    s = str1, fontsize = 14, color = "#f0f0f0",
                    backgroundcolor = "steelblue")
bb = signature.get_bbox_patch()
bb.set_boxstyle("ext", pad=0.6, width=ax.get_xlim()[1])

middle = (ax.get_xlim()[0] + ax.get_xlim()[1]) / 2
ax.text(x = middle, y = ax.get_ylim()[0]-ax.get_ylim()[1]/offset,
                    s = "DAILY INCREASE", fontsize = 14, color = "#f0f0f0",
                    ha="center", backgroundcolor = "steelblue")
ax.text(x = ax.get_xlim()[1], y = ax.get_ylim()[0]-ax.get_ylim()[1]/offset,
                    s = str2, fontsize = 14, color = "#f0f0f0",
                    ha="right", backgroundcolor = "steelblue")

# Save as jpg and show plot
fname = str(datetime.date.today()) + "_scandinavia_new_daily" + ".png"
plt.savefig(fname, format="png", dpi="figure", bbox_inches="tight", facecolor=fig.get_facecolor())
sns.despine()
plt.show()


# Plot confirmed cases from day with X cases
fig, ax = plt.subplots(figsize=(20,12))
sns.lineplot(data=scandi_outbreak, palette=sns.color_palette("Set1", n_colors=num_countries, desat=.8))

# Set background color
fig.patch.set_facecolor(bg_col)
ax.patch.set_facecolor(bg_col)

# Set title and axis parameters
plt.title('Confirmed cases of Coronavirus per day\nfrom day with 5 cases or more', size=30)

plt.xlabel('Days from confirmed cases > 5', size=20)
plt.ylabel('Cases', size=20)

# Format xticks
plt.xticks(range(0, len(scandi_outbreak), 10), size=15, rotation=45)

# Set the linewidth
for l in range(len(ax.lines)):
    ax.lines[l].set_linewidth(l_w)

# Position legend
plt.legend(loc="upper left")

# The signature bar
offset = 3.8
str1 = f"Latest data: {scandic_df.index[-1].date()}"
str2 = "Source: CSSE at Johns Hopkins University"
signature = ax.text(x = ax.get_xlim()[0]-2, y = ax.get_ylim()[0]*offset,
                    s = str1, fontsize = 14, color = "#f0f0f0",
                    backgroundcolor = "steelblue")
bb = signature.get_bbox_patch()
bb.set_boxstyle("ext", pad=0.6, width=ax.get_window_extent().width*2)

middle = (ax.get_xlim()[0] + ax.get_xlim()[1]) / 2
ax.text(x = middle, y = ax.get_ylim()[0]*offset,
                    s = "CUMULATIVE CASES FROM DAY n > 5", fontsize = 14, color = "#f0f0f0",
                    ha="center", backgroundcolor = "steelblue")
ax.text(x = ax.get_xlim()[1], y = ax.get_ylim()[0]*offset,
                    s = str2, fontsize = 14, color = "#f0f0f0",
                    ha="right", backgroundcolor = "steelblue")

# Save as jpg and show plot
fname = str(datetime.date.today()) + "_scandinavia_conf_outbreak" + ".png"
#plt.savefig(fname, format="png", dpi="figure", bbox_inches="tight", facecolor=fig.get_facecolor())
sns.despine()
plt.show()


# Plot deaths
fig, ax = plt.subplots(figsize=(20,12))
textstr = ""

# Set background color
fig.patch.set_facecolor(bg_col)
ax.patch.set_facecolor(bg_col)

# Plot per country
sns.lineplot(x=scandic_deaths_df[from_date:to_date].index, y="Deaths", data=scandic_deaths_df[from_date:to_date], palette=sns.color_palette("Set1", n_colors=num_countries, desat=.8), hue="Country")

# Plot mean if set to true
plot_mean = False
if plot_mean:
    # Plot Scandinavian mean
    sns.lineplot(x=scandic_deaths_df[from_date:to_date].index, y="Deaths", data=scandic_deaths_df[from_date:to_date], color="purple", label="Scandinavia (mean)", alpha=.6, ci="sd")

    # Format confidence interval
    ax.get_children()[0].set_alpha(.1)
    
    # Create and display infotext
    textstr = "ci = standard deviation"
    props = dict(boxstyle='round', facecolor='steelblue', alpha=.9)
    ax.text(0.05, 0.2, textstr, transform=ax.transAxes, fontsize=14,
            color="#f0f0f0", verticalalignment='top', bbox=props)

# Set xticks location and format
ax.xaxis.set_major_locator(mdates.DayLocator(interval=14))
#ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=[0,1,2,3,4,5,6]))
ax.xaxis.set_major_formatter(mdates.DateFormatter('get_ipython().run_line_magic("d", " %b'));")

# Set the linewidth
for l in range(len(ax.lines)):
    ax.lines[l].set_linewidth(l_w)



# Set title and axis parameters
plt.title('Deaths with Coronavirus\nScandinavia', size=30)


#plt.xlabel('Date', size=20)
ax.xaxis.label.set_visible(False)
plt.ylabel('Deaths', size=20)

# Format xticks
plt.xticks(rotation=90, size=15)

# The signature bar
offset = 4
str1 = f"Latest data: {scandic_df.index[-1].date()}"
str2 = "Source: CSSE at Johns Hopkins University"
signature = ax.text(x = ax.get_xlim()[0], y = ax.get_ylim()[0]*offset,
                    s = str1, fontsize = 14, color = "#f0f0f0",
                    backgroundcolor = "steelblue")
bb = signature.get_bbox_patch()
bb.set_boxstyle("ext", pad=0.6, width=ax.get_xlim()[1])

middle = (ax.get_xlim()[0] + ax.get_xlim()[1]) / 2
ax.text(x = middle, y = ax.get_ylim()[0]*offset,
                    s = "CUMULATIVE DEATHS", fontsize = 14, color = "#f0f0f0",
                    ha="center", backgroundcolor = "steelblue")
ax.text(x = ax.get_xlim()[1], y = ax.get_ylim()[0]*offset,
                    s = str2, fontsize = 14, color = "#f0f0f0",
                    ha="right", backgroundcolor = "steelblue")

# Save as jpg and show plot
fname = str(datetime.date.today()) + "_scandinavia_deaths" + ".png"
plt.savefig(fname, format="png", dpi="figure", bbox_inches="tight", facecolor=fig.get_facecolor())
sns.despine()
plt.show()


## Den här är letalitet (case fatality rate) och inte mortalitet (mortality).
## CFR är andelen av insjuknade som dör medans mortalitet är andel av populationen

# Plot Case Fatality Rate
fig, ax = plt.subplots(figsize=(20,12))

# Set background color
fig.patch.set_facecolor(bg_col)
ax.patch.set_facecolor(bg_col)

f_d = 47
genomsnitt = np.mean(mortality_rate[["Sweden", "Norway", "Denmark"]][f_d:])
print(genomsnitt)
sns.lineplot(data=mortality_rate[f_d:], palette=sns.color_palette("Set1", n_colors=num_countries, desat=.8))
#sns.lmplot(x=scandic_deaths_df.index.unique()[f_d:], y=mortality_rate_swe[f_d:])
ax.axhline(genomsnitt.mean(), linestyle="-.", color="grey", alpha=.4)

ax.xaxis.set_major_locator(mdates.DayLocator(interval=14))
#ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=range(0,7)))
ax.xaxis.set_major_formatter(mdates.DateFormatter("get_ipython().run_line_magic("d", " %b\"))")

#ax.lines[0].set_linestyle("--")
#ax.lines[0].set_alpha(.5)

# Set the linewidth
for l in range(len(ax.lines)):
    ax.lines[l].set_linewidth(l_w)

# Format x-ticks
plt.xticks(rotation=90, size=15)

# Set title and axis parameters
plt.title("Case Fatality Rate (deaths/confirmed)", size=30)
#plt.xlabel("Date", size=20)
ax.xaxis.label.set_visible(False)
plt.ylabel("CFR", size=20)

# The signature bar
offset = 4
str1 = f"Latest data: {scandic_df.index[-1].date()}"
str2 = "Source: CSSE at Johns Hopkins University"
signature = ax.text(x = ax.get_xlim()[0], y = ax.get_ylim()[0]*offset,
                    s = str1, fontsize = 14, color = "#f0f0f0",
                    backgroundcolor = "steelblue")
bb = signature.get_bbox_patch()
bb.set_boxstyle("ext", pad=0.6, width=ax.get_xlim()[1])

middle = (ax.get_xlim()[0] + ax.get_xlim()[1]) / 2
ax.text(x = middle, y = ax.get_ylim()[0]*offset,
                    s = "CASE FATALITY RATE", fontsize = 14, color = "#f0f0f0",
                    ha="center", backgroundcolor = "steelblue")
ax.text(x = ax.get_xlim()[1], y = ax.get_ylim()[0]*offset,
                    s = str2, fontsize = 14, color = "#f0f0f0",
                    ha="right", backgroundcolor = "steelblue")

# Show plot
sns.despine()
plt.show()


## Denna är mortalitet

# Plot deaths per capita
fig, ax = plt.subplots(figsize=(20,12))
textstr = ""

# Set background color
fig.patch.set_facecolor(bg_col)
ax.patch.set_facecolor(bg_col)

# Plot per country
sns.lineplot(x=scandic_deaths_df[from_date:to_date].index, y="Deaths per capita", data=scandic_deaths_df[from_date:to_date], palette=sns.color_palette("Set1", n_colors=num_countries, desat=.8), hue="Country")

# Plot mean if set to true
plot_mean = True
if plot_mean:
    # Plot Scandinavian mean
    sns.lineplot(x=scandic_deaths_df[from_date:to_date].index, y="Deaths per capita", data=scandic_deaths_df[from_date:to_date], color="purple", label="Scandinavia (mean)", alpha=.6, ci="sd")

    # Format confidence interval
    ax.get_children()[0].set_alpha(.1)
    textstr = "ci = standard deviation\n"

# Set xticks location and format
ax.xaxis.set_major_locator(mdates.DayLocator(interval=14))
#ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=[0,1,2,3,4,5,6]))
ax.xaxis.set_major_formatter(mdates.DateFormatter('get_ipython().run_line_magic("d", " %b'));")

# Create and display infotext
textstr = textstr + "Population data:\n2018 from from www.nordicstatistics.org"
props = dict(boxstyle='round', facecolor='steelblue', alpha=.9)
ax.text(0.05, 0.2, textstr, transform=ax.transAxes, fontsize=14,
        color="#f0f0f0", verticalalignment='top', bbox=props)

# Set title and axis parameters
plt.title("Deaths per capita with Coronavirus\nScandinavia", size=30)

#plt.xlabel("Date", size=20)
ax.xaxis.label.set_visible(False)
plt.ylabel("Deaths per capita (per mille)", size=20)

# Set the linewidth
for l in range(len(ax.lines)):
    ax.lines[l].set_linewidth(l_w)

# Format xticks
plt.xticks(rotation=90, size=15)

# Find last values
swe_last_dpc = scandic_deaths_df[scandic_deaths_df["Country"] == "Sweden"]["Deaths per capita"][-1]
nor_last_dpc = scandic_deaths_df[scandic_deaths_df["Country"] == "Norway"]["Deaths per capita"][-1]
den_last_dpc = scandic_deaths_df[scandic_deaths_df["Country"] == "Denmark"]["Deaths per capita"][-1]

# Add annotations
ann_col = "salmon"
txt_col = "black"
alpha = .4
ax.annotate(str(round(swe_last_dpc,3)) + "‰", xy=(scandic_deaths_df.index[-1], swe_last_dpc),  xycoords="data",
             xytext=(-30, 0), textcoords="offset points",
             size=13, ha="right", va="center",
             bbox=dict(boxstyle="round", alpha=alpha, color=ann_col),
             arrowprops=dict(arrowstyle="wedge,tail_width=0.5", alpha=alpha, color=ann_col), color=txt_col)
ax.annotate(str(round(nor_last_dpc,3)) + "‰", xy=(scandic_deaths_df.index[-1], nor_last_dpc),  xycoords="data",
             xytext=(-30, 0), textcoords="offset points",
             size=13, ha="right", va="center",
             bbox=dict(boxstyle="round", alpha=alpha, color=ann_col),
             arrowprops=dict(arrowstyle="wedge,tail_width=0.5", alpha=alpha, color=ann_col), color=txt_col)
ax.annotate(str(round(den_last_dpc,3)) + "‰", xy=(scandic_deaths_df.index[-1], den_last_dpc),  xycoords="data",
             xytext=(-30, 0), textcoords="offset points",
             size=13, ha="right", va="center",
             bbox=dict(boxstyle="round", alpha=alpha, color=ann_col),
             arrowprops=dict(arrowstyle="wedge,tail_width=0.5", alpha=alpha, color=ann_col), color=txt_col)

# The signature bar
offset = 2.8

str1 = f"Latest data: {scandic_df.index[-1].date()}"
str2 = "Source: CSSE at Johns Hopkins University"
signature = ax.text(x = ax.get_xlim()[0], y = ax.get_ylim()[0]*offset,
                    s = str1, fontsize = 14, color = "#f0f0f0",
                    backgroundcolor = "steelblue")
bb = signature.get_bbox_patch()
bb.set_boxstyle("ext", pad=0.6, width=ax.get_xlim()[1])

middle = (ax.get_xlim()[0] + ax.get_xlim()[1]) / 2
ax.text(x = middle, y = ax.get_ylim()[0]*offset,
                    s = "MORTALITY RATE", fontsize = 14, color = "#f0f0f0",
                    ha="center", backgroundcolor = "steelblue")
ax.text(x = ax.get_xlim()[1], y = ax.get_ylim()[0]*offset,
                    s = str2, fontsize = 14, color = "#f0f0f0",
                    ha="right", backgroundcolor = "steelblue")

# Save as jpg and show plot
fname = str(datetime.date.today())+ "_scandinavia_dpc" + ".png"
#plt.savefig(fname, format="png", dpi="figure", bbox_inches="tight", facecolor=fig.get_facecolor())
sns.despine()
plt.show()


## Denna är också mortalitet men per 100k istället för per 1k

# Plot deaths per capita
fig, ax = plt.subplots(figsize=(20,12))
textstr = ""

# Set background color
fig.patch.set_facecolor(bg_col)
ax.patch.set_facecolor(bg_col)

# Plot per country
sns.lineplot(x=scandic_deaths_df[from_date:to_date].index, y="Deaths per 100k", data=scandic_deaths_df[from_date:to_date], palette=sns.color_palette("Set1", n_colors=num_countries, desat=.8), hue="Country")

# Plot mean if set to true
plot_mean = True
if plot_mean:
    # Plot Scandinavian mean
    sns.lineplot(x=scandic_deaths_df[from_date:to_date].index, y="Deaths per 100k", data=scandic_deaths_df[from_date:to_date], color="purple", label="Scandinavia (mean)", alpha=.6, ci="sd")

    # Format confidence interval
    ax.get_children()[0].set_alpha(.1)
    textstr = "ci = standard deviation\n"

# Set xticks location and format
ax.xaxis.set_major_locator(mdates.DayLocator(interval=14))
#ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=[0,1,2,3,4,5,6]))
ax.xaxis.set_major_formatter(mdates.DateFormatter('get_ipython().run_line_magic("d", " %b'));")

# Create and display infotext
textstr = textstr + "Population data:\n2018 from from www.nordicstatistics.org"
props = dict(boxstyle='round', facecolor='steelblue', alpha=.9)
ax.text(0.05, 0.2, textstr, transform=ax.transAxes, fontsize=14,
        color="#f0f0f0", verticalalignment='top', bbox=props)

# Set title and axis parameters
plt.title("Deaths per 100 000 citizens with Coronavirus\nScandinavia", size=30)

#plt.xlabel("Date", size=20)
ax.xaxis.label.set_visible(False)
plt.ylabel("Deaths per 100k", size=20)

# Set the linewidth
for l in range(len(ax.lines)):
    ax.lines[l].set_linewidth(l_w)

# Format xticks
plt.xticks(rotation=90, size=15)

# Find last values
swe_last_dpc = scandic_deaths_df[scandic_deaths_df["Country"] == "Sweden"]["Deaths per 100k"][-1]
nor_last_dpc = scandic_deaths_df[scandic_deaths_df["Country"] == "Norway"]["Deaths per 100k"][-1]
den_last_dpc = scandic_deaths_df[scandic_deaths_df["Country"] == "Denmark"]["Deaths per 100k"][-1]

# Add annotations
ann_col = "salmon"
txt_col = "black"
alpha = .4
ax.annotate(str(round(swe_last_dpc,3)), xy=(scandic_deaths_df.index[-1], swe_last_dpc),  xycoords="data",
             xytext=(-30, 0), textcoords="offset points",
             size=13, ha="right", va="center",
             bbox=dict(boxstyle="round", alpha=alpha, color=ax.lines[0].get_color()),
             arrowprops=dict(arrowstyle="wedge,tail_width=0.5", alpha=alpha, color=ax.lines[0].get_color()), color=txt_col)
ax.annotate(str(round(nor_last_dpc,3)), xy=(scandic_deaths_df.index[-1], nor_last_dpc),  xycoords="data",
             xytext=(-30, 0), textcoords="offset points",
             size=13, ha="right", va="center",
             bbox=dict(boxstyle="round", alpha=alpha, color=ax.lines[1].get_color()),
             arrowprops=dict(arrowstyle="wedge,tail_width=0.5", alpha=alpha, color=ax.lines[1].get_color()), color=txt_col)
ax.annotate(str(round(den_last_dpc,3)), xy=(scandic_deaths_df.index[-1], den_last_dpc),  xycoords="data",
             xytext=(-30, 0), textcoords="offset points",
             size=13, ha="right", va="center",
             bbox=dict(boxstyle="round", alpha=alpha, color=ax.lines[2].get_color()),
             arrowprops=dict(arrowstyle="wedge,tail_width=0.5", alpha=alpha, color=ax.lines[2].get_color()), color=txt_col)

# The signature bar
offset = 2.8
str1 = f"Latest data: {scandic_df.index[-1].date()}"
str2 = "Source: CSSE at Johns Hopkins University"
signature = ax.text(x = ax.get_xlim()[0], y = ax.get_ylim()[0]*offset,
                    s = str1, fontsize = 14, color = "#f0f0f0",
                    backgroundcolor = "steelblue")
bb = signature.get_bbox_patch()
bb.set_boxstyle("ext", pad=0.6, width=ax.get_xlim()[1])

middle = (ax.get_xlim()[0] + ax.get_xlim()[1]) / 2
ax.text(x = middle, y = ax.get_ylim()[0]*offset,
                    s = "MORTALITY RATE", fontsize = 14, color = "#f0f0f0",
                    ha="center", backgroundcolor = "steelblue")
ax.text(x = ax.get_xlim()[1], y = ax.get_ylim()[0]*offset,
                    s = str2, fontsize = 14, color = "#f0f0f0",
                    ha="right", backgroundcolor = "steelblue")



# Save as jpg and show plot
fname = str(datetime.date.today())+ "_scandinavia_dp100k" + ".png"
plt.savefig(fname, format="png", dpi="figure", bbox_inches="tight", facecolor=fig.get_facecolor())
sns.despine()
plt.show()


# Number of people difference between Sweden and Denmark 
(swe_pop * (swe_last_dpc)/100000) - (swe_pop * (den_last_dpc)/100000)


# Number of people difference between Denmark and Sweden
(den_pop * (swe_last_dpc)/100000) - (den_pop * (den_last_dpc)/100000)


scandic_deaths_df["2020-03-31":]


sweden_mortality_2018 = pd.read_csv("mortalitet_sverige_2018.csv", sep=";", decimal=",")
#sweden_mortality_2018.rename({" Antal döda per 100 000": "Deaths per 100k"}, axis=1, inplace=True, errors="raise")



sweden_mortality_2018.columns = sweden_mortality_2018.columns.str.strip()
sweden_mortality_2018.head()


sweden_mortality_2018.drop(["Total", "Age0-85+", "BothGender", "2018"], axis=1, inplace=True)


sweden_mortality_2018.replace("\xad\xad", np.nan, inplace=True)
sweden_mortality_2018.fillna(0, inplace=True)
sweden_mortality_2018["DeadPer100k"] = [str(x).replace(",", ".") for x in sweden_mortality_2018["DeadPer100k"]]


sweden_mortality_2018["DeadPer100k"] = pd.to_numeric(sweden_mortality_2018["DeadPer100k"])


sweden_mortality_2018.sort_values(by="DeadPer100k", axis=0, ascending=False, na_position="first").head(20)


scandic_deaths_df["logDeaths"] = scandic_deaths_df["Deaths"].apply(lambda x: np.log(x))
scandic_deaths_df.tail()


# Plot log of deaths
fig, ax = plt.subplots(figsize=(20,12))
textstr = ""
from_date = "2020-03-14"

# Set background color
fig.patch.set_facecolor(bg_col)
ax.patch.set_facecolor(bg_col)

# Plot per country
sns.lineplot(x=scandic_deaths_df[from_date:to_date].index, y=scandic_deaths_df[from_date:to_date]["Deaths"].apply(lambda x: np.log(x)), data=scandic_deaths_df[from_date:to_date], palette=sns.color_palette("Set1", n_colors=num_countries, desat=.8), hue="Country")

# Plot mean if set to true
plot_mean = False
if plot_mean:
    # Plot Scandinavian mean
    sns.lineplot(x=scandic_deaths_df[from_date:to_date].index, y="Deaths", data=scandic_deaths_df[from_date:to_date], color="purple", label="Scandinavia (mean)", alpha=.6, ci="sd")

    # Format confidence interval
    ax.get_children()[0].set_alpha(.1)
    
    # Create and display infotext
    textstr = "ci = standard deviation"
    props = dict(boxstyle='round', facecolor='steelblue', alpha=.9)
    ax.text(0.05, 0.2, textstr, transform=ax.transAxes, fontsize=14,
            color="#f0f0f0", verticalalignment='top', bbox=props)

# Set xticks location and format
ax.xaxis.set_major_locator(mdates.DayLocator(interval=14))
#ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=[0,1,2,3,4,5,6]))
ax.xaxis.set_major_formatter(mdates.DateFormatter('get_ipython().run_line_magic("d", " %b'));")

# Set the linewidth
for l in range(len(ax.lines)):
    ax.lines[l].set_linewidth(l_w)
    ax.lines[l].set_linestyle("--")

# Set title and axis parameters
plt.title('Log Deaths with Coronavirus\nScandinavia', size=30)
ax.set(yscale="log")
ax.set_ylim(1e0,)
#plt.xlabel('Date', size=20)
ax.xaxis.label.set_visible(False)
plt.ylabel('Log Deaths', size=20)

# Format xticks
plt.xticks(rotation=90, size=15)

# The signature bar
offset = .7
str1 = f"Latest data: {scandic_df.index[-1].date()}"
str2 = "Source: CSSE at Johns Hopkins University"
signature = ax.text(x = ax.get_xlim()[0], y = ax.get_ylim()[0]*offset,
                    s = str1, fontsize = 14, color = "#f0f0f0",
                    backgroundcolor = "steelblue")
bb = signature.get_bbox_patch()
bb.set_boxstyle("ext", pad=0.6, width=ax.get_xlim()[1])

middle = (ax.get_xlim()[0] + ax.get_xlim()[1]) / 2
ax.text(x = middle, y = ax.get_ylim()[0]*offset,
                    s = "LOG OF CUMULATIVE DEATHS", fontsize = 14, color = "#f0f0f0",
                    ha="center", backgroundcolor = "steelblue")
ax.text(x = ax.get_xlim()[1], y = ax.get_ylim()[0]*offset,
                    s = str2, fontsize = 14, color = "#f0f0f0",
                    ha="right", backgroundcolor = "steelblue")

# Save as jpg and show plot
fname = str(datetime.date.today()) + "_scandinavia_log_deaths" + ".png"
#plt.savefig(fname, format="png", dpi="figure", bbox_inches="tight", facecolor=fig.get_facecolor())
sns.despine()
plt.show()


days_of_dead = scandic_deaths_df[scandic_deaths_df["Country"] == "Sweden"]["2020-03-11":]
print(len(days_of_dead))
print(days_of_dead["Deaths"][-1])
print("Genomsnitt:", days_of_dead["Deaths"][-1] / len(days_of_dead))


swe_death_stats = pd.read_csv("sweden_death_stats10.csv", sep=";", parse_dates=["Datum"], index_col=["Datum"])
swe_death_stats["Sum 2020"] = swe_death_stats["Deaths 2020"].cumsum()
swe_death_stats["Mean Sum"] = swe_death_stats["Mean Deaths 2015-2019"].cumsum()
swe_death_stats.drop(["Deaths 2020", "Mean Deaths 2015-2019"], axis=1, inplace=True)
#swe_death_stats.head()


# Plot deaths in 2020 and mean deaths in Sweden per day
fig, ax = plt.subplots(figsize=(20,12))
textstr = ""

# Set background color
fig.patch.set_facecolor(bg_col)
ax.patch.set_facecolor(bg_col)

# Plot per country
sns.lineplot(data=swe_death_stats["2020-02-24":], palette=sns.color_palette("Set1", n_colors=2, desat=.8))

# Set xticks location and format
ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
#ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=[0,1,2,3,4,5,6]))
ax.xaxis.set_major_formatter(mdates.DateFormatter('get_ipython().run_line_magic("d", " %b'));")

# Set the linewidth
for l in range(len(ax.lines)):
    ax.lines[l].set_linewidth(l_w)

# Set title and axis parameters
plt.title('Cumulative Deaths in 2020 and mean of 2015-2019\nSweden\n(not normalized for population)', size=30)


#plt.xlabel('Date', size=20)
ax.xaxis.label.set_visible(False)
plt.ylabel('Deaths', size=20)

# Format xticks
plt.xticks(rotation=90, size=12)

# The signature bar
offset = 6500
str1 = f"Latest data: {swe_death_stats.index[-1].date()}"
str2 = "Source: Statistiska centralbyrån, www.scb.se"
signature = ax.text(x = ax.get_xlim()[0], y = ax.get_ylim()[0]-offset,
                    s = str1, fontsize = 14, color = "#f0f0f0",
                    backgroundcolor = "steelblue")
bb = signature.get_bbox_patch()
bb.set_boxstyle("ext", pad=0.6, width=ax.get_xlim()[1])

middle = (ax.get_xlim()[0] + ax.get_xlim()[1]) / 2
ax.text(x = middle, y = ax.get_ylim()[0]-offset,
                    s = "CUMULATIVE DEATHS", fontsize = 14, color = "#f0f0f0",
                    ha="center", backgroundcolor = "steelblue")
ax.text(x = ax.get_xlim()[1], y = ax.get_ylim()[0]-offset,
                    s = str2, fontsize = 14, color = "#f0f0f0",
                    ha="right", backgroundcolor = "steelblue")

# Save as jpg and show plot
fname = str(datetime.date.today()) + "_sweden_death_stats" + ".png"
#plt.savefig(fname, format="png", dpi="figure", bbox_inches="tight", facecolor=fig.get_facecolor())
sns.despine()
plt.show()


swe_death_stats.tail()


swe_death_compare = pd.read_csv("swe_death_compare.csv", sep=";", parse_dates=["Datum"], index_col=["Datum"])


swe_death_compare.tail()


swe_death_compare.loc[:,"2015":"2019"].tail()


swe_death_compare["Mean 2015 2019"] = swe_death_compare.loc[:, "2015":"2019"].mean(axis=1)


swe_death_compare.info()


## DEPRACETED - NEW VERSION FURTHER DOWN

"""
# Plot deaths in 2020 and mean deaths in Sweden per day
fig, ax = plt.subplots(figsize=(20,12))
textstr = ""

# Set color palette and background color
#color_pal = ["grey", "grey", "grey", "grey", "grey", "red"]
color_pal = ["red", "grey"]

fig.patch.set_facecolor(bg_col)
ax.patch.set_facecolor(bg_col)


# Plot per country
#sns.lineplot(data=swe_death_compare, palette=sns.color_palette("Set1", n_colors=6, desat=.8), alpha=.7)
sns.lineplot(data=swe_death_compare.loc[:"2020-10-01", "2020":], palette=sns.color_palette(color_pal, n_colors=2), alpha=.7)

# Set xticks location and format
ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
##ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=[0,1,2,3,4,5,6]))
ax.xaxis.set_major_formatter(mdates.DateFormatter('get_ipython().run_line_magic("d", " %b'));")

# Set the linewidth
for l in range(len(ax.lines)):
    ax.lines[l].set_linewidth(l_w/2)
    ax.lines[l].set_linestyle("-")
ax.lines[0].set_linewidth((l_w/2)+1)
#ax.lines[5].set_color("black")

# Fix legend
handles, labels = ax.get_legend_handles_labels()
#handles[5].set_color("black")
ax.legend(handles=handles, labels=labels)    

# Set title and axis parameters
plt.title('Daily Deaths 2015-2020\nSweden\n(not normalized for population)', size=30)


#plt.xlabel('Date', size=20)
ax.xaxis.label.set_visible(False)
plt.ylabel('Deaths', size=20)

# Format xticks
plt.xticks(rotation=90, size=12)

# The signature bar
offset = 35
str1 = f"Latest data: {swe_death_stats.index[-1].date()}"
str2 = "Source: Statistiska centralbyrån, www.scb.se"
signature = ax.text(x = ax.get_xlim()[0], y = ax.get_ylim()[0]-offset,
                    s = str1, fontsize = 14, color = "#f0f0f0",
                    backgroundcolor = "steelblue")
bb = signature.get_bbox_patch()
bb.set_boxstyle("ext", pad=0.6, width=ax.get_xlim()[1])

middle = (ax.get_xlim()[0] + ax.get_xlim()[1]) / 2
ax.text(x = middle, y = ax.get_ylim()[0]-offset,
                    s = "DEATHS PER DAY", fontsize = 14, color = "#f0f0f0",
                    ha="center", backgroundcolor = "steelblue")
ax.text(x = ax.get_xlim()[1], y = ax.get_ylim()[0]-offset,
                    s = str2, fontsize = 14, color = "#f0f0f0",
                    ha="right", backgroundcolor = "steelblue")

# Save as jpg and show plot
fname = str(datetime.date.today()) + "_sweden_death_compare" + ".png"
#plt.savefig(fname, format="png", dpi="figure", bbox_inches="tight", facecolor=fig.get_facecolor())
sns.despine()
plt.show()
"""
print()


scandic_deaths_df["2020-03-20":].head(20)


"""
# Plot daily new deaths
fig, ax = plt.subplots(figsize=(20,12))

#x_dates = scandic_deaths_df[from_date:].index.strftime('get_ipython().run_line_magic("d", " %b')")

# Plot per country
sns.barplot(x=scandic_deaths_df[from_date:].index, y="Increase", data=scandic_deaths_df[from_date:], palette=sns.color_palette("Set1", n_colors=num_countries, desat=.8), hue="Country")

# Set xticks location and format
x_dates = scandic_deaths_df[from_date:].index.strftime('get_ipython().run_line_magic("d", " %b').unique()")
ax.set_xticklabels(labels=x_dates, rotation=45, ha='center')
#ax2.set_xticklabels(labels=x_dates, rotation=45, ha='center')

# Set title and axis parameters
plt.title('Daily deaths of Coronavirus\nScandinavia', size=30)

ax.xaxis.label.set_visible(False)
plt.ylabel('Deaths', size=20)

# Format xticks
plt.xticks(rotation=90, size=15)

# The signature bar
str1 = f"Latest data: {scandic_df.index[-1].date()}"
str2 = "Source: CSSE at Johns Hopkins University"
print(ax.get_ylim())
print(ax.get_xlim())
signature = ax.text(x = ax.get_xlim()[0], y = ax.get_ylim()[0]-ax.get_ylim()[1]/8,
                    s = str1, fontsize = 14, color = "#f0f0f0",
                    backgroundcolor = "steelblue")
bb = signature.get_bbox_patch()
bb.set_boxstyle("ext", pad=0.6, width=ax.get_xlim()[1]*60)

middle = (ax.get_xlim()[0] + ax.get_xlim()[1]) / 2
ax.text(x = middle, y = ax.get_ylim()[0]-ax.get_ylim()[1]/8,
                    s = "DAILY DEATHS", fontsize = 14, color = "#f0f0f0",
                    ha="center", backgroundcolor = "steelblue")
ax.text(x = ax.get_xlim()[1], y = ax.get_ylim()[0]-ax.get_ylim()[1]/8,
                    s = str2, fontsize = 14, color = "#f0f0f0",
                    ha="right", backgroundcolor = "steelblue")


# Save as jpg and show plot
fname = str(datetime.date.today()) + "_scandinavia_daily_deaths" + ".png"
plt.savefig(fname, format="png", dpi="figure", bbox_inches="tight")
plt.show()
"""
print()


scandic_deaths_df["2020-10-28":].tail(15)


# Plot daily new deaths and rolling average
fig= plt.figure(figsize=(20,12), dpi=600)

tmp_date = from_date

# Create dates for x-axis
x_dates = scandic_deaths_df[from_date:].index.strftime('get_ipython().run_line_magic("d", " %b') #.unique()")

# Add subplot
ax = fig.add_subplot(111)
ax2 = fig.add_subplot(111)
#ax3 = fig.add_subplot(111)

# Set background color
fig.patch.set_facecolor(bg_col)
ax.patch.set_facecolor(bg_col)

# Plot per country
ax = sns.lineplot(x=x_dates, y="Rolling7", data=scandic_deaths_df[from_date:], sort=False, palette=sns.color_palette("Set1", n_colors=3, desat=.8), hue="Country", linestyle="--")
ax.set_xlabel('Date', fontsize=16)
ax.set_ylabel('Increase', fontsize=16)

#ax3 = sns.lineplot(x=x_dates, y="Rolling30", data=scandic_deaths_df[from_date:], sort=False, palette=sns.color_palette("Set1", n_colors=3, desat=.8), hue="Country", linestyle="-.")

ax2 = sns.barplot(x=x_dates, y="Increase", data=scandic_deaths_df[from_date:], palette=sns.color_palette("Set1", n_colors=3, desat=.8), hue="Country", linewidth=0) #edgecolor="none")

# Set the linestyle
for l in range(len(ax.lines)):
    ax.lines[l].set_linestyle("--")
    ax.lines[l].set_linewidth(l_w)
    

# Fix legend
handles, labels = ax.get_legend_handles_labels()
#print(*handles)
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
#print(x_date_ticks)

# Set xticks location and format
ax.xaxis.set_major_locator(MultipleLocator(base=5))
ax.set_xticklabels(labels=x_date_ticks[0::date_jump], rotation=90, ha='center', size=10)


# Set title and axis parameters
plt.title('Daily deaths with Coronavirus\nScandinavia', size=30)

ax.xaxis.label.set_visible(False)
plt.ylim(bottom=0)

# The signature bar
offset = 9
str1 = f"Latest data: {scandic_df.index[-1].date()}"
str2 = "Source: CSSE at Johns Hopkins University"
#print(ax.get_ylim())
#print(ax.get_xlim())
signature = ax.text(x = ax.get_xlim()[0], y = ax.get_ylim()[0]-ax.get_ylim()[1]/offset,
                    s = str1, fontsize = 14, color = "#f0f0f0",
                    backgroundcolor = "steelblue")
bb = signature.get_bbox_patch()
bb.set_boxstyle("ext", pad=0.6, width=ax.get_xlim()[1]*80)

middle = (ax.get_xlim()[0] + ax.get_xlim()[1]) / 2
ax.text(x = middle, y = ax.get_ylim()[0]-ax.get_ylim()[1]/offset,
                    s = "DAILY DEATHS", fontsize = 14, color = "#f0f0f0",
                    ha="center", backgroundcolor = "steelblue")
ax.text(x = ax.get_xlim()[1], y = ax.get_ylim()[0]-ax.get_ylim()[1]/offset,
                    s = str2, fontsize = 14, color = "#f0f0f0",
                    ha="right", backgroundcolor = "steelblue")


# Save as jpg and show plot
fname = str(datetime.date.today()) + "_scandinavia_daily_deaths" + ".png"
plt.savefig(fname, format="png", dpi="figure", bbox_inches="tight", facecolor=fig.get_facecolor())
sns.despine()
plt.show()








# Set locale to Swedish to read name of months
locale.setlocale(locale.LC_TIME, "sv_SE.UTF-8")

# Read in preliminary death data from SCB
url = "https://scb.se/hitta-statistik/statistik-efter-amne/befolkning/befolkningens-sammansattning/befolkningsstatistik/pong/tabell-och-diagram/preliminar-statistik-over-doda/"
scb_deaths = pd.read_excel(url, sheet_name="Tabell 1", parse_dates=True)
scb_death_preset_page = pd.read_excel(url, sheet_name="Info", parse_dates=True)

# Drop unused rows and columns
scb_deaths.drop(range(5), inplace=True)
scb_deaths = scb_deaths.iloc[:-1, :-6]

# Set header to years
new_header = scb_deaths.iloc[0] #grab the first row for the header

for i, x in enumerate(new_header):
    if isinstance(x, float):
        new_header[i] = str(int(x))

scb_deaths = scb_deaths[1:] #take the data less the header row
scb_deaths.columns = new_header #set the header row as the df header

# Drop leap year dates
scb_deaths = scb_deaths[scb_deaths["DagMånad"] get_ipython().getoutput("= "29 februari"]")

# Convert the dates to date time
scb_deaths["DagMånad"] = pd.to_datetime(scb_deaths["DagMånad"], format="get_ipython().run_line_magic("d", " %B\", errors=\"coerce\")")

# Fix incorrect data types
scb_deaths.loc[:, "2015":] = scb_deaths.loc[:, "2015":].astype(int)
#scb_deaths["2015-2019"] = scb_deaths["2015-2019"].astype(float)

# Set index to date column
scb_deaths.set_index("DagMånad", inplace=True)

# Drop last two unused columns
#scb_deaths.drop(["-", "Sort"], axis=1, inplace=True)


#scb_death_preset_page.iloc[1:2,-1:].head()["Senaste registrerade dödsfallet"].values[0]
scb_latest_update = scb_death_preset_page["Senaste registrerade dödsfallet"].iloc[1:2].values[0]
scb_deaths_latest_update = "1900" + scb_latest_update[4:]
scb_deaths_latest_update = pd.to_datetime(scb_deaths_latest_update)
print(scb_deaths_latest_update.date())


print(new_header)
for i, x in enumerate(new_header):
    print(f"{i}: {x}", type(x))



scb_deaths.replace(0, np.nan, inplace=True)
#df.loc[df.date.astype(str) == '2019-06-01', 'price'] = np.nan
two_weeks_ago = scb_deaths_latest_update - pd.offsets.Day(14)
scb_deaths.loc[scb_deaths.index >= two_weeks_ago, "2021"] = np.nan 
scb_deaths.loc[:"1900-03-15", :].iloc[:, 5:]


scb_deaths.info()


norm_pct = 1.03418
scb_deaths["Mean norm"] = scb_deaths["2015-2019"] * norm_pct
scb_deaths.head()


# Plot deaths in 2020 and mean deaths in Sweden per day
fig, ax = plt.subplots(figsize=(20,12))
textstr = ""

# Set color palette and background color
#color_pal = ["grey", "grey", "grey", "grey", "grey", "red"]
color_pal = ["orange", "red", "grey", "grey", "steelblue"]

fig.patch.set_facecolor(bg_col)
ax.patch.set_facecolor(bg_col)


# Plot per country
#sns.lineplot(data=swe_death_compare, palette=sns.color_palette("Set1", n_colors=6, desat=.8), alpha=.7)
sns.lineplot(data=scb_deaths.loc[:"1900-12-31", :].iloc[:, 5:], palette=sns.color_palette(color_pal, n_colors=5), alpha=.7)

# Set xticks location and format
ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
##ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=[0,1,2,3,4,5,6]))
ax.xaxis.set_major_formatter(mdates.DateFormatter('get_ipython().run_line_magic("d", " %b'));")

# Set the linewidth
for l in range(len(ax.lines)):
    ax.lines[l].set_linewidth(l_w/2)
    ax.lines[l].set_linestyle("-")
ax.lines[0].set_linewidth((l_w/2)+1)
#ax.lines[5].set_color("black")

# Fix legend
handles, labels = ax.get_legend_handles_labels()
#handles[5].set_color("black")
ax.legend(handles=handles, labels=labels)    

# Set title and axis parameters
plt.title("Daily Deaths 2015-2020\nSweden", size=30)


#plt.xlabel('Date', size=20)
ax.xaxis.label.set_visible(False)
plt.ylabel("Deaths", size=20)

# Format xticks
plt.xticks(rotation=90, size=12)

# The signature bar
offset = 40
str1 = f"Latest data: {scb_latest_update}"
str2 = "Source: Statistiska centralbyrån, www.scb.se"
signature = ax.text(x = ax.get_xlim()[0], y = ax.get_ylim()[0]-offset,
                    s = str1, fontsize = 14, color = "#f0f0f0",
                    backgroundcolor = "steelblue")
bb = signature.get_bbox_patch()
bb.set_boxstyle("ext", pad=0.6, width=ax.get_xlim()[1])

middle = (ax.get_xlim()[0] + ax.get_xlim()[1]) / 2
ax.text(x = middle, y = ax.get_ylim()[0]-offset,
                    s = "DEATHS PER DAY", fontsize = 14, color = "#f0f0f0",
                    ha="center", backgroundcolor = "steelblue")
ax.text(x = ax.get_xlim()[1], y = ax.get_ylim()[0]-offset,
                    s = str2, fontsize = 14, color = "#f0f0f0",
                    ha="right", backgroundcolor = "steelblue")

# Save as jpg and show plot
fname = str(datetime.date.today()) + "_sweden_death_compare" + ".png"
plt.savefig(fname, format="png", dpi="figure", bbox_inches="tight", facecolor=fig.get_facecolor())
sns.despine()
plt.show()


import pandas_profiling
from pivottablejs import pivot_ui
from pydqc import distribution_compare_pretty


pandas_profiling.ProfileReport(scb_deaths).to_file("report.html")


pivot_ui(scb_deaths)



