import numpy as np
import pandas as pd
import datetime
import locale
import pickle
import os


# Population per country 2018 (data from https://www.nordicstatistics.org/population/)
swe_pop = 10120242
nor_pop = 5295619
den_pop = 5781190


# Read the data from Johns Hopkins
def read_corona_data():
    conf_url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
    confirmed_df = pd.read_csv(conf_url)

    death_url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv"
    deaths_df = pd.read_csv(death_url)

    # Extract the Scandinavian countries and combine to a dataframe
    scandic = ["Sweden", "Norway", "Denmark"]
    scandic_conf_df = confirmed_df[confirmed_df["Country/Region"].isin(scandic)]
    scandic_deaths_df = deaths_df[deaths_df["Country/Region"].isin(scandic)]

    # Drop not used columns
    scandic_conf_df = (
        scandic_conf_df.drop(["Province/State", "Lat", "Long"], axis=1)
        .groupby("Country/Region")
        .sum()
    )
    scandic_deaths_df = (
        scandic_deaths_df.drop(["Province/State", "Lat", "Long"], axis=1)
        .groupby("Country/Region")
        .sum()
    )

    return scandic_conf_df, scandic_deaths_df


def create_long_dfs(scandic_conf_df, scandic_deaths_df):
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

    for i in dates:
        sweden_sum = scandic_conf_df[i]["Sweden"]
        norway_sum = scandic_conf_df[i]["Norway"]
        denmark_sum = scandic_conf_df[i]["Denmark"]
        # death_sum = deaths[i].sum()
        sweden_death_sum = scandic_deaths_df[i]["Sweden"]
        norway_death_sum = scandic_deaths_df[i]["Norway"]
        denmark_death_sum = scandic_deaths_df[i]["Denmark"]

        sweden_cases.append(sweden_sum)
        norway_cases.append(norway_sum)
        denmark_cases.append(denmark_sum)

        # total_deaths.append(death_sum)
        sweden_deaths.append(sweden_death_sum)
        norway_deaths.append(norway_death_sum)
        denmark_deaths.append(denmark_death_sum)

        mortality_rate_swe.append(sweden_death_sum / sweden_sum)
        mortality_rate_nor.append(norway_death_sum / norway_sum)
        mortality_rate_den.append(denmark_death_sum / denmark_sum)

    # Calculate mortality rate
    mortality_rate = pd.DataFrame(
        {
            "Date": scandic_deaths_df.keys(),
            "Sweden": mortality_rate_swe,
            "Norway": mortality_rate_nor,
            "Denmark": mortality_rate_den,
        }
    )
    mortality_rate["Date"] = pd.to_datetime(mortality_rate["Date"])
    mortality_rate.set_index("Date", inplace=True)

    # Make dataframe that starts from a value of confirmed cases

    # above5 S: 36, N: 37, D: 41
    # above100 S: 44, N: 45, D: 48

    s5 = 36
    # s100 = 44
    n5 = 37
    # n100 = 44
    d5 = 41
    # d100 = 48

    scandi_outbreak = pd.DataFrame(
        {
            "Sweden": sweden_cases[s5:],
            "Norway": norway_cases[n5:]
            + (len(sweden_cases[s5:]) - len(norway_cases[n5:])) * [None],
            "Denmark": denmark_cases[d5:]
            + (len(sweden_cases[s5:]) - len(denmark_cases[d5:])) * [None],
        }
    )

    # Rearrange the confirmed dataframe to long format with date as index
    d = {
        "Date": scandic_conf_df.keys(),
        "Sweden": sweden_cases,
        "Norway": norway_cases,
        "Denmark": denmark_cases,
    }
    scandic_df = pd.DataFrame(data=d)
    scandic_df = scandic_df.melt(
        id_vars=["Date"], var_name="Country", value_name="Confirmed"
    )
    scandic_df["Date"] = pd.to_datetime(scandic_df["Date"], format="%m/%d/%y")
    scandic_df.set_index("Date", inplace=True)

    # Rearrange the deaths dataframe to long format with date as index
    d = {
        "Date": scandic_deaths_df.keys(),
        "Sweden": sweden_deaths,
        "Norway": norway_deaths,
        "Denmark": denmark_deaths,
    }
    scandic_deaths_df = pd.DataFrame(data=d)
    scandic_deaths_df = scandic_deaths_df.melt(
        id_vars=["Date"], var_name="Country", value_name="Deaths"
    )
    scandic_deaths_df["Date"] = pd.to_datetime(scandic_deaths_df["Date"])
    scandic_deaths_df.set_index("Date", inplace=True)

    # Calculate deaths per capita (per mille)
    country_normalize = {"Sweden": swe_pop, "Norway": nor_pop, "Denmark": den_pop}
    scandic_deaths_df["Pop"] = [
        country_normalize[x] for x in scandic_deaths_df["Country"]
    ]
    scandic_deaths_df["Deaths per capita"] = (
        scandic_deaths_df["Deaths"] / scandic_deaths_df["Pop"]
    ) * 1000
    scandic_deaths_df["Deaths per 100k"] = (
        scandic_deaths_df["Deaths"] / scandic_deaths_df["Pop"]
    ) * 100000
    scandic_deaths_df.drop("Pop", axis=1, inplace=True)

    # Calculate the daily increase in cases and deaths and add them to the dataframes
    diff_list = []
    diff_death = []
    rolling_7 = []
    rolling_30 = []
    for c in scandic_df["Country"].unique():
        diff_list.extend(
            scandic_df[scandic_df["Country"] == c]["Confirmed"].diff().values
        )
        diff_death.extend(
            scandic_deaths_df[scandic_deaths_df["Country"] == c]["Deaths"].diff().values
        )

    len(diff_list)
    scandic_df["Increase"] = diff_list
    scandic_deaths_df["Increase"] = diff_death

    for c in scandic_deaths_df["Country"].unique():
        rolling_7.extend(
            scandic_deaths_df[scandic_deaths_df["Country"] == c]["Increase"]
            .rolling(7)
            .mean()
        )
        rolling_30.extend(
            scandic_deaths_df[scandic_deaths_df["Country"] == c]["Increase"]
            .rolling(30)
            .mean()
        )

    scandic_deaths_df["Rolling7"] = rolling_7
    scandic_deaths_df["Rolling30"] = rolling_30

    # Calculate 7 day rolling average for deaths
    mean_list = []
    for c in scandic_df["Country"].unique():
        mean_list.extend(
            scandic_df[scandic_df["Country"] == c]["Confirmed"].rolling(7).mean()
        )

    scandic_df["Mean7"] = mean_list

    # Calculate number of countries in dataframe
    num_countries = len(scandic_df["Country"].unique())

    # Calculate logaritmic deaths
    scandic_deaths_df["logDeaths"] = scandic_deaths_df["Deaths"].apply(
        lambda x: np.log(x)
    )
    scandic_deaths_df.tail()

    return (
        scandic_df,
        scandic_deaths_df,
        scandi_outbreak,
        mortality_rate,
    )


def get_swe_death_stats():
    # Read and calculate mean for swedish death stats
    swe_death_stats = pd.read_csv(
        "../sweden_death_stats10.csv",
        sep=";",
        parse_dates=["Datum"],
        index_col=["Datum"],
    )
    swe_death_stats["Sum 2020"] = swe_death_stats["Deaths 2020"].cumsum()
    swe_death_stats["Mean Sum"] = swe_death_stats["Mean Deaths 2015-2019"].cumsum()
    swe_death_stats.drop(["Deaths 2020", "Mean Deaths 2015-2019"], axis=1, inplace=True)

    # Read swedish deaths per year
    swe_death_compare = pd.read_csv(
        "../swe_death_compare.csv", sep=";", parse_dates=["Datum"], index_col=["Datum"]
    )

    swe_death_compare["Mean 2015 2019"] = swe_death_compare.loc[:, "2015":"2019"].mean(
        axis=1
    )

    return swe_death_stats


def get_scb_deaths():
    # Set locale to Swedish to read name of months
    locale.setlocale(locale.LC_TIME, "sv_SE.UTF-8")

    # Read in preliminary death data from SCB
    url = "https://scb.se/hitta-statistik/statistik-efter-amne/befolkning/befolkningens-sammansattning/befolkningsstatistik/pong/tabell-och-diagram/preliminar-statistik-over-doda/"
    scb_deaths = pd.read_excel(url, sheet_name="Tabell 1", parse_dates=True)

    # Drop unused rows and columns
    scb_deaths.drop(range(5), inplace=True)
    scb_deaths = scb_deaths.iloc[:-1, :-6]

    # Set header to years
    new_header = scb_deaths.iloc[0].copy()  # grab the first row for the header

    for i, x in enumerate(new_header):
        if isinstance(x, float):
            new_header[i] = str(int(x))

    scb_deaths = scb_deaths[1:]  # take the data less the header row
    scb_deaths.columns = new_header  # set the header row as the df header

    # Drop leap year dates
    scb_deaths = scb_deaths[scb_deaths["DagMånad"] != "29 februari"]

    # Convert the dates to date time
    scb_deaths["DagMånad"] = pd.to_datetime(
        scb_deaths["DagMånad"], format="%d %B", errors="coerce"
    )

    # Fix incorrect data types
    scb_deaths.loc[:, "2015":] = scb_deaths.loc[:, "2015":].astype(int)
    # scb_deaths["2015-2019"] = scb_deaths["2015-2019"].astype(float)

    # Set index to date column
    scb_deaths.set_index("DagMånad", inplace=True)

    # Drop last two unused columns
    # scb_deaths.drop(["-", "Sort"], axis=1, inplace=True)

    norm_pct = 1.03418
    scb_deaths["Mean norm"] = scb_deaths["2015-2019"] * norm_pct

    return scb_deaths


def latest_data():
    pass


def load_data():
    cases, deaths = read_corona_data()

    pickle.dump(cases, open("cases.pkl", "wb"))
    pickle.dump(cases, open("deaths.pkl", "wb"))

    # cases = pickle.load(open("cases.pkl", "rb"))
    # deaths = pickle.load(open("deaths.pkl", "rb"))

    return create_long_dfs(cases, deaths)


# load_data()
# get_scb_deaths()
