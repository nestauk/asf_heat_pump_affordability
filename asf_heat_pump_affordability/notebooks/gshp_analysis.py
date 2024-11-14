# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: asf_heat_pump_affordability
#     language: python
#     name: python3
# ---

# %%
from asf_heat_pump_affordability import json_schema
from asf_heat_pump_affordability.getters import get_data
from asf_heat_pump_affordability import config
from asf_heat_pump_affordability.pipeline.preprocess_data import (
    join_df_supplementary_variables,
    generate_df_adjusted_costs,
)
from asf_heat_pump_affordability.pipeline import preprocess_cpi
import pandas
from matplotlib import pyplot
from scipy.stats import gaussian_kde
import numpy

# %% [markdown]
# ## Ground Source Heat Pump Analysis
# #### Prepare Data

# %%
# config
cost_year_min = 2021
cost_year_max = None

# %%
mcs_epc_data = pandas.read_csv(
    "s3://asf-core-data/outputs/MCS/mcs_installations_epc_most_relevant_240717.csv",
    dtype=json_schema["mcs_epc_data"],
    parse_dates=["commission_date", "INSPECTION_DATE"],
)

# %%
# Initial exclusion criteria
if cost_year_min:
    mcs_epc_data = mcs_epc_data[mcs_epc_data["commission_year"] >= cost_year_min]
if cost_year_max:
    mcs_epc_data = mcs_epc_data[mcs_epc_data["commission_year"] <= cost_year_max]

mcs_epc_data = mcs_epc_data.replace("(?i)unknown", pandas.NA, regex=True)

key_variables = [
    "cost",
    "tech_type",
    "original_epc_index",
    # "CONSTRUCTION_AGE_BAND",
    # "BUILT_FORM",
    "PROPERTY_TYPE",
]
mcs_epc_data = mcs_epc_data.dropna(subset=key_variables, how="any")
mcs_epc_data = mcs_epc_data[
    mcs_epc_data["tech_type"] == "Ground/Water Source Heat Pump"
]
mcs_epc_data["postcode"] = mcs_epc_data["postcode"].str.upper().replace(" ", "")

# %%
# 16,825
mcs_epc_data.shape

# %%
# Join some interesting extra variables
mcs_epc_data = join_df_supplementary_variables(mcs_epc_data)

# %%
# Import and process CPI data
cpi_05_3_df = get_data.get_df_from_csv_url(config["data_source"]["cpi_source_url"])
cpi_quarterly_df = preprocess_cpi.get_df_quarterly_cpi_with_adjustment_factors(
    ref_year=2023,
    cpi_df=cpi_05_3_df,
    cpi_col_header=config["cpi_data"]["cpi_column_header"],
)

# Get MCS-EPC df with adjusted costs
mcs_epc_data = generate_df_adjusted_costs(
    mcs_epc_df=mcs_epc_data, cpi_quarters_df=cpi_quarterly_df
)


# %% [markdown]
# #### Explore Building Types
#
# **Flats and maisonettes** - where a ground source heat pump has been installed into a flat, it is likely to be a shared ground loop. This is reflected in the ~82% of flats or maisonettes that use a Kensa heat pump (although some of these will likely have private boreholes or slinkies). Most installations are made on a commercial or non-domestic basis, which suggests they were not contracted by individual homeowners, but by housing assocations, developers or other organisations.
#
# NB Because we're only looking at records that have been linked to Domestic EPCs, we can make the assumption that we're considering domestic properties already.

# %%
mcs_epc_data.loc[mcs_epc_data["PROPERTY_TYPE"].isin(["Flat", "Maisonette"])].shape

# %%
# ~89% of flats/maisonettes are Kensa heat pumps
# These are likely to be shared ground loop schemes.
(
    mcs_epc_data.loc[mcs_epc_data["PROPERTY_TYPE"].isin(["Flat", "Maisonette"])]
    .groupby("manufacturer")
    .size()
    .sort_values(ascending=False)
    / mcs_epc_data.loc[
        mcs_epc_data["PROPERTY_TYPE"].isin(["Flat", "Maisonette"])
    ].shape[0]
    * 100
)

# %%
(
    mcs_epc_data.loc[mcs_epc_data["PROPERTY_TYPE"].isin(["Flat", "Maisonette"])]
    .groupby("manufacturer")["installation_type"]
    .value_counts()
)

# %% [markdown]
# **Houses and bungalows**
# We're mostly interested in how installations vary by size. As a proxy for size we're using the number of habitable rooms.
#
# Habitable rooms include any living room, sitting room, dining room, bedroom, study and similar; and also a non-separated conservatory. A kitchen/diner having a discrete seating area (with space for a table and four chairs) also counts as a habitable room. A non-separated conservatory adds to the habitable room count if it has an internal quality door between it and the dwelling. Excluded from the room count are any room used solely as a kitchen, utility room, bathroom, cloakroom, en-suite accommodation and similar and any hallway, stairs or landing; and also any room not having a window.
#
# There are quite a lot of EPC records missing number of habitable rooms, this is slightly higher as a proportion of all records in Wales and Scotland.
#
# As such, we'll look at habitable rooms, then attempt to do something similar with floor sizes.

# %%
if cost_year_min == 2021:
    # there is a single non-domestic observation for houses/bungalows installed 2021+, so we'll drop it for convenience.
    idx = mcs_epc_data.loc[
        lambda df: df["PROPERTY_TYPE"].isin(["House", "Bungalow"])
        & (df["installation_type"] == "Non-Domestic")
    ].index
    mcs_epc_data = mcs_epc_data.drop(index=idx)

# %%
mcs_epc_data.loc[lambda df: df["PROPERTY_TYPE"].isin(["House", "Bungalow"])].shape

# %%
# Country variable
mcs_epc_data["country"] = (
    mcs_epc_data["lsoa11"].str[0].map({"E": "England", "W": "Wales", "S": "Scotland"})
)
mcs_epc_data.loc[
    mcs_epc_data["PROPERTY_TYPE"].isin(["House", "Bungalow"])
    & (mcs_epc_data["NUMBER_HABITABLE_ROOMS"] > 0)
]["country"].value_counts()

# %%
mcs_epc_data.loc[
    mcs_epc_data["PROPERTY_TYPE"].isin(["House", "Bungalow"])
    & (mcs_epc_data["NUMBER_HABITABLE_ROOMS"] > 0)
]["country"].value_counts() / mcs_epc_data.loc[
    mcs_epc_data["PROPERTY_TYPE"].isin(["House", "Bungalow"]), "country"
].value_counts()


# %%
# Create property size categories based on habitable rooms.
def get_property_size(x):
    if isinstance(x, pandas._libs.missing.NAType):
        return pandas.NA
    elif x == 0:
        return pandas.NA
    elif (x >= 1) & (x <= 3):
        return "small"
    elif (x >= 4) & (x <= 6):
        return "medium"
    elif (x >= 7) & (x <= 9):
        return "large"
    else:
        return "very large"


# Habitable rooms needs a sensible classification
mcs_epc_data["property_size"] = mcs_epc_data["NUMBER_HABITABLE_ROOMS"].apply(
    lambda x: get_property_size(x)
)

# %%
mcs_epc_data.loc[
    mcs_epc_data["PROPERTY_TYPE"].isin(["House", "Bungalow"]), "property_size"
].value_counts(dropna=False) / mcs_epc_data["PROPERTY_TYPE"].isin(
    ["House", "Bungalow"]
).sum()

# %%
# Looks like NA on property size is likely to be new builds (2007+).
test = mcs_epc_data.loc[mcs_epc_data["PROPERTY_TYPE"].isin(["House", "Bungalow"])].copy(
    deep=True
)
test["property_size"] = test["property_size"].fillna("NA")
pandas.crosstab(test["property_size"], test["CONSTRUCTION_AGE_BAND"])

# %%
test[test["property_size"] == "NA"]["installation_type"].value_counts()

# %%
test["installation_type"].value_counts()

# %%
# These records are thus likely to represent retrofits or EPC subsequent to the new build issuance.
f, ax = pyplot.subplots(figsize=(8, 6))

colours = {
    "small": "#e41a1c",
    "medium": "#377eb8",
    "large": "#4daf4a",
    "very large": "#984ea3",
}
mode_50hdi = []

for property_size in [
    "small",
    "medium",
    "large",
    "very large",
]:
    temp = mcs_epc_data.loc[
        mcs_epc_data["PROPERTY_TYPE"].isin(["House", "Bungalow"])
        & (mcs_epc_data["property_size"] == property_size)
    ]
    z = gaussian_kde(temp["adjusted_cost"]).evaluate(list(range(0, 100_000, 1)))
    ax.plot(
        list(range(0, 100_000, 1)), z, color=colours[property_size], label=property_size
    )
    # ax.axvline(x=z.argmax(), label=f"{property_size} mode")
    # print(f"{property_size} mode: {z.argmax()}")

    # Fill under the curve with 50% HDI
    idx = numpy.argsort(z)[::-1]
    mass_cum = 0
    indices = []
    for i in idx:
        mass_cum += z[i]
        indices.append(i)
        if mass_cum >= 0.5:
            break
    lower, upper = numpy.sort(indices)[[0, -1]]

    x = list(range(0, 100_000, 1))
    ax.fill_between(
        x=x,
        y1=z,
        where=(lower < x) & (x < upper),
        color=colours[property_size],
        alpha=0.2,
    )

    mode_50hdi.append((property_size, z.argmax(), lower, upper))

ax.legend(title="Property Size")

ax.set_xlabel("Installation Cost (2023 £)")
ax.set_ylabel("Density")
ax.set_xticks(list(range(0, 110_000, 10_000)))

pyplot.savefig("./property_size_cost_plot.png", dpi=300, bbox_inches="tight")

# %%
summary_df = pandas.DataFrame(
    mode_50hdi, columns=["property size", "mode", "lower_50_hdi", "upper_50_hdi"]
)

# %%
summary_df.round(-2)

# %% [markdown]
# Many of the records that are NA on habitable rooms have commercial HP installs. My guess is that new build SAP doesn't enumerate habitable rooms because it's not always certain how a room is going to be used. Instead, we'll try to include these data using the floor area.

# %%
# Cut points are: 95, 183, 299
f, ax = pyplot.subplots(figsize=(8, 6))

zs = []
for property_size in [
    "small",
    "medium",
    "large",
    "very large",
]:
    temp = mcs_epc_data.loc[
        mcs_epc_data["PROPERTY_TYPE"].isin(["House", "Bungalow"])
        & (mcs_epc_data["property_size"] == property_size)
    ].copy(deep=True)
    z = gaussian_kde(temp["TOTAL_FLOOR_AREA"]).evaluate(list(range(0, 750, 1)))
    zs.append(z)
    ax.plot(list(range(0, 750, 1)), z, label=property_size)
    print(f"{property_size} mode: {z.argmax()}")

# add cut points
old_x = 0
for z in range(1, len(zs)):
    xs = numpy.argwhere(numpy.diff(numpy.sign(zs[z - 1] - zs[z]))).flatten()
    for x in xs:
        if x > old_x:
            ax.axvline(x, color="0.25", linestyle="dashed")
            old_x = x
            print(x)
            break

ax.legend(title="property size")

ax.set_xlabel("Total Floor Area, Square Metres")
ax.set_ylabel("Density")

pyplot.savefig("./floor_area_cuts.png", dpi=300, bbox_inches="tight")

# %%
mcs_epc_data["floor_size_cat"] = pandas.cut(
    x=mcs_epc_data["TOTAL_FLOOR_AREA"],
    bins=[0, 95, 183, 297, 10_000],
    labels=["small", "medium", "large", "very large"],
)

# %%
mcs_epc_data.loc[
    mcs_epc_data["PROPERTY_TYPE"].isin(["House", "Bungalow"]), "floor_size_cat"
].value_counts() / mcs_epc_data["PROPERTY_TYPE"].isin(["House", "Bungalow"]).sum()

# %%
f, ax = pyplot.subplots(figsize=(8, 6))

colours = {
    "small": "#e41a1c",
    "medium": "#377eb8",
    "large": "#4daf4a",
    "very large": "#984ea3",
}
mode_50hdi = []

for property_size in [
    "small",
    "medium",
    "large",
    "very large",
]:
    temp = mcs_epc_data.loc[
        mcs_epc_data["PROPERTY_TYPE"].isin(["House", "Bungalow"])
        & (mcs_epc_data["installation_type"] == "Domestic")
        & (mcs_epc_data["floor_size_cat"] == property_size)
    ].copy(deep=True)
    z = gaussian_kde(temp["adjusted_cost"]).evaluate(list(range(0, 100_000, 1)))
    ax.plot(
        list(range(0, 100_000, 1)), z, color=colours[property_size], label=property_size
    )
    # ax.axvline(x=z.argmax(), label=f"{property_size} mode")
    # print(f"{property_size} mode: {z.argmax()}")

    # Fill under the curve with 50% HDI
    idx = numpy.argsort(z)[::-1]
    mass_cum = 0
    indices = []
    for i in idx:
        mass_cum += z[i]
        indices.append(i)
        if mass_cum >= 0.5:
            break
    lower, upper = numpy.sort(indices)[[0, -1]]

    x = list(range(0, 100_000, 1))
    ax.fill_between(
        x=x,
        y1=z,
        where=(lower < x) & (x < upper),
        color=colours[property_size],
        alpha=0.2,
    )

    mode_50hdi.append((property_size, z.argmax(), lower, upper, temp.shape[0]))

ax.legend(title="Property Size")

ax.set_xlabel("Installation Cost (2023 £)")
ax.set_ylabel("Density")
ax.set_xticks(list(range(0, 110_000, 10_000)))
pyplot.savefig("./floor_area_cost_gshp.png", dpi=300, bbox_inches="tight")

# %%
pandas.DataFrame(
    mode_50hdi, columns=["size", "mode", "lower", "upper", "n"]
)  # .round(-2)

# %%
f, ax = pyplot.subplots(figsize=(8, 6))

colours = {
    "small": "#e41a1c",
    "medium": "#377eb8",
    "large": "#4daf4a",
    "very large": "#984ea3",
}
mode_50hdi = []

for property_size in ["small", "medium", "large", "very large"]:
    temp = mcs_epc_data.loc[
        mcs_epc_data["PROPERTY_TYPE"].isin(["House", "Bungalow"])
        & (mcs_epc_data["installation_type"] == "Commercial")
        & (mcs_epc_data["floor_size_cat"] == property_size)
    ].copy(deep=True)
    z = gaussian_kde(temp["adjusted_cost"]).evaluate(list(range(0, 100_000, 1)))
    ax.plot(
        list(range(0, 100_000, 1)), z, color=colours[property_size], label=property_size
    )
    # ax.axvline(x=z.argmax(), label=f"{property_size} mode")
    # print(f"{property_size} mode: {z.argmax()}")

    # Fill under the curve with 50% HDI
    idx = numpy.argsort(z)[::-1]
    mass_cum = 0
    indices = []
    for i in idx:
        mass_cum += z[i]
        indices.append(i)
        if mass_cum >= 0.5:
            break
    lower, upper = numpy.sort(indices)[[0, -1]]

    x = list(range(0, 100_000, 1))
    ax.fill_between(
        x=x,
        y1=z,
        where=(lower < x) & (x < upper),
        color=colours[property_size],
        alpha=0.2,
    )

    mode_50hdi.append((property_size, z.argmax(), lower, upper, temp.shape[0]))

ax.legend(title="Property Size")

ax.set_xlabel("Installation Cost (2023 £)")
ax.set_ylabel("Density")
ax.set_xticks(list(range(0, 110_000, 10_000)))

pyplot.savefig("./floor_area_cost_gshp_commercial.png", dpi=300, bbox_inches="tight")

# %%
pandas.DataFrame(mode_50hdi, columns=["size", "mode", "lower", "upper", "n"]).round(-2)
