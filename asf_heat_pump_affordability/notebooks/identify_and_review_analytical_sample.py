# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Identify and review analytical sample
# Purposes of notebook:
# - Identify analytical sample to use for heat pump cost analysis across different archetypes by filtering joined MCS-EPC data according to a set of inclusion/exclusion criteria (defined below)
# - Review proportions of NaN values in dataset for key variables
# - Review proportions and distributions of data lost when removing rows based on the exclusion criteria

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

from asf_heat_pump_affordability import json_schema

# %% [markdown]
# ## Import data
#
# Here we use the most recent available EPC data joined to MCS installations data which contains data up to end of 2023 Q2. This dataset is filtered to the 'most relevant' EPC record for each property where available which aims to best reflect the status of the property at the time of the installation: this is the latest EPC record before the installation if one exists, else the earliest EPC record after the installation.

# %%
date_cols = ["commission_date", "INSPECTION_DATE"]

# %%
mcs_epc_data = pd.read_csv(
    "s3://asf-core-data/outputs/MCS/mcs_installations_epc_most_relevant_231009.csv",
    dtype=json_schema["mcs_epc_data"],
    parse_dates=date_cols,
)

# %%
filter_by_year = mcs_epc_data[mcs_epc_data["commission_year"] >= 2021]

# %%
df = filter_by_year.copy()

# %% [markdown]
# Exclusion criteria:
# - Cost NaN
# - Tech type NaN
# - Tech type != ASHP
# - No EPC record
# - Insufficient property data to assign property archetype

# %%
## Replace 'unknown' string values with NaN for review
df = df.replace("(?i)unknown", np.nan, regex=True)

# %%
## Add cols to indicate rows where key variables are NaN
## original_epc_index col indicates whether MCS record has a joined EPC record
_cols = ["cost", "tech_type", "original_epc_index"]
for _col in _cols:
    df[f"{_col}_na"] = df[_col].isna()

## Add col to indicate if the MCS installation is missing any of the property characteristic data that is required to identify
## property archetype
property_archetype_cols = ["CONSTRUCTION_AGE_BAND", "BUILT_FORM", "PROPERTY_TYPE"]
df["property_archetype_data_na"] = df[property_archetype_cols].isna().any(axis=1)

## Add col to indicate rows with non-ASHPs
df["not_ashp"] = df["tech_type"] != "Air Source Heat Pump"

# %%
## Add col to indicate rows which will be dropped based on exclusion criteria
exclusion_criteria = [
    "cost_na",
    "tech_type_na",
    "original_epc_index_na",
    "property_archetype_data_na",
    "not_ashp",
]

df["drop_row"] = df[exclusion_criteria].any(axis=1)

# %%
## Plot number of rows removed for each individual exclusion criterion

exclusion_criteria.append("drop_row")

plot_counts = {"labels": exclusion_criteria}
plot_counts["counts"] = [df[ec].sum() for ec in exclusion_criteria]
plot_df = pd.DataFrame(plot_counts).sort_values("counts")

fig, ax = plt.subplots()
bar = ax.bar(x=plot_df["labels"], height=plot_df["counts"])
ax.bar_label(bar)
ax.set_ylim([0, round(max(plot_df["counts"]) + 1000, -3)])
ax.set_title(
    "Number of rows removed for individual exclusion criteria\nand net exclusion effect"
)
plt.xticks(rotation=90)
plt.show()

# %%
## See count and proportion of NaN values for each key variable
## See count and proportion of non-NaN values lost for each key variable when dropping rows based on exclusion criteria

_cols = [
    "cost_na",
    "tech_type_na",
    "original_epc_index_na",
    "property_archetype_data_na",
]
lost_vals = {
    k: []
    for k in [
        "variable",
        "na_count",
        "proportion_na",
        "non_na_lost_count",
        "non_na_proportion_lost",
    ]
}

for _col in _cols:
    lost_count = df[(~df[_col]) & (df["drop_row"])]["drop_row"].sum()
    lost_vals["variable"].append(_col[:-3])
    lost_vals["na_count"].append(df[_col].sum())
    lost_vals["proportion_na"].append(df[_col].sum() / len(df))
    lost_vals["non_na_lost_count"].append(lost_count)
    lost_vals["non_na_proportion_lost"].append(lost_count / len(df[~df[_col]]))

pd.DataFrame(lost_vals)

# %%
print(f"Total rows dropped: {df['drop_row'].sum()}")
print(f"Total proportion of rows dropped: {df['drop_row'].sum() / len(df)}")

# %%
## Keep rows where drop_row is False, i.e. drop rows based on exclusion criteria
sample = df[~df["drop_row"]]

# %%
## Conduct checks to ensure appropriate rows dropped

for _col in ["cost", "tech_type", "original_epc_index"]:
    assert sample[_col].isna().sum() == 0

for _col in [
    "cost_na",
    "tech_type_na",
    "original_epc_index_na",
    "property_archetype_data_na",
]:
    assert sample[_col].unique() == False

assert sample["tech_type"].unique() == "Air Source Heat Pump"

# %% [markdown]
# ## Compare removed cost data and retained cost data
#
# In this particular case, I'm interested in how similar/different the dropped and retained groups are in terms of cost.
#
# I also am not interested in the water source/ground source heat pumps at all, intuitively I know that they'll inflate the mean cost and give me a distribution with a heavier tail. Really, my key concern is: do costs for ASHP look different for installs where we don't have an EPC?
#
# If costs are similar, I can be relatively happy that the sample isn't massively missing something.
#
# Descriptively the dropped rows and sample look similar - similar mean values, dropped rows have a higher std. deviation, which is interesting, sample a bit higher at each quartile.
#
# Let's see what this actually looks like graphed out. NB I'm going to truncate the cost variable at £30,000 which is roughly the 99th percentile of the distribution, as I'm not actually that interested in the tail.
#
# Ultimately, I can see there is a slight trend for sampled costs to be a bit higher. However, I can't know whether this is a 'true' difference, or a difference in composition of the sample. Ultimately, I'm relatively happy that the sample isn't that different from the dropped rows in cost terms.

# %%
## Get dropped rows
dropped_rows = filter_by_year.loc[~filter_by_year.index.isin(sample.index)]
assert (len(filter_by_year) - len(sample)) == len(dropped_rows)

## Filter for just ASHP in dropped rows
dropped_rows = dropped_rows.loc[lambda df: df["tech_type"] == "Air Source Heat Pump", :]

# %%
dropped_rows["cost"].describe()

# %%
sample["cost"].describe()

# %%
binned_data, bin_ranges = pd.cut(
    sample[sample["cost"] < 30000]["cost"], bins=20, retbins=True
)

f, ax = plt.subplots(figsize=(8, 5))

ax.hist(
    dropped_rows.loc[lambda df: df["cost"] < 30_000, "cost"],
    density=True,
    histtype="bar",
    alpha=0.5,
    bins=bin_ranges,
    color="indianred",
    label="Dropped",
)
ax.hist(
    sample.loc[lambda df: df["cost"] < 30_000, "cost"],
    density=True,
    histtype="bar",
    alpha=0.5,
    bins=bin_ranges,
    color="dodgerblue",
    label="Sampled",
)

ax.axvline(
    dropped_rows.loc[lambda df: df["cost"] < 30_000, "cost"].mean(),
    color="indianred",
    label="Dropped (mean)",
)
ax.axvline(
    sample.loc[lambda df: df["cost"] < 30_000, "cost"].mean(),
    color="dodgerblue",
    label="Sampled (mean)",
)

ax.legend()

ax.set_xlabel("Cost (Unadjusted £)")

ax.set_title(
    "Comparison of MCS Heat Pump install costs for sampled and dropped records"
)

plt.show()

# %%
