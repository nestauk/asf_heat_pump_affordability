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
#     display_name: asf_heat_pump_affordability
#     language: python
#     name: asf_heat_pump_affordability
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

# %% [markdown]
# ## Import data
#
# Here we use the most recent available EPC data joined to MCS installations data which contains data up to end of 2023 Q2. This dataset is filtered to the 'most relevant' EPC record for each property where available which aims to best reflect the status of the property at the time of the installation: this is the latest EPC record before the installation if one exists, else the earliest EPC record after the installation.

# %%
mcs_epc_data = pd.read_csv(
    "s3://asf-core-data/outputs/MCS/mcs_installations_epc_most_relevant_231009.csv"
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
df["property_archetype_data_na"] = df[property_archetype_cols].isna().apply(any, axis=1)

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
df["drop_row"] = df[exclusion_criteria].apply(any, axis=1)

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
    lost_count = df[(df[_col] == False) & (df["drop_row"] == True)]["drop_row"].sum()
    lost_vals["variable"].append(_col[:-3])
    lost_vals["na_count"].append(df[_col].sum())
    lost_vals["proportion_na"].append(df[_col].sum() / len(df))
    lost_vals["non_na_lost_count"].append(lost_count)
    lost_vals["non_na_proportion_lost"].append(lost_count / len(df[df[_col] == False]))

pd.DataFrame(lost_vals)

# %%
print(f"Total rows dropped: {(df['drop_row'] == True).sum()}")
print(f"Total proportion of rows dropped: {(df['drop_row'] == True).sum() / len(df)}")

# %%
## Keep rows where drop_row is False, i.e. remove rows based on exclusion criteria
sample = df[df["drop_row"] == False]

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
# ## See distribution of removed cost data

# %% [markdown]
# Removed rows either have no EPC data or are not an ASHP.
# We will see if the distribution of costs in removed rows is the same as that of rows which have been kept in the sample for analysis.

# %%
dropped_rows = filter_by_year.loc[~filter_by_year.index.isin(sample.index)]
assert (len(filter_by_year) - len(sample)) == len(dropped_rows)

# %%
fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(10, 3))
ax[0].hist(sample["cost"])
ax[0].set_ylabel("Count")
ax[1].hist(dropped_rows["cost"])
for axis, title in zip(ax, ["kept", "dropped"]):
    axis.set_title(f"Distribution of installation cost in {title} rows")
    axis.set(xlabel="Installation cost, £")
plt.show()

# %%
fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(10, 3))
ax[0].hist(sample["cost"], log=True)
ax[0].set_ylabel("Count (log scale)")
ax[1].hist(dropped_rows["cost"], log=True)
for axis, title in zip(ax, ["kept", "dropped"]):
    axis.set_title(f"Distribution of installation cost in {title} rows")
    axis.set(xlabel="Installation cost, £")
plt.show()

# %%
