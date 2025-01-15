# %% [markdown]
# ## Counts and percentages of domestic ASHP installations across cost brackets
#
# This notebook fulfils a data request from ASF team for a breakdown of the costs of domestic installations of heat pumps, in cost brackets, with a distribution of installations (number and percentage if possible) across these cost brackets since the launch of BUS, in Wales and in the UK (since May 2022).

# %%
import pandas as pd
import numpy as np

from asf_heat_pump_affordability import config
from asf_heat_pump_affordability.getters import get_data
from asf_heat_pump_affordability.pipeline import preprocess_data, preprocess_cpi

# %% [markdown]
# ## Load MCS data

# %%
cols = ["commission_date", "postcode", "tech_type", "installation_type", "cost"]

# %%
mcs_df = pd.read_csv(
    "s3://asf-core-data/outputs/MCS/mcs_installations_241113.csv", usecols=cols
)

# %%
mcs_df.head()

# %%
# Check when last record is from
mcs_df["commission_date"].max()

# %%
# Filter to May 2022 onwards
mcs_df = mcs_df[(mcs_df["commission_date"] >= "2022-05-01")]

# %%
# Check NA in each key column. Fine to drop these records
mcs_df[["tech_type", "installation_type", "cost"]].isna().sum()

# %%
# See how installation type is coded
mcs_df["installation_type"].value_counts(dropna=False)

# %%
# Filter data to relevant records
mcs_df = mcs_df[
    (mcs_df["tech_type"] == "Air Source Heat Pump")
    & (mcs_df["installation_type"] == "Domestic")
    & (~mcs_df["cost"].isna())
]

# %% [markdown]
# ## Adjust cost data to 2024 average and add bands

# %%
# CPI data
cpi_05_3_df = get_data.get_df_from_csv_url(config["data_source"]["cpi_source_url"])

# %%
# Get adjustment factors
cpi_quarterly_df = preprocess_cpi.get_df_quarterly_cpi_with_adjustment_factors(
    ref_year=2024,
    cpi_df=cpi_05_3_df,
    cpi_col_header=config["cpi_data"]["cpi_column_header"],
)

# %%
mcs_df = preprocess_data.generate_df_adjusted_costs(
    mcs_epc_df=mcs_df, cpi_quarters_df=cpi_quarterly_df
)

# %%
# Add cost bands
conditions = [
    mcs_df["adjusted_cost"] <= 3000,
    (mcs_df["adjusted_cost"] > 3000) & (mcs_df["adjusted_cost"] <= 6000),
    (mcs_df["adjusted_cost"] > 6000) & (mcs_df["adjusted_cost"] <= 9000),
    (mcs_df["adjusted_cost"] > 9000) & (mcs_df["adjusted_cost"] <= 12000),
    (mcs_df["adjusted_cost"] > 12000) & (mcs_df["adjusted_cost"] <= 15000),
    mcs_df["adjusted_cost"] > 15000,
]

choices = [
    "£0-3000",
    "£3001-6000",
    "£6001-9000",
    "£9001-12,000",
    "£12,001-£15,0000",
    "£15,001+",
]

mcs_df["adjusted_cost_band"] = np.select(conditions, choices)

# %% [markdown]
# ## Add country column with ONSPD

# %%
onspd = pd.read_csv(
    "s3://asf-heat-pump-affordability/ONSPD_NOV_2024_UK.csv", usecols=["pcd", "ctry"]
)

# %%
country_mapping = {
    "S92000003": "Scotland",
    "E92000001": "England",
    "N92000002": "Northern Ireland",
    "W92000004": "Wales",
    "L93000001": "Channel Islands",
    "M83000003": "Isle of Man",
}

onspd["pcd"] = onspd["pcd"].str.upper().str.replace(r"\s+", "", regex=True)
onspd["country"] = onspd["ctry"].map(country_mapping)

# %%
mcs_df["postcode"] = mcs_df["postcode"].str.upper().str.replace(r"\s+", "", regex=True)
mcs_df = mcs_df.merge(
    onspd[["pcd", "country"]], how="left", left_on="postcode", right_on="pcd"
)

# %%
mcs_df["country"].value_counts(dropna=False)

# %% [markdown]
# ## Calculate results

# %%
# UK-wide results
results = mcs_df["adjusted_cost_band"].value_counts().reset_index()
results["UK_percentage_of_installations"] = (
    results["count"] / results["count"].sum() * 100
).round(2)
results = results.rename(columns={"count": "UK_ASHP_installation_count"})

# %%
# Filter to each nation
for country in ["England", "Scotland", "Wales"]:
    df = mcs_df[mcs_df["country"] == country]
    df = df["adjusted_cost_band"].value_counts().reset_index()
    df[f"{country}_percentage_of_installations"] = (
        df["count"] / df["count"].sum() * 100
    ).round(2)
    df = df.rename(columns={"count": f"{country}_ASHP_installation_count"})
    results = results.merge(df, how="left", on="adjusted_cost_band")

# %%
results["adjusted_cost_band"] = pd.Categorical(
    results["adjusted_cost_band"], categories=choices
)
results = results.sort_values(by="adjusted_cost_band")
results

# %%
results.to_csv(
    "s3://asf-heat-pump-affordability/May2022_March2024_ASHP_domestic_installation_costs_2024GBP.csv"
)

# %%
