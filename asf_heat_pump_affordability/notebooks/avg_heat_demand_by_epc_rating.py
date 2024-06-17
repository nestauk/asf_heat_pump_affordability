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

# %%
import polars as pl
import polars.selectors as cs
from datetime import datetime
import statsmodels.formula.api as sm
import s3fs

# %%
mcs_epc_path = (
    "s3://asf-core-data/outputs/MCS/mcs_installations_epc_most_relevant_231009.csv"
)

# %%
data = pl.read_csv(
    mcs_epc_path,
    columns=[
        "commission_date",
        "commission_year",
        "heat_demand",
        "installation_type",
        "INSPECTION_DATE",
        "CURRENT_ENERGY_RATING",
    ],
)

# %% [markdown]
# ## Prepare data for estimates
#
# We require current energy rating and heat demand data. We start by dropping rows missing this data.
#
# We are also only interested in domestic installations.

# %%
print(len(data))

# %%
# Replace 'unknown' current energy rating with Null so they can be filtered out
# Replace nulls in installation_type col with 'unknown' so they can be kept as they could be domestic
df = data.with_columns(
    [
        pl.col("CURRENT_ENERGY_RATING").replace("unknown", None),
        pl.col("heat_demand").cast(pl.String).replace("unknown", None).cast(pl.Float64),
        pl.col("installation_type").fill_null("unknown"),
        pl.col("INSPECTION_DATE").str.to_datetime("%Y-%m-%d"),
    ]
)

# %%
min_date = datetime.strptime("2013-01-01", "%Y-%m-%d")

# %%
# Keep only rows with EPC rating, heat demand data, and where installation is not Non-domestic
df = df.filter(
    pl.col("CURRENT_ENERGY_RATING").is_not_null(),
    pl.col("heat_demand").is_not_null(),
    ~pl.col("installation_type").is_in(["Non-Domestic"]),
    pl.col("INSPECTION_DATE") >= min_date,
)

# Check what % of rows remain
len(df) / len(data) * 100

# %% [markdown]
# ## Get avg heat demand by EPC rating

# %%
# Get median and mean heat demand for each EPC rating
hd_estimates = (
    df.group_by("CURRENT_ENERGY_RATING")
    .agg(
        [
            pl.col("heat_demand").median().name.suffix("_median"),
            pl.col("heat_demand").mean().name.suffix("_mean"),
        ]
    )
    .sort("CURRENT_ENERGY_RATING")
)

# %%
hd_estimates = hd_estimates.with_columns(cs.numeric().round_sig_figs(4))

# %%
hd_estimates

# %%
save_as = "s3://asf-heat-pump-affordability/avg_heat_demand_by_epc_rating.csv"
fs = s3fs.S3FileSystem()
with fs.open(save_as, mode="w") as f:
    hd_estimates.write_csv(f)

# %%
