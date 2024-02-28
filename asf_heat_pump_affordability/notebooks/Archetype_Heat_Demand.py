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

# %%
import pandas
import numpy
from matplotlib import pyplot
import statsmodels.formula.api as sm

from asf_heat_pump_affordability import config
from asf_heat_pump_affordability.pipeline import produce_costs_dataset, preprocess_data

# %%
quantiles = produce_costs_dataset.main(
    "231009",
    2023,
    2021,
    cost_quantiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    save_to_s3=True,
)

# %%
quantiles

# %%
# mcs-epc data
data = pandas.read_csv(
    "s3://asf-heat-pump-affordability/mcs_installations_epc_most_relevant_231009_preprocessed_yearRange_20212023.csv",
    index_col=0,
)

# %%
# cpi data
cpi_05_3_df = produce_costs_dataset.get_data.get_df_from_url(
    config["data_source"]["cpi_source_url"]
)
cpi_quarterly_df = (
    produce_costs_dataset.preprocess_cpi.get_df_quarterly_cpi_with_adjustment_factors(
        ref_year=2023,
        cpi_df=cpi_05_3_df,
        cpi_col_header=config["cpi_data"]["cpi_column_header"],
    )
)

# Get MCS-EPC df with adjusted costs
data = produce_costs_dataset.preprocess_data.generate_df_adjusted_costs(
    mcs_epc_df=data, cpi_quarters_df=cpi_quarterly_df
)

# %%
for (
    archetype,
    filter,
) in produce_costs_dataset.archetypes.classify_dict_archetypes_masks(data).items():
    data.loc[filter, "archetype"] = archetype

# %% [markdown]
# ### What is a representative heat demand for each of the archetypes
#
# We can use the heat demand estimates provided to MCS by heat pump installers.

# %%
(
    data.groupby("archetype", as_index=False)["heat_demand"]
    .describe()
    .sort_values(
        by="archetype",
        key=lambda x: x.map(dict(zip(quantiles.index, range(len(quantiles.index))))),
    )
    .reset_index(drop=True)
)

# %% [markdown]
# The median heat demands seem to line up reasonably well with other data I've seen with the exception of detached houses which seem quite high.
#
# As these heat demands could be influenced by the size of the properties within each categories, let's see how our results compare after some basic adjustment.

# %%
# Remove data where habitable room count is na (7268), 0 (1316), or > 12 (191). Final n=53,054
data = data.loc[lambda df: df["NUMBER_HABITABLE_ROOMS"].between(1, 12), :]

# %%
# This is a decent model that explains ~24% of the variation in heat_demand.
# Total floor area is a better predictor, but this allows us to map to the archtype for cost more easily.
model = sm.quantreg("heat_demand ~ archetype + NUMBER_HABITABLE_ROOMS", data=data).fit(
    0.5
)

# %%
pred = (
    model.get_prediction(
        pandas.DataFrame(
            {
                "archetype": [
                    label
                    for labels in [[archetype] * 13 for archetype in quantiles.index]
                    for label in labels
                ],
                "NUMBER_HABITABLE_ROOMS": list(numpy.linspace(0, 12, 13)) * 8,
            }
        )
    )
    .summary_frame()
    .assign(
        archetype=[
            label
            for labels in [[archetype] * 13 for archetype in quantiles.index]
            for label in labels
        ],
        NUMBER_HABITABLE_ROOMS=list(numpy.linspace(0, 12, 13)) * 8,
    )
)

# %%
f, ax = pyplot.subplots(figsize=(8, 6))

for archetype in quantiles.index:
    ax.plot(
        pred.loc[lambda df: df["archetype"] == archetype, "NUMBER_HABITABLE_ROOMS"],
        pred.loc[lambda df: df["archetype"] == archetype, "mean"],
        label=archetype,
    )

ax.legend()
ax.set_xlabel("Total Floor Area (m2)")
ax.set_ylabel("Estimated Annual Household Heat Demand (kWh)")

# %%
# This model is a slightly better fit, not clear if it's worth it.
model = sm.quantreg("heat_demand ~ archetype * NUMBER_HABITABLE_ROOMS", data=data).fit(
    0.5
)

# %%
pred = (
    model.get_prediction(
        pandas.DataFrame(
            {
                "archetype": [
                    label
                    for labels in [[archetype] * 13 for archetype in quantiles.index]
                    for label in labels
                ],
                "NUMBER_HABITABLE_ROOMS": list(numpy.linspace(0, 12, 13)) * 8,
            }
        )
    )
    .summary_frame()
    .assign(
        archetype=[
            label
            for labels in [[archetype] * 13 for archetype in quantiles.index]
            for label in labels
        ],
        NUMBER_HABITABLE_ROOMS=list(numpy.linspace(0, 12, 13)) * 8,
    )
)

# %%
# Hmm. really not sure about the pre-1950 flats in particular.
f, ax = pyplot.subplots(figsize=(8, 6))

for archetype in quantiles.index:
    ax.plot(
        pred.loc[lambda df: df["archetype"] == archetype, "NUMBER_HABITABLE_ROOMS"],
        pred.loc[lambda df: df["archetype"] == archetype, "mean"],
        label=archetype,
    )

ax.legend()
ax.set_xlabel("Total Floor Area (m2)")
ax.set_ylabel("Estimated Annual Household Heat Demand (kWh)")

# %%
collapse_archetypes = {
    "pre_1950_flat": "flat",
    "post_1950_flat": "flat",
    "pre_1950_semi_terraced_house": "semi_terraced_house",
    "post_1950_semi_terraced_house": "semi_terraced_house",
    "pre_1950_detached_house": "detached_house",
    "post_1950_detached_house": "detached_house",
    "pre_1950_bungalow": "bungalow",
    "post_1950_bungalow": "bungalow",
}

(
    data.assign(simple_archetype=lambda df: df["archetype"].map(collapse_archetypes))
    .groupby("simple_archetype", as_index=False)["NUMBER_HABITABLE_ROOMS"]
    .describe()
)

# %% [markdown]
# The medians look like reasonable sizes for each archetype, so we'll adjust the model by total habitabe room numbers of:
# Flat: 2 rooms (e.g. 1 bed flat)
# Semi_terraced_house: 5 rooms (e.g. 3 bedrooms, 1 reception, 1 dining)
# Detached_house: 7 rooms (e.g. 4 bedrooms, 1 dining, 2 reception)
# Bungalow: 4 rooms (e.g. 2 bedrooms, 1 reception, 1 dining)

# %%
# This is a decent model that explains ~24% of the variation in heat_demand.
model = sm.quantreg("heat_demand ~ archetype + NUMBER_HABITABLE_ROOMS", data=data).fit(
    0.5
)

# %%
pred = (
    model.get_prediction(
        pandas.DataFrame(
            {
                "archetype": list(quantiles.index),
                "NUMBER_HABITABLE_ROOMS": [2, 2, 5, 5, 7, 7, 4, 4],
            }
        )
    )
    .summary_frame()
    .assign(
        archetype=list(quantiles.index), NUMBER_HABITABLE_ROOMS=[2, 2, 5, 5, 7, 7, 4, 4]
    )
)

# %%
# This looks reasonable.
pred[["archetype", "NUMBER_HABITABLE_ROOMS", "mean"]].assign(
    mean=lambda df: df["mean"].round(-2)
).rename(columns={"mean": "heat demand"})
