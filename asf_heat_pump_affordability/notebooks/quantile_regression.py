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
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Use quantile regression to estimate cost deciles
#
# Explore using quantile regression models to estimate cost of installation of ASHP at each cost decile for each of the 8 property archetypes.

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
    save_to_s3=False,
)

# %%
quantiles

# %%
# quantiles.to_excel("./20240213_mcs_epc_archetype_heat_pump_cost_deciles.xlsx")

# %%
f, ax = pyplot.subplots(figsize=(8, 6))

for idx, row in quantiles.iterrows():
    ax.plot([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], row.values, label=idx)

ax.set_ylim([0, 22_500])
ax.legend()

# %%
# mcs-epc data
data = pandas.read_csv(
    "s3://asf-heat-pump-affordability/mcs_installations_epc_most_relevant_231009_preprocessed_yearRange_20212023.csv",
    index_col=0,
)

# %%
# cpi data
cpi_05_3_df = produce_costs_dataset.get_data.get_df_from_csv_url(
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
# Classify age to pre/post 1950
data["AGE_BAND_2CAT"] = data["CONSTRUCTION_AGE_BAND"].map(
    {
        "England and Wales: before 1900": "Pre-1950",
        "Scotland: before 1919": "Pre-1950",
        "1900-1929": "Pre-1950",
        "1930-1949": "Pre-1950",
        "1950-1966": "Post-1950",
        "1965-1975": "Post-1950",
        "1976-1983": "Post-1950",
        "1983-1991": "Post-1950",
        "1991-1998": "Post-1950",
        "1996-2002": "Post-1950",
        "2003-2007": "Post-1950",
        "2007 onwards": "Post-1950",
    }
)

# %% [markdown]
# ## Quantile Regression
#
# ### Model 1: basic model
# Binary built age variable only.
#
# NB mean_ci is the confidence interval of the model e.g. based on the standard errors of the model coefficients. obs_ci is the confidence interval of the model plus the residuals, accounting for the sum of the variance in the predicted means and the residual variance.

# %%
res = []
for quantile in numpy.linspace(0.1, 0.9, 9):
    model = sm.quantreg("adjusted_cost ~ AGE_BAND_2CAT", data).fit(quantile)
    res.append(
        model.get_prediction(
            pandas.Series(["Pre-1950", "Post-1950"], name="AGE_BAND_2CAT")
        )
        .summary_frame()
        .assign(AGE_BAND_2CAT=["Pre-1950", "Post-1950"], QUANTILE=quantile)
    )
model1 = pandas.concat(res)

# %%
f, ax = pyplot.subplots(figsize=(8, 6))

for archetype in ["Pre-1950", "Post-1950"]:
    ax.plot(
        numpy.linspace(0.1, 0.9, 9),
        model1.loc[lambda df: df["AGE_BAND_2CAT"] == archetype, "mean"],
        label=archetype,
    )
    ax.fill_between(
        numpy.linspace(0.1, 0.9, 9),
        y1=model1.loc[lambda df: df["AGE_BAND_2CAT"] == archetype, "obs_ci_lower"],
        y2=model1.loc[lambda df: df["AGE_BAND_2CAT"] == archetype, "obs_ci_upper"],
        alpha=0.33,
    )

ax.set_ylim([0, 20_000])
ax.set_xlabel("Quantile")
ax.set_ylabel("Cost (£2023)")
ax.legend()

# %% [markdown]
# ### Model 2: Archetypes Model

# %%
for (
    archetype,
    filter,
) in produce_costs_dataset.archetypes.classify_dict_archetypes_masks(data).items():
    data.loc[filter, "archetype"] = archetype

# %%
res = []
for quantile in numpy.linspace(0.1, 0.9, 9):
    model = sm.quantreg("adjusted_cost ~ archetype", data).fit(quantile)
    res.append(
        model.get_prediction(
            pandas.Series(
                [
                    "pre_1950_flat",
                    "post_1950_flat",
                    "pre_1950_bungalow",
                    "post_1950_bungalow",
                    "pre_1950_semi_terraced_house",
                    "post_1950_semi_terraced_house",
                    "pre_1950_detached_house",
                    "post_1950_detached_house",
                ],
                name="archetype",
            )
        )
        .summary_frame()
        .assign(
            archetype=[
                "pre_1950_flat",
                "post_1950_flat",
                "pre_1950_bungalow",
                "post_1950_bungalow",
                "pre_1950_semi_terraced_house",
                "post_1950_semi_terraced_house",
                "pre_1950_detached_house",
                "post_1950_detached_house",
            ],
            QUANTILE=quantile,
        )
    )
model2 = pandas.concat(res)

# %%
f, ax = pyplot.subplots(figsize=(8, 6))

for archetype in [
    "pre_1950_flat",
    "post_1950_flat",
    "pre_1950_bungalow",
    "post_1950_bungalow",
    "pre_1950_semi_terraced_house",
    "post_1950_semi_terraced_house",
    "pre_1950_detached_house",
    "post_1950_detached_house",
]:
    ax.plot(
        numpy.linspace(0.1, 0.9, 9),
        model2.loc[lambda df: df["archetype"] == archetype, "mean"],
        label=archetype,
    )
    ax.fill_between(
        numpy.linspace(0.1, 0.9, 9),
        y1=model2.loc[lambda df: df["archetype"] == archetype, "obs_ci_lower"],
        y2=model2.loc[lambda df: df["archetype"] == archetype, "obs_ci_upper"],
        alpha=0.33,
    )

ax.set_ylim([0, 22_500])
ax.set_xlabel("Quantile")
ax.set_ylabel("Cost (£2023)")
ax.legend()

# %% [markdown]
# ### Model 3: Independent attributes model
#
# The archetypes model is really a kind of partial interaction. We can fit an equivalent model, but looking at the predictions if it performs better it is not by a meaningful amount. It's also not immediately apparent how you would collapse some of the categories to form an archetype - an average weighted by representation in the data might be the answer.

# %%
# Enclosed terrace types are relatively rare, so let's collapse into regular terraces.
data["BUILT_FORM"] = (
    data["BUILT_FORM"]
    .map({"Enclosed End-Terrace": "End-Terrace", "Enclosed Mid-Terrace": "Mid-Terrace"})
    .fillna(data["BUILT_FORM"])
)

# %%
# property-type, built-form - age interaction.
# Exclude park homes as they're not an archetype of interest.
# Model is unstable - presumably due to low cell sizes, model struggles to predict on Maisonette.
model = sm.quantreg(
    "adjusted_cost ~ PROPERTY_TYPE:BUILT_FORM:AGE_BAND_2CAT",
    data.loc[lambda df: df["PROPERTY_TYPE"] != "Park home", :],
).fit(0.2, max_iter=10_000)

# %%
pred = model.get_prediction(
    pandas.DataFrame(
        {
            "AGE_BAND_2CAT": ["Pre-1950", "Post-1950"],
            "PROPERTY_TYPE": ["House", "House"],
            "BUILT_FORM": ["Detached", "Detached"],
        }
    )
)

# %%
pred.summary_frame()

# %%
model2[model2["QUANTILE"] == 0.2].tail(2)

# %% [markdown]
# ### Model 4: Archetypes + Room Adjustment
#
# Rooms actually makes more of a difference to model fit (variance explained) than the archetype factors measured. Unfortunately using it reduces the sample by ~7,000 records.
#
# However, this adjustment calls into question the validity of the age-based archetypes (at least for flats at the median).
#
# Here we're modelling it as an independendent effect as we're currently treating it as a continuous variable.

# %%
model = sm.quantreg("adjusted_cost ~ archetype + NUMBER_HABITABLE_ROOMS", data).fit(0.5)

# %%
# Predict at the mean for each archetype
pred = model.get_prediction(
    pandas.DataFrame(
        {
            "archetype": ["pre_1950_flat", "post_1950_flat"],
            "NUMBER_HABITABLE_ROOMS": [3.150685, 2.378056],
        }
    )
)

# %%
# These estimates should be similar to the unadjusted values from model 2.
# Take an example of flats.
pred.summary_frame()

# %%
model2.loc[
    lambda df: (df["QUANTILE"] == 0.5)
    & df["archetype"].isin(["pre_1950_flat", "post_1950_flat"])
]

# %%
# However, compare like for like flat estimates.
# Predict at the mean for each archetype
pred = (
    model.get_prediction(
        pandas.DataFrame(
            {
                "archetype": [
                    "pre_1950_flat",
                    "post_1950_flat",
                    "pre_1950_flat",
                    "post_1950_flat",
                    "pre_1950_flat",
                    "post_1950_flat",
                ],
                "NUMBER_HABITABLE_ROOMS": [1, 1, 2, 2, 3, 3],
            }
        )
    )
    .summary_frame()
    .assign(
        archetype=[
            "pre_1950_flat",
            "post_1950_flat",
            "pre_1950_flat",
            "post_1950_flat",
            "pre_1950_flat",
            "post_1950_flat",
        ],
        NUMBER_HABITABLE_ROOMS=[1, 1, 2, 2, 3, 3],
    )
)

# %%
pred

# %% [markdown]
# This suggests that - at the median - differences in the cost to install a heat pump in a flat is explained by the size of the flat (number of rooms), rather than the age of the flat (which seems to have been previously assumed). The difference exists because the composition of flat archetypes for which we have data is different for the two archetypes.
