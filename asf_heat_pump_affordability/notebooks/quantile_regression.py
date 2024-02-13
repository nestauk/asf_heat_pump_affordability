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
    save_to_s3=False,
)

# %%
quantiles

# %%
quantiles.to_excel("./20240213_mcs_epc_archetype_heat_pump_cost_deciles.xlsx")

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
# Binary built age variable.

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
model = sm.quantreg(
    "adjusted_cost ~ PROPERTY_TYPE:BUILT_FORM:AGE_BAND_2CAT -1", data
).fit(0.2)

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
# ### Model 4: Archetypes + Independent effect of Rural-Urban Setting.
#
# Fit the model "adjusted_cost ~ archetype + rural_urban", see what the independent effect of setting does to the estimates.

# %%

# %% [markdown]
# ### Model 5: Archetpes interacted with Rural-Urban Setting.
#
# Fit the model "adjusted_cost ~ archetype * rural_urban" or "adjusted_cost ~ archetype : rural_urban", see what the effect of setting does to the estimates, in particular see whether it produces better results than model 4. Explore uncertainty in predictions, bigger risk here of small group sizes leading to poor estimates.

# %%

# %% [markdown]
# Further Models - explore models for IMD (in Eng and Wales) or off gas grid homes. Not sure I have a strong hypothesis for either - off gas grid maybe more expensive if more work needed to central heating system? IMD data might need aggregation if too few data points in low income deciles.
