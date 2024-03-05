# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# ## Question: Does pre/post 1950 explain differences in ASHP installation costs within archetypes?
#
# Or does property size (i.e. number of rooms) explain the difference in cost within archetypes?
#
# To investigate this, we will:
# 1. Preprocess the MCS-EPC dataset to get adjusted costs of ASHP installation against a base year of 2023. This dataset will be used in the subsequent analyses.
# 2. Look at the distribution of number of rooms across the pre- and post-1950 groupings of the different 4 property types and see whether there are differences.
# 3. Create 2 series of quantile regression models to compare how predictions of installation costs change when number of rooms is included as a feature or not.

# +
from io import StringIO
import regex as re
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm

from asf_heat_pump_affordability import config
from asf_heat_pump_affordability.getters import get_data
from asf_heat_pump_affordability.pipeline import (
    preprocess_cpi,
    preprocess_data,
    archetypes,
)

# -

# ## 1. Get and preprocess dataset

## Get MCS-EPC data
data = pd.read_csv(
    "s3://asf-heat-pump-affordability/mcs_installations_epc_most_relevant_231009_preprocessed_yearRange_20212023.csv",
    index_col=0,
)

# +
## Get CPI data with adjustment factors for 2023 base year
cpi_05_3_df = get_data.get_df_from_csv_url(config["data_source"]["cpi_source_url"])
cpi_quarterly_df = preprocess_cpi.get_df_quarterly_cpi_with_adjustment_factors(
    ref_year=2023,
    cpi_df=cpi_05_3_df,
    cpi_col_header=config["cpi_data"]["cpi_column_header"],
)

## Get MCS-EPC df with adjusted costs
data = preprocess_data.generate_df_adjusted_costs(
    mcs_epc_df=data, cpi_quarters_df=cpi_quarterly_df
)

# +
## Add archetypes

for (
    archetype,
    filter,
) in archetypes.classify_dict_archetypes_masks(data).items():
    data.loc[filter, "archetype"] = archetype
# -

data.groupby("archetype")["archetype"].count()

# +
## As we will use number of habitable rooms as a feature, we need to drop rows with NaN values or 0 values
## View count of lost values from each archetype

data[
    (data["NUMBER_HABITABLE_ROOMS"].isna()) | (data["NUMBER_HABITABLE_ROOMS"] == 0)
].groupby("archetype")["archetype"].count().sort_values(ascending=False)

# +
_data = data.copy()

# Remove park homes as low count
_data = _data.loc[_data["PROPERTY_TYPE"] != "Park home"]

# Remove rows with 0 or NaN number of rooms
_data = _data[_data["NUMBER_HABITABLE_ROOMS"] != 0]
_data.dropna(subset=["NUMBER_HABITABLE_ROOMS"], inplace=True)

# Create archetype4 column (i.e. property type)
_data["archetype4"] = _data["archetype"].str.split("_", n=2, expand=True)[2]
# -

## View total rows removed
print(data.shape)
print(_data.shape)
print(len(data) - len(_data))


# ## 2. Explore distribution of number of rooms for each archetype


def generate_plots(feature):
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    for archetype, ax in zip(_data["archetype4"].unique(), axs.ravel()):
        pre_1950 = _data[_data["archetype"] == f"pre_1950_{archetype}"]
        post_1950 = _data[_data["archetype"] == f"post_1950_{archetype}"]

        binned_data, bin_ranges = pd.cut(pre_1950[feature], bins=20, retbins=True)

        ax.hist(
            pre_1950[feature],
            density=True,
            histtype="bar",
            alpha=0.5,
            bins=bin_ranges,
            color="indianred",
            label="pre_1950",
        )
        ax.hist(
            post_1950[feature],
            density=True,
            histtype="bar",
            alpha=0.5,
            bins=bin_ranges,
            color="dodgerblue",
            label="post_1950",
        )

        ax.axvline(
            pre_1950[feature].mean(),
            color="indianred",
            label="pre_1950 (mean)",
        )
        ax.axvline(
            post_1950[feature].mean(),
            color="dodgerblue",
            label="post_1950 (mean)",
        )

        ax.axvline(
            pre_1950[feature].median(),
            color="indianred",
            linestyle="--",
            label="pre_1950 (median)",
        )
        ax.axvline(
            post_1950[feature].median(),
            color="dodgerblue",
            linestyle="--",
            label="post_1950 (median)",
        )

        ax.legend()

        ax.set_xlabel(f"{feature}")

        ax.set_title(f"pre_1950 and post_1950 {archetype}s")


_data.groupby("archetype")["NUMBER_HABITABLE_ROOMS"].agg(["median", "mean"])

generate_plots("NUMBER_HABITABLE_ROOMS")


# ## Findings
#
# - On average (median) pre-1950 flats in our dataset have one more room than post-1950 flats
# - On average (median and mean) pre- and post-1950 detached houses in our dataset have almost exactly the same number of rooms
# - On average, for both semi-detached terraced houses and for bungalows, the median number of rooms is the same in pre- and post-1950 properties but the mean average is slightly different. Pre-1950 bungalows have ~0.3 more rooms on mean average than post-1950, and pre-1950 semi-detached/terraced houses have ~0.6 more rooms on mean average than post-1950.
# - We can see that there are some minor differences between pre- and post-1950 distributions for semi-detached/terraced properties and bungalows, significant different for pre- and post-1950 flats, and a slight difference between pre- and post-1950 detached houses although both pre- and post-1950 in this latter group don't appear to be skewed unlike the other groups.


def generate_plot_archetype(feature):
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    colours = plt.cm.rainbow(np.linspace(0, 1, _data["archetype4"].nunique()))
    labels = []

    for archetype, colour in zip(_data["archetype4"].unique(), colours):
        archetype_df = _data[_data["archetype4"] == archetype]

        binned_data, bin_ranges = pd.cut(
            _data[_data["archetype4"] == "flat"][feature], bins=20, retbins=True
        )

        ax.hist(
            archetype_df[feature],
            density=True,
            histtype="bar",
            alpha=0.5,
            bins=bin_ranges,
            color=colour,
            label=archetype,
        )

        ax.axvline(
            archetype_df[feature].median(),
            color=colour,
            alpha=0.5,
            label=f"{archetype} (median)",
        )

        labels.extend([archetype, f"{archetype}_median"])

    ax.legend(labels=labels)

    ax.set_title(f"{feature}")


for f in ["NUMBER_HABITABLE_ROOMS", "TOTAL_FLOOR_AREA"]:
    generate_plot_archetype(f)

# ## Findings
# There are differences in average size of home between archetypes:
# - Flats are smallest on average (~50m2) with fewer rooms (2)
# - Bungalows and semi-detached/terraced houses are larger (~90m2) with more rooms (4 and 5, respectively)
# - Detached houses are largest on average (~180m2) with the most rooms on average (7)

# ## 3. Compare coefficients of quantile regression models
#
# The code below aims to do the following:
# 1. Create 2 series of quantile regression models. Each series contains one regression model for each percentile 10-90. One series of models will be trained using the 8 different archetypes as the only features (a_models), the other with the 8 archetypes + number of rooms as the features (ar_models).
# 2. For each model within both series of models: calculate the difference between pre- and post-1950 coefficients for each of the 4 property types (detached house, semi-detached/terraced house, flat, bungalow). E.g. For Quantile 0.1 model: pre-1950 flats coef - post-1950 flats coef, etc. Calculate the confidence intervals for the difference as well. Ultimately, we get: the difference at each quantile between pre- and post-1950 installation cost for each archetype and the confidence interval for this difference.
# 3. For each property type, compare the difference in pre- and post-1950 installation cost at each quantile between both model groups (a_models vs ar_models).
#
# Note: confidence intervals spanning zero indicate there is a chance the difference in coefficients for pre- and post-1950 archetype is zero and therefore that there may be no meaningful difference in cost of installation when splitting the archetype by pre- or post-1950 build year.

archetypes4 = _data.archetype4.unique()
X_y = _data[["archetype", "NUMBER_HABITABLE_ROOMS", "adjusted_cost"]]

## Regex pattern to extract archetype names from summary table
pattern = r"\[(.*?)\]"

# ## Series 1: models with archetypes as only features

# +
a_model_pre_post_coefs = {
    "quantile": [],
    "archetype": [],
    "pre_1950_coef": [],
    "post_1950_coef": [],
    "pre_1950_coef_lower": [],
    "post_1950_coef_lower": [],
    "pre_1950_coef_upper": [],
    "post_1950_coef_upper": [],
}

for quantile in np.arange(0.1, 1, 0.1):
    quantile = round(quantile, 1)
    # Train model
    archetype_model = sm.quantreg("adjusted_cost ~ archetype - 1", X_y).fit(quantile)

    # Get summary table containing coefficients, p-values, and confidence intervals for the model
    table = archetype_model.summary(alpha=math.sqrt(0.05)).tables[1]
    _df = pd.read_csv(
        StringIO(table.as_csv()), index_col=0
    ).reset_index()  # convert statsmodels table to df
    _df.columns = ["feature", "coef", "std err", "t", "P>|t|", "lower", "upper"]
    _df["feature"] = _df["feature"].apply(lambda x: re.search(pattern, x).group(1))

    # Create dict to generate df from for calculating coefficient difference and confidence intervals from
    for a4 in archetypes4:
        pre = _df[_df["feature"] == f"pre_1950_{a4}"]
        post = _df[_df["feature"] == f"post_1950_{a4}"]

        a_model_pre_post_coefs["quantile"].append(quantile)
        a_model_pre_post_coefs["archetype"].append(a4)
        a_model_pre_post_coefs["pre_1950_coef"].append(pre["coef"].values[0])
        a_model_pre_post_coefs["post_1950_coef"].append(post["coef"].values[0])
        a_model_pre_post_coefs["pre_1950_coef_lower"].append(pre["lower"].values[0])
        a_model_pre_post_coefs["post_1950_coef_lower"].append(post["lower"].values[0])
        a_model_pre_post_coefs["pre_1950_coef_upper"].append(pre["upper"].values[0])
        a_model_pre_post_coefs["post_1950_coef_upper"].append(post["upper"].values[0])


## NB:
## Set alpha=np.sqrt(0.05) in `model.summary()` because we want to alter the confidence limits reported for the coefficients.
## This is because we are interested in comparing the differences between 2 pairs of coefficients.
## Leaving alpha=0.05 would reflect a confidence interval much larger than 95%, i.e. 99.75%,
## because it would involve the comparison of extreme values at 2 * 95 percentile intervals instead of just 1.
## We control for that by using sqrt 0.05.

# +
## Get diff between pre- and post-1950 coefficients (and upper and lower bounds) for each archetype

a_model_coefs = pd.DataFrame(a_model_pre_post_coefs)
a_model_coefs["coef_diff"] = (
    a_model_coefs["pre_1950_coef"] - a_model_coefs["post_1950_coef"]
)
a_model_coefs["max_coef_diff"] = (
    a_model_coefs["pre_1950_coef_upper"] - a_model_coefs["post_1950_coef_lower"]
)
a_model_coefs["min_coef_diff"] = (
    a_model_coefs["pre_1950_coef_lower"] - a_model_coefs["post_1950_coef_upper"]
)
# -

# ## Series 2: models with archetype + number of rooms as features

# +
ar_model_pre_post_coefs = {
    "quantile": [],
    "number_rooms": [],
    "archetype": [],
    "pre_1950_coef": [],
    "post_1950_coef": [],
    "pre_1950_coef_lower": [],
    "post_1950_coef_lower": [],
    "pre_1950_coef_upper": [],
    "post_1950_coef_upper": [],
}

for quantile in np.arange(0.1, 1, 0.1):
    quantile = round(quantile, 1)
    # Train model
    archetype_room_model = sm.quantreg(
        "adjusted_cost ~ NUMBER_HABITABLE_ROOMS + archetype -1", X_y
    ).fit(quantile)

    # Get summary table
    table = archetype_room_model.summary(alpha=math.sqrt(0.05)).tables[1]
    _df = pd.read_csv(StringIO(table.as_csv()), index_col=0).reset_index()
    _df.columns = ["feature", "coef", "std err", "t", "P>|t|", "lower", "upper"]
    _df["feature"] = _df["feature"].apply(
        lambda x: re.search(pattern, x).group(1)
        if x.startswith("archetype")
        else x.strip()
    )

    ## Create dict to generate df from
    for a4 in archetypes4:
        pre = _df[_df["feature"] == f"pre_1950_{a4}"]
        post = _df[_df["feature"] == f"post_1950_{a4}"]

        ar_model_pre_post_coefs["quantile"].append(quantile)
        ar_model_pre_post_coefs["number_rooms"].append(
            _df[_df["feature"] == "NUMBER_HABITABLE_ROOMS"]["coef"].values[0]
        )
        ar_model_pre_post_coefs["archetype"].append(a4)
        ar_model_pre_post_coefs["pre_1950_coef"].append(pre["coef"].values[0])
        ar_model_pre_post_coefs["post_1950_coef"].append(post["coef"].values[0])
        ar_model_pre_post_coefs["pre_1950_coef_lower"].append(pre["lower"].values[0])
        ar_model_pre_post_coefs["post_1950_coef_lower"].append(post["lower"].values[0])
        ar_model_pre_post_coefs["pre_1950_coef_upper"].append(pre["upper"].values[0])
        ar_model_pre_post_coefs["post_1950_coef_upper"].append(post["upper"].values[0])

# +
## Get diff between pre- and post-1950 coefficients (and upper and lower bounds) for each archetype

ar_model_coefs = pd.DataFrame(ar_model_pre_post_coefs)
ar_model_coefs["coef_diff"] = (
    ar_model_coefs["pre_1950_coef"] - ar_model_coefs["post_1950_coef"]
)
ar_model_coefs["max_coef_diff"] = (
    ar_model_coefs["pre_1950_coef_upper"] - ar_model_coefs["post_1950_coef_lower"]
)
ar_model_coefs["min_coef_diff"] = (
    ar_model_coefs["pre_1950_coef_lower"] - ar_model_coefs["post_1950_coef_upper"]
)

# +
## Plot results of both models

fig, axs = plt.subplots(2, 2, figsize=(10, 5))

for a4, ax in zip(archetypes4, axs.ravel()):
    a_df = a_model_coefs[a_model_coefs["archetype"] == a4]

    ax.plot(a_df["quantile"], a_df["coef_diff"], label="archetype_model")
    ax.fill_between(
        np.linspace(0.1, 0.9, 9),
        y1=a_df["min_coef_diff"],
        y2=a_df["max_coef_diff"],
        alpha=0.33,
    )

    ar_df = ar_model_coefs[ar_model_coefs["archetype"] == a4]

    ax.plot(ar_df["quantile"], ar_df["coef_diff"], label="archetype_room_model")
    ax.fill_between(
        np.linspace(0.1, 0.9, 9),
        y1=ar_df["min_coef_diff"],
        y2=ar_df["max_coef_diff"],
        alpha=0.33,
    )

    ax.set_title(f"{a4}")
    ax.set_xlabel("Quantile")
    ax.set_ylabel("pre-1950 - post-1950")

plt.legend()
plt.suptitle(
    "Difference in pre- and post-1950 regression coefficients by archetype by quantile\nfor 2 different model groups (with/without number of rooms as feature)"
)
plt.tight_layout()
plt.show()
# -

# ## Use archetype + number of rooms model to estimate costs
#
# - We see that some of the difference in pre- and post-1950 installation costs within semi-detached/terraced houses, bungalows, and flats is explained by difference in number of rooms, but not for detached houses.
# - We will therefore estimate pre- and post-1950 installation costs at each quantile for each property type using the archetype + number of rooms models. We will use the median number of rooms for each property type to predict installation costs in pre- and post-1950 properties.

# +
## Create df of data to predict on

median_rooms = _data.groupby("archetype4")["NUMBER_HABITABLE_ROOMS"].median().to_dict()
archetypes = []
rooms = []

for a4 in archetypes4:
    archetypes.extend([f"pre_1950_{a4}", f"post_1950_{a4}"])
    rooms.extend([int(median_rooms[a4]), int(median_rooms[a4])])

predict_on = pd.DataFrame({"archetype": archetypes, "NUMBER_HABITABLE_ROOMS": rooms})

predict_on["archetype_label"] = (
    predict_on["archetype"]
    + "_"
    + predict_on["NUMBER_HABITABLE_ROOMS"].map(str)
    + "_rooms"
)

# +
## Train models and run on archetype + median room number to estimate installation costs

results = []

for quantile in np.arange(0.1, 1, 0.1):
    quantile = round(quantile, 1)
    archetype_room_model = sm.quantreg(
        "adjusted_cost ~ NUMBER_HABITABLE_ROOMS + archetype - 1", X_y
    ).fit(quantile)

    results.append(
        archetype_room_model.get_prediction(predict_on)
        .summary_frame()
        .assign(
            archetype_label=predict_on.archetype_label.to_list(),
            quantile=quantile,
        )
    )

predictions = pd.concat(results)

# +
## Plot estimated installation cost for each archetype

fig, ax = plt.subplots(1, 1, figsize=(10, 5))

for archetype in predictions.archetype_label.unique():
    ax.plot(
        np.arange(0.1, 1, 0.1),
        predictions[predictions["archetype_label"] == archetype]["mean"],
        label=archetype,
    )

ax.set_xlabel("Quantile")
ax.set_ylabel("Cost (equiv. 2023 GBP)")
ax.set_title("Cost of ASHP installation in different property archetypes")
ax.legend()
plt.tight_layout()
plt.show()

# +
## Format df to save to Excel

predictions["rounded_cost"] = round(predictions["mean"], -1).apply(int)

final = predictions.pivot(
    index="archetype_label", columns="quantile", values="rounded_cost"
).reindex(predict_on.archetype_label.to_list())
final.columns.name = None
final.columns = [
    "_".join(["cost_percentile", str(int(col * 100))]) for col in final.columns
]

final.to_excel(
    "s3://asf-heat-pump-affordability/2021_2023_ashp_installation_costs_deciles_2023GBP.xlsx"
)

final
