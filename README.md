# asf_heat_pump_affordability

## Description

This repo contains the functions and run script to generate a dataframe of the costs of installing Air Source Heat Pumps
(ASHP) in 8 different types of home at a series of percentiles given by the user. The 8 housing archetypes are: flats;
semi-detached & terraced houses and maisonettes; detached houses; and bungalows; with each group split into pre- and
post-1950 construction. Costs are adjusted for inflation against a chosen base year.

## Setup

- Clone the repo
- Meet the data science cookiecutter [requirements](http://nestauk.github.io/ds-cookiecutter/quickstart), in brief:
  - Install: `direnv` and `conda`
- Run `make install` to configure the development environment. This will:
  - Setup the conda environment
  - Configure `pre-commit`
- Navigate to your local copy of the repo
- Activate the conda environment with `conda activate asf_heat_pump_affordability`

## Recreate cost policy analysis dataset

To recreate the data used in the Q1 2024 cost policy analysis, run the following lines in your terminal:

```
pip install jupytext
jupytext --to notebook asf_heat_pump_affordability/notebooks/investigate_effect_of_room_number.py
```

Run all lines of the resulting `investigate_effect_of_room_number.ipynb` notebook. The analytical output dataset will be saved into the `asf-heat-pump-affordability` bucket on S3.

## Data sources - Q1 2024 cost policy analysis

- MCS-EPC `most_relevant` version featuring data up and including Q2 2023: mcs_installations_epc_most_relevant_231009.csv
- [Xoserve off-gas postcode register October 2023](https://www.xoserve.com/help-centre/supply-points-metering/supply-point-administration-spa/)
- [ONS Postcode Directory (August 2023)](https://geoportal.statistics.gov.uk/datasets/487a5ba62c8b4da08f01eb3c08e304f6/about) (contains 2011 rural-urban classification indicator data)
- [Indices of Deprivation 2019: income and employment domains combined for England and Wales](https://www.gov.uk/government/statistics/indices-of-deprivation-2019-income-and-employment-domains-combined-for-england-and-wales)
- [Scottish Index of Multiple Deprivation 2020v2 - ranks](https://www.gov.scot/publications/scottish-index-of-multiple-deprivation-2020v2-ranks/) [(see further documentation)](https://www.gov.scot/collections/scottish-index-of-multiple-deprivation-2020/)
- Conversions for rural-urban classification codes to a 2-fold ("rural"/"urban") classification can be found on p19, section 39 in the [ONS Postcode Directory (August 2023) User Guide](https://geoportal.statistics.gov.uk/datasets/a8db59f77e7542d092458426dbacfb90/about)
  for England and Wales, and in Table 2.3 of [Scottish Government Urban Rural Classification 2020](https://www.gov.scot/publications/scottish-government-urban-rural-classification-2020/pages/2/) for Scotland.

NB: the URLs used by getter functions to load source data can be found and updated in `asf_heat_pump_affordability/config/base.yaml`

## Repo structure

Key directories and files:

```
asf_heat_pump_affordability
├───config
│    base.yaml - core variables, including source URLs used by getter functions
│    schema.json - data types for loading flat files
├───getters
│    get_data.py - functions to retrieve data from external sources
├───notebooks
│    identify_and_review_analytical_sample.py - notebook to explore impact of applying exclusion criteria on core MCS-EPC dataset
│    investigate_effect_of_room_number.py - notebook using quantile regression models to estimate installation costs by archetype and room number
├───pipeline
│    archetypes.py - functions to classify housing archetypes
│    generate_cost_percentiles.py - functions to generate cost percentiles by archetype
│    preprocess_cpi.py - functions to preprocess Consumer Price Index dataset
│    preprocess_data.py - functions to preprocess MCS-EPC dataset to produce sample used to calculate cost distributions
│    produce_costs_dataset.py - run script with main() function
```

## Contributor guidelines

[Technical and working style guidelines](https://github.com/nestauk/ds-cookiecutter/blob/master/GUIDELINES.md)

---

<small><p>Project based on <a target="_blank" href="https://github.com/nestauk/ds-cookiecutter">Nesta's data science project template</a>
(<a href="http://nestauk.github.io/ds-cookiecutter">Read the docs here</a>).
</small>
