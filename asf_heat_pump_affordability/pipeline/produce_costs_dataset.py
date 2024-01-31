import pandas as pd
import numpy as np
from argparse import ArgumentParser
from typing import Optional, Iterable
from asf_heat_pump_affordability import config, json_schema
from asf_heat_pump_affordability.pipeline import (
    archetypes,
    preprocess_data,
    preprocess_cpi,
    generate_cost_percentiles,
)
from asf_heat_pump_affordability.getters import get_data


def run():
    """
    Creates ArgumentParser and passes arguments to main() in order to run main() to produce outputs.
    """
    parser = ArgumentParser()
    parser.add_argument(
        "--mcs_epc_join_date",
        help="Specify which batch of `most_relevant` joined MCS-EPC dataset to use, by date in the format YYMMDD.",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--cost_year_min",
        help="Min year of heat pump installation cost data to include in analysis. Default min year in dataset.",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--cost_year_max",
        help="Max year of heat pump installation cost data to include in analysis. Default max year in dataset.",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--cpi_data_year",
        help="Reference year to adjust heat pump installation costs to.",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--cost_quantiles",
        help="Quantile(s) at which to extract cost values for each property archetype (range 0 to 1).",
        nargs="+",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--save_to_s3",
        help="Save sample and output datasets to `asf-heat-pump-affordability` S3 bucket.",
        action="store_true",
    )

    args = parser.parse_args()
    main(**vars(args))


def main(
    mcs_epc_join_date: int,
    cpi_data_year: int,
    cost_year_min: Optional[int] = None,
    cost_year_max: Optional[int] = None,
    cost_quantiles: Optional[Iterable[float]] = None,
    save_to_s3: bool = True,
) -> pd.DataFrame:
    """
    Import MCS-EPC joined dataset, apply preprocessing, and calculate specified cost percentiles (adjusted for inflation
    against a given base year) for MCS-certified installations of Air Source Heat Pumps (ASHP) in eight different property
    archetypes, where ASHPs were installed within the given year range.

    Args:
        mcs_epc_join_date (int): which batch of most_relevant joined MCS-EPC dataset to use, by date in the format YYMMDD.
        cpi_data_year (int): reference year to adjust heat pump installation costs to
        cost_year_min (int): min year of heat pump installation cost data to include in analysis
        cost_year_max (int): max year of heat pump installation cost data to include in analysis
        cost_quantiles (Iterable[float]): quantile(s) at which to extract cost values for each property archetype (range 0 to 1).
                                          Default produces cost deciles.
        save_to_s3 (bool): save analytical sample and output dataset to `asf-heat-pump-affordability` bucket on S3. Default True.

    Returns
        pd.DataFrame: cost percentiles for each property archetype adjusted for inflation
    """
    # Import MCS-EPC data
    mcs_epc_data = pd.read_csv(
        f"s3://asf-core-data/outputs/MCS/mcs_installations_epc_most_relevant_{mcs_epc_join_date}.csv",
        dtype=json_schema["mcs_epc_data"],
        parse_dates=["commission_date", "INSPECTION_DATE"],
    )

    # Preprocess MCS-EPC data - apply exclusion criteria
    sample = preprocess_data.apply_exclusion_criteria(
        df=mcs_epc_data, cost_year_min=cost_year_min, cost_year_max=cost_year_max
    )

    # Import and process CPI data
    cpi_05_3_df = get_data.get_df_from_url(config["data_source"]["cpi_source_url"])
    cpi_quarterly_df = preprocess_cpi.get_df_quarterly_cpi_with_adjustment_factors(
        ref_year=cpi_data_year,
        cpi_df=cpi_05_3_df,
        cpi_col_header=config["cpi_data"]["cpi_column_header"],
    )

    # Get MCS-EPC df with adjusted costs
    mcs_epc_inf = preprocess_data.generate_df_adjusted_costs(
        mcs_epc_df=sample, cpi_quarters_df=cpi_quarterly_df
    )

    # Get archetypes
    archetypes_dict = archetypes.classify_dict_archetypes_masks(mcs_epc_inf)

    # Get cost quantiles
    if cost_quantiles is None:
        cost_quantiles = np.arange(0, 1.1, 0.1)
    archetypes_costs_dict = (
        generate_cost_percentiles.generate_dict_cost_quantiles_by_archetype(
            cost_series=mcs_epc_inf["adjusted_cost"],
            archetypes_masks=archetypes_dict,
            quantiles=cost_quantiles,
        )
    )
    archetypes_costs_df = (
        generate_cost_percentiles.generate_df_cost_percentiles_by_archetype_formatted(
            archetype_costs_dict=archetypes_costs_dict,
            quantiles=cost_quantiles,
            ref_year=cpi_data_year,
        )
    )

    # Save files to S3 bucket
    if save_to_s3:
        year_range = "".join(
            [str(sample["commission_year"].min()), str(sample["commission_year"].max())]
        )
        ym_range = "_".join(
            [
                sample["commission_date"].dt.strftime("%Y%m").min(),
                sample["commission_date"].dt.strftime("%Y%m").max(),
            ]
        )

        # Save analytical sample
        sample.to_csv(
            f"s3://asf-heat-pump-affordability/mcs_installations_epc_most_relevant_{mcs_epc_join_date}_preprocessed_yearRange_{year_range}.csv"
        )

        # Save output
        archetypes_costs_df.to_excel(
            f"s3://asf-heat-pump-affordability/inflation_adjusted_costs_GBP_by_property_archetype_{cpi_data_year}_ashp_mcs_installations_{ym_range}.xlsx"
        )

    return archetypes_costs_df


if __name__ == "__main__":
    run()
