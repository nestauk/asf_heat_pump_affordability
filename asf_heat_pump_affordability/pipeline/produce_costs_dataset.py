import pandas as pd
from argparse import ArgumentParser
from asf_heat_pump_affordability import config
from asf_heat_pump_affordability.pipeline import preprocess_data, preprocess_cpi
from asf_heat_pump_affordability.getters import get_data


def run():
    """
    Creates ArgumentParser and passes arguments to main() in order to run main() to produce outputs.
    """
    parser = ArgumentParser()
    parser.add_argument(
        "--mcs_epc_join_date",
        help="Specify which batch of most_relevant joined MCS-EPC dataset to use, by date in the format YYMMDD.",
        type=str,
    )

    parser.add_argument(
        "--cost_year_min",
        help="Min year of heat pump installation cost data to include in analysis. Default min year in dataset.",
        default=None,
        type=int,
    )

    parser.add_argument(
        "--cost_year_max",
        help="Max year of heat pump installation cost data to include in analysis. Default max year in dataset.",
        default=None,
        type=int,
    )

    parser.add_argument(
        "--cpi_data_year",
        help="Reference year to adjust heat pump installation costs to.",
        type=int,
    )

    args = parser.parse_args()
    main(**vars(args))


def main(mcs_epc_join_date, cpi_data_year, cost_year_min=None, cost_year_max=None):
    """
    IN DEV: currently imports MCS-EPC joined dataset, applies exclusion criteria to it and saves the output to S3.

    Args:
        mcs_epc_join_date (str): which batch of most_relevant joined MCS-EPC dataset to use, by date in the format YYMMDD.
        cpi_data_year (int): reference year to adjust heat pump installation costs to
        cost_year_min (int): min year of heat pump installation cost data to include in analysis
        cost_year_max (int): max year of heat pump installation cost data to include in analysis

    Returns
        IN DEV
            target return: Excel file containing cost deciles by property archetype (save to S3)
            currently returns: analytical sample dataset with adjusted costs
    """
    # Import MCS-EPC data
    mcs_epc_data = pd.read_csv(
        f"s3://asf-core-data/outputs/MCS/mcs_installations_epc_most_relevant_{mcs_epc_join_date}.csv"
    )

    # Preprocess MCS-EPC data - apply exclusion criteria
    sample = preprocess_data.apply_exclusion_criteria(
        df=mcs_epc_data, cost_year_min=cost_year_min, cost_year_max=cost_year_max
    )

    # Save analytical sample
    year_range = "".join(
        [str(sample["commission_year"].min()), str(sample["commission_year"].max())]
    )
    sample.to_csv(
        f"s3://asf-heat-pump-affordability/mcs_installations_epc_most_relevant_{mcs_epc_join_date}_preprocessed_yearRange_{year_range}.csv"
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

    return mcs_epc_inf


if __name__ == "__main__":
    run()
