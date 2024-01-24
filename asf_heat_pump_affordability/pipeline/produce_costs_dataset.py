import pandas as pd
from argparse import ArgumentParser
from asf_heat_pump_affordability.pipeline import preprocess_data


def run():
    """
    Creates ArgumentParser and passes arguments to main() in order to run main() to produce outputs.
    """
    parser = ArgumentParser()
    parser.add_argument(
        "--mcs_epc_join_date",
        help="Specify which batch of most_relevant joined MCS-EPC dataset to use, by date in the format YYMMDD.",
        type=int,
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

    args = parser.parse_args()
    main(**vars(args))


def main(
    mcs_epc_join_date: int, cost_year_min: int = None, cost_year_max: int = None
) -> pd.DataFrame:
    """
    IN DEV: currently imports MCS-EPC joined dataset, applies exclusion criteria to it and saves the output to S3.

    Args:
        mcs_epc_join_date (int): which batch of most_relevant joined MCS-EPC dataset to use, by date in the format YYMMDD.
        cost_year_min (int): min year of heat pump installation cost data to include in analysis
        cost_year_max (int): max year of heat pump installation cost data to include in analysis

    Returns
        IN DEV
            target return: Excel file containing cost deciles by property archetype (save to S3)
            currently returns: analytical sample dataset
    """
    # Import data
    mcs_epc_data = pd.read_csv(
        f"s3://asf-core-data/outputs/MCS/mcs_installations_epc_most_relevant_{mcs_epc_join_date}.csv"
    )

    # Preprocess - apply exclusion criteria
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

    return sample


if __name__ == "__main__":
    run()
