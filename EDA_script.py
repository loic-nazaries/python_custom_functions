"""Exploratory Data Analysis (EDA)."""

# supress unnecessary warnings so that presentation looks clean
import warnings

warnings.filterwarnings("ignore")
# handling data
import pandas as pd

# data vizualisation
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# default folder to save files
file_path = "C:/python_projects/la_piscine/projet_pro/files/"


def load_data():
    """Load the dataset."""
    return pd.read_csv(
        file_path + "ghg_flux_data_aggregated_no_mdf_clean.csv",
        sep="[;,]",
        engine="python",
        skipinitialspace=True,
        na_values=" ",
    )


ghg_flux_data = load_data()

categorical_variables = (
    ghg_flux_data.iloc[:, :10].astype("category").set_index("Sampling_Date")
)
dummy_variables = (
    ghg_flux_data.iloc[:, 14:]
    .astype("int64")
    .join(ghg_flux_data["Sampling_Date"])
    .set_index("Sampling_Date")
    .astype("bool")
)
date_variables = (
    ghg_flux_data.iloc[:, [13]]
    .join(ghg_flux_data["Sampling_Date"])
    .set_index("Sampling_Date")
)
numerical_variables = (
    ghg_flux_data.iloc[:, 10:13]
    .astype("float64")
    .join(ghg_flux_data["Sampling_Date"])
    .set_index("Sampling_Date")
)
all_variables_loaded = [
    categorical_variables,
    dummy_variables,
    date_variables,
    numerical_variables,
]
ghg_flux_data = pd.concat(all_variables_loaded, sort=False, axis=1)


def exploratory_data_analysis(numerical_variables):
    """Print out the summary descriptive statistics from the dataset."""
    print(
        f"\nData shape: {numerical_variables.shape}\n",
        f"\nDataFrame head: {numerical_variables.head()}\n",
        f"\nDataFrame tail: {numerical_variables.tail()}\n",
        f"\nList of DataFrame columns: \n{numerical_variables.columns}\n",
    )
    summary_stats = numerical_variables.describe()
    pct_variation = pd.DataFrame(
        numerical_variables.std() / numerical_variables.mean() * 100
    ).T.rename(index={0: "pct_var"})
    kurtosis = pd.DataFrame(numerical_variables.kurt()).T.rename(
        index={0: "kurt"}
    )
    skewness = pd.DataFrame(numerical_variables.skew()).T.rename(
        index={0: "skew"}
    )
    dataframes = [summary_stats, pct_variation, kurtosis, skewness]
    summary_stats_non_cat_vars = pd.concat(dataframes, sort=False, axis=0)

    summary_stats_non_cat_vars.to_csv(
        file_path + "ghg_flux_data_aggregated_EDA.csv",
        sep=",",
        encoding="utf-8",
        index=True,
    )
    print(f"\nExploratory Data Analysis:\n{summary_stats_non_cat_vars}")
    # table of % of missing values
    non_nan_values = numerical_variables.notna().sum()
    nan_values = numerical_variables.isna().sum()
    pct_nan = numerical_variables.isna().sum() / len(numerical_variables) * 100
    nan_table = pd.DataFrame([non_nan_values, nan_values, pct_nan]).T.rename(
        columns={
            0: "Number of non-NA values",
            1: "Number of NA values",
            2: "Percentage of NaN values",
        }
    )
    print(f"\nSummary of missing data:\n{nan_table}\n")


def box_plot(args, *kwargs):
    """Draw a box plot of numerical variables."""
    ch4_flux = numerical_variables["ch4_flux"]
    co2_flux = numerical_variables["co2_flux"]
    n2o_flux = numerical_variables["n2o_flux"]

    plot_name = "ghg_flux_data_box_plot"
    title = "Distribution of GHG fluxes\n"

    plt.subplots(figsize=(12, 5))

    ax = plt.subplot(131)
    box_plot = sns.boxplot(data=ch4_flux, color="blue")
    ax.tick_params(labelsize=10)
    ax.set(xticklabels=[], xticks=[])
    ax.set_xlabel("CH4 flux", fontsize=12)
    ax.set_ylabel("CH4 flux (µg-C/m2/h)", color="blue", fontsize=14)
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.0f"))

    ax1 = plt.subplot(132)
    box_plot = sns.boxplot(data=co2_flux, color="orange")
    ax1.tick_params(labelsize=10)
    ax1.set(xticklabels=[], xticks=[])
    ax1.set_xlabel("CO2 flux", fontsize=12)
    ax1.set_ylabel("CO2 flux (mg-C/m2/h)", color="orange", fontsize=14)
    ax1.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.0f"))

    ax3 = plt.subplot(133)
    box_plot = sns.boxplot(data=n2o_flux, color="green")
    ax3.tick_params(labelsize=10)
    ax3.set(xticklabels=[], xticks=[])
    ax3.set_xlabel("N2O flux", fontsize=12)
    ax3.set_ylabel("N2O flux (µg-N/m2/h)", color="green", fontsize=14)
    ax3.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.0f"))

    plt.tight_layout(pad=1)
    plt.title(title, fontsize=18, horizontalalignment="right")
    plt.savefig(file_path + plot_name + ".png", dpi=100, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    load_data()
    exploratory_data_analysis(numerical_variables)
    box_plot(numerical_variables)
