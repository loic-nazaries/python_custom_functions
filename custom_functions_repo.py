"""Analysis of the detection parameters for defects/chewing gums.

CUSTOM FUNCTIONS

Definition of the functions customised for the Rockwool analyses.
"""
# Call the libraries required
# import glob
from typing import Any
import re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
# import openpyxl
import pandas as pd
import pingouin as pg
import scipy as sp
import seaborn as sns
import sidetable as stb
import statsmodels.api as sm
import statsmodels.stats.multicomp as mc
from scipy.io import loadmat
from scipy.stats import chi2
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.formula.api import ols
from statsmodels.multivariate.manova import MANOVA
# Uncomment next import when '.set_output(transform="pandas")' is fixed
# from sklearn import set_config

# set_config(transform_output="pandas")


# ----------------------------------------------------------------------------

# INPUT/OUTPUT


def get_mat_file_list(directory_name: str) -> list[str]:
    """Get the list of file from a directory.

    Args:
        directory_name (_type_): _description_

    Returns:
        _type_: _description_
    """
    paths = Path(directory_name=directory_name).glob("**/*.mat")
    dictionary_name_list = [str(path) for path in paths]
    return dictionary_name_list  # check list content type


# check dict content type
def load_mat_file(mat_file_name: str) -> dict[str, float]:
    """Load data from '.mat' file as a dictionary and print it.

    Args:
        mat_file_name (_type_): _description_

    Returns:
        _type_: _description_
    """
    dictionary = loadmat(file_name=mat_file_name)
    return dictionary  # check dict content type


def load_excel_file(
    file_path_name: str,
    sheet_name: Any = 0,
    nrows: int = None,
) -> pd.DataFrame:
    """Load an Excel file.

    Args:
        file_path_name (_type_): _description_

    Returns:
        _type_: _description_
    """
    dataframe = pd.read_excel(
        io=file_path_name,
        sheet_name=sheet_name,
        header=0,
        nrows=nrows,
        index_col=None,
        decimal=".",
    )
    return dataframe


def save_csv_file(
    dataframe: pd.DataFrame,
    file_name: str
) -> None:
    """Save a dataframe as a '.csv()' file.

    Args:
        dataframe (_type_): _description_
        file_name (_type_): _description_
    """
    # Save the EDA ANOVA output for EACH defect category
    dataframe.to_csv(
        path_or_buf=f"./output/{file_name}.csv",
        sep=",",
        encoding="utf-8",
        index=True,
    )
    return


def save_excel_file(
    dataframe: pd.DataFrame,
    file_name: str
) -> None:
    """Save the dataframe to an MS Excel '.xlsx' file.

    Args:
        dataframe (_type_): _description_
        file_name (_type_): _description_
    """
    dataframe.to_excel(
        excel_writer=f"./output/{file_name}.xlsx"
    )
    return


def save_figure(figure_name: str, dpi: int = 300) -> None:
    """Save a figure as '.png' AND '.svg' file.

    Args:
        figure_name (_type_): _description_
    """
    for extension in (
        [
            "png",
            # "svg"
        ]
    ):
        plt.savefig(
            fname=f"./figures/{figure_name}.{extension}",
            format=extension,
            bbox_inches="tight",
            dpi=dpi
        )
    return


# ----------------------------------------------------------------------------


def remove_key_from_dictionary(
    dictionary: dict,  # check dict content type
    key_name: str
) -> dict:  # check dict content type
    """Delete a key and its content from a dictionary.

    Args:
        dictionary (_type_): _description_
        key_name (_type_): _description_

    Returns:
        _type_: _description_
    """
    del dictionary[key_name]
    return dictionary  # check dict content type


def extract_pattern_from_file_name(file_name: str) -> list[str]:
    r"""Use regular expressions to extract a pattern from a list of file names.

    Specifically, it splits the string (here, file name) each time an
    underscore '_' is encountered using the pattern '_+'. A list of strings is
    returned.
    Then, in our case, the first element of the new string list is selected
    using '[0]'.
    Finally, string slicing is used to extract the string from the 8th
    character up to the 15th, and ignore everything til the end of the string
    applying the slice '[8:16:]'.

    Args:
        file_name (_type_): _description_

    Returns:
        _type_: _description_
    """
    pattern_output_list = [
        re.split(pattern="_+", string=name)[0][8:16:] for name in file_name
    ]
    return pattern_output_list  # check list content type


def extract_pattern_from_file_name_2(file_name):
    r"""Use regular expressions to extract a pattern from a list of file names.

    Specifically, it splits the string each time an underscore '_' is
    encountered using the pattern '_+'. A list of strings is returned.

    Here, the first element of the list is selected using '[0]'.
    Then, string slicing is used to extract the string from the 19th
    character up to the end of the element using '[19:]'.

    Args:
        file_name (_type_): _description_

    Returns:
        _type_: _description_
    """
    pattern_output_list = [
        re.split(pattern="_+", string=name)[0][19:] for name in file_name
    ]
    # pattern_output = pattern_output.replace("\\", "")
    # pattern_output = list(pattern_output)
    return pattern_output_list  # check list content type


# ----------------------------------------------------------------------------

# ARRAYS


def convert_dataframe_to_array(
    dataframe: pd.DataFrame,
    column_name: str
) -> np.ndarray:
    """Convert dataframe to array.

    Note: the use of the present function could be avoided if the dataframe is
    transformed using one of the transformer function of the 'scikit-learn'
    library and its new parameter 'set_output(transform="pandas").
    See 'standardise_features' function below for an exemple.

    Args:
        dataframe (_type_, optional): _description_. Defaults to dataframe.
        column_name (_type_, optional): _description_. Defaults to column_name.

    Returns:
        _type_: _description_
    """
    array = dataframe[column_name].to_numpy()
    return array


def calculate_array_length(array: np.ndarray) -> int:
    """Calculate the number of rows in the array.

    Args:
        array (_type_): _description_

    Returns:
        _type_: _description_
    """
    array_length = array.shape[0]
    print(f"The array contains {array_length} rows.")
    return array_length


def convert_array_to_series(array: np.ndarray, array_name: str) -> pd.Series:
    """Convert the means of the array to a Pandas Series.

    Args:
        array (_type_, optional): _description_. Defaults to array.
        array_name (_type_, optional): _description_. Defaults to array_name.
    """
    series = pd.Series(array).rename(array_name)
    return series


def convert_array_to_dataframe(array: np.ndarray) -> pd.DataFrame:
    """Convert an array to a dataframe.

    Args:
        array (_type_): _description_

    Returns:
        _type_: _description_
    """
    dataframe = pd.DataFrame(data=array)
    return dataframe


# ----------------------------------------------------------------------------

# CONVERT OBJECTS


# check dict content type
def convert_dictionary_to_dataframe(dictionary_file: dict) -> pd.DataFrame:
    """Convert a dictionary into a dataframe.

    First, the 'len()' function gets the number of items in the dictionary.
    Second, 'range()' is used to set a range from 0 to length of dictionary.
    Finally, 'list()' converts the items into NUMERICAL index values.

    Args:
        dictionary_file (_type_): _description_

    Returns:
        _type_: _description_
    """
    dataframe = pd.DataFrame(
        data=dictionary_file,
        index=[list(range(len(dictionary_file)))]
    )
    return dataframe


def convert_list_to_dataframe(
    items_list: list[str],
    # column_names: list[str]
) -> pd.DataFrame:
    """Convert a nested list into a dataframe.

    Args:
        items_list (_type_): _description_

    Returns:
        _type_: _description_
    """
    dataframe = pd.DataFrame(data=items_list)
    # # BUG Below NOT working
    # dataframe = pd.DataFrame(
    #     data=items_list,
    #     columns=column_names
    # )
    return dataframe


# -----------------------------------------------------------------------------

# DATA MANIPULATION

def concatenate_processed_dataframes(dataframe_list):
    """Concatenate the dataframes from the various processing steps.

    Args:
        dataframe_list (_type_): _description_

    Returns:
        _type_: _description_
    """
    data_concatenation = pd.concat(objs=dataframe_list, axis=1)
    return data_concatenation


def remove_column_list(column, column_list, column_to_remove):
    """_summary_.

    Args:
        column (_type_): _description_
        column_list (_type_): _description_
        column_to_remove (_type_): _description_

    Returns:
        _type_: _description_
    """
    column_list_reduced = [
        column for column in column_list
        if column not in column_to_remove
    ]
    return column_list_reduced


def drop_column(dataframe, column_list):
    """Drop columns from the dataframe.

    Args:
        dataframe (_type_): _description_

    Returns:
        _type_: _description_
    """
    dataframe_reduced = dataframe.drop(
        labels=column_list,
        axis=1,
        # inplace=True,
    )
    return dataframe_reduced


def drop_row_by_value(dataframe, column_name, value_name):
    """Drop rows from a column in the dataframe.

    A list a 'value_name' can be passed with a for-loop.

    Args:
        dataframe (_type_): _description_

    Returns:
        _type_: _description_
    """
    dataframe_reduced = dataframe[dataframe[column_name] != value_name]

    return dataframe_reduced


def get_list_of_unique_values(dataframe, column_name):
    """Get a list of unique values from a dataframe.

    Since an array is produced, the output is converted to a list.

    Args:
        dataframe (_type_): _description_

    Returns:
        _type_: _description_
    """
    column_unique_values = dataframe[column_name].unique()
    unique_names = list(column_unique_values)
    return unique_names


def get_numerical_features(dataframe):
    """get_numerical_features _summary_.

    Args:
        dataframe (_type_): _description_

    Returns:
        _type_: _description_
    """
    # Select numerical variables ONLY and make a list
    numerical_features = dataframe.select_dtypes(include=np.number)
    numerical_features_list = numerical_features.columns.to_list()
    print(f"\nList of Numerical Features: \n{numerical_features_list}\n")
    print(numerical_features.describe())
    return numerical_features, numerical_features_list


def get_categorical_features(dataframe):
    """get_categorical_features _summary_.

    Args:
        dataframe (_type_): _description_

    Returns:
        _type_: _description_
    """
    # Select categorical variables ONLY and make a list
    categorical_features = dataframe.select_dtypes(exclude=(np.number))
    categorical_features_list = categorical_features.columns.to_list()
    print(f"\nList of Categorical Features: \n{categorical_features_list}\n")
    print(categorical_features.describe())
    return categorical_features, categorical_features_list


# -----------------------------------------------------------------------------

# EXPLORATORY DATA ANALYSIS


def run_exploratory_data_analysis(dataframe):
    """Print out the summary descriptive statistics from the dataset.

    Also includes a missing table summary.

    Args:
        dataframe (_type_): _description_

    Returns:
        _type_: _description_

    TODO Include the 'mode' function to analyse the existence of a bimodal
    distribution within the transmissivity variables.
    """
    print(
        f"\nData Shape: {dataframe.shape}\n",
        f"\nData Types:\n{dataframe.dtypes}\n",
        # f"\nData Preview:\n{dataframe.sample(n=10)}\n",
        f"\nList of DataFrame Columns:\n{dataframe.columns.to_list()}\n",
    )
    # Select "all" value in 'include' parameter to include non numerical data
    # in the EDA
    summary_stats = dataframe.describe(include="all").T

    # # Mode
    # mode = dataframe.mode(axis=0).T
    # # Rename the mode column names
    # mode_col_names = [
    #     "mode" + str(col) for col in mode.columns
    # ]
    # mode.columns = mode_col_names

    # Percent of variance (standard deviation actually) compared to mean value
    pct_variation = (
        dataframe.std() / dataframe.mean() * 100
    ).rename("pct_var")
    # Calculate mean absolute deviation (MAD)
    mad = dataframe.mad().rename("mad")
    # Kurtosis
    kurtosis = dataframe.kurt().rename("kurt")
    # Skewness
    skewness = dataframe.skew().rename("skew")

    dataframe_list = [
        summary_stats,
        # mode,
        pct_variation,
        mad,
        kurtosis,
        skewness,
    ]
    summary_stats_table = pd.concat(
        objs=dataframe_list,
        sort=False,
        axis=1
    )

    # Save statistics summary to .csv file
    save_csv_file(
        dataframe=summary_stats_table,
        file_name="eda_output"
    )
    print(f"\nExploratory Data Analysis:\n{summary_stats_table}\n")
    return summary_stats_table


# Trying to generate numerical and categorical subsets
def run_exploratory_data_analysis_nums_cats(dataframe):
    """Print out the summary descriptive statistics from the dataset.

    Also includes a missing table summary.

    Args:
        dataframe (_type_): _description_

    Returns:
        _type_: _description_

    TODO Include the 'mode' function to analyse the existence of a bimodal
    distribution within the transmissivity variables.
    """
    print(
        f"\nData Shape: {dataframe.shape}\n",
        f"\nData Types:\n{dataframe.dtypes}\n",
        # f"\nData Preview:\n{dataframe.sample(n=10)}\n",
        f"\nList of DataFrame Columns:\n{dataframe.columns.to_list()}\n",
    )
    # Select "all" value in 'include' parameter to include non numerical data
    # in the EDA
    summary_stats_nums = dataframe.select_dtypes(
        include="number").describe(include="all").T

    # # Mode
    # mode = dataframe.mode(axis=0).T
    # # Rename the mode column names
    # mode_col_names = [
    #     "mode" + str(col) for col in mode.columns
    # ]
    # mode.columns = mode_col_names

    # Percent of variance (standard deviation actually) compared to mean value
    pct_variation = (
        dataframe.std() / dataframe.mean() * 100
    ).rename("pct_var")
    # Calculate mean absolute deviation (MAD)
    mad = dataframe.mad().rename("mad")
    # Kurtosis
    kurtosis = dataframe.kurt().rename("kurt")
    # Skewness
    skewness = dataframe.skew().rename("skew")

    dataframe_list = [
        summary_stats_nums,
        # mode,
        pct_variation,
        mad,
        kurtosis,
        skewness,
    ]
    summary_stats_nums_table = pd.concat(
        objs=dataframe_list,
        sort=False,
        axis=1
    )

    # Save statistics summary to .csv file
    save_csv_file(
        dataframe=summary_stats_nums_table,
        file_name="eda_output"
    )
    print(f"\nExploratory Data Analysis:\n{summary_stats_nums_table}\n")
    return summary_stats_nums_table


def get_missing_values_table(dataframe):
    """Use 'Sidetable' library to produce a table of metrics on missing values.

    Exclude columns that have 0 missing values using 'clip_0=True' parameter.
    Use parameter 'style=True' to style the display in the output table.

    Args:
        dataframe (_type_): _description_

    Returns:
        _type_: _description_
    """
    nan_table = dataframe.stb.missing(
        clip_0=True,
        # # BUG Below NOT working
        # style=True
    )
    print(f"\nPercentage of Missing Values:\n{nan_table}\n")
    return nan_table


def pivot_to_aggregate(
    dataframe,
    values=None,
    index_list=None,
    column_list=None,
    aggfunc_list=None
):
    """Use 'pivot_table' function to aggregate data.

    Args:
        dataframe (_type_): _description_
        index (_type_): _description_
        aggfunc_list (_type_): _description_

    Returns:
        _type_: _description_
    """
    pivot_table = pd.pivot_table(
        data=dataframe,
        values=values,
        index=index_list,
        columns=column_list,
        fill_value="",
        aggfunc=aggfunc_list
    )
    return pivot_table


def calculate_mahalanobis_distance(var, data, cov=None):
    """Compute the Mahalanobis Distance between each row of x and the data.

    Args:
        var (_type_, optional): vector or matrix of data with, say, p columns.
        data (_type_, optional): ndarray of the distribution from which
            Mahalanobis distance of each observation of x is to be computed.
        cov (_type_, optional): covariance matrix (p x p) of the distribution.
            If None, will be computed from data.
            Defaults to None.

    Returns:
        _type_: _description_
    """
    var_minus_mu = var - np.mean(data)
    if not cov:
        cov = np.cov(data.values.T)
    inv_covmat = sp.linalg.inv(cov)
    left_term = np.dot(var_minus_mu, inv_covmat)
    mahalanobis = np.dot(left_term, var_minus_mu.T)
    return mahalanobis.diagonal()


def apply_mahalanobis_test(dataframe, alpha=0.01):
    """_summary_.

    Args:
        dataframe (_type_): _description_
        alpha (float, optional): _description_. Defaults to 0.01.

    Returns:
        _type_: _description_
    """
    # Run the Mahalanobis test
    # outliers_detection = dataframe.copy()
    dataframe["mahalanobis"] = (
        calculate_mahalanobis_distance(
            var=dataframe,
            data=dataframe
        )
    )

    # Get the critical value for the test
    mahalanobis_test_critical_value = chi2.ppf(
        (1-alpha),
        df=len(dataframe.columns) - 1
    )
    print(
        f"""
        \nMahalanobis Test Critical Value for alpha < {alpha} : \
{mahalanobis_test_critical_value:.2f}\n
        """
    )

    # Get p-values from chi2 distribution
    dataframe["mahalanobis_p_value"] = (
        1 - chi2.cdf(
            dataframe["mahalanobis"],
            df=len(dataframe.columns) - 1)
    )
    outliers_dataframe = dataframe[
        dataframe.mahalanobis_p_value < alpha
    ]
    print(
        f"""
        \nTable of outliers based on Mahalanobis distance:\n
        {outliers_dataframe}\n
        """
    )

    # Get the index values to create a list of outlier IDs
    outlier_list = outliers_dataframe.index.values.tolist()
    print(f"\nOutlier List:\n{outlier_list}\n")
    return outliers_dataframe, outlier_list


def get_iqr_outliers(dataframe, column):
    """Build a dataframe containing outliers details.

    Args:
        dataframe (_type_): _description_
        column (_type_): _description_

    Returns:
        _type_: _description_
    """
    # Calculate Q1, Q3
    q_1, q_3 = np.percentile(
        a=dataframe[column],
        q=[25, 75]
    )
    print(f"\nFor {column}:")
    # Calculate IQR, upper limit and lower limit
    iqr = q_3 - q_1
    upper_limit = q_3 + 1.5 * iqr
    print(f"Upper Limit of IQR = {upper_limit :.2f}")
    lower_limit = q_1 - 1.5 * iqr
    print(f"Lower Limit of IQR = {lower_limit :.2f}")

    # Find outliers
    outlier_dataframe = (
        dataframe[
            (dataframe[column] > upper_limit) |
            (dataframe[column] < lower_limit)
        ]
    )
    # outlier_dataframe = dataframe.query(
    #     "column > upper_limit | column < lower_limit"
    # )
    print(
        f"""
        \nTable of outliers for {column} based on IQR value:\n
        {outlier_dataframe}\n
        """
    )
    return outlier_dataframe


def standardise_features(features):
    """Standardise the features to get them on the same scale.

    When used for Principal Component Analysis (PCA), this MUST be performed
    BEFORE applying PCA.

    NOTE: since the 'StandardScaler()' function from scikit-learn library
    produces a Numpy array, the 'convert_array_to_dataframe()' custom function
    is applied to convert the array to a Pandas dataframe.
    As a consequence, the column names must also be renamed using the selected
    feature names.

    Args:
        feature_list (_type_): _description_

    Returns:
        _type_: _description_

    TODO Why is '.set_output(transform="pandas")' not working ?!
        Once fixed, 'features_scaled = convert_array_to_dataframe
        (features_scaled)' can be deleted.
    """
    # Standardise the features
    scaler = StandardScaler()
    # BUGcBelow NOT working
    # scaler = StandardScaler().set_output(transform="pandas")  # NOT working
    # print(f"Data Type for scaler: {type(scaler).__name__}")
    features_scaled = scaler.fit_transform(features)
    # Delete next function call when .set_output(transform="pandas") is fixed
    features_scaled = convert_array_to_dataframe(features_scaled)
    features_scaled.columns = features.columns.to_list()
    # print(f"Data Type for features_scaled: {type(features_scaled).__name__}")
    return features_scaled


# -----------------------------------------------------------------------------

# STATISTICAL ANALYSIS


def apply_pca(n_components, features_scaled):
    """Run a Principal Component Analysis (PCA) on scaled data.

    The output is of type array.

    Args:
        n_components (_type_): _description_
        features_scaled (_type_): _description_

    Returns:
        _type_: _description_
    """
    pca_model = PCA(n_components, random_state=42)
    pca_array = pca_model.fit_transform(features_scaled)
    return pca_model, pca_array


def explain_pca_variance(pca_model, pca_components):
    """Build a dataframe of variance explained.

    The variable dataframe 'variance_explained_df' is created and its index is
    renamed for better comprehension of the output

    Args:
        pca_model (_type_): _description_
        pca_components (_type_): _description_

    Returns:
        _type_: _description_
    """
    variance_explained = pca_model.explained_variance_ratio_ * 100
    variance_explained_cumulated = np.cumsum(
        pca_model.explained_variance_ratio_) * 100
    variance_explained_df = pd.DataFrame(
        data=[variance_explained, variance_explained_cumulated],
        columns=pca_components,
    ).rename(index={0: "percent_variance_explained", 1: "cumulated_percent"})
    print(f"\nVariance explained by the PCA:\n{variance_explained_df}")
    return variance_explained_df, variance_explained_cumulated


def apply_anova(dataframe, dependent_variable, independent_variable):
    """Perform Analysis of Variance using Ordinary Least Squares (OLS) model.

    Args:
        dataframe (_type_): _description_
        dependent_variable (_type_): _description_
        independent_variable (_type_): _description_

    Returns:
        _type_: _description_
    """
    # Fit the model
    model = ols(
        formula=f"{dependent_variable} ~ C({independent_variable})",
        data=dataframe
    ).fit()

    # Display ANOVA table
    anova_table = sm.stats.anova_lm(model, typ=1)
    print(
        f"""
        \n\nANOVA Test Table for {dependent_variable} - {independent_variable}:
        {anova_table}\n
        """
    )
    return model, anova_table


def apply_manova(dataframe, formula):
    """Perform Multiple Analysis of Variance using OLS model.

    In this instance, it is better to give the formula of the model.

    Args:
        dataframe (_type_): _description_
        formula (_type_): _description_

    Returns:
        _type_: _description_
    """
    # Run MANOVA test
    model = MANOVA.from_formula(
        formula=formula,
        data=dataframe
    )
    manova_test = model.mv_test()
    return manova_test


def check_normality_assumption(data, alpha=0.05):
    """Check assumptions for normality of data.

    Use the following 'normality' tests:
        - Shapiro_Wilk
        - Normality
        - Jarque-Bera (kurtosis and skewness together)

    Then, concatenate the test outputs into a dataframe.

    The parameter 'data' can take a 1D-array (e.g. model output) or a
    dataframe and one of its column (e.g. dataframes[column]).

    Args:
        data (_type_): _description_
        alpha (float, optional): _description_. Defaults to 0.05.

    Returns:
        _type_: _description_
    """
    shapiro_wilk = pg.normality(
        data=data,
        method="shapiro",
        alpha=alpha
    )
    shapiro_wilk.rename(index={0: "shapiro_wilk"}, inplace=True)

    normality = pg.normality(
        data=data,
        method="normaltest",
        alpha=alpha
    )
    normality.rename(index={0: "normality"}, inplace=True)

    jarque_bera = pg.normality(
        data=data,
        method="jarque_bera",
        alpha=alpha
    )
    jarque_bera.rename(index={0: "jarque_bera"}, inplace=True)

    # Concatenate the tests output
    normality_tests = pd.concat(
        objs=[shapiro_wilk, normality, jarque_bera],
        axis=0,
    )
    normality_tests.rename(
        columns={"W": "statistic", "pval": "p-value"},
        inplace=True
    )
    print(f"Normality Tests Results:\n{normality_tests}\n")

    # # BUG Below NOT working
    # # Print a message depending on the value ('True' or 'False') of the
    # # 'jarque_bera' output
    # print("Normal Distribution of data ?")
    # if normality_tests.iloc[2, 2] == "False":
    #     print("The data are NOT normally distributed.")
    # elif normality_tests.iloc[2, 2] is True:
    #     print("The data are normally distributed.")
    return normality_tests


def check_equal_variance_assumption(
    data,
    dependent_variable,
    group,
    alpha=0.05
):
    """Check assumption of equal variance between groups.

    Use the following 'equality of variance' tests:
        - Bartlett
        - Levene

    Then, concatenate the test outputs into a dataframe.

    The parameter 'data' can take a 1D-array (e.g. model output) or a
    dataframe and one of its column (e.g. dataframes[column]).

    BUG This is NOT working with the 'data' argument.
        However, it works with 'dataframe' and 'dv'.

    Args:
        model (_type_): _description_

    Returns:
        _type_: _description_
    """
    bartlett = pg.homoscedasticity(
        data=data,
        dv=dependent_variable,
        group=group,
        method="bartlett",
        alpha=alpha
    )
    bartlett.rename(
        index={"mean_intensities": "bartlett"},
        columns={"T": "statistic", "pval": "p-value", "normal": "normal"},
        inplace=True
    )

    levene = pg.homoscedasticity(
        data=data,
        dv=dependent_variable,
        group=group,
        method="levene",
        alpha=alpha
    )
    levene.rename(
        index={"mean_intensities": "levene"},
        columns={"W": "statistic", "pval": "p-value", "normal": "normal"},
        inplace=True
    )

    # Concatenate the tests output
    equal_variance_tests = pd.concat(
        objs=[bartlett, levene],
        axis=0,
    )
    print(
        f"""
        \nEquality of Variance Tests Results for {dependent_variable} - {group}
        \n{equal_variance_tests}\n
        """
    )

    # # BUG Below NOT working
    # print("Equal Variance of Data Between Groups:")
    # if levene["equal_var"] is False:
    #     print("The data do NOT have equal variance between groups.")
    # elif levene["equal_var"] is True:
    #     print("The data have equal variance between groups")
    return equal_variance_tests


def run_tukey_post_hoc_test(dataframe, dependent_variable, group_list):
    """Use Tukey's HSD post-hoc test for multiple comparison between groups.

    It produces a table of significant differences.

    Args:
        dataframe (_type_): _description_
        dependent_variable (_type_): _description_
        group (_type_): _description_

    Returns:
        _type_: _description_
    """
    tukey = pg.pairwise_tukey(
        data=dataframe,
        dv=dependent_variable,
        between=group_list
    )
    return tukey


def perform_multicomparison_correction(p_values, method="bonferroni"):
    """Apply the Bonferroni correction method to the p-values.

    Args:
        p_values (_type_): _description_
        method (str, optional): _description_. Defaults to "bonferroni".

    Returns:
        _type_: _description_
    """
    reject, p_values_corrected = pg.multicomp(
        pvals=p_values,
        method=method
    )
    # Concatenate the two arrays together
    concatenate_arrays = np.column_stack((reject, p_values_corrected))
    concatenate_arrays.astype(np.bool)

    correction_dataframe = pd.DataFrame(concatenate_arrays)
    correction_dataframe.rename(
        columns={0: "reject_hypothesis", 1: "corrected_p_values"},
        inplace=True
    )
    return correction_dataframe


# -----------------------------------------------------------------------------

# DATA VISUALISATION


def draw_scree_plot(x_axis, y_axis):
    """Draw scree plot following a PCA.

    Args:
        x (_type_): _description_
        y (_type_): _description_

    Returns:
        _type_: _description_
    """
    scree_plot = sns.lineplot(
        x=x_axis,
        y=y_axis,
        color='black',
        linestyle='-',
        linewidth=2,
        marker='o',
        markersize=8,
    )
    return scree_plot


def draw_scatterplot(
    dataframe,
    x_axis,
    y_axis,
    size=None,
    hue=None,
    palette=None,
):
    """Draw a scatter plot of TWO variables.

    Use the parameter 'size' to choose a grouping variable that will produce
    points with different sizes.

    Args:
        dataframe (_type_): _description_
        x (_type_): _description_
        y (_type_): _description_
        hue (_type_): _description_
        size (_type_): _description_
        palette (_type_): _description_

    Returns:
        _type_: _description_
    """
    scatterplot = sns.scatterplot(
        data=dataframe,
        x=x_axis,
        y=y_axis,
        hue=hue,
        palette=palette,
        size=size,
        sizes=(20, 200),
        legend="full",
        s=100,  # size of the markers
    )
    return scatterplot


def draw_heatmap(data, xticklabels, yticklabels):
    """Draw a heat map of ONE variable.

    Args:
        data (_type_): _description_
        xticklabels (_type_): _description_
        yticklabels (_type_): _description_

    Returns:
        _type_: _description_
    """
    heatmap = sns.heatmap(
        data=data,
        xticklabels=xticklabels,
        yticklabels=yticklabels,
        vmin=0,
        vmax=1,
        annot=True,
        fmt=".0%",
        square=True,
        linewidths=0.2,
        cbar=True,
        cbar_kws={
            "orientation": "vertical",
            "shrink": .8
        },
        cmap="YlGnBu",
    )
    heatmap.set_xticklabels(
        labels=heatmap.get_xticklabels(),
        rotation=45,
        ha="right"
    )
    return heatmap


def draw_kdeplot(
    dataframe,
    x_axis,
    xlabel,
    ylabel=None,
    hue=None,
    palette=None,
):
    """Draw a density curve of data distribution.

    Args:
        dataframe (_type_): _description_
        x (_type_): _description_
        xlabel (_type_): _description_
        ylabel (_type_): _description_
        hue (_type_, optional): _description_. Defaults to None.
        palette (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    kdeplot = sns.kdeplot(
        data=dataframe,
        x=x_axis,
        hue=hue,
        palette=palette,
        legend=True,
    )
    kdeplot.set(
        xlabel=xlabel,
        ylabel=ylabel,
    )
    return kdeplot


def draw_kdeplot_subplots(
    dataframe,
    x_axis,
    item_list,
    nb_columns,
    xlabel,
    ylabel=None,
    hue=None,
    palette=None
):
    """Draw several bar plots on the same image using an iterable.

    This means the figure will contains m rows and n columns of barplots.

    Depending on the data to be summarise/displayed, it can be very fiddly to
    choose the write amount of space between plots and ensure that everything
    is displayed properly. In these conditions, better use the
    'plt.figure(figsize=(rows, columns))' settings and the
    'plt.subplots_adjust()' parameter and play around with the 'wspace' (width
    space) and 'hspace' (height space) values.

    The layout of the subplots is dynamically changed depending on the number
    of columns desired.
    For example, if the iterable contains 9 items & the number of columns is
    set to 3, then the size of the subplot matrix will be 3 rows by 3columns.
    Similarly, if one wants 2 columns, the size of the plot array will be 4*3,
    hence their will be 4 rows by 2 columns (that is 8 plots), plus a 5th row
    with only 1 plot and 1 more 'blank'.

    NOTE: This function calls the function 'draw_barplot()' function defined
    earlier.

    Args:
        dataframe (_type_): _description_
        x_axis (_type_): _description_
        item_list (_type_): _description_
        nb_columns (_type_): _description_
        errorbar (str, optional): _description_. Defaults to "ci".
        palette (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    plt.subplots_adjust(wspace=0.3, hspace=0.4)

    # set number of columns
    ncols = nb_columns

    # calculate number of corresponding rows
    nrows = len(item_list) // ncols + \
        (len(item_list) % ncols > 0)

    # loop through the length of tickers and keep track of index
    for index, item in enumerate(item_list):
        # add a new subplot iteratively using nrows and cols
        axis = plt.subplot(
            nrows,
            ncols,
            index + 1,  # indexing starts from 0
            # sharex=True
        )
        # fig.tight_layout(pad=2)

        # filter df and plot ticker on the new subplot axis
        kdeplot_subplots = draw_kdeplot(
            dataframe=dataframe,
            x_axis=x_axis,
            xlabel=xlabel,
            hue=hue,
            palette=palette
        )
        kdeplot_subplots.set(
            xlabel=xlabel,
            ylabel=ylabel,
            title=item.upper()
        ),

        # # chart formatting
        # axis.set_title(item.upper())
        # axis.set_xticklabels(
        #     labels=kdeplot_subplots.get_xticklabels(),
        #     size=10,
        #     rotation=45,
        #     ha="right"
        # )
        axis.set_xlabel("")
    return kdeplot_subplots


def draw_boxplot(
    dataframe,
    x_axis,
    y_axis,
    hue=None,
    palette=None
):
    """Draw a boxplot of data distribution.

    Args:
        dataframe (_type_): _description_
        x (_type_): _description_
        y (_type_): _description_
        hue (_type_, optional): _description_. Defaults to None.
        palette (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    boxplot = sns.boxplot(
        data=dataframe,
        x=x_axis,
        y=y_axis,
        hue=hue,
        palette=palette,
        orient="h",
    )
    return boxplot


def draw_barplot(
    dataframe,
    x_axis,
    y_axis,
    errorbar="ci",
    palette=None
):
    """Draw a barplot which can include colour bars for treatment options.

    The default error bar is set to 'ci' for 'confidence interval'.
    'sd' or 'se' can be used in place for, respectively, 'standard deviation'
    and 'standard error'.

    Args:
        dataframe (_type_): _description_
        x (_type_): _description_
        y (_type_): _description_
        errorbar (str, optional): _description_. Defaults to "ci".
        palette (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    barplot = sns.barplot(
        data=dataframe,
        x=x_axis,
        y=y_axis,
        errorbar=errorbar,
        palette=palette
    )
    return barplot


def draw_barplot_subplots(
    dataframe,
    x_axis,
    item_list,
    nb_columns,
    errorbar="ci",
    palette=None
):
    """Draw several bar plots on the same image using an iterable.

    This means the figure will contains m rows and n columns of barplots.

    Depending on the data to be summarise/displayed, it can be very fiddly to
    choose the write amount of space between plots and ensure that everything
    is displayed properly. In these conditions, better use the
    'plt.figure(figsize=(rows, columns))' settings and the
    'plt.subplots_adjust()' parameter and play around with the 'wspace' (width
    space) and 'hspace' (height space) values.

    The layout of the subplots is dynamically changed depending on the number
    of columns desired.
    For example, if the iterable contains 9 items & the number of columns is
    set to 3, then the size of the subplot matrix will be 3 rows by 3columns.
    Similarly, if one wants 2 columns, the size of the plot array will be 4*3,
    hence their will be 4 rows by 2 columns (that is 8 plots), plus a 5th row
    with only 1 plot and 1 more 'blank'.

    NOTE: This function calls the function 'draw_barplot()' function defined
    earlier.

    Args:
        dataframe (_type_): _description_
        x (_type_): _description_
        item_list (_type_): _description_
        nb_columns (_type_): _description_
        errorbar (str, optional): _description_. Defaults to "ci".
        palette (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    plt.subplots_adjust(wspace=0.3, hspace=0.4)

    # set number of columns
    ncols = nb_columns

    # calculate number of corresponding rows
    nrows = len(item_list) // ncols + \
        (len(item_list) % ncols > 0)

    # loop through the length of tickers and keep track of index
    for index, item in enumerate(item_list):
        # add a new subplot iteratively using nrows and cols
        axis = plt.subplot(
            nrows,
            ncols,
            index + 1,  # indexing starts from 0
            # sharex=True
        )
        # fig.tight_layout(pad=2)

        # filter df and plot ticker on the new subplot axis
        barplot_subplots = draw_barplot(
            dataframe=dataframe,
            x_axis=x_axis,
            y_axis=item,
            errorbar=errorbar,
            palette=palette
        )
        barplot_subplots.set(
            xlabel="",
            ylabel="Transmissivité \u03C4 (-)",
            title=item.upper()
        ),
        axis.set_xticklabels(
            labels=barplot_subplots.get_xticklabels(),
            size=10,
            rotation=45,
            ha="right"
        )
        # axis.set_xlabel("")
    return barplot_subplots


def draw_correlation_heatmap(dataframe, method="pearson"):
    """Draw a correlation matrix between variables.

    Also, add a mask to be applied to the 'upper triangle'.

    Args:
        dataframe (_type_): _description_
        method (str, optional): _description_. Defaults to "pearson".

    Returns:
        _type_: _description_
    """
    correlation_matrix = dataframe.corr(method=method)

    # Generate a mask for the upper triangle
    # Use 'ones_like' to include the diagonal of ones
    mask = np.ones_like(correlation_matrix, dtype=np.bool)
    # Return the indices for the lower-triangle with 'tril'
    mask[np.tril_indices_from(mask)] = False  # Set to True to remove display

    # Draw the heatmap with the mask
    correlation_heatmap = sns.heatmap(
        data=correlation_matrix,
        mask=mask,
        vmin=-1,
        vmax=1,
        annot=True,
        annot_kws={"size": 10},
        square=True,
        cbar=True,
        cmap="coolwarm",
        center=0
    )
    correlation_heatmap.set_xticklabels(
        labels=correlation_heatmap.get_xticklabels(),
        rotation=45,
        ha="right"
    )
    correlation_heatmap.tick_params(left=True, bottom=True)
    return correlation_matrix, correlation_heatmap


def draw_qqplot(
    data,
    confidence=0.95
):
    """Draw the Q-Q plot using the residuals of fitted model, for example.

    The parameter 'data' can take a 1D-array (e.g. model output) or a
    dataframe and one of its column (e.g. dataframes[column]).

    Args:
        data (_type_): _description_
        confidence (float, optional): _description_. Defaults to 0.95.

    Returns:
        _type_: _description_
    """
    qqplot = pg.qqplot(x=data, dist="norm", confidence=confidence)
    return qqplot


def draw_anova_quality_checks(
    dataframe,
    dependent_variable,
    independent_variable,
    model
):
    """Draw Q-Q plots and run post-hoc tests on selected variables."""
    # Q-Q plot of the dependent variable residuals
    draw_qqplot(
        data=model.resid,
        confidence=0.95
    )
    plt.title(
        label=f"""
        Q-Q Plot of Model Residuals ({dependent_variable} - \
{independent_variable})
        """,
        fontsize=14,
        loc="center"
    )
    # Save figure
    save_figure(
        figure_name=f"qqplot_anova_{independent_variable}_{dependent_variable}"
    )
    # plt.show()

    # -------------------------------------------------------------------------

    # Run post-hoc test
    comparison = mc.MultiComparison(
        dataframe[dependent_variable],
        dataframe[independent_variable],
    )
    tukey = comparison.tukeyhsd(alpha=0.01)
    print(f"\nTukey's Test Output:\n{tukey.summary()}\n")

    tukey.plot_simultaneous(
        ylabel="Defects",
        xlabel="Score Difference"
    )
    plt.suptitle(
        t=f"""
        Tukey's HSD Post-hoc Test on {dependent_variable} for \
{independent_variable}
        """,
        fontsize=14
    )
    save_figure(
        figure_name=f"tukey_anova_{independent_variable}_{dependent_variable}"
    )
    # plt.show()

    tukey_post_hoc_test = run_tukey_post_hoc_test(
        dataframe=dataframe,
        dependent_variable=dependent_variable,
        group_list=independent_variable
    )
    # Apply p-values correction
    corrected_tukey_dataframe = (
        perform_multicomparison_correction(
            p_values=tukey_post_hoc_test["p-tukey"],
        )
    )
    # Add output to 'tukey_post_hoc_test' dataframe
    corrected_tukey_post_hoc_test = pd.concat(
        objs=[tukey_post_hoc_test, corrected_tukey_dataframe], axis=1
    )
    print(corrected_tukey_post_hoc_test)
    print("\n==============================================================\n")
    return corrected_tukey_post_hoc_test
