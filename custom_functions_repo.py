"""Analysis of the detection parameters for defects/chewing gums.

CUSTOM FUNCTIONS

Definition of the functions customised for the Rockwool analyses.

Content Sections:
    - INPUT/OUTPUT
    - ARRAYS
    - CONVERT OBJECTS
    - DATA MANIPULATION
    - EXPLORATORY DATA ANALYSIS
    - STATISTICAL ANALYSIS
    - DATA VISUALISATION
    - DATA MODELLING (MACHINE LEARNING)
"""
# Call the libraries required
# import glob
import sys
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple
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
from statsmodels.formula.api import ols
from statsmodels.multivariate.manova import MANOVA
from statsmodels.stats.multicomp import MultiComparison

# Handle constant/duplicates and missing features/columns
from feature_engine.selection import (
    DropFeatures,
    DropConstantFeatures,
    DropDuplicateFeatures,
)

# Assemble pipeline(s)
from sklearn import set_config, tree
from sklearn.compose import (
    ColumnTransformer,
    make_column_selector as selector
)
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold, RFECV
from sklearn.impute import SimpleImputer

from sklearn.model_selection import (
    # cross_validate,
    cross_val_score,
    GridSearchCV,
    RandomizedSearchCV,
    RepeatedStratifiedKFold,
    StratifiedShuffleSplit,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
    MinMaxScaler,
)
from sklearn.tree import plot_tree, DecisionTreeClassifier

# Sampling
from fast_ml.model_development import train_valid_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# Models
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    multilabel_confusion_matrix,
    f1_score,
    roc_curve,
    roc_auc_score,
)
from sklearn.inspection import permutation_importance

# Uncomment next import when '.set_output(transform="pandas")' is fixed
# from sklearn import set_config

# set_config(transform_output="pandas")


# ----------------------------------------------------------------------------

# INPUT/OUTPUT


def get_folder_name_list_from_directory(
    directory_name: str | Path
) -> List[str]:
    """Get the list of file (name) from a directory.

    Args:
        directory_name (_type_): _description_
        extension (_type_): _description_

    Returns:
        _type_: _description_
    """
    paths = Path(directory_name).glob(pattern="*")
    folder_name_list = [str(path) for path in paths if path.is_dir()]
    return folder_name_list


def get_file_name_list_from_extension(
    directory_name: str | Path,
    extension: str
) -> List[str]:
    """Get the list of file (name) from a directory.

    Args:
        directory_name (_type_): _description_
        extension (_type_): _description_

    Returns:
        _type_: _description_
    """
    paths = Path(directory_name).glob(pattern=f"**/*.{extension}")
    file_name_list = [str(path) for path in paths]
    return file_name_list


def load_pickle_file(file_path_name: str | Path) -> pd.DataFrame:
    """Load the pickle file into a dataframe.

    Args:
        file_path_name (str  |  Path]): _description_

    Returns:
        pd.DataFrame: _description_
    """
    dataframe = pd.read_pickle(filepath_or_buffer=file_path_name)
    print("\nDataset details:\n")
    print(dataframe.info())
    return dataframe


def load_mat_file(mat_file_name: str) -> Dict[str, float]:
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
    nrows: int = 0,
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
    csv_path_name: str
) -> None:
    """Save a dataframe as a '.csv()' file.

    Args:
        dataframe (_type_): _description_
        file_name (_type_): _description_
    """
    # Save the EDA ANOVA output for EACH defect category
    dataframe.to_csv(
        path_or_buf=csv_path_name,
        sep=",",
        encoding="utf-8",
        index=True,
    )
    # return


def save_excel_file(
    dataframe: pd.DataFrame,
    excel_file_name: str
) -> None:
    """Save the dataframe to an MS Excel '.xlsx' file.

    Args:
        dataframe (_type_): _description_
        file_name (_type_): _description_
    """
    dataframe.to_excel(
        excel_writer=f"./output/{excel_file_name}.xlsx"
    )
    # return


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
    # return


def save_image_show(
    image: np.ndarray,
    image_title: str,
    save_image_name: str,
):
    """Image _summary_.

    Args:
        image (np.ndarray): _description_
        image_title (str): _description_
    """
    image = plt.imshow(
        X=image,
        cmap="Greens", vmin=0, vmax=1,
        # cmap='prism', vmin=0, vmax=255,
    )
    # # Testing the 'matshow()' function
    # plt.matshow(image)

    # # Using 'seaborn' library
    # cmap = ListedColormap(sns.color_palette("Spectral", 256))
    # image_plot2 = sns.heatmap(image, cmap=cmap)

    plt.title(label=f"{image_title}\n", fontsize=14)
    plt.grid(visible=False)
    # plt.axis("off")
    plt.tight_layout()
    plt.show()
    save_figure(figure_name=save_image_name)
    return image


def save_npz_file(
    file_name: str | Path,
    data_array: np.ndarray,
):
    """save_npz _summary_.

    Args:
        file_name (np.ndarray): _description_
        data_array (str): _description_
        save_image_name (str): _description_
    """
    np.savez(file=file_name, data=data_array)
    return


def save_pickle_file(dataframe: pd.DataFrame, file_path_name: str):
    """Save the dataframe as a pickle object.

    Args:
        dataframe (pd.DataFrame): _description_
        file_path_name (str): _description_
    """
    dataframe.to_pickle(path=file_path_name, protocol=-1)
    return


def save_console_output(file_name: str):
    """Save the console output."""
    # Save the original stdout
    original_stdout = sys.stdout
    # Open a file for writing
    with open(file=file_name, mode="x", encoding="utf-8") as output_file:
        # Redirect stdout to the file
        sys.stdout = output_file
        yield
        # Reset stdout back to the original
        sys.stdout = original_stdout


# # Use the function
# with save_console_output('console_output.txt'):
#     print('This text will be saved to the file.')
#     print('This text will also be saved to the file.')

# print('This text will be printed to the console.')


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


def extract_pattern_from_file_name(file_name: str) -> List[str]:
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


# ----------------------------------------------------------------------------

# CONVERT OBJECTS


def convert_data_to_dataframe(
    data: Dict[int | str, float] | np.ndarray
) -> pd.DataFrame:
    """Convert an array or dictionary to a dataframe.

    Args:
        data (_type_): _description_
        Note: include list ?

    Returns:
        _type_: _description_
    """
    dataframe = pd.DataFrame(data=data)
    return dataframe


# Merge with above ?
def convert_list_to_dataframe(
    items_list: List[str],
    # column_names: List[str]
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


def convert_to_datetime_type(
    dataframe: pd.DataFrame,
    datetime_variable_list: List[str]
) -> pd.DataFrame:
    """date_time_variables _summary_.

    Args:
        dataframe (_type_): _description_

    Returns:
        _type_: _description_
    """
    # Convert data to their proper type
    dataframe[datetime_variable_list] = (
        dataframe[datetime_variable_list].astype("datetime64[ns]")
    )
    return dataframe


def convert_to_category_type(
    dataframe: pd.DataFrame,
    category_variable_list: List[str]
) -> pd.DataFrame:
    """category_variables _summary_.

    Args:
        processed_data (_type_): _description_

    Returns:
        _type_: _description_
    """
    dataframe[category_variable_list] = (
        dataframe[category_variable_list].astype("category")
    )
    return dataframe


def convert_to_number_type(
    dataframe: pd.DataFrame,
    numeric_variable_list: List[str]
) -> pd.DataFrame:
    """category_variables _summary_.

    Args:
        processed_data (_type_): _description_

    Returns:
        _type_: _description_
    """
    dataframe[numeric_variable_list] = (
        dataframe[numeric_variable_list].astype("number")
    )
    return dataframe


def convert_to_integer_type(
    dataframe: pd.DataFrame,
    integer_variable_list: List[str]
) -> pd.DataFrame:
    """integer_variables _summary_.

    Args:
        processed_data (_type_): _description_

    Returns:
        _type_: _description_
    """
    dataframe[integer_variable_list] = (
        dataframe[integer_variable_list].astype("int")
    )
    return dataframe


def convert_to_float_type(
    dataframe: pd.DataFrame,
    float_variable_list: List[str]
) -> pd.DataFrame:
    """integer_variables _summary_.

    Args:
        processed_data (_type_): _description_

    Returns:
        _type_: _description_
    """
    dataframe[float_variable_list] = (
        dataframe[float_variable_list].astype("float")
    )
    return dataframe


def convert_to_string_type(
    dataframe: pd.DataFrame,
    string_variable_list: List[str]
) -> pd.DataFrame:
    """string_variables _summary_.

    Args:
        processed_data (_type_): _description_

    Returns:
        _type_: _description_
    """
    dataframe[string_variable_list] = (
        dataframe[string_variable_list].astype("string")
    )
    return dataframe


def convert_to_proper_types(
    dataframe: pd.DataFrame,
    datetime_variable_list: List[str] = None,
    category_variable_list: List[str] = None,
    numeric_variable_list: List[str] = None,
    integer_variable_list: List[str] = None,
    string_variable_list: List[str] = None,
) -> Tuple[pd.DataFrame]:
    """convert_to_proper_types _summary_.

    Args:
        dataframe (pd.DataFrame): _description_
        datetime_variable_list (List[str], optional): _description_.
        Defaults to None.
        category_variable_list (List[str], optional): _description_.
        Defaults to None.
        numeric_variable_list (List[str], optional): _description_.
        Defaults to None.
        integer_variable_list (List[str], optional): _description_.
        Defaults to None.
        string_variable_list (List[str], optional): _description_.
        Defaults to None.

    Returns:
        Tuple[pd.DataFrame]: _description_
    """
    # Convert variables to proper type
    datetime_features = convert_to_datetime_type(
        dataframe=dataframe,
        datetime_variable_list=datetime_variable_list
    )

    categorical_features = convert_to_category_type(
        dataframe=dataframe,
        category_variable_list=category_variable_list
    )

    numeric_features = convert_to_number_type(
        dataframe=dataframe,
        numeric_variable_list=numeric_variable_list
    )

    integer_features = convert_to_integer_type(
        dataframe=dataframe,
        integer_variable_list=integer_variable_list
    )

    string_features = convert_to_string_type(
        dataframe=dataframe,
        string_variable_list=string_variable_list
    )
    return (
        datetime_features,
        categorical_features,
        numeric_features,
        integer_features,
        string_features,
    )

# -----------------------------------------------------------------------------

# DATA MANIPULATION


def concatenate_dataframes(
    dataframe_list: List[str],
    axis: str | int,
) -> pd.DataFrame:
    """Concatenate the dataframes from the various processing steps.

    Use either 'axis="horizontal"' or 'axis=1' for an analysis column-wise
    or 'axis="vertical"' or 'axis=0' for an analysis row-wise.

    Args:
        dataframe_list (_type_): _description_

    Returns:
        _type_: _description_
    """
    concatenated_dataframes = pd.concat(objs=dataframe_list, axis=axis)
    return concatenated_dataframes


def remove_column_based_on_list(
    # column,  # delete ?
    column_list: List[str],
    column_to_remove: List[str],
) -> List[str]:  # to be checked; not dataframe ?
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


def remove_duplicated_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """remove_duplicated_columns _summary_.

    The function 'duplicated()' is applied to the dataframe columns and only
    the 'first' appearance is kept.

    Args:
        dataframe (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
    dataframe = dataframe.loc[:, ~dataframe.columns.duplicated(keep="first")]
    return dataframe


def remove_duplicated_rows(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicated values row-wide and reset index.

    Args:
        dataframe (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
    dataframe = (
        dataframe.drop_duplicates().reset_index(drop=True)
    )
    return dataframe


def drop_column(
    dataframe: pd.DataFrame,
    column_list: List[str]
) -> pd.DataFrame:
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


def drop_row_by_value(
        dataframe: pd.DataFrame,
        column_name: str,
        value_name: str | int | float,  # to be checked
) -> pd.DataFrame:
    """Drop rows from a column in the dataframe.

    A list a 'value_name' can be passed with a for-loop.

    Args:
        dataframe (_type_): _description_

    Returns:
        _type_: _description_
    """
    dataframe_reduced = dataframe[dataframe[column_name] != value_name]

    return dataframe_reduced


def select_row_by_query(
    dataframe: pd.DataFrame,
    query_content: str,
) -> pd.DataFrame:
    """select_row_by_value _summary_."""
    dataframe_reduced = dataframe.query(query_content)
    return dataframe_reduced


def get_list_of_unique_values(
        dataframe: pd.DataFrame,
        column_name: str
) -> List[str | int | float]:  # to be checked
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


def get_numeric_features(
    dataframe: pd.DataFrame
) -> Tuple[pd.DataFrame, List[str]]:
    """get_numeric_features _summary_.

    Args:
        dataframe (_type_): _description_

    Returns:
        _type_: _description_
    """
    # Select numeric variables ONLY and make a list
    numerical_features = dataframe.select_dtypes(include=np.number)
    numerical_features_list = numerical_features.columns.to_list()
    print(f"\nList of Numeric Features: \n{numerical_features_list}\n")
    print(numerical_features.describe())
    return numerical_features, numerical_features_list


def get_categorical_features(
    dataframe: pd.DataFrame
) -> Tuple[pd.DataFrame, List[str]]:
    """get_categorical_features _summary_.

    Args:
        dataframe (_type_): _description_

    Returns:
        _type_: _description_
    """
    # Select categorical variables ONLY and make a list
    categorical_features = dataframe.select_dtypes(include="category")
    categorical_features_list = categorical_features.columns.to_list()
    print(f"\nList of Categorical Features: \n{categorical_features_list}\n")
    print(categorical_features.describe())
    return categorical_features, categorical_features_list


def get_dictionary_key(
    dictionary: Dict[int, str],
    target_key_string: str
) -> int:
    """Get the key (as an integer) from the dictionary based on its values.

    Args:
        dictionary (_type_): _description_
        target_string (str): _description_

    Returns:
        int: _description_
    """
    key_list = list(dictionary.keys())
    value_list = list(dictionary.values())
    # print key based on its value
    target_key = value_list.index(target_key_string)
    print(f"\nTarget key: {key_list[target_key_string]}\n")
    return target_key


def get_missing_columns(dataframe: pd.DataFrame) -> pd.DataFrame | pd.Series:
    """Get_missing_columns _summary_.

    Args:
        dataframe (pd.DataFrame): _description_

    Returns:
        [pd.DataFrame | pd.Series]: _description_
    """
    missing_columns = dataframe.columns[dataframe.isna().any()].values[0]
    print(f"\nThe following columns have missing values: {missing_columns}\n")
    return missing_columns


# -----------------------------------------------------------------------------

# EXPLORATORY DATA ANALYSIS


def run_exploratory_data_analysis(dataframe: pd.DataFrame) -> pd.DataFrame:
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
    # Select "all" value in 'include' parameter to include non numeric data
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
        csv_path_name="eda_output"
    )
    print(f"\nExploratory Data Analysis:\n{summary_stats_table}\n")
    return summary_stats_table


# Trying to generate numeric and categorical subsets
def run_exploratory_data_analysis_nums_cats(
    dataframe: pd.DataFrame
) -> pd.DataFrame:
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
    # Select "all" value in 'include' parameter to include non numeric data
    # in the EDA
    summary_stats_nums = dataframe.select_dtypes(
        include="number"
    ).describe(include="all").T

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


def group_by_columns_mean_std(dataframe, by_column_list, column_list):
    """Aggregate column values based on an aggregate function.

    Args:
        dataframe (_type_): _description_
        by_column_list (_type_): _description_
        column_list (_type_): _description_

    Returns:
        _type_: _description_
    """
    dataframe_groups = dataframe.groupby(
        by=by_column_list,
        axis=0,
        as_index=True,
        sort=True,
    ).agg(
        mean=pd.NamedAgg(column=column_list, aggfunc="mean"),
        std=pd.NamedAgg(column=column_list, aggfunc="std"),
        # mad_intensity=("intensity", "mad"),  # NOT working
    ).dropna(
        axis=0,
        how="any",
        # inplace=True,
    )  # Drop rows with NA values to focus on columns with defects
    return dataframe_groups


def calculate_mahalanobis_distance(
    var: np.ndarray,
    data: pd.DataFrame,
    cov=None
):
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
    iqr_outlier_dataframe = (
        dataframe[
            (dataframe[column] > upper_limit) |
            (dataframe[column] < lower_limit)
        ]
    )
    print(
        f"""
        \nTable of outliers for {column} based on IQR value:\n
        {iqr_outlier_dataframe}\n
        """
    )
    return iqr_outlier_dataframe


def get_zscore_outliers(
    dataframe: pd.DataFrame,
    column: str,
    zscore_threshold: int = 3
):
    """Build a dataframe containing outliers details based on their Z-score.

    Args:
        dataframe (pd.DataFrame): _description_
        column (str): _description_
        zscore_threshold (int, optional): _description_. Defaults to 3.

    Returns:
        _type_: _description_
    """
    # Calculate Z-score
    dataframe[f"{column}_zscore"] = np.abs(zscore(a=dataframe[column]))
    # OR
    z_score = np.abs(zscore(a=dataframe[column]))  # or without abs ?

    print(f"\nFor {column}:")
    # Find outliers based on Z-score threshold value
    zscore_outlier_dataframe = (
        dataframe[
            # (dataframe[column].z_score > zscore_threshold) |
            # (dataframe[column].z_score < -zscore_threshold)
            # OR
            (z_score > zscore_threshold) |
            (z_score < zscore_threshold)
        ]
        # OR
        (z_score > zscore_threshold) |
        (z_score < zscore_threshold)
    )
    print(
        f"""
        \nTable of outliers for {column} based on Z-score value:\n
        {zscore_outlier_dataframe}\n
        """
    )
    return zscore_outlier_dataframe


def standardise_features(features):
    """Standardise the features to get them on the same scale.

    When used for Principal Component Analysis (PCA), this MUST be performed
    BEFORE applying PCA.

    NOTE: since the 'StandardScaler()' function from scikit-learn library
    produces a Numpy array, the 'convert_data_to_dataframe()' custom function
    is applied to convert the array to a Pandas dataframe.
    As a consequence, the column names must also be renamed using the selected
    feature names.

    Args:
        feature_list (_type_): _description_

    Returns:
        _type_: _description_

    TODO Why is '.set_output(transform="pandas")' not working ?!
        Once fixed, 'features_scaled = convert_data_to_dataframe
        (features_scaled)' can be deleted.
    """
    # Standardise the features
    scaler = StandardScaler()
    # BUGcBelow NOT working
    # scaler = StandardScaler().set_output(transform="pandas")  # NOT working
    # print(f"Data Type for scaler: {type(scaler).__name__}")
    features_scaled = scaler.fit_transform(features)
    # Delete next function call when .set_output(transform="pandas") is fixed
    features_scaled = convert_data_to_dataframe(features_scaled)
    features_scaled.columns = features.columns.to_list()
    # print(f"Data Type for features_scaled: {type(features_scaled).__name__}")
    return features_scaled


def remove_low_variance_features(
    features: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    """Remove variables with low variance.

    Args:
        features (pd.DataFrame): _description_
        threshold (float, optional): _description_

    Returns:
        pd.DataFrame: _description_
    """
    var_thres = VarianceThreshold(threshold=threshold)
    _ = var_thres.fit(features)
    # Get a boolean mask
    mask = var_thres.get_support()
    # Subset the data
    features_reduced = features.loc[:, mask]
    print("\nThe following features were retained:")
    print(f"{features_reduced.columns}")
    return features_reduced


def identify_highly_correlated_features(
    dataframe: pd.DataFrame,
    correlation_threshold: float = 0.80,
) -> List[str]:
    """Identify highly correlated features.

    Args:
        dataframe (pd.DataFrame): _description_
        correlation_threshold (float, optional): _description_.
        Defaults to 0.80.

    Returns:
        List[str]: _description_
    """
    # Compute correlation matrix with absolute values
    correlation_matrix = dataframe.corr(method="pearson").abs()

    # Create a boolean mask
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

    # Subset the matrix
    reduced_correlation_matrix = correlation_matrix.mask(cond=mask)

    # Find cols that meet the threshold
    features_to_drop = [
        feature for feature in reduced_correlation_matrix.columns
        if any(reduced_correlation_matrix[feature] > correlation_threshold)
    ]
    print(
        f"\nThere are {len(features_to_drop)} features to drop due to high \
            correlation:\n{features_to_drop}\n"
    )
    return features_to_drop


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


def check_normality_assumption_residuals(data, alpha=0.05):
    """Check assumptions for normality of model residual data.

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


def check_equal_variance_assumption_residuals(
    dataframe,
    model,
    group_variable,
    alpha=0.05
):
    """Check assumption of equal variance between groups of a model residuals.

    Use the following 'equality of variance' tests:
        - Bartlett
        - Levene

    Then, concatenate the test outputs into a dataframe.

    The parameter 'dataframe' can take a 1D-array (e.g. model output) or a
    dataframe and one of its column (e.g. dataframes[column]).

    Args:
        dataframe (_type_): _description_
        model (_type_): _description_
        group_variable (_type_): _description_
        alpha (float, optional): _description_. Defaults to 0.05.

    Returns:
        _type_: _description_
    """
    # Prepare a dataframe of the ANOVA model residuals
    model_residuals_dataframe = pd.concat(
        [dataframe[group_variable], model.resid], axis=1
    ).rename(columns={0: "residuals"})

    bartlett = pg.homoscedasticity(
        data=model_residuals_dataframe,
        dv="residuals",
        group=group_variable,
        method="bartlett",
        alpha=alpha
    )
    bartlett.rename(
        index={"mean_intensities": "bartlett"},
        columns={"T": "statistic", "pval": "p-value", "normal": "normal"},
        inplace=True
    )

    levene = pg.homoscedasticity(
        data=model_residuals_dataframe,
        dv="residuals",
        group=group_variable,
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
        \nEquality of Variance Tests Results - {group_variable}
        \n{equal_variance_tests}\n
        """
    )
    # BUG
    print(levene.iloc[0]["equal_var"])
    print("Equal Variance of Data Between Groups:")
    if levene.iloc[0]["equal_var"] is False:
        print(
            f"Data do NOT have equal variance between {group_variable} groups."
        )
    else:
        print(f"Data have equal variance between {group_variable} groups.\n")
    return equal_variance_tests


def perform_multicomparison(
    dataframe: pd.DataFrame,
    groups: np.ndarray | pd.Series | pd.DataFrame,
    alpha: float = 0.05
):
    """perform_multicomparison _summary_.

    Args:
        dataframe (pd.DataFrame): _description_
        groups (np.ndarray  |  pd.Series  |  pd.DataFrame]): _description_
        alpha (float, optional): _description_. Defaults to 0.05.
    """
    multicomparison = MultiComparison(
        data=dataframe,
        groups=groups
    )
    tukey_result = multicomparison.tukeyhsd(alpha=alpha)
    print(f"\nMulticomparaison between groups:\n{tukey_result}\n")
    print(f"Unique groups: {multicomparison.groupsunique}\n")


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


def draw_lineplot(x_axis, y_axis):
    """Draw scree plot following a PCA.

    Args:
        x (_type_): _description_
        y (_type_): _description_

    Returns:
        _type_: _description_
    """
    lineplot = sns.lineplot(
        x=x_axis,
        y=y_axis,
        color='black',
        linestyle='-',
        linewidth=2,
        marker='o',
        markersize=8,
    )
    return lineplot


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

        # chart formatting (check it is working ?)
        axis.set_title(item.upper())
        axis.set_xticklabels(
            labels=kdeplot_subplots.get_xticklabels(),
            size=10,
            rotation=45,
            ha="right"
        )
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
    # plt.grid(visible=False)
    # plt.axis("off")
    plt.tight_layout()
    plt.show()
    save_figure(
        figure_name=f"qqplot_anova_{independent_variable}_{dependent_variable}"
    )

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
    # plt.grid(visible=False)
    # plt.axis("off")
    plt.tight_layout()
    plt.show()
    save_figure(
        figure_name=f"tukey_anova_{independent_variable}_{dependent_variable}"
    )

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


# -----------------------------------------------------------------------------

# DATA MODELLING (MACHINE LEARNING)


def target_label_encoder(
    target_selection: List[str | int] | pd.Series
) -> Tuple[np.ndarray, List[str]]:
    """Encode the target labels (usually strings) into integers.

    Args:
        target (List[str  |  int] | pd.Series): _description_

    Returns:
        np.ndarray: _description_
    """
    label_encoder = LabelEncoder()
    target_encoded = label_encoder.fit_transform(target_selection)
    # Convert the encoded labels from a np.ndarray to a list
    target_encoded_list = label_encoder.classes_.tolist()
    print(f"\nList of the target encoded classes:\n{target_encoded_list}\n")
    return target_encoded, target_encoded_list


def train_test_split_pipeline(
    feature_selection: pd.DataFrame,
    target_selection: List[str | int] | pd.Series,
    train_size: float = 0.6
) -> Pipeline:
    """Train_test_split_pipeline _summary_.

    Args:
        feature_selection (pd.DataFrame): _description_
        target_selection (List[str  |  int]  |  pd.Series]): _description_
        test_size (float, optional): _description_. Defaults to 0.6.

    Returns:
        Pipeline: _description_
    """
    train_test_split_pipe = Pipeline(
        steps=[
            ("train_test_split", train_test_split(
                feature_selection,
                target_selection,
                train_size=train_size,
                random_state=42,
                shuffle=True,
                stratify=target_selection,
            )),
        ],
    )
    return train_test_split_pipe


def train_valid_test_split_fast(
    dataframe: pd.DataFrame,
    target: pd.Series,
    train_size: float = 0.6,
    valid_size: float = 0.2,
    test_size: float = 0.2,
    random_state=42,
):
    """Split a dataframe into three sets for training, validation and testing.

    The difference of this function with the 'train_test_split' function from
    the 'scikit-learn' library is that a second split is operated for
    validation purposes, whereas 'train_test_split' needs to be applied twice
    in order to create a validation set.
    The function 'train_valid_test_split' is imported from the 'fast_ml'
    library.

    Args:
        dataframe (pd.DataFrame): _description_
        target (pd.Series): _description_
        train_size (float, optional): _description_. Defaults to 0.6.
        valid_size (float, optional): _description_. Defaults to 0.2.
        test_size (float, optional): _description_. Defaults to 0.2.
        random_state (int, optional): _description_. Defaults to 42.

    Returns:
        _type_: _description_
    """
    (
        features_train,
        target_train,
        features_valid,
        target_valid,
        features_test,
        target_test
    ) = train_valid_test_split(
        df=dataframe,
        target=target,
        train_size=train_size,
        valid_size=valid_size,
        test_size=test_size,
        method="random",
        random_state=random_state,
    )
    return (
        features_train,
        target_train,
        features_valid,
        target_valid,
        features_test,
        target_test
    )


def drop_feature_pipeline(
    feature_selection: pd.DataFrame,
    features_to_keep: List[str]
) -> Pipeline:
    """Drop_feature_pipeline _summary_.

    Args:
        feature_selection (pd.DataFrame): _description_
        features_to_keep (List[str]): _description_

    Returns:
        _type_: _description_
    """
    drop_feature_pipe = Pipeline(
        steps=[
            ("drop_columns", DropFeatures(
                features_to_drop=[
                    feature for feature in feature_selection
                    if feature not in features_to_keep
                ]
            )),
            ("drop_constant_values", DropConstantFeatures(
                tol=1,
                missing_values="ignore"
            )),
            ("drop_duplicates", DropDuplicateFeatures(
                missing_values="ignore"
            )),
            # ("drop_low_variance", VarianceThreshold(threshold=0.03)),
        ],
    )
    print(f"\nPipeline structure:\n{drop_feature_pipe}\n")
    return drop_feature_pipe


def select_numeric_feature_pipeline() -> Pipeline:
    """Create the preprocessing pipeline for numeric data."""
    numeric_feature_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(
                missing_values=np.nan,
                strategy="median",
            )),
            # ("scaler", StandardScaler()),
            # ("scaler", MinMaxScaler()),
            ("scaler", RobustScaler()),
        ],
        verbose=True,
    )
    print(f"\nNumeric Data Pipeline Structure:\n{numeric_feature_pipeline}\n")
    return numeric_feature_pipeline


def select_categorical_feature_pipeline() -> Pipeline:
    """Create the preprocessing pipeline for categorical data."""
    categorical_feature_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(
                strategy="most_frequent",
                fill_value="missing",
            )),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
            # ("encoder", OrdinalEncoder(handle_unknown="ignore")),
        ],
        verbose=True,
    )
    print("\nCategorical Data Pipeline Structure:")
    print(categorical_feature_pipeline)
    return categorical_feature_pipeline


def calculate_model_scores(
    pipeline: Pipeline,
    model_name: str,
    features_train: pd.DataFrame,
    target_train: pd.Series,
    features_test: pd.DataFrame,
    target_test: pd.Series,
    target_pred: pd.Series,
    target_label_list: List[str],
    # ) -> None:
):
    """Calculate_model_scores _summary_.

    Args:
        pipeline (Pipeline): _description_
        model_name (str): _description_
        features_train (pd.DataFrame): _description_
        target_train (pd.Series): _description_
        features_test (pd.DataFrame): _description_
        target_test (pd.Series): _description_
    """
    # Calculate the pipeline train and test scores
    train_score = pipeline.score(X=features_train, y=target_train)
    test_score = pipeline.score(X=features_test, y=target_test)

    # Compute model scores
    accuracy_score_ = accuracy_score(
        y_true=target_test,
        y_pred=target_pred,
    )
    f1_score_ = f1_score(
        y_true=target_test,
        y_pred=target_pred,
        average="weighted"
    )

    # Build a dictionary containing the model scores and convert to a dataframe
    model_score_dictionary = {
        "Model Name": model_name,
        "Train Score": train_score,
        "Test Score": test_score,
        "Model Accuracy Score": accuracy_score_,
        "Model F1-score": f1_score_,
    }
    # Convert the dictionary to a dataframe and drop duplicated rows
    model_score_dataframe = convert_dictionary_to_dataframe(
        model_score_dictionary
    ).set_index(keys="Model Name").drop_duplicates()
    # Format scores as percentages and rotate the df for better viewing
    model_score_dataframe = model_score_dataframe.applymap(
        lambda float_: f"{float_:.1%}"
    ).T
    print(f"\nModel Score Output:\n{model_score_dataframe}\n")

    # Produce a classification report
    classification_report_ = classification_report(
        y_true=target_test,
        y_pred=target_pred,
        target_names=target_label_list,
        # output_dict=True,  # not working inside the function...
    )
    print(f"\nClassification Report:\n{classification_report_}\n")
    return (
        train_score,
        test_score,
        accuracy_score_,
        f1_score_,
        classification_report_,
    )


def draw_confusion_matrix_heatmap(
    target_test: pd.Series,
    target_pred: pd.Series,
    target_label_list: List[str],
    figure_path_name: str,
) -> None:
    """Draw_confusion_matrix_heatmap _summary_.

    Args:
        target_test (pd.Series): _description_
        target_pred (pd.Series): _description_
        target_label_list (List[str]): _description_
        figure_path_name (str): _description_

    Returns:
        _type_: _description_
    """
    # Compute model scores
    accuracy_score_ = accuracy_score(
        y_true=target_test,
        y_pred=target_pred,
    )
    f1_score_ = f1_score(
        y_true=target_test,
        y_pred=target_pred,
        average="weighted"
    )

    # Compute the confusion matrix
    confusion_matrix_ = confusion_matrix(
        y_true=target_test,
        y_pred=target_pred,
        labels=target_label_list,
    )
    print(f"\nConfusion Matrix (Test Set):\n{confusion_matrix_}\n")

    # Show the confusion matrix as a heatmap
    plt.figure()
    sns.heatmap(
        data=confusion_matrix_,
        # Get % of predictions
        # data=confusion_matrix_/np.sum(confusion_matrix_),
        annot=True,
        # fmt=".1%",
        cmap="Greens",
        cbar=False,
        xticklabels=target_label_list,
        yticklabels=target_label_list,
    )
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.title(
        label=(
            f"Confusion matrix of predictions\n \
    Model Accuracy = {accuracy_score_:.1%} - F1-score = {f1_score_:.1%}"),
        fontsize=16
    )
    plt.tight_layout()
    plt.show()
    save_figure(figure_name=figure_path_name)
    # print(confusion_matrix_heatmap)
    return None


# # ????????????
# multilabel_confusion_matrix_ = multilabel_confusion_matrix(
#     y_true=target_test,
#     y_pred=target_pred,
#     labels=target_label_list,
# )
# print(f"\nMulti-Label Confusion Matrix:\n{multilabel_confusion_matrix_}\n")


def perform_roc_auc_analysis(
    target_test: pd.Series,
    target_pred: pd.Series,
) -> float:
    """Compute the ROC curve and AUC score.

    This is NOT suitable for multi-class classification.

    Args:
        target_test (pd.Series): _description_
        target_pred (pd.Series): _description_

    Returns:
        float: _description_
    """
    fpr, tpr, thresholds = roc_curve(y_true=target_test, y_score=target_pred)
    roc_auc_score_ = roc_auc_score(y_true=target_test, y_score=target_pred)
    print(f"\nArea Under the Curve Score:\n{roc_auc_score_}\n")
    return roc_auc_score_
