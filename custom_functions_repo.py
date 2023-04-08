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
import re
# Call the libraries required
# import glob
import sys
from collections import defaultdict
from datetime import datetime
import time
import itertools
import joblib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import colourmap as colourmap
import docx
import dtale
# from matplotlib.colors import ListedColormap
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import missingno as msno
import numpy as np
from numpy import array, array_equal, load
# import openpyxl
import pandas as pd
from pca import pca
import pingouin as pg
import plotly.graph_objs as go
from reportlab.pdfgen import canvas
import researchpy as rp
import scipy as sp
from scipy.io import savemat
import seaborn as sns
import sidetable as stb
import statsmodels.api as sm
# import statsmodels.stats.multicomp as mc
import sweetviz as sv
from yellowbrick.features import PCA as yb_pca, Rank1D
# from yellowbrick.cluster import InterclusterDistance
# Sampling
from fast_ml.model_development import train_valid_test_split
# Handle constant/duplicates and missing features/columns
from feature_engine.selection import (
    DropConstantFeatures,
    DropDuplicateFeatures,
    DropFeatures
)
# from imblearn.over_sampling import SMOTE
# from imblearn.under_sampling import RandomUnderSampler
from pyod.models.mad import MAD
from scipy.io import loadmat
from scipy.stats import chi2, zscore
from scipy import stats
# Assemble pipeline(s)
from sklearn import set_config
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
from sklearn.decomposition import PCA
# from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_selection import (
    # RFECV,
    VarianceThreshold
)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
# from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    # accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve
)
from sklearn.model_selection import (
    # GridSearchCV,
    cross_validate,
    # RandomizedSearchCV,
    # RepeatedStratifiedKFold,
    # StratifiedShuffleSplit,
    cross_val_predict,
    cross_val_score,
    train_test_split
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler
)
from sklearn.tree import (
    DecisionTreeClassifier,
    plot_tree
)
from statsmodels.formula.api import ols
from statsmodels.multivariate.manova import MANOVA
from statsmodels.stats.multicomp import MultiComparison
import treeplot as tree

# Models
# from xgboost import XGBClassifier

set_config(transform_output="pandas")

# Default directories
INPUT_DIRECTORY = Path("./sources")
OUTPUT_DIRECTORY = Path("./output")
OUTPUT_DIR_IMAGES = Path("./images")
OUTPUT_DIR_FIGURES = Path("./figures")


# -----------------------------------------------------------------------------

# DECORATOR FUNCTIONS


# def cache(func: Callable[any, ...]) -> Callable[any, ...]:
def cache(func):
    """AI is creating summary for cache.

    Args:
        func ([type]): [description]

    Returns:
        [type]: [description]
    """
    cached_results = {}

    def wrapper(*args, **kwargs):
        key = str(args) + str(kwargs)
        if key not in cached_results:
            cached_results[key] = func(*args, **kwargs)
        return cached_results[key]

    return wrapper


def timing(func):
    """AI is creating summary for timing.

    Args:
        func ([type]): [description]
    """
    def wrapper(*args, **kwargs):
        # Do something before calling the function
        start_time = time.time()
        result = func(*args, **kwargs)
        # Do something after calling the function
        end_time = time.time()
        print(
            f"\nFunction '{func.__name__}' took {end_time - start_time:.2} "
            f"seconds to run.\n"
        )
        return result
    return wrapper


# -----------------------------------------------------------------------------

# USER INTERFACE


def ask_user_outlier_removal(
        dataframe_with_outliers: pd.DataFrame,
        dataframe_no_outliers: pd.DataFrame
) -> pd.DataFrame:
    """Let the user choose whether to bypass outlier removal.

    This will affect which dataframe is to be used within the next steps.

    Args:
        dataframe_with_outliers (pd.DataFrame): _description_
        dataframe_no_outliers (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
    while True:
        choice = input("Apply Outlier Removal? (y/n): ").lower()
        try:
            assert choice == 'y' or choice == 'n', "Enter 'y' or 'n'"
            break
        except AssertionError as error:
            print(error)
        except ValueError:
            print("Please enter a valid letter choice.")
    if choice == 'y':
        dataframe = dataframe_no_outliers.copy()
        print("\nOutliers were removed.\n")
    else:
        dataframe = dataframe_with_outliers.copy()
        print("\nOutliers were NOT removed.\n")
    return dataframe


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


def load_pickle_file(
    file_name: str,
    output_directory: Path
) -> pd.DataFrame:
    """Load the pickle file into a dataframe.

    Args:
        file_path_name (str  |  Path]): _description_

    Returns:
        pd.DataFrame: _description_
    """
    dataframe = pd.read_pickle(
        filepath_or_buffer=output_directory.joinpath(file_name + ".pkl")
    )
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
    dataframe.to_excel(excel_writer=f"./output/{excel_file_name}.xlsx")


def save_pipeline_model(
    pipeline_name: Pipeline,
    file_name: str,
    output_directory: Path,
) -> Pipeline:
    """save_pipeline_model _summary_.

    Use 'joblib' library and/or 'pickle' / 'feather' format)

    To load the pipeline, use the following command:
    pipeline_name = joblib.load(filename="pipeline_file_name.joblib")

    Args:
        pipeline_name (Pipeline): _description_
        file_name (str): _description_
        output_directory (Path): _description_

    Returns:
        Pipeline: _description_
    """
    pipeline_model_file = joblib.dump(
        value=pipeline_name,
        filename=output_directory.joinpath(file_name + ".joblib"),
    )
    return pipeline_model_file


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
    # plt.show()
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


def save_console_output(file_name: str) -> None:
    """Save the console output.

    BUG NOT working as a function...

    # Use the function as shown below:
    with save_console_output(file_name="console_output.txt"):
        print("This text will be saved to the file.")
        print("This text will also be saved to the file.")
    print("This text will be printed to the console.")

    Args:
        file_name (str): _description_
    """
    # Save the original stdout
    original_stdout = sys.stdout
    # Open a file for writing
    with open(file=file_name, mode="x", encoding="utf-8") as output_file:
        # Redirect stdout to the file
        sys.stdout = output_file
        yield
        # Reset stdout back to the original
        sys.stdout = original_stdout
    print("The console output was saved.")


def convert_text_file_to_docx(
    file_name: str,
    output_directory: Path
) -> None:
    """convert_text_to_docx _summary_.

    Args:
        file_path_name (str): _description_
        output_directory (Path): _description_
    """
    # Open the text file and read its contents
    with open(
            file=output_directory.joinpath(file_name + ".txt"),
            mode="r",
            encoding="utf-8"
    ) as output_file:
        output_text = output_file.read()

    # Create a new Word document
    word_doc = docx.Document()
    # Add the text to the document
    word_doc.add_paragraph(output_text)
    # Save the document as a Word file
    word_doc.save(output_directory.joinpath(file_name + ".docx"))
    print("The console output was converted to a MS Word document file.")


def convert_text_file_to_pdf(
    file_name: str,
    output_directory: Path
) -> None:
    """convert_text_file_to_pdf _summary_.

    Args:
        file_name (str): _description_
        output_directory (Path): _description_
    """
    # Open the text file and read its contents
    with open(
            file=output_directory.joinpath(file_name + ".txt"),
            mode="r",
            encoding="utf-8"
    ) as output_file:
        output_text = output_file.read()

    # Create a new PDF file
    pdf_file = canvas.Canvas(
        filename=f"{file_name}.pdf"
    )
    pdf_file = canvas.Canvas(
        filename=f"{output_directory}/{file_name}.pdf"
        # filename=output_directory.joinpath(file_name + ".pdf")  # BUG
    )

    # Set the font and font size
    pdf_file.setFont("Helvetica", 12)

    # Add the text to the PDF file
    pdf_file.drawString(100, 750, output_text)

    # Save the PDF file
    pdf_file.save()
    # pdf_file.save(output_directory.joinpath(file_name + ".pdf"))
    print("The console output was converted to a PDF document file.")


@timing
def convert_npz_to_mat(input_directory: str | Path) -> None:
    """AI is creating summary for convert_npz_to_mat.

    Args:
        input_directory (str): [description]
    """
    input_directory = input_directory
    output_directory = input_directory

    none_array = array(None)
    NB_ERRORS = 0
    NB_CONVERTED = 0

    npz_file_list = list(input_directory.glob(pattern="*.npz"))

    for input_path in npz_file_list:
        try:
            with load(file=str(input_path), allow_pickle=True) as npz_file:
                data_dict = dict(npz_file)

            for key, value in data_dict.items():
                if array_equal(value, none_array):
                    data_dict[key] = array([])

            output_path = output_directory.joinpath(input_path.stem + ".mat")
            savemat(file_name=str(output_path), mdict=data_dict)
        except Exception:
            NB_ERRORS += 1
        else:
            NB_CONVERTED += 1

    print(f"{NB_ERRORS} error(s) occurred")
    print(f"{NB_CONVERTED} file(s) converted")


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
    """Use regular expressions to extract a pattern from a list of file names.

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
    """Use regular expressions to extract a pattern from a list of file names.

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
    See 'standardise_features' function below for an example.

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
    datetime_variable_list: List[str | datetime]
) -> pd.DataFrame:
    """convert_to_datetime_type _summary_.

    Args:
        dataframe (pd.DataFrame): _description_
        datetime_variable_list (List[str  |  datetime]): _description_

    Returns:
        pd.DataFrame: _description_
    """
    dataframe[datetime_variable_list] = (
        dataframe[datetime_variable_list].astype("datetime64[ns]")
    )
    return dataframe


def convert_to_category_type(
    dataframe: pd.DataFrame,
    category_variable_list: List[str]
) -> pd.DataFrame:
    """convert_to_category_type _summary_.

    Args:
        dataframe (pd.DataFrame): _description_
        category_variable_list (List[str]): _description_

    Returns:
        pd.DataFrame: _description_
    """
    dataframe[category_variable_list] = (
        dataframe[category_variable_list].astype("category")
    )
    return dataframe


def convert_to_number_type(
    dataframe: pd.DataFrame,
    numeric_variable_list: List[str]
) -> pd.DataFrame:
    """convert_to_number_type _summary_.

    Args:
        dataframe (pd.DataFrame): _description_
        numeric_variable_list (List[str]): _description_

    Returns:
        pd.DataFrame: _description_
    """
    dataframe[numeric_variable_list] = (
        dataframe[numeric_variable_list].astype("number")
    )
    return dataframe


def convert_to_integer_type(
    dataframe: pd.DataFrame,
    integer_variable_list: List[str]
) -> pd.DataFrame:
    """convert_to_integer_type _summary_.

    Args:
        dataframe (pd.DataFrame): _description_
        integer_variable_list (List[str]): _description_

    Returns:
        pd.DataFrame: _description_
    """
    dataframe[integer_variable_list] = (
        dataframe[integer_variable_list].astype("int")
    )
    return dataframe


def convert_to_float_type(
    dataframe: pd.DataFrame,
    float_variable_list: List[str]
) -> pd.DataFrame:
    """convert_to_float_type _summary_.

    Args:
        dataframe (pd.DataFrame): _description_
        float_variable_list (List[str]): _description_

    Returns:
        pd.DataFrame: _description_
    """
    dataframe[float_variable_list] = (
        dataframe[float_variable_list].astype("float")
    )
    return dataframe


def convert_to_string_type(
    dataframe: pd.DataFrame,
    string_variable_list: List[str]
) -> pd.DataFrame:
    """convert_to_string_type _summary_.

    Args:
        dataframe (pd.DataFrame): _description_
        string_variable_list (List[str]): _description_

    Returns:
        pd.DataFrame: _description_
    """
    dataframe[string_variable_list] = (
        dataframe[string_variable_list].astype("string")
    )
    return dataframe


def convert_variables_to_proper_type(
    dataframe: pd.DataFrame,
    datetime_variable_list: Optional[List[str | datetime]] = None,
    category_variable_list: Optional[List[str]] = None,
    numeric_variable_list: Optional[List[str]] = None,
    integer_variable_list: Optional[List[str]] = None,
    float_variable_list: Optional[List[str]] = None,
    string_variable_list: Optional[List[str]] = None,
) -> pd.DataFrame:
    """convert_variables_to_proper_type _summary_.

    Apply the '.pipe()' method to the defined functions.

    Args:
        dataframe (pd.DataFrame): _description_
        datetime_variable_list (Optional[List[str  |  datetime]], optional):
        _description_. Defaults to None.
        category_variable_list (Optional[List[str]], optional): _description_.
        Defaults to None.
        numeric_variable_list (Optional[List[str]], optional): _description_.
        Defaults to None.
        integer_variable_list (Optional[List[str]], optional): _description_.
        Defaults to None.
        float_variable_list (Optional[List[str]], optional): _description_.
        Defaults to None.
        string_variable_list (Optional[List[str]], optional): _description_.
        Defaults to None.

    Returns:
        pd.DataFrame: _description_
    """
    processed_dataframe_pipe = dataframe.pipe(
        func=convert_to_datetime_type,
        datetime_variable_list=datetime_variable_list
    ).pipe(
        func=convert_to_category_type,
        category_variable_list=category_variable_list
    ).pipe(
        func=convert_to_number_type,
        numeric_variable_list=numeric_variable_list
    ).pipe(
        func=convert_to_integer_type,
        integer_variable_list=integer_variable_list
    ).pipe(
        func=convert_to_float_type,
        float_variable_list=float_variable_list
    ).pipe(
        func=convert_to_string_type,
        string_variable_list=string_variable_list
    )
    print("\nSummary of Data Types:")
    print(f"\n{processed_dataframe_pipe.info()}\n")
    return processed_dataframe_pipe


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
        value: str | int | float,  # to be checked
) -> pd.DataFrame:
    """Drop rows from a column in the dataframe.

    A list a 'value_name' can be passed with a for-loop.

    Args:
        dataframe (_type_): _description_

    Returns:
        _type_: _description_
    """
    dataframe_reduced = dataframe[dataframe[column_name] != value]

    return dataframe_reduced


def filter_dataframe(
    dataframe: pd.DataFrame,
    filter_content: str,
) -> pd.DataFrame:
    """filter_dataframe _summary_.

    Args:
        dataframe (pd.DataFrame): _description_
        filter_content (str): _description_

    Returns:
        pd.DataFrame: _description_
    """
    print(f"The following filter was applied:\n{filter_content}\n")

    # Apply the filter
    filtered_dataframe = dataframe.query(filter_content)
    print(
        f"Filtered Cluster Dataset Shape:\n{filtered_dataframe.shape}"
    )
    return filtered_dataframe


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


def prepare_scaled_features_encoded_target(
    dataframe: pd.DataFrame,
    target_name: str
) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """Define feature and target dataframes for modelling.

    Args:
        dataframe (pd.DataFrame): _description_
        target (str): _description_

    Returns:
        Tuple[pd.DataFrame, np.ndarray, List[str]]: _description_
    """
    features = dataframe.select_dtypes(include="number")
    features_scaled = standardise_features(features=features)
    target = dataframe[target_name]
    target_encoded, target_class_list = target_label_encoder(
        dataframe=dataframe,
        target_name=target
    )
    return (
        features,
        features_scaled,
        target_encoded,
        target_class_list,
    )


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


def produce_sweetviz_eda_report(
    dataframe: pd.DataFrame,
    eda_report_name: str,
    output_directory: Path
) -> None:
    """Exploratory Data Analysis using the Sweetviz library.

    Produce an '.html' file of the main steps of the EDA
    """
    print("\nPreparing SweetViz Report:\n")
    sweetviz_eda_report = sv.analyze(
        source=dataframe,
        pairwise_analysis="auto"
    )
    sweetviz_eda_report.show_html(
        # filepath=output_directory/"sweetviz_eda_report.html",
        filepath=output_directory.joinpath(eda_report_name + ".html"),
        open_browser=True,
        layout="widescreen"
    )
    return


def compare_sweetviz_eda_report(
    train_set: pd.DataFrame,
    test_set: pd.DataFrame,
    eda_report_name: str,
    output_directory: Path,
) -> None:
    """Exploratory Data Analysis using the Sweetviz library.

    Particularly, this report compares the data split between the training
    and testing sets.

    Produce an '.html' file of the main steps of the EDA
    """
    print("\nPreparing SweetViz Report:\n")
    sweetviz_eda_report = sv.compare(
        [train_set, "Training Data"], [test_set, "Test Data"]
    )
    sweetviz_eda_report.show_html(
        filepath=output_directory.joinpath(eda_report_name + ".html"),
        open_browser=True,
        layout="widescreen"
    )


def produce_dtale_eda_report(dataframe: pd.DataFrame) -> None:
    """Exploratory Data Analysis using the Dtale library.

    Args:
        dataframe (pd.DataFrame): _description_
    """
    # Create a D-Tale instance
    dtale_eda_report = dtale.show(data=dataframe)
    # Launch the report in a web browser
    dtale_eda_report.open_browser()


def get_missing_values_table(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Use 'Sidetable' library to produce a table of metrics on missing values.

    Exclude columns that have 0 missing values using 'clip_0=True' parameter.
    Use parameter 'style=True' to style the display in the output table.

    Args:
        dataframe (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
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
        values (_type_, optional): _description_. Defaults to None.
        index_list (_type_, optional): _description_. Defaults to None.
        column_list (_type_, optional): _description_. Defaults to None.
        aggfunc_list (_type_, optional): _description_. Defaults to None.

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


def apply_mahalanobis_test(
    dataframe: pd.DataFrame,
    alpha: float = 0.01
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run the Mahalanobis test.

    Args:
        dataframe (_type_): _description_
        alpha (float, optional): _description_. Defaults to 0.01.

    Returns:
        _type_: _description_
    """
    # Create a copy to enable row dropping within the loop instead of
    # overwriting the 'original' (here, reduced) dataframe
    mahalanobis_dataframe = dataframe.copy()
    mahalanobis_dataframe["mahalanobis_score"] = (
        calculate_mahalanobis_distance(
            var=mahalanobis_dataframe,
            data=mahalanobis_dataframe
        )
    )

    # Get the critical value for the test
    mahalanobis_test_critical_value = chi2.ppf(
        (1-alpha),
        df=len(mahalanobis_dataframe.columns) - 1
    )
    print(
        f"\n\nMahalanobis Test Critical Value for alpha < {alpha}: \
{mahalanobis_test_critical_value:.2f}"
    )

    # Get p-values from chi2 distribution
    mahalanobis_dataframe["mahalanobis_p_value"] = (
        1 - chi2.cdf(
            mahalanobis_dataframe["mahalanobis_score"],
            df=len(mahalanobis_dataframe.columns) - 1)
    )
    # Select the rows below the alpha threshold
    mahalanobis_outlier_dataframe = mahalanobis_dataframe[
        mahalanobis_dataframe.mahalanobis_p_value < alpha
    ]
    print("\nTable of Outliers based on Mahalanobis Distance:")
    print(mahalanobis_outlier_dataframe)
    return mahalanobis_dataframe, mahalanobis_outlier_dataframe


def remove_mahalanobis_outliers(
    mahalanobis_dataframe: pd.DataFrame,
    mahalanobis_outlier_dataframe: pd.DataFrame
) -> pd.DataFrame:
    """Drop the outliers from the 'mahalanobis_dataframe'.

    Args:
        mahalanobis_dataframe (pd.DataFrame): _description_
        mahalanobis_outlier_dataframe (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
    # Make a list of outliers
    mahalanobis_outlier_list = mahalanobis_outlier_dataframe.index.to_list()

    # Select rows of outliers using 'isin' based on 'mahalanobis_outlier_list'
    # Then, take opposite rows (as it would be 'notin') using '~' on the df
    no_outlier_dataframe = (
        mahalanobis_dataframe[
            ~ mahalanobis_dataframe.index.isin(mahalanobis_outlier_list)
        ]
    )

    # Drop the mahalanobis column as not needed any more
    no_outlier_dataframe.drop(
        labels=["mahalanobis_score", "mahalanobis_p_value"],
        axis=1,
        inplace=True,
    )
    print(f"\nData without outliers:\n{no_outlier_dataframe}\n")
    return no_outlier_dataframe


def get_iqr_outliers(
    dataframe: pd.DataFrame,
    column_name: str
):
    """Build a dataframe containing outliers details.

    Args:
        dataframe (_type_): _description_
        column_name (_type_): _description_

    Returns:
        _type_: _description_
    """
    print("\n==============================================================\n")
    print(f"For {column_name}:")
    # Calculate Q1, Q3
    q_1, q_3 = np.percentile(
        a=dataframe[column_name],
        q=[25, 75]
    )

    # Calculate IQR, upper limit and lower limit
    iqr = q_3 - q_1
    upper_limit = q_3 + 1.5 * iqr
    print(f"Upper Limit of IQR = {upper_limit :.2f}")
    lower_limit = q_1 - 1.5 * iqr
    print(f"Lower Limit of IQR = {lower_limit :.2f}")

    # Find outliers based on upper and lower limit values
    iqr_outlier_dataframe = (
        dataframe[
            (dataframe[column_name] > upper_limit) |
            (dataframe[column_name] < lower_limit)
        ]
    )

    # Isolate the target column from the rest of the dataframe
    iqr_outlier_dataframe = iqr_outlier_dataframe[column_name]
    iqr_outlier_ratio = len(iqr_outlier_dataframe) / len(dataframe)
    print(
        f"There are {len(iqr_outlier_dataframe)} \
({iqr_outlier_ratio:.1%}) IQR outliers."
    )
    # print(f"Table of outliers for {column_name} based on IQR value:")
    # print(iqr_outlier_dataframe)
    return iqr_outlier_dataframe


def get_zscore_outliers(
    dataframe: pd.DataFrame,
    column_name: str,
    zscore_threshold: int = 3
):
    """Build a dataframe containing outliers details based on their Z-score.

    Args:
        dataframe (pd.DataFrame): _description_
        column_name (str): _description_
        zscore_threshold (int, optional): _description_. Defaults to 3.

    Returns:
        _type_: _description_
    """
    print("\n==============================================================\n")
    print(f"For {column_name}:")
    # Calculate Z-score
    z_score = np.abs(zscore(a=dataframe[column_name]))

    # Find outliers based on Z-score threshold value
    zscore_outlier_dataframe = dataframe[z_score > zscore_threshold]

    # Isolate the target column from the rest of the dataframe
    zscore_outlier_dataframe = zscore_outlier_dataframe[column_name]
    zscore_outlier_ratio = len(zscore_outlier_dataframe) / len(dataframe)
    print(
        f"There are {len(zscore_outlier_dataframe)} \
({zscore_outlier_ratio:.1%}) Z-score outliers."
    )
    # print(f"Table of outliers for {column_name} based on Z-score value:")
    # print(zscore_outlier_dataframe)
    return zscore_outlier_dataframe


def get_mad_outliers(
    dataframe: pd.DataFrame,
    column_name: str,
) -> pd.DataFrame:
    """get_mad_outliers _summary_.

    Args:
        dataframe (pd.DataFrame): _description_
        column_name (str): _description_

    Returns:
        pd.DataFrame: _description_
    """
    print("\n==============================================================\n")
    print(f"For {column_name}:")
    # Reshape the target column to make it 2D
    column_2d = dataframe[column_name].values.reshape(-1, 1)
    # Fit to the target column
    mad = MAD().fit(column_2d)

    # Extract the inlier/outlier labels
    labels = mad.labels_
    # print(f"\nMean Absolute Deviation Labels:\n{labels}\n")

    # Extract the outliers
    # use '== 0' to get inliers
    mad_outliers = dataframe[column_name][labels == 1]
    # dataframe["mad_score"] = mad_outliers

    # Isolate the target column from the rest of the dataframe
    mad_outlier_dataframe = dataframe[column_name]
    mad_outlier_dataframe["mad_score"] = mad_outliers
    mad_outlier_ratio = len(mad_outliers) / len(dataframe)
    print(
        f"There are {len(mad_outliers)} \
({mad_outlier_ratio:.1%}) MAD outliers."
    )
    # print(f"Table of outliers for {column_name} based on MAD value:")
    # print(mad_outlier_dataframe)
    return mad_outliers


def concatenate_outliers_with_target_category_dataframe(
    dataframe: pd.DataFrame,
    target_category_list: List[str],
    data_outliers: pd.DataFrame,
    feature: str,
    outlier_method: str,
    output_directory: str | Path,
) -> pd.DataFrame:
    """Concatenate the outliers with their corresponding target categories.

    Args:
        dataframe (pd.DataFrame): _description_
        target_category_list (List[str]): _description_
        data_outliers (pd.DataFrame): _description_
        feature (str): _description_
        outlier_method (str): _description_
        output_directory (str | Path): _description_

    Returns:
        pd.DataFrame: _description_
    """
    outlier_dataframe = pd.merge(
        left=dataframe.loc[:, target_category_list],
        right=data_outliers,
        on="file_name",
        how="right"
    )
    print(
        f"\nTable of outliers for {feature} based on {outlier_method} values:"
    )
    print(outlier_dataframe)

    # Save output as a '.csv()' file
    save_csv_file(
        dataframe=outlier_dataframe,
        csv_path_name=output_directory /
        f"{feature}_{outlier_method}_outliers.csv"
    )
    return outlier_dataframe


def detect_univariate_outliers(
    dataframe: pd.DataFrame,
    target_category_list: List[str],
    output_directory: str | Path
) -> pd.DataFrame:
    """detect_univariate_outliers _summary_.

    Args:
        dataframe (pd.DataFrame): _description_
        output_directory (str | Path): _description_

    Returns:
        pd.DataFrame: _description_
    """
    # Create a list of methods as tuples containing the method name and the
    # 'getter' to the function definition
    methods = [
        ("iqr", get_iqr_outliers),
        ("zscore", get_zscore_outliers),
        ("mad", get_mad_outliers)
    ]

    # Get list of numeric features
    selected_numeric_feature_list = (
        dataframe.select_dtypes(include=np.number).columns.to_list()
    )
    # Run 'concatenate_outliers_with_target_category_dataframe' function on all
    # methods
    for method_name, method in methods:  # tuple unpacking
        for feature in selected_numeric_feature_list:
            outliers = method(dataframe=dataframe, column_name=feature)
            concatenate_outliers_with_target_category_dataframe(
                dataframe=dataframe,
                target_category_list=target_category_list,
                data_outliers=outliers,
                feature=feature,
                outlier_method=method_name,
                output_directory=output_directory
            )

    # TODO Merge ALL outlier dataframes using only common file names (inner ?)
    # return merged_outliers


def detect_multivariate_outliers(
    dataframe: pd.DataFrame,
    target_category_list: List[str],
    output_directory: str | Path
) -> Tuple[pd.DataFrame]:
    """detect_multivariate_outliers _summary_.

    Args:
        dataframe (pd.DataFrame): _description_
        target_category_list (List[str]): _description_
        output_directory (str | Path): _description_

    Returns:
        Tuple[pd.DataFrame]: _description_
    """
    # MAHALANOBIS TEST
    # Perform Mahalanobis test and get outlier list
    mahalanobis_dataframe, mahalanobis_outliers = apply_mahalanobis_test(
        dataframe.select_dtypes(include=np.number),
        alpha=0.01
    )

    # Concatenate the outliers with their corresponding target categories
    mahalanobis_outlier_dataframe = pd.merge(
        left=dataframe.loc[:, target_category_list],
        right=mahalanobis_outliers,
        on="file_name",
        how="right"
    )
    print("\nTable of outliers based on Mahalanobis distance:")
    print(mahalanobis_outlier_dataframe)

    # Save output as a '.csv()' file
    save_csv_file(
        dataframe=mahalanobis_outlier_dataframe,
        csv_path_name=output_directory/"mahalanobis_outliers.csv"
    )

    # Compile final outlier-free dataframe
    no_outlier_dataframe = remove_mahalanobis_outliers(
        mahalanobis_dataframe=mahalanobis_dataframe,
        mahalanobis_outlier_dataframe=mahalanobis_outlier_dataframe,
    )
    # print(f"\nData without outliers:\n{no_outlier_dataframe}\n")
    return no_outlier_dataframe


def standardise_features(features: pd.DataFrame) -> pd.DataFrame:
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
    features_scaled = scaler.fit_transform(features)

    # Convert the Numpy array to a Pandas dataframe
    features_scaled = pd.DataFrame(
        data=features_scaled,
        columns=features.columns
    )
    return features_scaled


def remove_low_variance_features(
    features: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame | np.ndarray:
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


# Calculate eigenvalues and eigenvectors from PCA
def get_pca_eigen_values_vectors(pca_model: PCA) -> Tuple[np.ndarray]:
    """Get eigenvalues and eigenvectors from a Principal Component Analysis.

    Args:
        pca_model (PCA): [description]

    Returns:
        Tuple[np.ndarray]: [description]
    """
    pca_eigen_values = pca_model.explained_variance_
    pca_eigen_values = np.round(pca_eigen_values, decimals=2)
    print(f"\nPCA Eigenvalues:\n{pca_eigen_values}")

    pca_eigen_vectors = pca_model.components_
    pca_eigen_vectors = np.round(pca_eigen_vectors, decimals=2)
    print(f"\nPCA Eigenvectors:\n{pca_eigen_vectors}\n")
    return pca_eigen_values


def apply_pca(
    n_components: int | float,
    features_scaled: pd.DataFrame | np.ndarray
) -> Tuple[PCA, np.ndarray]:
    """Run a Principal Component Analysis (PCA) on scaled data.

    Args:
        n_components (int | float): If 0 < n_components < 1, the float value
        corresponds to the amount of variance that needs to be explained.
        So if n_components = 0.95, we want to explain 95% of the variance.
        Conversely, if 'n_components' is an integer value, this represents the
        number of PC to include in the model.

        features_scaled (pd.DataFrame | np.ndarray): _description_

    Returns:
        Tuple[PCA, np.ndarray]: _description_
    """
    pca_model = PCA(n_components=n_components, random_state=42)
    pca_array = pca_model.fit_transform(features_scaled)

    eigen_values = pca_model.explained_variance_
    print(f"\nPCA Eigenvalues:\n{eigen_values}")

    eigen_vectors = pca_model.components_
    print(f"PCA Eigenvectors:\n{eigen_vectors}\n")
    return pca_model, pca_array


def explain_pca_variance(
    pca_eigen_values: np.ndarray,
    pca_model: PCA,
    pca_components: List[str]
) -> Tuple[pd.DataFrame]:
    """explain_pca_variance _summary_.

    Args:
        pca_eigen_values (np.ndarray): _description_
        pca_model (PCA): _description_
        pca_components (List[str]): _description_

    Returns:
        Tuple[pd.DataFrame]: _description_
    """
    variance_explained = pca_model.explained_variance_ratio_ * 100
    variance_explained_cumulated = np.cumsum(
        pca_model.explained_variance_ratio_) * 100
    variance_explained_df = pd.DataFrame(
        data=[
            pca_eigen_values,
            variance_explained,
            variance_explained_cumulated
        ],
        columns=pca_components,
    ).rename(
        index={
            0: "Eigenvalues",
            1: "Variance Explained (%)",
            2: "Cumulated Variance (%)",
        }
    )
    print(f"\nVariance Explained by the PCA:\n{variance_explained_df}")
    return variance_explained_df


def find_best_pc_axes(variance_explained_df: pd.DataFrame) -> Tuple[List[str]]:
    """find_best_pc_axes _summary_.

    Args:
        variance_explained_df (pd.DataFrame): _description_

    Returns:
        Tuple[List[str]]: _description_
    """
    # First, subset 'variance_explained_df' to keep cumulated values
    cumulative_values = variance_explained_df.loc["Cumulated Variance (%)", :]

    # Then, find the index of the last element in the array that is < 0.95
    index = np.searchsorted(cumulative_values, 95, side='right')

    # Include the first element that is >= 0.95
    if cumulative_values[index-1] < 95:
        # Add 1 to include the first element greater than or equal to 0.95
        index += 1

    # Slice the array to keep only the elements up to and including that index
    best_pc_axes = cumulative_values[:index]
    print(f"\nThe following filtered values are kept:\n{best_pc_axes}\n")

    best_pc_axis_names = best_pc_axes.index.to_list()
    best_pc_axis_values = best_pc_axes.to_list()
    return best_pc_axis_names, best_pc_axis_values


# Then, perform PCA to observe clusters based on defect types
def run_pc_analysis(
    features: pd.DataFrame,
) -> Tuple[pd.DataFrame | np.ndarray, List[str]]:
    """run_pc_analysis _summary_.

    Args:
        features (pd.DataFrame): _description_

    Returns:
        Tuple[pd.DataFrame | np.ndarray], List[str]: _description_
    """
    # Scale features data
    pca_features_scaled = standardise_features(features=features)

    # Run the PC analysis
    pca_model = PCA(n_components=0.95, random_state=42)
    pca_array = pca_model.fit_transform(pca_features_scaled)

    # Get eigenvalues and eigenvectors
    pca_eigen_values = get_pca_eigen_values_vectors(pca_model=pca_model)

    # Convert PCA array to a dataframe
    pca_dataframe = pd.DataFrame(data=pca_array)
    pca_dataframe.reset_index()

    # Get PC axis labels and insert into the dataframe
    pca_components = [
        "pc" + str(col+1) for col in pca_dataframe.columns.to_list()
    ]
    pca_dataframe.columns = pca_components

    # Calculate variance explained by PCA
    pca_variance_explained = (
        explain_pca_variance(
            pca_eigen_values=pca_eigen_values,
            pca_model=pca_model,
            pca_components=pca_components,
        )
    )
    # Set index as dataframe
    pca_dataframe = pca_dataframe.set_index(keys=features.index)
    # print(f"\nFinal PCA Dataframe:\n{pca_dataframe}\n")

    return (
        pca_model,
        pca_array,
        pca_dataframe,
        pca_variance_explained,
        pca_features_scaled
    )

    # Keep the PC axes that correspond to AT LEAST 95% of the cumulated
    # explained variance
    best_pc_axis_names, best_pc_axis_values = find_best_pc_axes(
        variance_explained_df=variance_explained_df,
        percent_cut_off_threshold=95
    )

    # Subset the PCA dataframe to include ONLY the best PC axes
    final_pca_df = pca_df.loc[:, best_pc_axis_names]
    print(f"\nFinal PCA Dataframe:\n{final_pca_df}\n")

    # -------------------------------------------------------------------------

    # Produce an '.html' file of the main steps of the EDA
    produce_sweetviz_eda_report(
        dataframe=final_pca_df,
        eda_report_name=eda_report_name,
        output_directory=output_directory
    )

    # -------------------------------------------------------------------------

    # Draw a scree plot of the variance explained
    plt.figure(figsize=(15, 10))
    scree_plot = draw_scree_plot(
        x_axis=best_pc_axis_names,
        y_axis=best_pc_axis_values,
    )
    scree_plot.grid(False)  # remove the grid from the plot
    plt.axhline(
        y=95,
        color="blue",
        linestyle="--",
    )
    plt.text(
        x=0.05,
        y=96,
        s="95% Cut-off Threshold",
        color="blue",
        fontsize=12
    )
    plt.title(
        label="Scree Plot PCA",
        fontsize=16,
        loc="center"
    )
    plt.ylabel("Percentage of Variance Explained")
    save_figure(figure_name=output_directory/"pca_scree_plot.png")

    # -------------------------------------------------------------------------

    # Prepare loadings/weights heatmap
    # Feature Weight from PCA
    plt.figure(figsize=(15, 10))
    draw_heatmap(
        data=pca_model.components_**2,
        xticklabels=numeric_feature_list,
        yticklabels=pca_components,
    )
    plt.title(
        label="Table of parameter effects on defect detection",
        fontsize=16,
        loc="center"
    )
    save_figure(figure_name=output_directory/"pca_loading_table.png")

    return (
        pca_model,
        pca_array,
        final_pca_df,
        best_pc_axis_names,
        variance_explained_df,
    )


def get_outliers_from_pca(
    dataframe: pd.DataFrame,
    label: str,
    target_class_list: List[str],
    min_percent_variance_coverage: float = 0.95,
    hotellings_t2_alpha: float = 0.05,
    number_std_deviation: int = 2,
) -> pd.DataFrame:
    """Use the 'pca' library to detect outliers.

    Args:
        dataframe (pd.DataFrame): _description_
        min_percent_variance_coverage (float, optional): _description_.
        Defaults to 0.95.
        hotellings_t2_alpha (float, optional): _description_. Defaults to 0.05.
        number_std_deviation (int, optional): _description_. Defaults to 2.

    Returns:
        pd.DataFrame: _description_
    """
    pca_outlier_model = pca(
        n_components=min_percent_variance_coverage,
        method="pca",
        normalize=True,
        alpha=hotellings_t2_alpha,
        multipletests="fdr_bh",
        detect_outliers=["ht2", "spe"],
        n_std=number_std_deviation,  # or 3 standard deviations
        random_state=42,
    )
    # Fit and transform
    # NOTE MUST be NUMERIC variables
    # NOTE Alternatively, apply one-shot encoding to use CATEGORICAL variables
    pca_results = pca_outlier_model.fit_transform(
        dataframe.select_dtypes(include="number")
    )
    print(f"\nPCA Data Dictionary Content:\n{pca_results.keys()}\n")

    # Get top features with the most effect on sample variance
    pca_dataframe = pca_results["PC"]
    print(f"PCA Output:\n{pca_dataframe}\n")

    # Get variance ratio of each PC
    variance_ratio = pca_results["variance_ratio"]
    variance_ratio = np.around(variance_ratio * 100, decimals=1)
    print(f"Percent Variance Ratio of Each PC:\n{variance_ratio}\n")

    # Get (cumulated) explained variance of each PC
    cumulated_variance = pca_results["explained_var"]
    cumulated_variance = np.around(cumulated_variance * 100, decimals=1)
    print(f"Cumulated Variance Ratio of Each PC:\n{cumulated_variance}\n")

    # Get loadings from each feature
    pca_loadings = pca_results["loadings"]
    print(f"PC Loadings for Each Feature:\n{pca_loadings.round(2)}\n")

    # Get total explained variance
    pcp = pca_results["pcp"]
    print(f"Total Explained Variance for Selected PCs:\n{pcp:.1%}\n")

    # Get top features with the most effect on sample variance
    top_features = pca_results["topfeat"]
    # BUG Below NOT working ?!!
    # top_features = top_features.applymap(
    #     lambda float_: f"{float_:.2f}"
    # )
    print(f"Top Features for each PC:\n{top_features}\n")

    # Get outliers stats values
    outliers_params = pca_results["outliers_params"]
    print(f"Outlier Parameters:\n{outliers_params}\n")

    print("\nPCA Outlier Detection\n")
    # Get the outliers using Hotellings T2 method.
    outliers_ht2 = pca_results["outliers"]["y_bool"]
    outliers_ht2_filtered = filter_dataframe(
        dataframe=pca_results["outliers"],
        filter_content="y_bool == True",
    )
    print(
        f"\nList of Outliers using Hotellings T2:\n{outliers_ht2_filtered}\n"
    )

    # outlier_ht2_category_count = outliers_ht2.value_counts()
    # print(
    #     f"\nValue Count of 'Hotellings T2' Outliers ('True' Category):\n"
    #     f"{outlier_ht2_category_count}\n"
    # )

    # Get the outliers using SPE/DmodX method.
    outliers_spe = pca_results["outliers"]["y_bool_spe"]
    outliers_spe_filtered = filter_dataframe(
        dataframe=pca_results["outliers"],
        filter_content="y_bool_spe == True",
    )
    print(f"\nList of Outliers using SPE/DmodX:\n{outliers_spe_filtered}\n")

    # outlier_spe_category_count = outliers_spe.value_counts()
    # print(
    #     f"\nValue Count of 'SPE/DmodX' Outliers ('True' Category):\n"
    #     f"{outlier_spe_category_count}\n"
    # )

    # Grab overlapping outliers
    overlapping_outliers = np.logical_and(
        outliers_ht2,
        outliers_spe
    )
    pca_outlier_dataframe = dataframe.loc[overlapping_outliers, :]
    print(f"\nOverlapping Outliers:\n{pca_outlier_dataframe}\n")
    return pca_outlier_dataframe, pca_outlier_model


def run_anova_check_assumption(
    dataframe: pd.DataFrame,
    dependent_variable: pd.Series,
    independent_variable: pd.Series | str,
    group_variable: List[str],
    output_directory: str | Path,
    confidence_interval: float = 0.95,
) -> ols:
    """run_anova_check_assumption _summary_.

    Args:
        dataframe (pd.DataFrame): _description_
        dependent_variable (pd.Series): _description_
        independent_variable (pd.Series): _description_
        group_variable (List[str]): _description_
        output_directory (str | Path): _description_

    Returns:
        _type_: _description_
    """
    anova_model = run_anova_test(
        dataframe=dataframe,
        dependent_variable=dependent_variable,
        independent_variable=independent_variable,
        group_variable=group_variable,
        output_directory=output_directory,
        confidence_interval=confidence_interval,
    )

    # ---------------------------------------------------------------

    # Check assumptions of the model residuals are met
    check_normality_assumption_residuals(
        dataframe=anova_model.resid
    )

    check_equal_variance_assumption_residuals(
        dataframe=dataframe,
        model=anova_model,
        group_variable=group_variable,
    )

    # ---------------------------------------------------------------

    draw_anova_quality_checks(
        dependent_variable=dependent_variable,
        independent_variable=independent_variable,
        model=anova_model,
        output_directory=output_directory,
        confidence_interval=confidence_interval,
    )

    # ---------------------------------------------------------------

    draw_tukeys_hsd_plot(
        dataframe=dataframe,
        dependent_variable=dependent_variable,
        independent_variable=independent_variable,
        output_directory=output_directory,
        confidence_interval=confidence_interval,
    )
    return anova_model


def run_anova_test(
    dataframe: pd.DataFrame,
    dependent_variable: pd.Series,
    independent_variable: str,
    group_variable: str,
    output_directory: str | Path,
    confidence_interval: float = 0.95,
) -> None:
    """ANOVA test on numeric variables and calculate group confidence interval.

    Args:
        dataframe (pd.DataFrame): _description_
        dependent_variable (pd.Series): _description_
        group_variable (str): _description_
        confidence_interval (float, optional): _description_.
        Defaults to 0.95.
    """
    print(f"\n{dependent_variable.upper()}:")

    # Fit the OLS model
    ols_model = ols(
        formula=f"{dependent_variable} ~ C({independent_variable})",
        data=dataframe
    ).fit()

    # Display ANOVA table
    anova_table = sm.stats.anova_lm(ols_model, typ=1)
    print(f"\nANOVA Test Output Table:\n{anova_table}")

    # Build confidence interval
    ci_table_defect_type = rp.summary_cont(
        group1=dataframe[dependent_variable].groupby(
            dataframe[group_variable]
        ),
        conf=confidence_interval
    )
    print("One-way ANOVA and confidence intervals:")
    print(ci_table_defect_type)
    save_csv_file(
        dataframe=ci_table_defect_type,
        csv_path_name=output_directory /
        f"ci_table_defect_type_{dependent_variable}.csv"
    )

    # Find group differences
    perform_multicomparison(
        dataframe=dataframe[dependent_variable],
        groups=dataframe[group_variable],
        confidence_interval=confidence_interval
    )
    return ols_model


def perform_multicomparison(
    dataframe: pd.DataFrame,
    groups: List[str],
    confidence_interval: float = 0.95,
) -> pd.DataFrame:
    """perform_multicomparison _summary_.

    Args:
        dataframe (pd.DataFrame): _description_
        groups (List[str]): _description_
        confidence_interval (float, optional): _description_. Defaults to 0.95.

    Returns:
        _type_: _description_
    """
    multicomparison = MultiComparison(
        data=dataframe,
        groups=groups
    )
    tukey_result = multicomparison.tukeyhsd(alpha=1 - confidence_interval)
    print(f"\nTukey's Multicomparison Test between groups:\n{tukey_result}\n")
    # print(f"Unique groups: {multicomparison.groupsunique}\n")
    return tukey_result


def apply_manova(dataframe: pd.DataFrame, formula: str) -> pd.DataFrame:
    """Perform Multiple Analysis of Variance using OLS model.

    In this instance, it is better to give the formula of the model.

    Args:
        dataframe (_type_): _description_
        formula (_type_): _description_

    Returns:
        _type_: _description_
    """
    # Run MANOVA test
    manova_model = MANOVA.from_formula(
        formula=formula,
        data=dataframe
    )
    manova_test = manova_model.mv_test()
    return manova_test


def check_normality_assumption_residuals(
    dataframe: pd.DataFrame,
    confidence_interval: float = 0.95,
) -> pd.DataFrame:
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
        confidence_interval (float, optional): _description_. Defaults to 0.95.

    Returns:
        _type_: _description_
    """
    shapiro_wilk = pg.normality(
        data=dataframe,
        method="shapiro",
        alpha=(1 - confidence_interval)
    )
    shapiro_wilk.rename(index={0: "Shapiro-Wilk"}, inplace=True)

    normality = pg.normality(
        data=dataframe,
        method="normaltest",
        alpha=(1 - confidence_interval)
    )
    normality.rename(index={0: "Normality"}, inplace=True)

    jarque_bera = pg.normality(
        data=dataframe,
        method="jarque_bera",
        alpha=(1 - confidence_interval)
    )
    jarque_bera.rename(index={0: "Jarque-Bera"}, inplace=True)

    # Concatenate the tests output
    normality_tests = pd.concat(
        objs=[shapiro_wilk, normality, jarque_bera],
        axis=0,
    )
    normality_tests.rename(
        columns={"W": "Statistic", "pval": "p-value"},
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
    confidence_interval: float = 0.95,
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
        confidence_interval (float, optional): _description_. Defaults to 0.95.

    Returns:
        _type_: _description_
    """
    # Prepare a dataframe of the ANOVA model residuals
    model_residuals_dataframe = pd.concat(
        [dataframe[group_variable], model.resid], axis=1
    ).rename(columns={0: "Residuals"})

    bartlett = pg.homoscedasticity(
        data=model_residuals_dataframe,
        dv="residuals",
        group=group_variable,
        method="bartlett",
        alpha=(1 - confidence_interval)
    )
    bartlett.rename(
        index={"mean_intensities": "Bartlett's Test"},
        columns={"T": "Statistic", "pval": "p-value", "normal": "Normal"},
        inplace=True
    )

    levene = pg.homoscedasticity(
        data=model_residuals_dataframe,
        dv="residuals",
        group=group_variable,
        method="levene",
        alpha=(1 - confidence_interval)
    )
    levene.rename(
        index={"mean_intensities": "Levene's Test"},
        columns={"W": "Statistic", "pval": "p-value", "normal": "Normal"},
        inplace=True
    )

    # Concatenate the tests output
    equal_variance_tests = pd.concat(
        objs=[bartlett, levene],
        axis=0,
    )
    print(
        f"\nEquality of Variance Tests Results - '{group_variable}' as group"
        f"\n{equal_variance_tests}\n"
    )
    # BUG It does not the difference between 'if' and 'else' output
    print("Equal Variance of Data Between Groups:")
    if levene.iloc[0]["equal_var"] is False:
        print(
            f"Data do NOT have equal variance between {group_variable} groups."
        )
    else:
        print(f"Data have equal variance between {group_variable} groups.\n")
    return equal_variance_tests


def run_tukey_post_hoc_test(dataframe, dependent_variable, group_list):
    """Use Tukey's HSD post-hoc test for multiple comparison between groups.

    It produces a table of significant differences.

    Args:
        dataframe (_type_): _description_
        dependent_variable (_type_): _description_
        group_list (_type_): _description_

    Returns:
        _type_: _description_
    """
    tukey = pg.pairwise_tukey(
        data=dataframe,
        dv=dependent_variable,
        between=group_list
    )
    return tukey


def perform_multicomparison_correction(
    p_values,
    method: str = "bonferroni"
) -> pd.DataFrame:
    """Apply the Bonferroni correction method to the p-values.

    This version of 'multicomparison_correction' uses the Pingouin library.
    An equivalent version is also available from the 'statsmodels' library.

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
        columns={0: "Reject Hypothesis", 1: "Corrected p-values"},
        inplace=True
    )
    return correction_dataframe


# -----------------------------------------------------------------------------

# DATA VISUALISATION


def create_missing_data_matrix(
        dataframe: pd.DataFrame,
        output_directory: str | Path,
) -> None:
    # fix type hints for the content of the 'object'
    """Display missing values status for each column in a matrix.

    Args:
        dataframe (pd.DataFrame): _description_

    Returns:
        object: _description_
    """
    plt.figure(figsize=(15, 10))
    msno.matrix(
        dataframe,
        sort="descending",  # NOT working
        # figsize=(10, 5),
        fontsize=8,
        sparkline=False,
    )
    plt.title(
        label="Summary of Missing Data",
        fontsize=16,
        loc="center"
    )
    plt.tight_layout()
    # plt.show()
    save_figure(
        figure_name=output_directory/"missing_data_matrix.png"
    )


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
    hue=None,
    style=None,
    size=None,
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
        style (_type_): _description_
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
        style=style,
        palette=palette,
        size=size,
        sizes=(20, 200),
        legend="full",
        s=100,  # size of the markers
    )
    return scatterplot


def draw_all_pca_pairs_scatterplot(
    dataframe: pd.DataFrame,
    variance_explained_df: pd.DataFrame,
    pca_components: List[str],
    target_category: str,
    sub_category: str,
    output_directory: str | Path
) -> None:
    """Draw the scatter plots for all selected PC axes.

    The axes can be all those existing or a selection of them. The selection
    can be the axes that represent at least 95% of the explained variance
    AND/OR eigenvalues above 1.
    Here, selection of n-% variance explained.

    The function 'itertools.combinations()' generates all possible pairs of
    elements from the pc component list while keeping unique pairs of elements.
    This is possible via the use of the '2' parameters to work on 'pairs'.

    Args:
        dataframe (pd.DataFrame): _description_
        variance_explained_df (pd.DataFrame): _description_
        pca_components (List[str]): _description_
        target_category (str): _description_
        sub_category (str): _description_
        output_directory (str | Path): _description_

    TODO Change the symbol shapes for the 'product' sub-category
    """
    # Generate all possible pairs of elements from the pc component list
    # while keeping unique pairs of elements
    # BUT keep the first four PC components only
    # OR PC axes that are AT LEAST equal to 0.95
    pca_pair_combinations = itertools.combinations(pca_components, 2)
    pca_pair_combination_list = list(pca_pair_combinations)
    print(f"\nPairs of PC axes selected:\n{pca_pair_combination_list}\n")

    # Create a scatter plot for each pair in the 'pca_pair_combinations' object
    # for pair in pca_pair_combination_list:
    for pair_tuple in pca_pair_combination_list:
        # Convert the tuple of PC pairs to a list for subsequent slicing
        pair_list = list(pair_tuple)
        # Unpack each pc axis
        pc_x, pc_y = pair_list
        # Select the PC axes from the main explained variance dataframe
        # BUT select the individual variances, not the cumulated ones
        variance_dataframe = variance_explained_df.iloc[0]
        # Select the PC axes from the main explained variance dataframe
        variance_dataframe = variance_dataframe.loc[pair_list]
        # Extract the variance values
        variance_list = [
            variance for pc, variance in variance_dataframe.iteritems()
        ]

        plt.figure(figsize=(15, 10))
        pca_scatter_plot = draw_scatterplot(
            dataframe=dataframe,
            x_axis=dataframe[pc_x],
            y_axis=dataframe[pc_y],
            hue=target_category,
            style=sub_category,
        )
        pca_scatter_plot.set(
            xlabel=f"{pc_x.upper()} ({variance_list[0]:.1f} %)",
            ylabel=f"{pc_y.upper()} ({variance_list[1]:.1f} %)"
        )
        plt.legend(
            title="Legend",
            bbox_to_anchor=(1, 1),
            loc="upper left",
            fontsize="10",
            frameon=False,
            markerscale=2
        )
        plt.title(
            label=(
                f"Représentation des effets des paramètres de détection des"
                f"défauts {pc_x.upper()} vs. {pc_y.upper()}"
            ),
            fontsize=16,
            loc="center"
        )
        plt.tight_layout()
        # plt.show()
        save_figure(
            figure_name=output_directory /
            f"scatterplot_pca_{target_category}_{pc_x}_vs_{pc_y}.png"
        )


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
    dataframe: pd.DataFrame,
    x_axis: List[float],
    x_label: str,
    y_label: str,
    hue: List[str] = None,
    palette: str | List[str] | Dict[str, str] = None,
):
    """Draw a density curve of data distribution.

    Args:
        dataframe (pd.DataFrame): _description_
        x_axis (List[float]): _description_
        x_label (str): _description_
        y_label (str): _description_
        hue (List[str], optional): _description_. Defaults to None.
        palette (str | List[str] | Dict[str, str], optional): _description_.
        Defaults to None.

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
        xlabel=x_label,
        ylabel=y_label,
    )
    return kdeplot


def draw_kdeplot_subplots(
    dataframe: pd.DataFrame,
    x_axis: List[float],
    x_label: str,
    y_label: str,
    item_list: List[str],
    nb_columns: int,
    hue: List[str] = None,
    palette: str | List[str] | Dict[str, str] = None,
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
        dataframe (pd.DataFrame): _description_
        x_axis (List[float]): _description_
        x_label (str): _description_
        y_label (str): _description_
        item_list (List[str]): _description_
        nb_columns (int): _description_
        hue (List[str], optional): _description_. Defaults to None.
        palette (str | List[str] | Dict[str, str], optional): _description_.
        Defaults to None.

    Returns:
        _type_: _description_
    """
    plt.subplots_adjust(wspace=0.3, hspace=0.5)

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
            x_label=x_label,
            y_label="Density",
            hue=hue,
            palette=palette
        )
        axis.set_title(label=item.upper(), fontsize=20),
        axis.set_xticklabels(
            labels=kdeplot_subplots.get_xticklabels(),
            size=14,
        ),
        axis.set_xlabel(
            xlabel=x_label,
            fontsize=18
        )
        axis.set_ylabel(
            ylabel=y_label,
            fontsize=18
        )
    return kdeplot_subplots


def draw_boxplot(
    dataframe: pd.DataFrame,
    x_axis: List[float],
    y_axis: List[float],
    hue: List[str] = None,
    palette: str | List[str] | Dict[str, str] = None,
):
    """Draw a boxplot of data distribution.

    Args:
        dataframe (pd.DataFrame): _description_
        x_axis (List[float]): _description_
        y_axis (List[float]): _description_
        hue (List[str], optional): _description_. Defaults to None.
        palette (str | List[str] | Dict[str, str], optional): _description_.
        Defaults to None.

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
    x_axis: List[float],
    y_axis: List[float],
    errorbar: str = "ci",
    orient: str = "vertical",
    hue: List[str] = None,
    palette: str | List[str] | Dict[str, str] = None,
):
    """Draw a barplot which can include colour bars for treatment options.

    Args:
        dataframe (_type_): _description_
        x_axis (List[float]): _description_
        y_axis (List[float]): _description_
        errorbar (str, optional): _description_. Defaults to "ci".
        orient (str, optional): _description_. Defaults to "vertical".
        hue (List[str], optional): _description_. Defaults to None.
        palette (str | List[str] | Dict[str, str], optional): _description_.
        Defaults to None.

    Returns:
        _type_: _description_
    """
    barplot = sns.barplot(
        data=dataframe,
        x=x_axis,
        y=y_axis,
        hue=hue,
        errorbar=errorbar,
        orient=orient,
        palette=palette
    )
    return barplot


def draw_barplot_subplots(
    dataframe,
    x_axis: List[float],
    y_label: str,
    item_list: List[str],
    nb_columns: int,
    errorbar: str = "ci",
    hue: List[str] = None,
    palette: str | List[str] | Dict[str, str] = None,
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
        x_axis (List[float]): _description_
        y_label (str): _description_
        item_list (List[str]): _description_
        nb_columns (int): _description_
        errorbar (str, optional): _description_. Defaults to "ci".
        hue (List[str], optional): _description_. Defaults to None.
        palette (str | List[str] | Dict[str, str], optional): _description_.
        Defaults to None.

    Returns:
        _type_: _description_
    """
    plt.subplots_adjust(wspace=0.3, hspace=0.5)

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
            hue=hue,
            errorbar=errorbar,
            palette=palette
        )
        axis.set_title(label=item.upper(), fontsize=20),
        axis.set_xticklabels(
            labels=barplot_subplots.get_xticklabels(),
            size=14,
            rotation=45,
            ha="right"
        ),
        axis.set_xlabel(""),
        axis.set_ylabel(
            ylabel=y_label,
            fontsize=18
        )
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
    dataframe: pd.DataFrame,
    confidence_interval: float = 0.95
) -> None:
    """Draw the Q-Q plot using the residuals of fitted model, for example.

    The parameter 'data' can take a 1D-array (e.g. model output) or a
    dataframe and one of its column (e.g. dataframes[column]).

    Args:
        data (_type_): _description_
        confidence (float, optional): _description_. Defaults to 0.95.

    Returns:
        _type_: _description_
    """
    plt.figure()
    qqplot = pg.qqplot(
        x=dataframe,
        dist="norm",
        confidence=confidence_interval
    )
    return qqplot


def draw_anova_quality_checks(
    dependent_variable: str,
    independent_variable: str,
    model,
    output_directory: str | Path,
    confidence_interval: float = 0.95,
) -> None:
    """Draw Q-Q plots of the model dependent variable residuals.

    Args:
        dataframe (pd.DataFrame): _description_
        dependent_variable (str): _description_
        independent_variable (str): _description_
        model (_type_): _description_
        output_directory (str | Path): _description_
        confidence_interval (float, optional): _description_. Defaults to 0.95.
    """
    draw_qqplot(
        dataframe=model.resid,
        confidence_interval=confidence_interval
    )
    plt.title(
        label=(
            f"Q-Q Plot of Model Residuals ({dependent_variable}"
            f"vs. {independent_variable})"
        ),
        fontsize=14,
        loc="center"
    )
    plt.grid(visible=False)
    # plt.axis("off")
    plt.tight_layout()
    # plt.show()
    save_figure(
        figure_name=(
            output_directory /
            f"qqplot_anova_{independent_variable}_{dependent_variable}.png"
        )
    )


def draw_tukeys_hsd_plot(
    dataframe: pd.DataFrame,
    dependent_variable: str,
    independent_variable: str,
    output_directory: str | Path,
    confidence_interval: float = 0.95,
) -> None:
    """draw_tukeys_hsd_plot _summary_.

    Args:
        dataframe (pd.DataFrame): _description_
        dependent_variable (str): _description_
        independent_variable (str): _description_
        output_directory (str | Path): _description_
        confidence_interval (float, optional): _description_. Defaults to 0.95.
    """
    print("\nStats for Tukey's HSD Plots")
    # Run post-hoc test
    tukey_result = perform_multicomparison(
        dataframe=dataframe[dependent_variable],
        groups=dataframe[independent_variable],
        confidence_interval=confidence_interval
    )

    # Plot graphic output from Tukey's test
    tukey_result.plot_simultaneous(
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
    plt.grid(visible=False)
    # plt.axis("off")
    plt.tight_layout()
    # plt.show()
    save_figure(
        figure_name=(
            output_directory /
            f"tukey_anova_{independent_variable}_{dependent_variable}.png"
        )
    )

    # -------------------------------------------------------------------------

    # # Draw a different version of Tukey's table using Pingouin library
    # print("Draw a different version of Tukey's table using Pingouin library")

    # tukey_post_hoc_test = run_tukey_post_hoc_test(
    #     dataframe=dataframe,
    #     dependent_variable=dependent_variable,
    #     group_list=independent_variable
    # )
    # # Apply p-values correction
    # corrected_tukey_dataframe = (
    #     perform_multicomparison_correction(
    #         p_values=tukey_post_hoc_test["p-tukey"],
    #     )
    # )
    # # Add output to 'tukey_post_hoc_test' dataframe
    # corrected_tukey_post_hoc_test = pd.concat(
    #     objs=[tukey_post_hoc_test, corrected_tukey_dataframe], axis=1
    # )
    # print(
    #     f"\nTukey's Multicomparison Test Version 2:\n \
    #         {corrected_tukey_post_hoc_test}\n"
    # )


def draw_pca_outliers_biplot_3d(
    dataframe: pd.DataFrame,
    label: str,
    pca_outlier_model: pca,
    target_class_list: List[str],
    file_name: str,
    output_directory: Path,
    outlier_detection: bool = True,
) -> None:
    """Use the 'pca' library to detect outliers & display them in scatterplot.

    See 'get_outliers_from_pca()' function.

    Args:
        dataframe (pd.DataFrame): _description_
        label (str): _description_
        pca_outlier_model (pca): _description_
        target_class_list (List[str]): _description_
        file_name (str): _description_
        output_directory (Path): _description_
        outlier_detection (bool, optional): _description_. Defaults to True.
    """
    # Define the colour scheme for identifying the target classes ?
    # BUG Below code NOT working ?!!
    target_class_colour_list, _ = colourmap.fromlist(target_class_list)
    # target_class_colour_list = colourmap.generate(
    #     len(target_class_list),
    #     method="seaborn"
    # )

    plt.figure()
    pca_outlier_model.biplot3d(
        y=dataframe[label],  # (categorical) label
        SPE=outlier_detection,
        hotellingt2=outlier_detection,
        legend=True,
        label=False,
        figsize=(20, 12),
        color_arrow="k",
        fontdict={
            "weight": "bold",
            "size": 12,
            "ha": "center",
            "va": "center",
            "c": "color_arrow"
        },
        title="Outliers marked using Hotellings T2 & SPE/DmodX methods",
        # cmap="bwr_r",
        # cmap="Set2",
        c=target_class_colour_list,
        # cmap=target_class_colour_list,
        # gradient="#FFFFFF",
        visible=True,
    )
    plt.grid(False)
    plt.tight_layout()
    # plt.show()
    save_figure(figure_name=output_directory.joinpath(file_name + ".png"))


def draw_pca_outliers_biplot(
    dataframe: pd.DataFrame,
    label: str,
    pca_outlier_model: pca,
    target_class_list: List[str],
    file_name: str,
    output_directory: Path,
    outlier_detection: bool = True,
) -> None:
    """Use the 'pca' library to detect outliers & display them in scatterplot.

    See 'get_outliers_from_pca()' function.

    Args:
        dataframe (pd.DataFrame): _description_
        label (str): _description_
        pca_outlier_model (pca): _description_
        target_class_list (List[str]): _description_
        file_name (str): _description_
        output_directory (Path): _description_
        outlier_detection (bool, optional): _description_. Defaults to True.
    """
    # Define the colour scheme for identifying the target classes ?
    # BUG Below code NOT working ?!!
    target_class_colour_list, _ = colourmap.fromlist(target_class_list)
    # target_class_colour_list = colourmap.generate(
    #     len(target_class_list),
    #     method="seaborn"
    # )

    plt.figure()
    pca_outlier_model.biplot(
        y=dataframe[label],  # (categorical) label
        SPE=outlier_detection,
        hotellingt2=outlier_detection,
        legend=True,
        label=False,
        # figsize=(20, 12),
        color_arrow="k",
        fontdict={
            "weight": "bold",
            "size": 12,
            "ha": "center",
            "va": "center",
            "c": "color_arrow"
        },
        title="Outliers marked using Hotellings T2 & SPE/DmodX methods",
        # cmap="bwr_r",
        # cmap="Set2",
        c=target_class_colour_list,
        # cmap=target_class_colour_list,
        # gradient="#FFFFFF",
        visible=True,
    )
    plt.grid(False)
    plt.tight_layout()
    # plt.show()
    save_figure(figure_name=output_directory.joinpath(file_name + ".png"))


def draw_pca_biplot_3d(
    features_scaled: pd.DataFrame,
    target_encoded: pd.Series,
    target_class_list: List[str],
    file_name: str,
    output_directory: Path,
) -> None:
    """PCA plot in 3 dimensions using 'yellowbrick' library.

    The 'yellowbrick' library is used for Principle Component Analysis.
    Prior to fitting the features and target, the latter must be first
    label-encoded.

    Args:
        features_scaled (pd.DataFrame): _description_
        target_encoded (pd.Series): _description_
        target_class_list (List[str]): _description_
        output_directory (str | Path): _description_
    """
    # Define the colour scheme for identifying the target classes ?
    # target_class_colour_list, _ = colourmap.fromlist(target_class_list)
    target_class_colour_list = colourmap.generate(
        len(target_class_list),
        method="seaborn"
    )

    plt.figure(figsize=(15, 10))
    pca_biplot_3d = yb_pca(
        scale=True,
        projection=3,
        classes=target_class_list,
        proj_features=True,
        colors=target_class_colour_list,
        colormap="tab20",
        # heatmap=True
    )
    pca_biplot_3d.fit_transform(features_scaled, target_encoded)
    pca_biplot_3d.finalize()
    # pca_biplot_3d.show()

    # -------------------------------------------------------------------------

    # Save 3D scatter plot data as '.html' file to preserve its interactivity
    scatter_data = pca_biplot_3d.ax.collections[0]
    scatter_data_data = scatter_data._offsets3d

    # # Define the color map for the classes ?
    # colours = ["blue", "orange", "green", "red"]
    # # colour_map = {colour: colours[i] for i, colour in enumerate(classes)}

    # Create the Plotly trace
    trace = go.Scatter3d(
        x=scatter_data_data[0],
        y=scatter_data_data[1],
        z=scatter_data_data[2],
        mode="markers",
        marker=dict(
            size=5,
            # color=scatter_data.get_facecolors(),
            color=[
                target_class_colour_list[target_class]
                for target_class in target_class_list
            ],
            # color=[colour_map[colour] for colour in pca_biplot_3d.y]  # BUG
        )
    )

    # Create the Plotly layout
    layout = go.Layout(
        scene=dict(
            xaxis=dict(title="PC1"),
            yaxis=dict(title="PC2"),
            zaxis=dict(title="PC3"),
            aspectmode="data"
        ),
        title=dict(
            text="Table of parameter effects on defect detection",
            font=dict(size=24)
        )
    )

    # Create the Plotly figure
    fig = go.Figure(data=[trace], layout=layout)
    fig.show()
    # Save the Plotly figure as an HTML file
    fig.write_html(output_directory.joinpath(file_name + ".html"))


def draw_pca_biplot(
    pca_array: np.ndarray,
    features_scaled: pd.DataFrame,
    target_encoded: pd.Series,
    target_class_list: List[str],
    file_name: str,
    output_directory: Path,
) -> None:
    """draw_pca_biplot _summary_.

    The 'yellowbrick' library is used for Principle Component Analysis.
    Prior to fitting the features and target, the latter must be first
    label-encoded.

    Args:
        features_scaled (pd.DataFrame): _description_
        target_encoded (pd.Series): _description_
        target_class_list (List[str]): _description_
        output_directory (str | Path): _description_
    """
    # Define the colour scheme for identifying the target classes

    # Get a colour map
    # cmap = plt.get_cmap('tab10')
    # Create a list of colours based on the colour map and the number of
    # colours, i.e. the number of target classes
    # target_class_colour_list = [
    #     cmap[colour] for colour in range(len(target_class_list))
    # ]

    # Define the colour scheme for identifying the target classes
    target_class_colour_list, _ = colourmap.fromlist(target_class_list)

    plt.figure(figsize=(15, 10))
    pca_biplot = yb_pca(
        scale=True,
        classes=target_class_list,
        proj_features=True,
    )
    pca_biplot.fit_transform(features_scaled, target_encoded)
    pca_biplot.finalize()

    # plt.set(
    #     xlabel=f"{pc_x.upper()} ({variance_list[0]:.1f} %)",
    #     ylabel=f"{pc_y.upper()} ({variance_list[1]:.1f} %)"
    # )
    # plt.set(
    #     xlabel="PC1",
    #     ylabel="PC2"
    # )
    # plt.legend(
    #     title="Legend",
    #     bbox_to_anchor=(1, 1),
    #     loc="upper left",
    #     fontsize="10",
    #     frameon=False,
    #     markerscale=2
    # )
    plt.title(
        label="Parameter Effects on Defect Detection PC1 vs. PC2",
        fontsize=16,
        loc="center"
    )

    # -------------------------------------------------------------------------

    # Create ellipsis for 95% CI for each classe
    # # BUG Code below NOT working
    # add_confidence_interval_ellipses(
    #     pca_array=pca_array,
    #     target_class_list=target_class_list,
    #     confidence_interval=0.90,
    # )

    # Define the confidence level and alpha value for the ellipse
    confidence_interval = 0.95
    alpha = 0.3  # this is the transparency level of the ellipses

    # Iterate over each class in the dataset
    for target_class, colour in zip(
        range(len(target_class_list)), target_class_colour_list
    ):
        # Select the data for the current class
        pca_class = pca_array[target_encoded == target_class]
        # Calculate the mean and covariance matrix for the data
        mean = np.mean(pca_class, axis=0)
        cov = np.cov(pca_class.T)
        # Calculate the ellipse width and height based on the confidence level
        # and covariance matrix
        ellipse = Ellipse(
            xy=mean,
            width=2 * np.sqrt(cov[0, 0]) * stats.t.ppf(
                q=(1 + confidence_interval) / 2,
                df=pca_class.shape[0] - 1
            ),
            height=2 * np.sqrt(cov[1, 1]) * stats.t.ppf(
                q=(1 + confidence_interval) / 2,
                df=pca_class.shape[0] - 1
            )
        )
        width, height = ellipse.get_width(), ellipse.get_height()

        # Add the ellipse to the plot with the specified color and alpha value
        plt.gca().add_artist(Ellipse(
            xy=mean,
            width=width,
            height=height,
            edgecolor=colour,
            facecolor=colour,
            alpha=alpha
        ))
    # pca_biplot.show()
    save_figure(figure_name=output_directory.joinpath(file_name + ".png"))


def add_confidence_interval_ellipses(
    pca_array: np.ndarray,
    target_class_list: List[str],
    confidence_interval: float = 0.95,
    alpha: float = 0.2,
) -> None:
    """Create ellipses for 95% CI for each class in a PCA array.

    Define the confidence level and alpha value for the ellipsis.
    The alpha value is is the transparency level of the ellipses. It takes a
    value between 0 and 1, where 0 is completely transparent (i.e., invisible)
    and 1 is completely opaque (i.e., solid).

    IMPORTANT: the ellipses will be displayed on the latest figure object.

    Args:
        pca_array (np.ndarray): _description_
        target_class_list (List[str]): _description_
        confidence_interval (float, optional): _description_. Defaults to 0.95.
        alpha (float, optional): _description_. Defaults to 0.2.
    """
    # Define the colour scheme for identifying the target classes
    target_class_colour_list, _ = colourmap.fromlist(target_class_list)

    # Iterate over each class in the dataset
    for target_class, colour in zip(
        range(len(target_class_list)), target_class_colour_list
    ):
        # Select the data for the current class
        pca_class = pca_array[target_class_list == target_class]
        # Calculate the mean and covariance matrix for the data
        mean = np.mean(pca_class, axis=0)
        cov = np.cov(pca_class.T)
        # Calculate the ellipse width and height based on the confidence
        # interval value and covariance matrix
        ellipse = Ellipse(
            xy=mean,
            width=2*np.sqrt(cov[0, 0])*stats.t.ppf(
                q=(1+confidence_interval)/2,
                df=pca_class.shape[0]-1
            ),
            height=2*np.sqrt(cov[1, 1])*stats.t.ppf(
                q=(1+confidence_interval)/2,
                df=pca_class.shape[0]-1
            )
        )
        width, height = ellipse.get_width(), ellipse.get_height()
        # print(width, height)

        # Add ellipses to the plot with the specified color & alpha value
        plt.gca().add_artist(Ellipse(
            xy=mean,
            width=width,
            height=height,
            edgecolor=colour,
            facecolor=colour,
            alpha=alpha
        ))
    plt.tight_layout()
    plt.show()


# # RadViz with 'yellowbrick' library
# plt.figure(dpi=120)
# radviz = RadViz(classes=target_encoded_list)
# radviz.fit(features, target_encoded)
# radviz.transform(features)
# # radviz.show()


def draw_feature_rank(
    features: pd.DataFrame,
    target_encoded: pd.Series,
    output_directory: str | Path,
) -> None:
    """draw_feature_rank _summary_.

    The 'yellowbrick' library is used for ranking features.
    Prior to fitting the features and target, the latter must be first
    label-encoded.

    Args:
        features (pd.DataFrame): _description_
        target_encoded (pd.Series): _description_
        output_directory (str | Path): _description_
    """
    plt.figure(figsize=(15, 10))
    # Instantiate the 1D visualiser with the Shapiro ranking algorithm
    feature_rank = Rank1D(
        algorithm="shapiro",
        orient="h",
    )
    # Fit the data to the visualizer
    feature_rank.fit(features, target_encoded)
    feature_rank.transform(features)

    plt.title(
        label="Ranking of Features",
        fontsize=16,
        loc="center"
    )
    save_figure(
        figure_name=output_directory/"feature_ranking.png"
    )


def show_items_per_category(data: pd.Series) -> None:
    """show_items_per_category _summary_.

    Args:
        data (pd.Series): _description_
    """
    # Show number of items in each class of the target
    data_class_count = data.value_counts()
    print(
        f"\nItems number within each {data.name} class:\n{data_class_count}\n"
    )
    ax = data_class_count.sort_values().plot.barh()
    ax.set(xlabel="Number of items", ylabel=data.name)


# -----------------------------------------------------------------------------

# DATA MODELLING (MACHINE LEARNING)


def target_label_encoder(
    dataframe: pd.DataFrame,
    target_name: str
) -> Tuple[np.ndarray, List[str]]:
    """Encode the target labels (usually strings) into integers.

    Args:
        dataframe (pd.DataFrame): _description_
        target_name (str): _description_

    Returns:
        Tuple[np.ndarray, List[str]]: _description_
    """
    target_dataframe = dataframe[target_name]
    label_encoder = LabelEncoder()
    target_encoded = label_encoder.fit_transform(target_dataframe)
    # Convert the encoded labels from a np.ndarray to a list
    target_class_list = label_encoder.classes_.tolist()
    print(f"\nList of the target encoded classes:\n{target_class_list}\n")
    return target_encoded, target_class_list


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


def preprocess_robust_scaler_numeric_feature_pipeline() -> Pipeline:
    """Preprocess numeric data using robust scaling.

    Returns:
        Pipeline: _description_
    """
    numeric_feature_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(
                missing_values=np.nan,
                strategy="median",
            )),
            ("scaler", RobustScaler()),
        ],
        verbose=True,
    )
    print(f"\nNumeric Data Pipeline Structure:\n{numeric_feature_pipeline}\n")
    return numeric_feature_pipeline


def preprocess_minmax_scaler_numeric_feature_pipeline() -> Pipeline:
    """Preprocess numeric data using min-max normalising.

    Returns:
        Pipeline: _description_
    """
    numeric_feature_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(
                missing_values=np.nan,
                strategy="median",
            )),
            ("scaler", MinMaxScaler()),
        ],
        verbose=True,
    )
    print(f"\nNumeric Data Pipeline Structure:\n{numeric_feature_pipeline}\n")
    return numeric_feature_pipeline


def preprocess_std_scaler_numeric_feature_pipeline() -> Pipeline:
    """Preprocess numeric data using standard scaling.

    Returns:
        Pipeline: _description_
    """
    numeric_feature_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(
                missing_values=np.nan,
                strategy="median",
            )),
            ("scaler", StandardScaler()),
        ],
        verbose=True,
    )
    print(f"\nNumeric Data Pipeline Structure:\n{numeric_feature_pipeline}\n")
    return numeric_feature_pipeline


def preprocess_numeric_feature_pipeline(scaler: str) -> Pipeline:
    """preprocess_numeric_feature_pipeline _summary_.

    Args:
        scaler (str): _description_

    Returns:
        Pipeline: _description_
    """
    # Set up the dictionary for the scaler functions
    scaler_dictionary = {
        "standard_scaler": StandardScaler(),
        "min_max_scaler": MinMaxScaler(),
        "robust_scaler": RobustScaler()
    }
    scaler_class = scaler_dictionary.get(scaler)

    # Build the pipeline for the chosen scaler
    numeric_feature_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(
                missing_values=np.nan,
                strategy="median",
            )),
            ("scaler", scaler_class),
        ],
        verbose=True,
    )
    print(f"\nNumeric Data Pipeline Structure:\n{numeric_feature_pipeline}\n")
    return numeric_feature_pipeline


def preprocess_ordinal_categorical_feature_pipeline() -> Pipeline:
    """Preprocess categorical data using ordinal encoding.

    Returns:
        Pipeline: _description_
    """
    categorical_feature_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(
                strategy="most_frequent",
                fill_value="missing",
            )),
            ("encoder", OrdinalEncoder(handle_unknown="ignore")),
        ],
        verbose=True,
    )
    print("\nCategorical Data Pipeline Structure:")
    print(categorical_feature_pipeline)
    return categorical_feature_pipeline


def preprocess_one_hot_categorical_feature_pipeline() -> Pipeline:
    """Preprocess categorical data using one-hot encoding.

    Returns:
        Pipeline: _description_
    """
    categorical_feature_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(
                strategy="most_frequent",
                fill_value="missing",
            )),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ],
        verbose=True,
    )
    print("\nCategorical Data Pipeline Structure:")
    print(categorical_feature_pipeline)
    return categorical_feature_pipeline


def preprocess_categorical_feature_pipeline(encoder: str) -> Pipeline:
    """preprocess_numeric_feature_pipeline _summary_.

    Args:
        scaler (str): _description_

    Returns:
        Pipeline: _description_
    """
    # Set up the dictionary for the scaler functions
    encoder_dictionary = {
        "one_hot_encoder": OneHotEncoder(handle_unknown="ignore"),
        "ordinal_encoder": OrdinalEncoder(handle_unknown="ignore"),
    }
    encoder_class = encoder_dictionary.get(encoder)

    # Build the pipeline for the chosen scaler
    categorical_feature_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(
                strategy="most_frequent",
                fill_value="missing",
            )),
            ("encoder", encoder_class),
        ],
        verbose=True,
    )
    print("\nCategorical Data Pipeline Structure:")
    print(categorical_feature_pipeline)
    return categorical_feature_pipeline


def transform_feature_pipeline(
    numeric_feature_pipeline: Pipeline,
    categorical_feature_pipeline: Pipeline,
) -> ColumnTransformer:
    """Transform the selected features in the pipeline.

    For the 'transformers' parameters, the settings represent, respectively:
        - the transformer name
        - the transformer pipeline it represents
        - the columns included in the transformer

    Args:
        numeric_feature_pipeline (Pipeline): _description_
        categorical_feature_pipeline (Pipeline): _description_

    Returns:
        ColumnTransformer: _description_
    """
    feature_transformer = ColumnTransformer(
        transformers=[
            (
                "numeric features",
                numeric_feature_pipeline,
                selector(dtype_include="number")
            ),
            (
                "categorical features",
                categorical_feature_pipeline,
                selector(dtype_include="category")
            ),
        ],
        verbose_feature_names_out=False,  # if True, will display transformers
        remainder="drop",
        n_jobs=-1,
    )
    return feature_transformer


def get_transformed_feature_pipeline(
    pipeline: Pipeline,
    features_data: pd.DataFrame,
) -> pd.DataFrame:
    """Fit/transform preprocessed train or test data individually.

    This is to check data were processed through properly, i.e. categorical and
    numeric preprocessing was applied successfully.

    NOTE: If needed, it is possible to set the config pipeline output to a
    (dense, not sparse) Numpy array using the parameter below.
    set_config(transform_output="default")

    Args:
        pipeline (Pipeline): _description_
        features_data (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
    # Get a list of the pipeline features from an array
    # Using the slice '[:-1]' means we take the last step of the pipeline,
    # i.e. here, the model/classifier
    feature_names_from_pipe = pipeline[:-1].get_feature_names_out().tolist()

    # Similarly, we can extract the preprocessed (transformed) array data
    transformed_features_from_pipeline = (
        pipeline[:-1].fit_transform(features_data)
    )
    # Convert the array into a dataframe
    preprocessed_df = pd.DataFrame(
        data=transformed_features_from_pipeline,
        columns=feature_names_from_pipe,
        index=features_data.index,
    )
    print(f"\nPreprocessed Data:\n{preprocessed_df.head(3)}\n")
    return preprocessed_df


def calculate_cross_validation_scores(
    model,
    features_test: pd.DataFrame,
    target_test: pd.Series,
    target_pred: pd.Series,
    target_label_list: List[str],
    cv: int = 5
) -> None:
    """calculate_cross_validation_score _summary_.

    Args:
        model (_type_): _description_
        features_test (pd.DataFrame): _description_
        target_test (pd.Series): _description_
        target_pred (pd.Series): _description_
        target_label_list (List[str]): _description_
    """
    # Use below function when there is an imbalance in each target class
    balanced_accuracy_score_ = balanced_accuracy_score(
        y_true=target_test,
        y_pred=target_pred,
    )
    print(f"\nBalanced Accuracy Score = {balanced_accuracy_score_:.2%}\n")

    # Apply cross-validation to improve the variability of the score
    cv_accuracy_score = cross_val_score(
        estimator=model,
        X=features_test,
        y=target_test,
        scoring="accuracy",
        cv=cv,
        n_jobs=-1,
    )
    cv_accuracy_score_mean = cv_accuracy_score.mean()
    cv_accuracy_score_stdev = cv_accuracy_score.std()
    print(
        f"Cross-Validation Accuracy Score = {cv_accuracy_score_mean:.2%}"
        f" +/- {cv_accuracy_score_stdev.std():.2%}"
    )

    # Produce a classification report
    classification_report_ = classification_report(
        y_true=target_test,
        y_pred=target_pred,
        digits=3,
        target_names=target_label_list,
        # output_dict=True,
    )
    print(f"\nClassification Report:\n{classification_report_}\n")


def calculate_multiple_cross_validation_scores(
    model,
    features: pd.DataFrame | np.ndarray,
    target: pd.Series | np.ndarray,
    cv: int = 5
) -> None:
    """calculate_multiple_cross_validation_scores _summary_.

    Use the 'cross_validate()' function to calculate multiple model scores for
    EACH train and test sets.
    NOTE: to print out the train scores, the parameter 'return_train_score'
    must be set to 'True'.

    Args:
        model (_type_): _description_
        features (pd.DataFrame | np.ndarray): _description_
        target (pd.Series | np.ndarray): _description_
    """
    scorer_list = [
        "balanced_accuracy",
        "precision_macro",
        "recall_macro",
        "f1_weighted"
    ]
    valid_scores_cv = cross_validate(
        estimator=model,
        X=features,
        y=target,
        # groups=group_list,
        scoring=scorer_list,
        cv=cv,
        n_jobs=-1,
        verbose=1,
        return_train_score=False,
        # error_score=np.nan
    )

    # Convert output dictionary into a dataframe
    valid_scores_cv_dataframe = pd.DataFrame(
        data=valid_scores_cv,
        columns=valid_scores_cv.keys(),
    )

    # Create a new df to hold the mean & standard deviation for each scorer
    valid_scores_cv_means = pd.DataFrame()
    valid_scores_cv_means["CV Mean"] = valid_scores_cv_dataframe.mean()
    valid_scores_cv_means["CV StDev"] = valid_scores_cv_dataframe.std()
    # Format scores as percentages (delete 'fit_time' and 'score_time' columns)
    valid_scores_cv_means = valid_scores_cv_means.iloc[2:].applymap(
        lambda float_: f"{float_:.2%}"
    )
    print(f"\nModel Score Output*:\n{valid_scores_cv_means}\n")
    print("""
        * Note: the scores calculated for the 'test' set are derived from the
        'split' of the input data (i.e. the train set), hence the output values
        should be very close to the ones calculated with the function
        'calculate_cross_validation_prediction_scores', which uses the
        'cross_val_predict' function from Scikit-learn library.
    """)


def calculate_cross_validation_prediction_scores(
    model,
    features: pd.DataFrame,
    target: pd.Series,
    # groups: np.ndarray,
    cv: int = 5
) -> None:
    """calculate_cross_validation_predictions _summary_.

    Args:
        model (_type_): _description_
        features (pd.DataFrame): _description_
        target (pd.Series): _description_
        cv (int, optional): _description_. Defaults to 5.

    TODO code 'roc_auc_score' function
    """
    target_pred_cv = cross_val_predict(
        estimator=model,
        X=features,
        y=target,
        # groups=,  # how does it work ?
        cv=cv,
        n_jobs=-1,
    )

    # roc_auc_score_ = roc_auc_score(
    #     y_true=target,
    #     y_score=,
    #     average="macro",
    #     # max_fprfloat=1,
    #     # multi_class="ovr",
    #     multi_class="ovo",
    #     labels=,
    # )
    # TODO apply 'roc_curve' separately

    # Set a list of scoring methods to be applied (imported at top of script)
    score_list = [precision_score, recall_score, f1_score]

    # Create a dictionary of score means and standard deviations
    mean_scores = defaultdict(list)
    stdev_scores = defaultdict(list)

    # Apply each scoring method 'score' to a loop & append output to dictionary
    for score in score_list:
        score_name = (
            score.__name__.replace("_", " ").replace(
                "score", "").capitalize()
        )
        score_aggregation = score(
            y_true=target,
            y_pred=target_pred_cv,
            # labels=groups,  # how does it work ?
            average=None
        )
        # Calculate mean and standard deviation of each score
        mean_scores[f"{score_name}"].append(score_aggregation.mean())
        stdev_scores[f"{score_name}"].append(score_aggregation.std())

    # Convert the dictionaries and concatenate them
    score_mean_df = pd.DataFrame(data=mean_scores)
    score_stdev_df = pd.DataFrame(data=stdev_scores)
    prediction_scores_dataframe = pd.concat(
        objs=[score_mean_df, score_stdev_df],
        axis=0
    )
    prediction_scores_dataframe.index = ["Mean", "StDev"]

    # Format scores as percentages
    prediction_scores_dataframe = prediction_scores_dataframe.applymap(
        lambda float_: f"{float_:.2%}"
    )
    print(f"\nModel Prediction Scores:\n{prediction_scores_dataframe}\n")


def train_tree_classifier(
    features_train: pd.DataFrame | np.ndarray,
    target_train: pd.Series | np.ndarray,
    features_test: pd.DataFrame | np.ndarray,
    target_test: pd.Series | np.ndarray,
    target_pred: pd.Series | np.ndarray,
    index_name: str
) -> pd.DataFrame:
    """Train a tree classifier.

    Args:
        features_train (pd.DataFrame): _description_
        target_train (pd.Series): _description_
        features_test (pd.DataFrame): _description_
        target_test (pd.Series): _description_
        target_pred (pd.Series): _description_

    Returns:
        pd.DataFrame: _description_
    """
    tree_classifier = DecisionTreeClassifier(
        criterion="gini",
        max_leaf_nodes=5,
        # max_depth=3,
        # max_features="sqrt",
        random_state=42
    )
    tree_classifier.fit(X=features_train, y=target_train)
    tree_classifier.predict(
        X=features_test
    )

    # Build a dataframe of true/test values vs prediction values
    target_pred_tree_classifier_df = pd.DataFrame(
        data=target_pred,
        columns=["predictions"]
    )

    # Compare the predictions against the target
    prediction_list_tree_classifier = [
        target_test.reset_index(),  # need resetting index before concatenating
        target_pred_tree_classifier_df
    ]
    predictions_tree_classifier = pd.concat(
        objs=prediction_list_tree_classifier,
        axis=1
    ).set_index(keys=index_name)
    print("Test vs. Predictions for tree classifier:")
    print(predictions_tree_classifier)
    return tree_classifier, predictions_tree_classifier


def show_tree_classifier_feature_importances(
    tree_classifier,
    feature_name_list: List[str],
    features_train: pd.DataFrame | np.ndarray,
    target_train: pd.Series | np.ndarray,
    features_test: pd.DataFrame | np.ndarray,
    target_test: pd.Series | np.ndarray,
) -> None:
    """show_tree_classifier_feature_importances _summary_.

    Args:
        tree_classifier (_type_): _description_
        feature_name_list (List[str]): _description_
        features_train (pd.DataFrame): _description_
        target_train (pd.Series): _description_
        features_test (pd.DataFrame): _description_
        target_test (pd.Series): _description_
    """
    # Create a df to store the importance of features in tree classification
    # Get classification feature importance scores same way they are ordered
    # in the source dataset
    tree_classification_feat_importance = tree_classifier.feature_importances_
    # Sort the INDEX of classification feature importance scores in desc. order
    tree_classification_indices = np.argsort(
        tree_classification_feat_importance)[::-1]
    # Reorder classification importance scores according to the previous step
    reordered_feat_importance = [
        tree_classification_feat_importance[index]
        for index in tree_classification_indices
    ]
    # Reorder the classification feature names according to the previous step
    reordered_classification_names = [
        feature_name_list[index] for index in tree_classification_indices
    ]
    # # Keep 1st part of classification feat. name to show it pretty in a table
    # tree_classification_feat_names = [
    #     name.split(sep="_")[1] for name in reordered_classification_names
    # ]
    tree_classification_importances = pd.DataFrame(
        data=[reordered_feat_importance, reordered_classification_names],
    ).T  # transpose to rotate rows to columns
    tree_classification_importances.columns = ["Importance", "Feature"]
    print("\nImportance of Features in Tree Classification:")
    print(tree_classification_importances)

    tree_classifiers_train_score = tree_classifier.score(
        features_train, target_train
    )
    print(
        f"\nDecision Tree Mean Accuracy Score for Train Set = \
    {tree_classifiers_train_score:.1%}\n"
    )
    tree_classifiers_test_score = tree_classifier.score(
        features_test, target_test
    )
    print(
        f"\nDecision Tree Mean Accuracy Score for Test Set = \
    {tree_classifiers_test_score:.1%}\n"
    )


def draw_decision_tree(
    tree_classifier,
    feature_name_list: List[str],
    target_label_list: List[str],
    figure_name: str,
    output_directory: Path,
) -> None:
    """_summary_.

    Args:
        tree_classifier (_type_): _description_
        feature_name_list (List[str]): _description_
        target_label_list (List[str]): _description_
        figure_path_name (str | Path): _description_
    """
    # Define the tree classifier parameters
    n_nodes = tree_classifier.tree_.node_count
    children_left = tree_classifier.tree_.children_left
    children_right = tree_classifier.tree_.children_right
    feature = tree_classifier.tree_.feature
    threshold_value = tree_classifier.tree_.threshold

    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
    while len(stack) > 0:
        # `pop` ensures each node is only visited once
        node_id, depth = stack.pop()
        node_depth[node_id] = depth

        # If the left and right child of a node is not the same we have a split
        # node
        is_split_node = children_left[node_id] != children_right[node_id]
        # If a split node, append left and right children and depth to `stack`
        # so we can loop through them
        if is_split_node:
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
        else:
            is_leaves[node_id] = True

    print(
        f"\nThe decision tree has {n_nodes} nodes "
        f"and has the following structure:\n"
    )
    print("Feature order in the decision tree as described below:")
    print(feature_name_list)
    print(f"\n{target_label_list = }\n")

    for index in range(n_nodes):
        if is_leaves[index]:
            print(f"node={index} is a leaf node.")
        else:
            print(
                f"\nNode={index} is a split node: "
                f"go to node {children_left[index]} "
                f"if feature {feature[index]} <= {threshold_value[index]:.3f},"
                f" else go to node {children_right[index]}."
            )

    # Display the tree structure
    plt.subplots(figsize=(10, 10))
    plot_tree(
        decision_tree=tree_classifier,
        feature_names=feature_name_list,
        class_names=target_label_list,
        filled=True,
    )
    plt.title(
        label="Decision Tree for the Identification of Target Categories",
        fontsize=20
    )
    plt.tight_layout()
    # plt.show()
    save_figure(figure_name=output_directory.joinpath(figure_name + ".png"))


def draw_random_forest_tree(
    random_forest_classifier,
    feature_name_list: List[str],
    target_label_list: List[str],
    figure_name: str,
    output_directory: Path,
    ranked_tree: int = None,
) -> None:
    """_summary_.

    Args:
        random_forest_classifier (_type_): _description_
        feature_name_list (List[str]): _description_
        target_label_list (List[str]): _description_
        figure_path_name (str | Path): _description_
    """
    # plt.figure()
    tree.plot(
        model=random_forest_classifier,
        featnames=feature_name_list,
        num_trees=ranked_tree,
        plottype="vertical",
    )
    plt.title(
        label=(
            "Random Forest Tree for the Identification of Target Categories"
        ),
        fontsize=16
    )
    plt.tight_layout()
    # plt.show()
    save_figure(figure_name=output_directory.joinpath(figure_name + ".png"))


def draw_confusion_matrix_heatmap(
    target_test: pd.Series,
    target_pred: pd.Series,
    target_label_list: List[str],
    figure_name: str,
    output_directory: Path,
) -> None:
    """_summary_.

    Args:
        target_test (pd.Series): _description_
        target_pred (pd.Series): _description_
        target_label_list (List[str]): _description_
        figure_path_name (str): _description_
    """
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
        label="Confusion matrix of predictions",
        fontsize=16
    )
    plt.grid(visible=False)
    # plt.axis("off")
    plt.tight_layout()
    # plt.show()
    save_figure(figure_name=output_directory.joinpath(figure_name + ".png"))


def get_feature_importance_scores(
    model,
    feature_name_list: List[str],
    figure_name: str,
    output_directory: Path,
) -> pd.Series:
    """get_feature_importance_scores _summary_.

    Args:
        model (_type_): _description_
        feature_name_list (List[str]): _description_
        figure_path_name (str | Path): _description_

    Returns:
        pd.Series: _description_
    """
    # Get feature importance scores the same way they are ordered in the
    # source dataset
    feature_importances = model.feature_importances_
    # Sort the INDEX of the feature importance scores in descending order
    feature_indices = np.argsort(feature_importances)[::-1]
    # Reorder the feature names according to the previous step
    feature_names = [
        feature_name_list[index] for index in feature_indices
    ]
    # Calculate the standard deviation of all estimators
    estimator_std = np.std(
        [tree.feature_importances_ for tree in model.estimators_],
        axis=0
    )
    # Create a Pandas Series to plot the data
    model_feature_importances = pd.Series(
        data=feature_importances[feature_indices],
        index=feature_names
    )

    # Create a bar plot
    fig, ax = plt.subplots()
    model_feature_importances.plot.barh(
        # xerr=estimator_std,
        align="center",
        ax=ax
    )
    plt.ylabel("Parameters")
    plt.xlabel("Mean decrease in Impurity")
    plt.title(label=("Feature Importance in Predictions"), fontsize=16)
    plt.grid(visible=False)
    fig.tight_layout()
    # plt.show()
    save_figure(figure_name=output_directory.joinpath(figure_name + ".png"))
    return model_feature_importances


def get_feature_importance_scores_permutation(
    model,
    features_test: pd.DataFrame,
    target_test: pd.Series,
    feature_name_list: List[str],
    figure_name: str,
    output_directory: Path,
    n_repeats=10,
) -> pd.Series:
    """Feature importance based on feature permutation.

    This removes bias toward high-cardinality features.

    Args:
        model (_type_): _description_
        features_test (pd.DataFrame): _description_
        target_test (pd.Series): _description_
        feature_name_list (List[str]): _description_
        figure_path_name (str | Path): _description_
        n_repeats (int, optional): _description_. Defaults to 10.

    Returns:
        pd.Series: _description_
    """
    permutation_result = permutation_importance(
        estimator=model,
        X=features_test,
        y=target_test,
        n_repeats=n_repeats,
        random_state=42,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
    )
    # Try below
    print(f"\n{permutation_result.importances = }\n")

    # Create a Pandas Series to plot the data
    model_feature_importances_permutation = pd.Series(
        data=permutation_result.importances_mean,
        index=feature_name_list
    )
    fig, ax = plt.subplots()
    model_feature_importances_permutation.plot.barh(
        xerr=permutation_result.importances_std,
        align="center",
        ax=ax
    )
    plt.ylabel("Parameters")
    plt.xlabel("Mean accuracy decrease")
    plt.title(label=("Feature importance in predictions"), fontsize=16)
    plt.grid(visible=False)
    fig.tight_layout()
    # plt.show()
    save_figure(figure_name=output_directory.joinpath(figure_name + ".png"))
    return model_feature_importances_permutation


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
