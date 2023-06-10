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
from typing import Dict, List, Optional, Tuple

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
from scipy.spatial.distance import mahalanobis
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
    DropCorrelatedFeatures,
    DropDuplicateFeatures,
    DropFeatures,
    DropMissingData,
)
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import (
    BorderlineSMOTE,
    KMeansSMOTE,
    RandomOverSampler,
    SMOTE,
    SVMSMOTE,
)
from imblearn.under_sampling import RandomUnderSampler
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
# from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    # accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve
)
from sklearn.model_selection import (
    # GridSearchCV,
    cross_validate,
    RandomizedSearchCV,
    StratifiedKFold,
    # RepeatedStratifiedKFold,
    # StratifiedShuffleSplit,
    cross_val_predict,
    cross_val_score,
    train_test_split
)
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import (
    FunctionTransformer,
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
from skopt import BayesSearchCV, space
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
            assert choice in ("y", "n"), "Enter 'y' or 'n'"
            break
        except AssertionError as error:
            print(error)
        except ValueError:
            print("Please enter a valid letter choice.")
    if choice == "y":
        dataframe = dataframe_no_outliers.copy()
        print("\nOutliers were removed.\n")
    else:
        dataframe = dataframe_with_outliers.copy()
        print("\nOutliers were NOT removed.\n")
    return dataframe


# ----------------------------------------------------------------------------

# INPUT/OUTPUT


def get_folder_name_list_from_directory(directory_name: Path) -> List[str]:
    """Get the list of file (name) from a directory.

    Args:
        directory_name (Path): _description_

    Returns:
        _type_: _description_
    """
    paths = Path(directory_name).glob(pattern="*")
    folder_name_list = [str(path) for path in paths if path.is_dir()]
    return folder_name_list


def get_file_name_list_from_extension(
    directory_name: Path,
    extension: str
) -> List[str]:
    """Get the list of file (name) from a directory.

    Args:
        directory_name (Path): _description_
        extension (_type_): _description_

    Returns:
        _type_: _description_
    """
    paths = Path(directory_name).glob(pattern=f"**/*.{extension}")
    file_name_list = [str(path) for path in paths]
    return file_name_list


def load_pickle_file(
    file_name: str,
    input_directory: Path
) -> pd.DataFrame:
    """Load a pickle file into a dataframe.

    Args:
        file_name (str): Name of the pickle file, without extension.
        input_directory (Path): Directory path where the '.pkl' file resides.

    Returns:
        pd.DataFrame: _description_
    """
    dataframe = pd.read_pickle(
        filepath_or_buffer=input_directory.joinpath(f"{file_name}.pkl")
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
    file_name: str,
    input_directory: Path,
    sheet_name: str | int = 0,
    nrows: int = None,
) -> pd.DataFrame:
    """Load an Excel file.

    Args:
        file_name (str): The file name of the Excel file.
        output_directory (Path): The directory path where the file is located.
        sheet_name (str | int, optional): The name or index of the sheet to
            load. Defaults to 0.
        nrows (int, optional): The number of rows to read from the sheet.
            Defaults to None.

    Returns:
        pd.DataFrame: The loaded DataFrame object.
    """
    dataframe = pd.read_excel(
        io=input_directory.joinpath(f"{file_name}.xlsx"),
        sheet_name=sheet_name,
        header=0,
        nrows=nrows,
        index_col=None,
        decimal=".",
    )
    return dataframe


def load_csv_file(
    file_name: str,
    input_directory: Path,
    index_col: str = None,
) -> pd.DataFrame:
    """Load a CSV file.

    Args:
        file_name (str): Name of the CSV file, without extension.
        input_directory (Path): Directory path where the CSV file resides.
        index_col (str): Name of the column to be used as dataframe index.

    Returns:
        pd.DataFrame: DataFrame containing the data from the input CSV file.
    """
    dataframe = pd.read_csv(
        filepath_or_buffer=(
            input_directory.joinpath(f"{file_name}.csv")
        ),
        encoding="utf-8"
    )
    if index_col is not None:
        dataframe.set_index(keys=index_col, inplace=True)
    return dataframe


def save_csv_file(
    dataframe: pd.DataFrame,
    file_name: str,
    output_directory: Path,
) -> None:
    """Save a dataframe as a '.csv()' file.

    Args:
        dataframe (pd.DataFrame): Pandas DataFrame to be saved to CSV.
        file_name (str): File name of the CSV file. Without extension.
        output_directory (Path): Folder where the data is saved to.

    Returns:
        None.
    """
    dataframe.to_csv(
        path_or_buf=output_directory.joinpath(f"{file_name}.csv"),
        sep=",",
        encoding="utf-8",
        index=True,
    )
    return None


def save_excel_file(
    dataframe: pd.DataFrame,
    file_name: str,
    output_directory: Path,
) -> None:
    """Save a dataframe as a '.xlsx()' file.

    Args:
        dataframe (pd.DataFrame): Pandas DataFrame to be saved to Excel.
        file_name (str): File name of the Excel file. Without extension.
        output_directory (Path): Folder where the data is saved to.

    Returns:
        None.
    """
    dataframe.to_excel(
        excel_writer=output_directory.joinpath(f"{file_name}.xlsx"),
        index=False,
        sheet_name="Sheet1"
    )
    return None


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
        filename=output_directory.joinpath(f"{file_name}.joblib"),
    )
    return pipeline_model_file


def save_figure(file_name: str, output_directory: Path) -> None:
    """Save a figure as '.png' file.

    Args:
        file_name (str): File name of the image file. Without extension.
        output_directory (Path): Folder where the image is saved to.

    Returns:
        None.
    """
    plt.savefig(
        fname=output_directory.joinpath(f"{file_name}.png"),
        bbox_inches="tight",
        dpi=300
    )
    return None


def save_image_show(
    image: np.ndarray,
    image_title: str,
    file_name: str,
    output_directory: Path
):
    """save_image_show _summary_.

    Args:
        image (np.ndarray): _description_
        image_title (str): _description_
        file_name (str): _description_
        output_directory (Path): _description_

    Returns:
        _type_: _description_
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
    save_figure(file_name=file_name, output_directory=output_directory)
    # plt.show()
    return image


def save_npz_file(
    file_name: str,
    output_directory: Path,
    data_array: np.ndarray,
) -> None:
    """save_npz _summary_.

    Args:
        file_name (str): _description_
        output_directory (Path): _description_
        data_array (np.ndarray): _description_
    """
    np.savez(
        file=output_directory.joinpath(f"{file_name}"),
        data=data_array
    )


def save_pickle_file(
    dataframe: pd.DataFrame,
    file_name: str,
    output_directory: Path,
) -> None:
    """Save the dataframe as a pickle object.

    Args:
        dataframe (pd.DataFrame): _description_
        file_path_name (str): _description_

    Returns:
        None. Saves the dataframe as a pickle file.
    """
    dataframe.to_pickle(
        path=output_directory.joinpath(f"{file_name}.pkl"),
        protocol=-1
    )


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
        file=output_directory.joinpath(f"{file_name}.txt"),
        mode="r",
        encoding="utf-8"
    ) as output_file:
        # output_text = output_file.read()  # works fine
        output_text = Path(output_file).read_text(...)  # to be tested

    # Create a new Word document
    word_doc = docx.Document()
    # Add the text to the document
    word_doc.add_paragraph(output_text)
    # Save the document as a Word file
    word_doc.save(output_directory.joinpath(f"{file_name}.docx"))
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
        file=output_directory.joinpath(f"{file_name}.txt"),
        mode="r",
        encoding="utf-8"
    ) as output_file:
        # output_text = output_file.read()  # works fine
        output_text = Path(output_file).read_text(...)  # to be tested

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
def convert_npz_to_mat(input_directory: Path) -> None:
    """AI is creating summary for convert_npz_to_mat.

    Args:
        input_directory (Path): [description]
    """
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


def convert_dictionary_to_dataframe(dictionary_file: Dict) -> pd.DataFrame:
    """Convert a dictionary into a dataframe.

    First, the 'len()' function gets the number of items in the dictionary.
    Second, 'range()' is used to set a range from 0 to length of dictionary.
    Finally, 'list()' converts the items into NUMERIC index values.

    Args:
        dictionary_file (Dict): Dictionary containing data to be converted
            to a dataframe.

    Returns:
        pd.DataFrame: Dataframe representing the data in the input dictionary.
    """
    dataframe = pd.DataFrame(
        data=dictionary_file,
        index=[list(range(len(dictionary_file)))]
    )
    return dataframe


def convert_to_datetime_type(
    dataframe: pd.DataFrame,
    datetime_variable_list: List[str | datetime]
) -> pd.DataFrame:
    """Convert specified columns in a DataFrame to datetime type.

    Args:
        dataframe (pd.DataFrame): The DataFrame to be converted.
        datetime_variable_list (List[str | datetime]): List of column names or
            datetime variables to be converted to datetime type.

    Returns:
        pd.DataFrame: The DataFrame with specified columns converted to
            datetime type.
    """
    dataframe[datetime_variable_list] = (
        dataframe[datetime_variable_list].astype(dtype="datetime64[ns]")
    )
    return dataframe


def convert_to_category_type(
    dataframe: pd.DataFrame,
    category_variable_list: List[str]
) -> pd.DataFrame:
    """Convert specified columns in a DataFrame to category type.

    Args:
        dataframe (pd.DataFrame): The DataFrame to be converted.
        category_variable_list (List[str]): List of column names to be
            converted to category type.

    Returns:
        pd.DataFrame: The DataFrame with specified columns converted to
            category type.
    """
    dataframe[category_variable_list] = (
        dataframe[category_variable_list].astype(dtype="category")
    )
    return dataframe


def convert_to_number_type(
    dataframe: pd.DataFrame,
    numeric_variable_list: List[str]
) -> pd.DataFrame:
    """Convert specified columns in a DataFrame to numeric type.

    Args:
        dataframe (pd.DataFrame): The DataFrame to be converted.
        numeric_variable_list (List[str]): List of column names to be
            converted to string type.

    Returns:
        pd.DataFrame: The DataFrame with specified columns converted to
            number type.
    """
    dataframe[numeric_variable_list] = (
        dataframe[numeric_variable_list].astype(dtype="number")
    )
    return dataframe


def convert_to_integer_type(
    dataframe: pd.DataFrame,
    integer_variable_list: List[str]
) -> pd.DataFrame:
    """Convert specified columns in a DataFrame to integer type.

    Args:
        dataframe (pd.DataFrame): The DataFrame to be converted.
        integer_variable_list (List[str]): List of column names to be
            converted to integer type.

    Returns:
        pd.DataFrame: The DataFrame with specified columns converted to
            integer type.
    """
    dataframe[integer_variable_list] = (
        dataframe[integer_variable_list].astype(dtype="int")
    )
    return dataframe


def convert_to_float_type(
    dataframe: pd.DataFrame,
    float_variable_list: List[str]
) -> pd.DataFrame:
    """Convert specified columns in a DataFrame to float type.

    Args:
        dataframe (pd.DataFrame): The DataFrame to be converted.
        float_variable_list (List[str]): List of column names to be
            converted to float type.

    Returns:
        pd.DataFrame: The DataFrame with specified columns converted to
            float type.
    """
    dataframe[float_variable_list] = (
        dataframe[float_variable_list].astype(dtype="float")
    )
    return dataframe


def convert_to_string_type(
    dataframe: pd.DataFrame,
    string_variable_list: List[str]
) -> pd.DataFrame:
    """Convert specified columns in a DataFrame to string type.

    Args:
        dataframe (pd.DataFrame): The DataFrame to be converted.
        string_variable_list (List[str]): List of column names to be
            converted to string type.

    Returns:
        pd.DataFrame: The DataFrame with specified columns converted to
            string type.
    """
    dataframe[string_variable_list] = (
        dataframe[string_variable_list].astype(dtype="string")
    )
    return dataframe


def convert_variables_to_proper_type(
    dataframe: pd.DataFrame,
    datetime_variable_list: Optional[List[str | datetime]] = None,
    category_variable_list: Optional[List[str]] = None,
    numeric_variable_list: Optional[List[str]] = None,
    integer_variable_list: Optional[List[str]] = None,
    float_variable_list: Optional[List[str]] = None,
    string_variable_list: Optional[List[str]] = None
) -> pd.DataFrame:
    """Convert variables in a Pandas DataFrame to their proper data type.

    Apply the '.pipe()' method to the defined functions.

    Args:
        dataframe (pd.DataFrame): The DataFrame to convert variables to
            proper data type.
        datetime_variable_list (Optional[List[str  |  datetime]], optional):
            List of variable names to convert to 'datetime' type.
            Defaults to None.
        category_variable_list (Optional[List[str]], optional):  List of
            variable names to convert to 'category' type. Defaults to None.
        integer_variable_list (Optional[List[str]], optional): List of
            variable names to convert to 'integer' type. Defaults to None.
        float_variable_list (Optional[List[str]], optional): List of
            variable names to convert to 'float' type. Defaults to None.
        string_variable_list (Optional[List[str]], optional): List of
            variable names to convert to 'string' type. Defaults to None.

    Returns:
        pd.DataFrame: DataFrame with proper data types for selected variables.
    """
    if category_variable_list is None:
        category_variable_list = []
    if numeric_variable_list is None:
        numeric_variable_list = []
    if integer_variable_list is None:
        integer_variable_list = []
    if float_variable_list is None:
        float_variable_list = []
    if string_variable_list is None:
        string_variable_list = []

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
        dataframe (pd.DataFrame): DataFrame containing duplicated rows to be
            removed.

    Returns:
        pd.DataFrame: DataFrame with all duplicated rows removed.
    """
    # Drop all duplicated rows and reset the index
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
    """Filter a dataframe using the provided filter.

    Args:
        dataframe (pd.DataFrame): Pandas dataframe to be filtered.
        filter_content (str): Filter string to be applied on the dataframe.

    Returns:
        pd.DataFrame: The filtered pandas dataframe.
    """
    print(f"\nThe following filter was applied:\n{filter_content}\n")

    # Apply the filter
    filtered_dataframe = dataframe.query(filter_content)
    print(
        f"Filtered Cluster Dataset Shape:\n{filtered_dataframe.shape}\n"
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
    """Extract numeric features from a Pandas DataFrame.

    Args:
        dataframe (pd.DataFrame): DataFrame to select numeric features from.

    Returns:
        Tuple[pd.Dataframe, List[str]]: A tuple containing a pandas DataFrame
            with only numeric features and a list of the feature names.
    """
    # Select numeric variables ONLY and make a list
    numeric_features = dataframe.select_dtypes(include=np.number)
    numeric_feature_list = numeric_features.columns.to_list()
    print(f"\nList of Numeric Features:\n{numeric_feature_list}\n")
    print(f"\nSummary Statistics:\n{numeric_features.describe()}\n")
    return numeric_features, numeric_feature_list


def get_categorical_features(dataframe):
    """Extract categorical features from a Pandas DataFrame.

    Args:
        dataframe (pd.DataFrame): Df to select categorical features from.

    Returns:
        Tuple (pd.Dataframe, List[str]): A tuple containing a pandas DataFrame
            with only categorical features and a list of the feature names.
    """
    # Select categorical variables ONLY and make a list
    categorical_features = dataframe.select_dtypes(include="category")
    categorical_features_list = categorical_features.columns.to_list()
    print(f"\nList of Categorical Features:\n{categorical_features_list}\n")
    print(f"\nSummary Statistics:\n{categorical_features.describe()}\n")
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


def run_exploratory_data_analysis(
    dataframe: pd.DataFrame,
    file_name: str,
    output_directory: Path,
) -> pd.DataFrame:
    """Print out the summary descriptive statistics from the dataset.

    Also includes a missing table summary.

    Args:
        dataframe (pd.DataFrame): Input dataset.
        file_name (str): File name.
        output_directory (Path): Save directory.

    Returns:
        pd.DataFrame: Table with summary statistics.
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

    # Concatenate the metrics into a dataframe
    metrics_list = [
        summary_stats,
        pct_variation,
        mad,
        kurtosis,
        skewness,
    ]
    summary_stats_table = pd.concat(
        objs=metrics_list,
        sort=False,
        axis=1
    )
    print(f"\nExploratory Data Analysis:\n{summary_stats_table}\n")

    # Save statistics summary to .csv file
    save_csv_file(
        dataframe=summary_stats_table,
        file_name=file_name,
        output_directory=output_directory,
    )
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
    """Generate an exploratory data analysis (EDA) report.

    The Sweetviz library is used.

    Args:
        train_set (pd.DataFrame): The training dataset as a Pandas DataFrame.
        test_set (pd.DataFrame): The testing dataset as a Pandas DataFrame.
        eda_report_name (str): The name of the EDA report.
        output_directory (Path): Directory where the EDA report will be saved.

    Returns:
        None. Produces an '.html' file of the main steps of the EDA.
    """
    print("\nPreparing SweetViz Report:\n")
    sweetviz_eda_report = sv.analyze(
        source=dataframe,
        pairwise_analysis="auto"
    )
    sweetviz_eda_report.show_html(
        filepath=output_directory.joinpath(f"{eda_report_name}.html"),
        open_browser=True,
        layout="widescreen",
    )


def compare_sweetviz_eda_report(
    train_set: pd.DataFrame,
    test_set: pd.DataFrame,
    eda_report_name: str,
    output_directory: Path,
) -> None:
    """Generate an exploratory data analysis (EDA) report.

    The Sweetviz library is used.

    Particularly, this report compares the data split between the training
    and testing sets.

    Args:
        train_set (pd.DataFrame): The training dataset as a Pandas DataFrame.
        test_set (pd.DataFrame): The testing dataset as a Pandas DataFrame.
        eda_report_name (str): The name of the EDA report.
        output_directory (Path): Directory where the EDA report will be saved.

    Returns:
        None. Produces an '.html' file of the main steps of the EDA.
    """
    print("\nPreparing SweetViz Report:\n")
    sweetviz_eda_report = sv.compare(
        [train_set, "Training Data"], [test_set, "Test Data"]
    )
    sweetviz_eda_report.show_html(
        filepath=output_directory.joinpath(f"{eda_report_name}.html"),
        open_browser=True,
        layout="widescreen",
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
        dataframe (pd.DataFrame): The dataframe the metrics should be
            calculated on.

    Returns:
        pd.DataFrame: The metric table of missing values.
    """
    # Produce a heatmap of missing values
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
        var (np.ndarray): vector or matrix of data with, say, p columns.
        data (_type_, optional): ndarray of the distribution from which
            Mahalanobis distance of each observation of x is to be computed.
        cov (pd.DataFrame): covariance matrix (p x p) of the distribution.
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


def calculate_mahalanobis_distance_alt(dataframe: pd.DataFrame) -> List[float]:
    """Calculate Mahalanobis distance between each pair of rows in dataframe.

    Args:
        dataframe (pd.DataFrame): Input dataframe.

    Returns:
        List[float]: List of Mahalanobis distances.
    """
    # Compute the covariance matrix of the data
    cov = np.cov(dataframe.select_dtypes(include=np.number).T)

    # Compute the inverse covariance matrix of the data
    inv_cov = np.linalg.inv(cov)

    # Compute the Mahalanobis distance between each pair of rows using a nested
    # list comprehension
    mahalanobis_distance_list = [
        mahalanobis(dataframe.iloc[i], dataframe.iloc[j], inv_cov)
        for i in range(dataframe.shape[0])
        for j in range(i + 1, dataframe.shape[0])
    ]
    print(f"\n{mahalanobis_distance_list = }\n")
    return mahalanobis_distance_list


def apply_mahalanobis_test(
    dataframe: pd.DataFrame,
    alpha: float = 0.01
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the Mahalanobis test to identify outliers in the dataframe.

    Args:
        dataframe (pd.DataFrame): Input dataframe.
        alpha (float, optional): Significance level for test. Defaults to 0.01.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the Mahalanobis
        dataframe and the outlier dataframe.
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
        f"\n\nMahalanobis Test Critical Value for alpha < {alpha}:"
        f"{mahalanobis_test_critical_value:.2f}"
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


def concatenate_outliers_with_target_category_dataframe(
    dataframe: pd.DataFrame,
    target_category_list: List[str],
    data_outliers: pd.DataFrame,
    feature: str,
    outlier_method: str,
    output_directory: Path,
) -> pd.DataFrame:
    """Concatenate the outliers with their corresponding target categories.

    The output dataframe is saved as a CSV file.

    Args:
        dataframe(pd.DataFrame): Target category dataframe.
        target_category_list(List[str]): List of target categories.
        data_outliers(pd.DataFrame): Outliers dataframe.
        feature(str): Name of the feature.
        outlier_method(str): Name of the outlier detection method.
        output_directory(Path): Directory to save the output file.

    Returns:
        pd.DataFrame: Merged dataframe of target categories and outliers.
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
        file_name=f"{feature}_{outlier_method}_outliers",
        output_directory=output_directory,
    )
    return outlier_dataframe


def get_iqr_outliers(
    dataframe: pd.DataFrame,
    column_name: str
) -> pd.DataFrame:
    """Get IQR (Inter Quantile Range) outliers for a column in the dataframe.

    Args:
        dataframe (pd.DataFrame): Input dataframe.
        column_name (str): Name of the column to find IQR outliers.

    Returns:
        pd.Series: Series containing the IQR outliers.
    """
    print("\n==============================================================\n")
    print(f"\n{column_name.upper()}\n")

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
        ][column_name]
    )

    # Calculate the ratio of outliers
    iqr_outlier_ratio = len(iqr_outlier_dataframe) / len(dataframe)
    print(
        f"There are {len(iqr_outlier_dataframe)} "
        f"({iqr_outlier_ratio:.1%}) IQR outliers.\n"
    )
    # print(f"Table of outliers for {column_name} based on IQR value:")
    # print(iqr_outlier_dataframe)
    return iqr_outlier_dataframe


def get_zscore_outliers(
    dataframe: pd.DataFrame,
    column_name: str,
    zscore_threshold: int = 3
):
    """Get Z-score outliers for a specific column in the dataframe.

    Args:
        dataframe (pd.DataFrame): Input dataframe.
        column_name (str): Name of the column to find Z-score outliers.
        zscore_threshold (int, optional): Threshold value to reject outliers.
            Defaults to 3.

    Returns:
        pd.Series: Series containing the Z-score outliers.
    """
    print("\n==============================================================\n")
    print(f"\n{column_name.upper()}\n")

    # Calculate Z-score
    z_score = np.abs(zscore(a=dataframe[column_name]))

    # Find outliers based on Z-score threshold value
    zscore_outlier_dataframe = (
        dataframe[z_score > zscore_threshold][column_name]
    )

    # Calculate the ratio of outliers
    zscore_outlier_ratio = len(zscore_outlier_dataframe) / len(dataframe)
    print(
        f"There are {len(zscore_outlier_dataframe)} "
        f"({zscore_outlier_ratio:.1%}) Z-score outliers.\n"
    )
    # print(f"Table of outliers for {column_name} based on Z-score value:")
    # print(zscore_outlier_dataframe)
    return zscore_outlier_dataframe


def get_mad_outliers(
    dataframe: pd.DataFrame,
    column_name: str,
) -> pd.DataFrame:
    """Get MAD (Mean Absolute Deviation) outliers for a column in a dataframe.

    Args:
        dataframe (pd.DataFrame): Input dataframe.
        column_name (str): Name of the column to find MAD outliers.

    Returns:
        pd.Series: Series containing the MAD outliers.
    """
    print("\n==============================================================\n")
    print(f"\n{column_name.upper()}\n")

    # Reshape the target column to make it 2D
    column_2d = dataframe[column_name].values.reshape(-1, 1)
    # Fit to the target column
    mad = MAD().fit(column_2d)

    # Extract the inlier/outlier labels
    labels = mad.labels_
    # print(f"\nMean Absolute Deviation Labels:\n{labels}\n")

    # Extract the outliers
    # use '== 0' to get inliers
    mad_outlier_dataframe = dataframe[column_name][labels == 1]

    # Calculate the ratio of outliers
    mad_outlier_ratio = len(mad_outlier_dataframe) / len(dataframe)
    print(
        f"There are {len(mad_outlier_dataframe)} "
        f"({mad_outlier_ratio:.1%}) MAD outliers.\n"
    )
    # print(f"Table of outliers for {column_name} based on MAD value:")
    # print(mad_outlier_dataframe)
    return mad_outlier_dataframe


def detect_univariate_outliers(
    dataframe: pd.DataFrame,
    target_category_list: List[str],
    output_directory: Path
) -> pd.DataFrame:
    """Detect outliers using several methods: IQR, Z-score and MAD.

    It is a univariate analysis since each feature is analysed individually.
    See 'detect_multivariate_outliers' function for a parallel analysis of all
    features.

    Args:
        dataframe (pd.DataFrame): Input dataframe.
        output_directory (Path): Directory to save the output file.

    Returns:
        pd.DataFrame: Dataframe containing outliers from the various methods.
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
    output_directory: Path
) -> Tuple[pd.DataFrame]:
    """detect_multivariate_outliers _summary_.

    Args:
        dataframe (pd.DataFrame): _description_
        target_category_list (List[str]): _description_
        output_directory (Path): _description_

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
        file_name="mahalanobis_outliers",
        output_directory=output_directory,
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
        f"\nThere are {len(features_to_drop)} features to drop due to high "
        f"correlation:\n{features_to_drop}\n"
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


def run_pc_analysis(
    features: pd.DataFrame,
    eda_report_name: str,
    output_directory: Path
) -> Tuple[pd.DataFrame | np.ndarray, List[str]]:
    """run_pc_analysis _summary_.

    Args:
        features (pd.DataFrame): _description_
        eda_report_name (str): _description_
        output_directory (Path): _description_

    Returns:
        Tuple[pd.DataFrame | np.ndarray], List[str]: _description_
    """
    # Select only the numeric input variables, i.e. not mahalanobis variables
    numeric_feature_list = (
        features.select_dtypes(include="number").columns.to_list()
    )
    print(f"\nSelected (Numeric) Features for PCA:\n{numeric_feature_list}\n")

    # Scale features data
    features_scaled = standardise_features(features=features)

    # Run the PC analysis
    pca_model, pca_array = apply_pca(
        # n_components=len(features_scaled.columns),
        n_components=0.95,  # set to keep PCs with cumulated variance of 95%
        features_scaled=features_scaled
    )

    # Get eigenvalues and eigenvectors
    pca_eigen_values = get_pca_eigen_values_vectors(
        # n_components=len(features_scaled.columns),
        # n_components=0.95,  # set to keep PCs with cumulated variance of 95%
        # features_scaled=features_scaled,
        pca_model=pca_model
    )

    # Convert PCA array to a dataframe
    pca_dataframe = pd.DataFrame(data=pca_array)
    pca_dataframe.reset_index()

    # Get PC axis labels and insert into the dataframe
    pca_components = [
        # f"pc{str(col + 1)}" for col in pca_dataframe.columns.to_list()
        f"pc{col + 1}" for col in pca_dataframe.columns.to_list()  # to test
    ]
    pca_dataframe.columns = pca_components

    # Calculate variance explained by PCA
    variance_explained_df = (
        explain_pca_variance(
            pca_eigen_values=pca_eigen_values,
            pca_model=pca_model,
            pca_components=pca_components,
        )
    )
    # Set index as dataframe
    pca_dataframe = pca_dataframe.set_index(keys=[features.index])

    # Keep the PC axes that correspond to AT LEAST 95% of the cumulated
    # explained variance
    best_pc_axis_names, best_pc_axis_values = find_best_pc_axes(
        variance_explained_df=variance_explained_df,
        percent_cut_off_threshold=95
    )

    # Subset the PCA dataframe to include ONLY the best PC axes
    final_pca_df = pca_dataframe.loc[:, best_pc_axis_names]
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
    plt.title(label="Scree Plot PCA", fontsize=16)
    plt.ylabel("Percentage of Variance Explained")
    save_figure(file_name="pca_scree_plot", output_directory=output_directory)

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
    save_figure(
        file_name="pca_loading_table", output_directory=output_directory
    )

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


def run_anova_check_assumptions(
    dataframe: pd.DataFrame,
    dependent_variable: pd.Series,
    independent_variable: pd.Series | str,
    group_variable: List[str],
    output_directory: Path,
    confidence_interval: float = 0.95,
) -> ols:
    """AI is creating summary for run_anova_check_assumptions.

    Args:
        dataframe (pd.DataFrame): [description]
        dependent_variable (pd.Series): [description]
        independent_variable (pd.Series): [description]
        group_variable (List[str]): [description]
        output_directory (Path): [description]
        confidence_interval (float, optional): [description]. Defaults to 0.95.

    Returns:
        ols: [description]
    """
    anova_model, tukey_results = run_anova_test(
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
        independent_variable=independent_variable,
        dependent_variable=dependent_variable,
        tukey_results=tukey_results,
        output_directory=output_directory,
        confidence_interval=confidence_interval,
    )
    return anova_model


def run_anova_test(
    dataframe: pd.DataFrame,
    dependent_variable: pd.Series,
    independent_variable: str,
    group_variable: str,
    output_directory: Path,
    confidence_interval: float = 0.95,
) -> None:
    """ANOVA test on numeric variables and calculate group confidence interval.

    Args:
        dataframe (pd.DataFrame): [description]
        dependent_variable (pd.Series): [description]
        independent_variable (str): [description]
        group_variable (str): [description]
        output_directory (Path): [description]
        confidence_interval (float, optional): [description]. Defaults to 0.95.

    Returns:
        [type]: [description]
    """
    print("\n==============================================================\n")
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
    ci_table_target_categories = rp.summary_cont(
        group1=dataframe[dependent_variable].groupby(
            dataframe[group_variable]
        ),
        conf=confidence_interval
    )
    print("One-way ANOVA and confidence intervals:")
    print(ci_table_target_categories)
    save_csv_file(
        dataframe=ci_table_target_categories,
        file_name=f"ci_table_target_categories_{dependent_variable}",
        output_directory=output_directory,
    )

    # Find group differences
    tukey_results = perform_multicomparison(
        dataframe=dataframe[dependent_variable],
        groups=dataframe[group_variable],
        confidence_interval=confidence_interval
    )
    return ols_model, tukey_results


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
    tukey_results = multicomparison.tukeyhsd(alpha=1 - confidence_interval)
    print(f"\nTukey's Multicomparison Test between groups:\n{tukey_results}\n")
    return tukey_results


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


def calculate_jarque_bera_values(
    data: pd.Series | List[float]
) -> pd.Series | List[float]:
    """Calculate Jarque-Bera values for kurtosis and skewness.

    NOTE: this is the manual implementation of the test.
    # ? It seems data are not consistent with the statsmodels version (see
    # ? below)

    Args:
        data (pd.Series | List[float]): Input data.

    Returns:
        pd.DataFrame: DataFrame with 'jarque_bera' and 'jb_p_value' columns.
    """
    # Jarque-Bera (kurtosis and skewness together)
    jarque_bera, jb_p_value = stats.jarque_bera(
        x=data,
        nan_policy="propagate"
    )
    jarque_bera_values = {
        'jarque_bera': jarque_bera,
        'jb_p_value': jb_p_value
    }
    jarque_bera_df = pd.DataFrame.from_dict(
        data=jarque_bera_values,
        orient='index',
        columns=['Value']
    ).T
    print("\nJarque-Bera Test Results:")
    print(jarque_bera_df)
    return jarque_bera


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
    print(f"\nNormality Tests Results:\n{normality_tests}\n")

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
    concatenate_arrays.astype(dtype=np.bool)

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
    file_name: str,
    output_directory: Path,
) -> None:
    """Display missing values status for each column in a matrix.

    Args:
        dataframe (pd.DataFrame): Input dataframe.
        file_name (str): Output file name to save matrix.
        output_directory (Path): Output directory where figure will be saved.

    Returns:
        None. Saves the matrix as an image file.
    """
    plt.figure(figsize=(15, 10))
    msno.matrix(
        dataframe,
        sort="descending",  # NOT working
        fontsize=8,
        sparkline=False,
        color=(0.50, 0.50, 0.50),
    )
    plt.title(label="Summary of Missing Data", fontsize=16)
    plt.tight_layout()
    save_figure(file_name=file_name, output_directory=output_directory)
    # plt.show()


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
    output_directory: Path
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
        output_directory (Path): _description_

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
                f"Représentation des effets des paramètres de détection des "
                f"défauts {pc_x.upper()} vs. {pc_y.upper()}"
            ),
            fontsize=16,
            loc="center"
        )
        plt.tight_layout()
        save_figure(
            file_name=f"scatterplot_pca_{target_category}_{pc_x}_vs_{pc_y}",
            output_directory=output_directory
        )
        # plt.show()


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
    feature_list: List[str],
    nb_columns: int,
    hue: List[str] = None,
    palette: str | List[str] | Dict[str, str] = None,
) -> None:
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

    Args:
        dataframe (pd.DataFrame): The input dataframe.
        feature_list (List[str]): The list of features to plot.
        nb_columns (int): The number of columns for subplots.
        hue (List[str], optional): The variable to differentiate the KDE plots.
            Defaults to None.
        palette (str | List[str] | Dict[str, str], optional): The color palette
            for the KDE plots. Defaults to None.

    Returns:
        None: Returns a template to draw subplots. See function
        'run_exploratory_data_visualisation' for an example.
    """
    plt.subplots_adjust(wspace=0.3, hspace=0.5)

    # set number of columns
    ncols = nb_columns

    # calculate number of corresponding rows
    nrows = len(feature_list) // ncols + (len(feature_list) % ncols > 0)

    # loop through the length of tickers and keep track of index
    for index, feature in enumerate(feature_list):
        # add a new subplot iteratively using nrows and cols
        axis = plt.subplot(
            nrows,
            ncols,
            index + 1,  # indexing starts from 0
            # sharex=True
        )
        # fig.tight_layout(pad=2)

        kdeplot_subplots = sns.kdeplot(
            data=dataframe,
            x=feature,
            hue=hue,
            palette=palette,
            legend=True,
        )
        axis.set_xticklabels(
            labels=kdeplot_subplots.get_xticklabels(),
            size=14,
        ),
        axis.set_xlabel(
            # Split the feature name on '_' and capitalize each word
            xlabel=(
                ' '.join([word.capitalize() for word in feature.split('_')])
            ),
            fontsize=18
        )
        axis.set_ylabel(
            ylabel="Density",
            fontsize=18
        )
        # # BUG below NOT working
        # plt.legend(
        #     title=' '.join([word.capitalize() for word in hue.split('_')])
        # )
    return None


# ! Compare function below with function above
def draw_kde_plots(dataframe, columns, label):
    """draw_kde_plots _summary_.

    Args:
        dataframe (_type_): _description_
        columns (_type_): _description_
        label (_type_): _description_
    """
    num_rows, num_cols = 4, 4
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(12, 12))
    fig.suptitle(
        f"Distribution of {label.capitalize()} Features (with Skewness)",
        fontsize=20
    )

    for index, column in enumerate(dataframe[columns].columns):
        i, j = (index // num_cols, index % num_cols)
        graph = sns.kdeplot(
            dataframe[column],
            color="green",
            shade=True,
            label=f"{dataframe[column].skew():.2f}",
            ax=axes[i, j],
        )
        graph.legend(loc="best")
    fig.delaxes(axes[3, 2])
    fig.delaxes(axes[3, 3])
    plt.tight_layout()
    plt.show()


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
    dataframe: pd.DataFrame,
    x_axis: List[float],
    y_label: str | List[float],
    feature_list: List[str],
    nb_columns: int,
    errorbar: str = "ci",
    orient: str = "vertical",
    hue: List[str] = None,
    palette: str | List[str] | Dict[str, str] = None,
) -> None:
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

    Args:
        dataframe (pd.DataFrame): The input dataframe.
        x_axis (List[float]): The x-axis values.
        y_label (str | List[float]): The y-axis label.
        feature_list (List[str]): The list of features to plot.
        nb_columns (int): The number of columns for subplots.
        errorbar (str, optional): The type of error bar to display.
            Defaults to "ci".
        orient(str, optional): The orientation of the bar plots.
            Defaults to 'vertical'.
        hue (List[str], optional): The variable to differentiate the bar plots.
            Defaults to None.
        palette (str | List[str] | Dict[str, str], optional): The color palette
            for the bar plots. Defaults to None.

    Returns:
        None: Returns a template to draw subplots. See function
        'run_exploratory_data_visualisation' for an example.
    """
    plt.subplots_adjust(wspace=0.3, hspace=0.5)

    # set number of columns
    ncols = nb_columns

    # calculate number of corresponding rows
    nrows = len(feature_list) // ncols + (len(feature_list) % ncols > 0)

    # loop through the length of tickers and keep track of index
    for index, feature in enumerate(feature_list):
        # add a new subplot iteratively using nrows and cols
        axis = plt.subplot(
            nrows,
            ncols,
            index + 1,  # indexing starts from 0
            # sharex=True
        )
        # fig.tight_layout(pad=2)

        barplot_subplots = sns.barplot(
            data=dataframe,
            x=x_axis,
            y=feature,
            hue=hue,
            errorbar=errorbar,
            orient=orient,
            palette=palette
        )
        axis.set_title(
            # Split the feature name on '_' and capitalise each word
            label=' '.join([word.capitalize() for word in feature.split('_')]),
            fontsize=16
        ),
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
    return None


# ! Compare function below with function above
def draw_bar_plots(dataframe, columns, label):
    """draw_bar_plots _summary_.

    Args:
        dataframe (_type_): _description_
        columns (_type_): _description_
        label (_type_): _description_
    """
    num_rows, num_cols = 3, 4
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(12, 12))
    fig.suptitle(f"{label.capitalize()} Features Count", fontsize=20)

    for index, column in enumerate(dataframe[columns].columns):
        i, j = (index // num_cols, index % num_cols)
        graph = sns.countplot(
            data=dataframe,
            x=column,
            color="red",
            ax=axes[i, j],
        )
        graph.legend(loc="best")
    fig.delaxes(axes[2, 2])
    fig.delaxes(axes[2, 3])
    plt.tight_layout()
    plt.show()


def draw_correlation_heatmap(
    dataframe, method="pearson"
) -> Tuple[pd.DataFrame, sns.matrix.ClusterGrid]:
    """Draw a correlation matrix between numeric variables.

    Also, add a mask to be applied to the 'upper triangle'.

    Args:
        dataframe (_type_): The input dataframe.
        method (str, optional): The correlation method to use.
            Defaults to "pearson".

    Returns:
        Tuple[pd.DataFrame, sns.matrix.ClusterGrid]: A tuple containing the
            correlation matrix and the correlation heatmap.
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
        center=0,
        # linewidths=0.5,
        # cbar_kws={"shrink": 0.5},
    )
    correlation_heatmap.set_xticklabels(
        labels=correlation_heatmap.get_xticklabels(),
        rotation=45,
        horizontalalignment="right",
        size=12,
    )
    correlation_heatmap.set_yticklabels(
        labels=correlation_heatmap.get_yticklabels(),
        rotation="horizontal",
        size=12
    )
    correlation_heatmap.tick_params(left=True, bottom=True)
    plt.title("Correlation Matrix Heatmap", fontsize=20)
    plt.tight_layout()
    plt.show()
    return correlation_matrix, correlation_heatmap


def run_exploratory_data_visualisation(
        dataframe: pd.DataFrame,
        numeric_features: pd.DataFrame,
        numeric_feature_list: List[str],
        group_variable: str,
        nb_columns: int,
        output_directory: Path,
        palette: str | List[str] | Dict[str, str] = None
) -> None:
    """Produce data visualisation to observe target and features behaviour.

    Draw bar plots and kde plots for ALL variables for EACH category.

    The results are presented as a series of subplots.

    Args:
        dataframe (pd.DataFrame): The input dataframe.
        numeric_features (pd.DataFrame): The dataframe containing only numeric
            features.
        numeric_feature_list (List[str]): The list of numeric feature names.
        group_variable (str): The variable used for grouping in visualizations.
        nb_columns (int): The number of columns for subplots.
        output_directory (Path): The directory to save the output figures.
        palette (str | List[str] | Dict[str, str], optional): The color palette
            to use in the visualizations. Defaults to None.
    """
    # BAR PLOTS

    # Produce a BIG bar plot image with MANY subplots
    plt.figure(figsize=(25, 30))
    draw_barplot_subplots(
        dataframe=dataframe,
        x_axis=group_variable,
        y_label="Transmissivity \u03C4 (-)",
        feature_list=numeric_feature_list,
        nb_columns=nb_columns,
    )
    plt.suptitle(
        t=f"Effects of {group_variable} on detection parameters",
        fontsize=28,
        y=0.92
    )
    save_figure(
        file_name=f"barplot_num_vars_{group_variable}",
        output_directory=output_directory
    )
    # plt.show()

    # ---------------------------------------------------------------------

    # DENSITY PLOTS

    # Produce a BIG kde plot image with MANY subplots
    plt.figure(figsize=(25, 30))
    draw_kdeplot_subplots(
        dataframe=dataframe,
        feature_list=numeric_feature_list,
        nb_columns=nb_columns,
        hue=group_variable
    )
    plt.suptitle(
        t=f"Distribution of detection parameters for each {group_variable}",
        fontsize=28,
        y=0.92
    )
    save_figure(
        file_name=f"kdeplot_num_vars_{group_variable}",
        output_directory=output_directory
    )
    # plt.show()

    # -------------------------------------------------------------------------

    # ! Below NOT working any more...
    # # Produce an '.html' file of the main steps of the EDA
    # produce_sweetviz_eda_report(
    #     dataframe=dataframe,
    #     eda_report_name="clean_data_sweetviz_eda_report",
    #     output_directory=output_directory
    # )

    # -------------------------------------------------------------------------

    print("\nGenerating Pair Plots...\n")

    # Display pair plots for EACH target
    plt.figure(figsize=(30, 30))
    sns.pairplot(
        data=dataframe,
        kind="reg",
        hue=group_variable,
        diag_kind="kde",
    )
    plt.suptitle(
        t=f"Correlogram of detection features for each {group_variable}",
        fontsize=36,
        # y=1.12
    )
    plt.subplots_adjust(top=0.95)
    save_figure(
        file_name=f"pairplot_between_parameters_{group_variable}",
        output_directory=output_directory
    )
    # plt.show()

    # -------------------------------------------------------------------------

    # # Correlation analysis
    # # ! Below NOT working any more...
    # # Display output as a table AND a figure
    # plt.figure(figsize=(25, 20))
    # features_correlation_matrix = draw_correlation_heatmap(
    #     dataframe=numeric_features,
    #     method="pearson"
    # )
    # plt.title(
    #     label="Correlation analysis between engineered features\n",
    #     fontsize=16,
    #     loc="left"
    # )
    # print("\nFeatures Correlation Matrix:")
    # print(features_correlation_matrix)

    # save_figure(
    #     file_name="correlation_matrix_feat_engineering",
    #     output_directory=output_directory
    # )
    # # plt.show()
    return None


def draw_pair_plot(dataframe, hue=None):
    """draw_pair_plot _summary_.

    Args:
        dataframe (_type_): _description_
        hue (_type_, optional): _description_. Defaults to None.
    """
    fig, ax = plt.subplots(figsize=(16, 12))
    graph = sns.pairplot(
        dataframe,
        kind="reg",
        diag_kind="kde",
        hue=hue,
        # palette="husl",  # disabled to favour "colour-blind" setting at start
        corner=True,
        dropna=False,
        size=2.5,
    )
    # CANNOT change position of the legend
    graph.legend(
        #         title="Season",
        loc="center right"
    )  # or loc="upper right"
    plt.tight_layout()
    plt.suptitle("Features Relationships", y=1.05, fontsize=20)
    plt.show()


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
    output_directory: Path,
    confidence_interval: float = 0.95,
) -> None:
    """Draw Q-Q plots of the model dependent variable residuals.

    Args:
        dataframe (pd.DataFrame): _description_
        dependent_variable (str): _description_
        independent_variable (str): _description_
        model (_type_): _description_
        output_directory (Path): _description_
        confidence_interval (float, optional): _description_. Defaults to 0.95.

    Returns:
        None. Saves the plot as an image file.
    """
    draw_qqplot(
        dataframe=model.resid,
        confidence_interval=confidence_interval
    )
    plt.title(
        label=f"Q-Q Plot of Model Residuals for {dependent_variable}",
        fontsize=14,
        loc="center"
    )
    plt.grid(visible=False)
    # plt.axis("off")
    plt.tight_layout()
    save_figure(
        file_name=f"qqplot_anova_{independent_variable}_{dependent_variable}",
        output_directory=output_directory
    )
    # plt.show()
    return None


def draw_tukeys_hsd_plot(
    dataframe: pd.DataFrame,
    dependent_variable: str,
    independent_variable: str,
    output_directory: Path,
    confidence_interval: float = 0.95,
) -> None:
    """draw_tukeys_hsd_plot _summary_.

    Args:
        dataframe (pd.DataFrame): _description_
        dependent_variable (str): _description_
        independent_variable (str): _description_
        output_directory (Path): _description_
        confidence_interval (float, optional): _description_. Defaults to 0.95.

    Returns:
        None. Saves the plot as an image file.
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
        ylabel="Target Categories",
        xlabel="Score Difference"
    )
    plt.suptitle(
        t=f"Tukey's HSD Post-hoc Test on {dependent_variable}",
        fontsize=14
    )
    plt.grid(visible=False)
    # plt.axis("off")
    plt.tight_layout()
    save_figure(
        file_name=f"tukey_anova_{independent_variable}_{dependent_variable}",
        output_directory=output_directory
    )
    # plt.show()

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
    #     f"\nTukey's Multicomparison Test Version 2:\n"
    #     f"{corrected_tukey_post_hoc_test}\n"
    # )
    return None


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
    save_figure(file_name=file_name, output_directory=output_directory)
    # plt.show()


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
    save_figure(file_name=file_name, output_directory=output_directory)
    # plt.show()


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
        output_directory (Path): _description_
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
    fig.write_html(output_directory.joinpath(f"{file_name}.html"))


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
        output_directory (Path): _description_
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
    save_figure(file_name=file_name, output_directory=output_directory)


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
    output_directory: Path,
) -> None:
    """draw_feature_rank _summary_.

    The 'yellowbrick' library is used for ranking features.
    Prior to fitting the features and target, the latter must be first
    label-encoded.

    Args:
        features (pd.DataFrame): _description_
        target_encoded (pd.Series): _description_
        output_directory (Path): _description_
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

    plt.title(label="Ranking of Features", fontsize=16)
    save_figure(file_name="feature_ranking", output_directory=output_directory)


def show_items_per_category(
        data: pd.Series | pd.DataFrame,
        category_name: str,
) -> None:
    """Show the number of items in each class of a given category.

    Args:
        data (pd.Series | pd.DataFrame): The data containing the category to be
            analysed.
        category_name (str): The name of the category to be analysed.

    Returns:
        None. Saves a bar chart image for counts of each defect class.
    """
    # Show number of items in each class of the target
    data_class_count = data.value_counts()
    print(
        f"\nNumber of items within each '{category_name}' class:\n"
        f"{data_class_count}\n"
    )
    plt.figure()
    ax = data_class_count.sort_values().plot.barh()
    ax.set(xlabel="Number of items", ylabel=category_name)
    plt.title(
        label=(
            f"Number of items for each class of the category '{category_name}'"
        ),
        fontsize=16,
        loc="center"
    )
    # plt.axis("off")
    plt.tight_layout()
    save_figure(
        file_name="item_count_barchart",
        output_directory=OUTPUT_DIR_FIGURES
    )
    plt.show()


def generate_class_colour_list(class_list: List[str]) -> List[str]:
    """AI is creating summary for generate_class_colour_list.

    Args:
        class_list (List[str]): [description]

    Returns:
        List[str]: [description]
    """
    colour_list = colourmap.generate(
        len(class_list),
        method="seaborn"
    )
    return colour_list

# -----------------------------------------------------------------------------

# DATA MODELLING (MACHINE LEARNING)


def target_label_encoder(
    data: pd.DataFrame | pd.Series,
) -> Tuple[np.ndarray, List[str]]:
    """Encode the target labels (usually strings) into integers.

    Args:
        data (pd.DataFrame | pd.Series): The data containing the target labels
            to be encoded.

    Returns:
        Tuple[np.ndarray, List[str], LabelEncoder]: A tuple containing the
            encoded target labels, the list of target classes and the label
            encoder object.
    """
    label_encoder = LabelEncoder()
    target_encoded = label_encoder.fit_transform(data)

    # Convert the encoded labels from a np.ndarray to a list
    target_class_list = label_encoder.classes_.tolist()

    # Get the mapping dictionary of original labels to encoded labels
    label_mapping_dictionary = dict(
        zip(target_class_list, range(len(target_class_list)))
    )
    print("\nDictionary of the target encoded classes:")
    print(label_mapping_dictionary)
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


def remove_features_with_nans(
    dataframe: pd.DataFrame,
    nan_threshold: float = 0.7
) -> pd.DataFrame:
    """Remove features if there is a high proportion of NaN values.

    Args:
        dataframe (pd.DataFrame): The input DataFrame.
        nan_threshold (float, optional): The threshold for removing features
            with NaN values. Defaults to 0.7.

    Returns:
        pd.DataFrame: A DataFrame with the features removed.
    """
    # Get the percentage of missing values in each column
    missing_percent = dataframe.isnull().mean()

    # Get the indices of columns that have less than or equal to the threshold
    # of missing values
    column_indices = missing_percent[missing_percent <= nan_threshold].index

    # Return the subset of dataframe with those columns
    reduced_dataframe = dataframe[column_indices]

    # List the variables removed due to a high proportion of NaN values
    variable_list_before = dataframe.columns.to_list()
    variable_list_after = reduced_dataframe.columns.to_list()

    removed_column_list = [
        feature for feature in variable_list_before
        if feature not in variable_list_after
    ]
    print("\nColumns Removed due to a High Proportion of NaN Values:")
    print(removed_column_list)
    return reduced_dataframe


def remove_features_with_nans_transformer(
    nan_threshold: float
) -> FunctionTransformer:
    """Create a transformer to remove features with a high proportion of NaNs.

    Args:
        nan_threshold (float): The threshold for removing features with NaNs.

    Returns:
        FunctionTransformer: A transformer object that can be fit and
            transformed on a DataFrame.
    """
    remove_nan_transformer = FunctionTransformer(
        func=remove_features_with_nans,
        kw_args={"nan_threshold": nan_threshold},
        validate=False,
    )
    return remove_nan_transformer


def drop_feature_pipeline(
    features: pd.DataFrame,
    features_to_keep: List[str],
    nan_threshold: float = 0.7,
    correlation_threshold: float = 0.8,
    variables_with_nan_values: List[str] = None,
    variance_threshold: float = None,
) -> Pipeline:
    """Create a pipeline to drop features from a DataFrame.

    Args:
        features (pd.DataFrame): The input DataFrame.
        features_to_keep (List[str]): A list of feature names to keep.
        nan_threshold (float, optional): The threshold for removing features
            with NaN values. Defaults to 0.7.
        correlation_threshold (float, optional): The threshold for removing
            correlated features. Defaults to 0.8.
        variables_with_nan_values (List[str], optional): A list of variables
            with NaN values. Defaults to None.
        variance_threshold (float, optional): The threshold for removing low
            variance features. Defaults to None.

    Returns:
        Pipeline: Object that can be fit and transformed on a DataFrame.

    TODO Add a 'drop_low_variance' function to pipeline ?
    TODO Compare vs. 'DropCorrelatedFeatures' class.
    Try something like the code below:
            var_thres = VarianceThreshold(threshold=threshold)
            _ = var_thres.fit(features)
            # Get a boolean mask
            mask = var_thres.get_support()
            # Subset the data
            features_reduced = features.loc[:, mask]
            print("The following features were retained:")
            print(f"{features_reduced.columns}")
    BUG Fix error message below when using 'VarianceThreshold' class.
        ValueError: make_column_selector can only be applied to pandas df
    """
    steps = [
        (
            "Remove Rows With NaN Values",
            remove_features_with_nans_transformer(nan_threshold=nan_threshold)
        ),
        (
            "Remove Rows With NaN Values",
            DropMissingData(
                variables=variables_with_nan_values,
                missing_only=True,
                threshold=0.03,  # variable with =< 0.3 variance removed
                # threshold=None,  # rows with any NaNs will be removed
            )
        ),
        (
            "Drop Columns",
            DropFeatures(
                features_to_drop=[
                    feature for feature in features
                    if feature not in features_to_keep
                ]
            )
        ),
        (
            "Drop Constant Values",
            DropConstantFeatures(tol=0.95, missing_values="ignore")
        ),  # TODO to test vs. Scikit-learn 'VarianceThreshold()' (below)
        (
            "Drop Duplicates",
            DropDuplicateFeatures(missing_values="ignore")
        ),
        (
            "Drop Correlated Features",
            DropCorrelatedFeatures(
                method="pearson",
                threshold=correlation_threshold,
                missing_values="ignore"
            )
        ),
    ]

    if variance_threshold is not None:
        steps.append(
            (
                "drop_low_variance",
                VarianceThreshold(threshold=variance_threshold)
            )
        )

    drop_feature_pipe = Pipeline(steps=steps, verbose=True)
    print(f"\nDrop-Feature Pipeline Structure:\n{drop_feature_pipe}\n")
    return drop_feature_pipe


def get_dropped_features_from_pipeline(
    pipeline: Pipeline,
    features: pd.DataFrame
) -> List[str]:
    """Retrieve the list of features dropped by a Scikit-learn pipeline.

    NOTE: The 'named_steps[]' attributes allows accessing the various steps of
    the pipeline by calling its name, as described in the list of tuples (see
    'drop_feature_pipeline' function). This can be used to ANY scikit-learn (or
    scikit-learn like class/function) pipeline.

    Args:
        pipeline (Pipeline): The pipeline object.
        features (pd.DataFrame): The dataframe containing the features.

    Returns:
        List[str]: The list of dropped features.
    """
    # Make a list of features dropped through the pipeline model
    drop_constant_features = list(
        pipeline.named_steps["Drop Constant Values"].features_to_drop_
    )
    print(f"\nFeatures with Constant Values:\n{drop_constant_features}\n")

    drop_duplicate_features = list(
        pipeline.named_steps["Drop Duplicates"].features_to_drop_
    )
    print(f"\nFeatures Duplicated:\n{drop_duplicate_features}\n")

    drop_correlated_features = list(
        pipeline.named_steps[
            "Drop Correlated Features"].features_to_drop_
    )

    # Get a list of correlated features
    correlated_features = list(
        pipeline.named_steps[
            "Drop Correlated Features"].correlated_feature_sets_
    )
    print(f"\nFeatures Correlated:\n{correlated_features}\n", sep="\n")

    # Display a correlation matrix
    correlation_matrix = features.corr()
    print(f"\nCorrelation Matrix of Model Features:\n{correlation_matrix}\n")

    # Concatenate lists of dropped features
    dropped_feature_list = (
        *drop_constant_features,
        *drop_duplicate_features,
        *drop_correlated_features
    )
    print(f"\nFeatures Dropped:\n{dropped_feature_list}\n")
    return dropped_feature_list


def preprocess_numeric_feature_pipeline(scaler: str) -> Pipeline:
    """Create a pipeline to preprocess numeric features.

    Args:
        scaler (str): The name of the scaler to use. Can be either
            'standard_scaler', 'min_max_scaler' or 'robust_scaler'.

    Returns:
        Pipeline: An object that can be fit and transformed on a DataFrame.
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


def preprocess_categorical_feature_pipeline(encoder: str) -> Pipeline:
    """Create a pipeline to preprocess categorical features.

    Args:
        encoder (str): The name of the encoder to use. Can be either
            'one_hot_encoder' or 'ordinal_encoder'.

    Returns:
        Pipeline: An object that can be fit and transformed on a DataFrame.
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


def run_machine_learning_pipeline(
    features: pd.DataFrame,
    features_to_keep: List[str],
    label_encoder: LabelEncoder,
    model: Dict[str, object],
    resampling: bool = True,
    nan_threshold=0.70,
    correlation_threshold=0.80,
) -> Pipeline:
    """Perform a machine learning analysis based on several key steps.

    The model parameter is defined as a dictionary where the key represents the
    name (as a string) of the model/pipeline and the value represents the
    'definition' of the model, that is, for example, the pipeline object that
    is created outside the function.

    Args:
        features (pd.DataFrame): The dataframe containing the features.
        features_to_keep (List[str]): A list of feature names to keep.
        label_encoder (LabelEncoder): The instance of the label encoder.
        model (_type_): _description_
        resampling (bool, optional): Whether to integrate or not a resampler
            tool to the pipeline. Defaults to True.
        nan_threshold (float, optional): The threshold for removing features
            with NaN values. Defaults to 0.7.
        correlation_threshold (float, optional): The threshold for removing
            correlated features. Defaults to 0.8.

    Returns:
        Pipeline: The final pipeline structure/step.
    """
    # Step 1: Drop irrelevant columns/features

    drop_feature_pipe = drop_feature_pipeline(
        features=features,
        features_to_keep=features_to_keep,
        nan_threshold=nan_threshold,
        correlation_threshold=correlation_threshold,
    )

    # ------------------------------------------------------------------------

    # Step 2: Preprocess numeric and categorical features

    # Impute and encode categorical features  # TODO Test the 'one2hot' library
    categorical_feature_pipeline = (
        preprocess_categorical_feature_pipeline(encoder="one_hot_encoder")
    )

    # Impute and scale numeric features
    numeric_feature_pipeline = (
        preprocess_numeric_feature_pipeline(scaler="robust_scaler")
    )

    # Create a column transformer with both numeric and categorical pipelines
    preprocess_feature_pipeline = transform_feature_pipeline(
        categorical_feature_pipeline=categorical_feature_pipeline,
        numeric_feature_pipeline=numeric_feature_pipeline,
    )

    # ------------------------------------------------------------------------

    # Step 3: Resampling dataset to deal with class imbalance of TARGET only

    if resampling:
        smote_resampler = SMOTE(random_state=42)

    # ------------------------------------------------------------------------

    # Step 4: Set up the final pipeline
    # NOTE: this is not a 'scikit-learn' pipeline but a 'imblearn' pipeline.

    pipeline = Pipeline(
        steps=[
            # ! Pipeline class from imblearn library not support nested class
            # ("Drop Features", drop_feature_pipeline),
            # ! Unpacking the steps of the 'drop features' pipeline is required
            *drop_feature_pipe.steps,
            ("Transform Features", preprocess_feature_pipeline),
            ("Resampler", smote_resampler),
            (model["model_name"], model["model_definition"]),
        ],
        verbose=False,
    )
    # Set the config pipeline output to a (dense, not sparse) Numpy array
    # pipeline.set_output(transform="default")
    # print("\nDisplay Analytic Pipeline:")
    # print(pipeline)
    return pipeline


def select_best_features():
    """select_best_features _summary_."""
    return


def transform_feature_pipeline(
    numeric_feature_pipeline: Pipeline,
    categorical_feature_pipeline: Pipeline,
) -> ColumnTransformer:
    """Transform the selected features using the specified pipelines.

    For the 'transformers' parameters, the settings represent, respectively:
    - the transformer name
    - the transformer pipeline it represents
    - the columns included in the transformer

    Args:
        numeric_feature_pipeline (Pipeline): Pipeline for transforming numeric
            features.
        categorical_feature_pipeline (Pipeline): Pipeline for transforming
            categorical features.

    Returns:
        ColumnTransformer: A column transformer object that applies the
            specified transformers to the appropriate columns.
    """
    feature_transformer = ColumnTransformer(
        transformers=[
            (
                "Numeric Features",
                numeric_feature_pipeline,
                selector(dtype_include="number")
            ),
            (
                "Categorical Features",
                categorical_feature_pipeline,
                selector(dtype_include="category")
            ),
        ],
        verbose_feature_names_out=False,  # if True, will display transformers
        remainder="drop",
        n_jobs=-1,
    )
    print("\nModel Transformer Pipeline Structure ('ColumnTransformer'):")
    print(feature_transformer)
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


def join_parallel_pipelines(
    pipeline_one: Pipeline,
    pipeline_two: Pipeline,
) -> FeatureUnion:
    """Join pipelines for parallel processing.

    This approach is similar to that of the 'ColumnTransformer()' function.
    There are some subtleties though:
        -

    Args:
        pipeline_one (Pipeline): _description_
        pipeline_two (Pipeline): _description_

    Returns:
        FeatureUnion: _description_
    """
    union_transformer = FeatureUnion(
        transformer_list=[
            ("Pipeline #1", pipeline_one),
            ("Pipeline #2", pipeline_two)
        ],
        n_jobs=-1,
        verbose=True
    )
    print("\nModel Transformer Pipeline Structure ('FeatureUnion'):")
    print(union_transformer)
    return union_transformer


def predict_target(
    model: Pipeline,
    features_test: pd.DataFrame | np.ndarray,
    target_test: pd.Series | np.ndarray,
) -> np.ndarray:
    """Predict target using a model and create a dataframe of predictions.

    Args:
        model (Pipeline): The trained model for prediction.
        features_test (pd.DataFrame | np.ndarray): The test set features.
        target_test (pd.Series | np.ndarray): The true target values for
            the test set.
        label_encoder (LabelEncoder): The instance of the label encoder.

    Returns:
        np.ndarray: The predicted target values.
    """
    # Predict the target on the test set
    target_pred = model.predict(X=features_test)

    # Build a dataframe of true/test values vs. prediction values
    predictions_vs_test = pd.concat(
        objs=[pd.Series(target_test), pd.Series(target_pred)],
        keys=["test", "predictions"],
        axis=1,
    )
    predictions_vs_test.set_index(
        keys=features_test.index,
        inplace=True
    )
    print(f"\nTest vs. Predictions:\n{predictions_vs_test}\n")
    return target_pred


def predict_defect_class(
        data: np.ndarray,
        model: Pipeline,
        defect_class_dictionary: Dict[int, str],
) -> str:
    """Predict the defect class based on the input data using a trained model.

    The class can be a type of defect (e.g. CGGros, Melt, etc.) or a family of
    defect (e.g. CG or Melt).

    NOTE: As opposed to the previous function ('predict_target'), this function
    allows for prediction of a single sample at the time.

    Args:
        data (np.ndarray): The input data for prediction
        model (Pipeline): The trained model used for prediction. The model
        could also be a whole process pipeline as with Scikit-learn.
        defect_class_dictionary (Dict[int, str]): A dictionary mapping class
        numbers to defect names.

    Returns:
        str: Prints the predicted defect type or family.
    """
    target_pred = model.predict(X=data)

    # Convert the output array (a single score) to an integer of the defect
    # type class. This is necessary since a numpy array is unhashable which
    # will prevent the scaler to be used in a dictionary.
    target_pred = int(target_pred)

    # Get the defect name from its class number
    defect_predicted = defect_class_dictionary.get(target_pred)
    print(f"\nPredicted Defect Type or Family: {defect_predicted}\n")


def save_pipeline_model_joblib(
    pipeline_name: Pipeline,
    file_name: str,
    output_directory: Path,
) -> Pipeline:
    """Save a pipeline model using joblib.

    To load the pipeline, use the following command:
    pipeline_name = joblib.load(filename="pipeline_file_name.joblib")

    Args:
        pipeline_name (Pipeline): The pipeline model to be saved.
        file_name (str): The name of the output file.
        output_directory (Path): The directory to save the output file.

    Returns:
        Pipeline: The pipeline model that has been saved.
    """
    pipeline_model_file = joblib.dump(
        value=pipeline_name,
        filename=output_directory.joinpath(f"{file_name}.joblib"),
    )
    return pipeline_model_file


def calculate_cross_validation_scores(
    model: object,
    features_test: pd.DataFrame,
    target_test: pd.Series,
    target_pred: pd.Series,
    target_label_list: List[str],
    cv: int = 5
) -> None:
    """Calculate cross-validation scores.

    It includes the balanced accuracy score and production of a classification
    report that displays the model metrics for the different classes of the
    target.

    Args:
        features_test (pd.DataFrame): The test features data.
        target_test (pd.Series): The true target values.
        target_pred (pd.Series): The predicted target values.
        target_label_list (List[str]): A list of target labels for
        classification report.
        cv (int): Number of cross-validation folds. Defaults to 5).

    Returns:
        None. Prints the balanced accuracy score and classification report.
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
    model: object,
    features: pd.DataFrame | np.ndarray,
    target: pd.Series | np.ndarray,
    cv: int = 5
) -> None:
    """Calculate multiple cross-validation scores using the provided model.

    Use the 'cross_validate()' function to calculate multiple model scores for
    EACH train and test sets.
    NOTE: To print out the train scores, the parameter 'return_train_score'
    must be set to 'True'.

    Args:
        model (object): The model used for cross-validation.
        features (pd.DataFrame | np.ndarray): The input features data.
        target (pd.Series | np.ndarray): The target values.
        cv (int, optional): Number of cross-validation folds. Defaults to 5.

    Returns:
        None. Prints the model score output.
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
        f"\nDecision Tree Mean Accuracy Score for Train Set = "
        f"{tree_classifiers_train_score:.1%}\n"
    )
    tree_classifiers_test_score = tree_classifier.score(
        features_test, target_test
    )
    print(
        f"\nDecision Tree Mean Accuracy Score for Test Set = "
        f"{tree_classifiers_test_score:.1%}\n"
    )


def draw_decision_tree(
    tree_classifier,
    feature_name_list: List[str],
    target_label_list: List[str],
    file_name: str,
    output_directory: Path,
) -> None:
    """_summary_.

    Args:
        tree_classifier (_type_): _description_
        feature_name_list (List[str]): _description_
        target_label_list (List[str]): _description_
        file_name (str): _description_
        output_directory (Path): _description_
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
    while stack:
        # `pop` ensures each node is only visited once
        node_id, depth = stack.pop()
        node_depth[node_id] = depth

        # If the left and right child of a node is not the same we have a split
        # node
        is_split_node = children_left[node_id] != children_right[node_id]
        # If a split node, append left and right children and depth to `stack`
        # so we can loop through them
        if is_split_node:
            # stack.append((children_left[node_id], depth + 1))
            # stack.append((children_right[node_id], depth + 1))
            stack.extend(
                (children_right[node_id], depth + 1),
                (children_right[node_id], depth + 1),
            )  # use `.extend()` when appending multiple values to a list
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
    save_figure(file_name=file_name, output_directory=output_directory)
    # plt.show()


def draw_random_forest_tree(
    random_forest_classifier,
    feature_name_list: List[str],
    target_label_list: List[str],
    file_name: str,
    output_directory: Path,
    ranked_tree: int = None,
) -> None:
    """Draw a random forest tree for visualization.

    Args:
        random_forest_classifier (_type_): The random forest classifier model.
        feature_name_list (List[str]): The list of feature names.
        target_label_list (List[str]): The list of target label names.
        file_name (str): The name of the output file.
        output_directory (Path): Directory where the output file will be saved.
        ranked_tree (int, optional): The index of the tree to draw.

    Returns:
        None. Saves a figure of the decision tree.
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
    save_figure(file_name=file_name, output_directory=output_directory)
    # plt.show()


def draw_confusion_matrix_heatmap(
    target_test: pd.Series,
    target_pred: pd.Series,
    target_label_list: List[str],
    file_name: str,
    output_directory: Path,
) -> None:
    """
    Draw a heatmap of confusion matrix based on predicted & true target values.

    Args:
        target_test (pd.Series): The true target values.
        target_pred (pd.Series): The predicted target values.
        target_label_list (List[str]): A list of target labels.
        file_name (str): The name of the output file.
        output_directory (Path): The directory to save the output file.

    Returns:
        None. Saves the heatmap as an image file.
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
    plt.title(label="Confusion matrix of predictions", fontsize=16)
    plt.grid(visible=False)
    # plt.axis("off")
    plt.tight_layout()
    save_figure(file_name=file_name, output_directory=output_directory)
    # plt.show()


def get_feature_importance_scores(
    model: object,
    feature_name_list: List[str],
    file_name: str,
    output_directory: Path,
) -> pd.Series:
    """
    Retrieve the feature importance scores from a model & generates a bar plot.

    Args:
        model (object): The trained model.
        feature_name_list (List[str]): The list of feature names.
        file_name (str): The name of the output file.
        output_directory (Path): The directory to save the output file.

    Returns:
        pd.Series: A Pandas Series containing the feature importance scores.
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

    # # Calculate the standard deviation of all estimators
    # estimator_std = np.std(
    #     [tree.feature_importances_ for tree in model.estimators_],
    #     axis=0
    # )

    # Create a Pandas Series to plot the data
    model_feature_importances = pd.Series(
        data=feature_importances[feature_indices],
        index=feature_names
    )
    print(f"\nModel Features Importance:\n{model_feature_importances}\n")

    # Create a bar plot
    fig, ax = plt.subplots()
    model_feature_importances.plot.barh(
        # xerr=estimator_std,  # remove error bars
        align="center",
        ax=ax
    )
    plt.ylabel("Parameters")
    plt.xlabel("Mean decrease in Impurity")
    plt.title(label="Feature Importance in Predictions", fontsize=16)
    plt.grid(visible=False)
    fig.tight_layout()
    save_figure(file_name=file_name, output_directory=output_directory)
    # plt.show()
    return model_feature_importances


def apply_cross_validation_analysis(
    model: object,
    features_train: pd.DataFrame,
    target_train: pd.Series,
    features_test: pd.DataFrame,
    target_test: pd.Series,
    target_pred: pd.Series,
    target_label_list: str,
    cv: int,
    file_name: str,
    output_directory: Path
) -> None:
    """
    Apply cross-validation on train & test data to evaluate model performance.

    Args:
        model (object): The model to be used for cross validation analysis.
        features_train (pd.DataFrame): The training data features.
        target_train (pd.Series): The training data target.
        features_test (pd.DataFrame): The test data features.
        target_test (pd.Series): The test data target.
        target_pred (pd.Series): The predicted target values.
        target_label_list (str): The list of target labels.
        file_name (str): The name of the file to save the confusion matrix
            heatmap to.
        output_directory (Path): The directory to save the confusion matrix
            heatmap to.
        cv (int): The number of cross validation folds. Defaults to 5.
    """
    calculate_cross_validation_scores(
        features_test=features_test,
        target_test=target_test,
        target_pred=target_pred,
        target_label_list=target_label_list,
        cv=cv
    )

    draw_confusion_matrix_heatmap(
        target_test=target_test,
        target_pred=target_pred,
        target_label_list=target_label_list,
        file_name=file_name,
        output_directory=output_directory,
    )

    calculate_multiple_cross_validation_scores(
        model=model,
        features=features_train,
        target=target_train,
    )


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


def get_best_parameters_ensemble(
    pipeline: Pipeline,
    model: Dict[str, object],
) -> pd.DataFrame:
    """Retrieve the best parameters and scores from a pipeline.

    It also returns the parameters and metrics as a DataFrame.

    The model parameter is defined as a dictionary where the key represents the
    name (as a string) of the model/pipeline and the value represents the
    'definition' of the model, that is, for example, the pipeline object that
    is created outside the function.

    NOTE: This function is specific to the optimisation of the 'Ensemble
    Classifier' through the use of the 'VotingClassifier()' class of the
    'scikit-learn' package.

    Args:
        pipeline (Pipeline): The pipeline object.
        model (Dict[str, object]): A dictionary containing the model name.

    Returns:
        pd.DataFrame: A DataFrame containing the best parameters and scores.
    """
    # Get best score
    pipeline_best_score = pipeline.best_score_
    print(f"\nBest Score: {pipeline_best_score:.1%}\n")

    # Generate the dictionary of best parameters
    pipeline_best_parameters = pipeline.best_params_
    print(f"\nBest Parameters:\n{pipeline_best_parameters}\n")

    # Extract cross-validation results and convert to dataframe
    cv_results = pipeline.cv_results_
    cv_results_dataframe = pd.DataFrame(data=cv_results)
    cv_results_dataframe = cv_results_dataframe.sort_values(
        by=["mean_test_score"], ascending=False
    )

    # Get the optimised parameters of the Top 5 Ensemble Classifiers
    # First, reorder the columns
    cv_results_best_5 = cv_results_dataframe[[
        "mean_test_score",
        "std_test_score",
        "rank_test_score",

        "param_Transform Features__Numeric Features__scaler",

        "param_Resampler",

        f"param_{model['model_name']}__voting",

        # f"param_{model['model_name']}__Decision Tree Classifier__criterion",
        # f"param_{model['model_name']}__Decision Tree Classifier__max_depth",
        # f"param_{model['model_name']}__Decision Tree \
        #     Classifier__min_samples_leaf",

        # f"param_{model['model_name']}__Gaussian Naive Bayes__var_smoothing",
        # f"param_{model['model_name']}__Linear Discriminant Analysis__solver",

        f"param_{model['model_name']}__Random Forest Classifier__max_depth",
        f"param_{model['model_name']}__Random Forest "
        f"Classifier__max_leaf_nodes",
        f"param_{model['model_name']}__Random Forest "
        f"Classifier__min_samples_leaf",
        f"param_{model['model_name']}__Random Forest "
        f"Classifier__min_samples_split",
        f"param_{model['model_name']}__Random Forest Classifier__n_estimators",

        f"param_{model['model_name']}__Support Vector Machine Classifier__C",
        f"param_{model['model_name']}__Support Vector Machine "
        f"Classifier__gamma",
        f"param_{model['model_name']}__Support Vector Machine "
        f"Classifier__kernel",

        f"param_{model['model_name']}__XGBoost Classifier__gamma",
        f"param_{model['model_name']}__XGBoost Classifier__learning_rate",
        f"param_{model['model_name']}__XGBoost Classifier__max_depth",
    ]].head(5)

    # Set the column 'mean_test_score' to a percentage with one decimal
    cv_results_best_5["mean_test_score"] = (
        cv_results_best_5["mean_test_score"].apply(
            lambda percent: f"{percent:.1%}"
        )
    )
    print("\nTable of Top 5 Best Models:")
    print(cv_results_best_5)
    return cv_results_best_5


def random_search_cv_optimisation_ensemble(
    pipeline: Pipeline,
    model: Dict[str, object],
    cv: int = 5
) -> RandomizedSearchCV:
    """Perform random search cross-validation.

    This is to optimise the hyper-parameters of pipeline (e.g. here an ensemble
    model).

    Different steps of the pipeline are optimised:
        - Transformation of the numeric data via three types of transformation:
        'StandardScaler', MinMaxScaler' and 'RobustScaler'.
        - Several over- and under-resampling algorithms are tested, as well as
        NO resampling.
        - Various model parameters and range values are tested.
        NOTE: Not all parameters could be tested due to limitations of Python
        computing power.

    The model parameter is defined as a dictionary where the key represents the
    name (as a string) of the model/pipeline and the value represents the
    'definition' of the model, that is, for example, the pipeline object that
    is created outside the function.

    Args:
        pipeline (Pipeline): The pipeline object.
        model (Dict[str, object]): A dictionary containing the model name.
        cv (int, optional): The number of cross-validation folds. Defaults to 5

    Returns:
        RandomizedSearchCV: The optimised model.
    """
    # Set of optimisation options
    parameter_search = {
        "Transform Features__Numeric Features__scaler":
        [StandardScaler(), MinMaxScaler(), RobustScaler()],

        "Resampler": [
            None,
            BorderlineSMOTE(random_state=42),
            KMeansSMOTE(random_state=42),
            RandomOverSampler(random_state=42),
            RandomUnderSampler(random_state=42),
            SMOTE(random_state=42),
            SMOTEENN(random_state=42),  # combination of over- & under-sampling
            SVMSMOTE(random_state=42),
        ],

        f"{model['model_name']}__voting": ["soft", "hard"],

        # f"{model['model_name']}__Gaussian Naive Bayes__var_smoothing": \
        # [0, 1e-9, 0.01, 0.1, 0.2, 0.5, 1],
        # f"{model['model_name']}__Linear Discriminant Analysis__solver": \
        # ["svd", "lsqr", "eigen"],

        f"{model['model_name']}__Random Forest Classifier__max_depth": \
        [2, 3, 5, 7, 10],
        f"{model['model_name']}__Random Forest Classifier__max_leaf_nodes": \
        [2, 5, 7, 10],
        f"{model['model_name']}__Random Forest Classifier__min_samples_leaf": \
        [2, 3, 5, 7, 10],
        f"{model['model_name']}__Random Forest Classifier__min_samples_split":\
        [2, 5, 7],
        f"{model['model_name']}__Random Forest Classifier__n_estimators": \
        [100, 200, 500, 700],

        f"{model['model_name']}__Support Vector Machine Classifier__C": \
        [1, 10, 100, 500, 1000],
        f"{model['model_name']}__Support Vector Machine Classifier__gamma": \
        ["auto", "scale", 1, 0.1, 0.01, 0.001, 0.0001],
        f"{model['model_name']}__Support Vector Machine Classifier__kernel": \
        ["rbf", "linear"],

        f"{model['model_name']}__XGBoost Classifier__learning_rate": \
        [0.01, 0.05, 0.1, 0.2, 0.3],
        f"{model['model_name']}__XGBoost Classifier__max_depth": \
        [2, 5, 10, 15],
        f"{model['model_name']}__XGBoost Classifier__gamma": [0, 1, 2,  5, 10],
        # f"{model['model_name']}__XGBoost Classifier__min_child_weight": \
        # [10, 15, 20, 25],
        # f"{model['model_name']}__XGBoost Classifier__colsample_bytree": \
        # [0.8, 0.9, 1],
        # f"{model['model_name']}__XGBoost Classifier__n_estimators": \
        # [300, 400, 500, 600],
        # f"{model['model_name']}__XGBoost Classifier__reg_alpha": \
        # [0.5, 0.2, 1],
        # f"{model['model_name']}__XGBoost Classifier__reg_lambda": [2, 3, 5],
    }

    # Model optimisation using 'RandomizedSearchCV()'
    random_search_cv = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=parameter_search,
        # scoring="roc_auc",  # TODO Test on multi-class labelling ?
        scoring="accuracy",
        # verbose=True,
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
        random_state=42,
        n_jobs=-1
    )
    print("\nRandom Search CV Pipeline:")
    print(random_search_cv)
    return random_search_cv


def bayes_search_cv_optimisation(
    pipeline: Pipeline,
    model: Dict[str, object],
    cv: int = 5
) -> BayesSearchCV:
    """Perform random search cross-validation.

    This is to optimize the hyper-parameters of pipeline (e.g. here an LightGBM
    Classifier model).

    The model parameter is defined as a dictionary where the key represents the
    name (as a string) of the model/pipeline and the value represents the
    'definition' of the model, that is, for example, the pipeline object that
    is created outside the function.

    Different steps of the pipeline are optimised:
        - Transformation of the numeric data via three types of transformation:
        'StandardScaler', MinMaxScaler' and 'RobustScaler'.
        - Several over- and under-resampling algorithms are tested, as well as
        NO resampling.
        - Various model parameters and range values are tested.
        NOTE: Not all parameters could be tested due to limitations of Python
        computing power.

    Args:
        pipeline (Pipeline): The pipeline object.
        model (Dict[str, object]): A dictionary containing the model name.
        cv (int, optional): The number of cross-validation folds. Defaults to 5

    Returns:
        RandomizedSearchCV: The optimised model.
    """
    parameter_search = {
        "Transform Features__Numeric Features__scaler":
        [StandardScaler(), MinMaxScaler(), RobustScaler()],

        "Resampler": [
            None,
            BorderlineSMOTE(random_state=42),
            KMeansSMOTE(random_state=42),
            RandomOverSampler(random_state=42),
            RandomUnderSampler(random_state=42),
            SMOTE(random_state=42),
            SMOTEENN(random_state=42),  # combination of over- & under-sampling
            SVMSMOTE(random_state=42),
        ],

        f"{model['model_name']}__num_leaves": space.Integer(20, 200),
        f"{model['model_name']}__min_data_in_leaf": space.Integer(20, 50),
        # f"{model['model_name']}__boosting_type": ['gbdt',  'dart',  'rf'],
        f"{model['model_name']}__learning_rate": \
        space.Real(0.005, 1, prior='log-uniform'),
        f"{model['model_name']}__n_estimators": space.Integer(100, 500),
        f"{model['model_name']}__max_depth": space.Integer(4, 10)
    }

    bayes_search_cv = BayesSearchCV(
        estimator=pipeline,
        search_spaces=[(parameter_search, 100)],
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
        scoring=make_scorer(
            score_func=recall_score,
            average="weighted",
        ),
        random_state=42,
        n_jobs=-1,
    )
    print("\nBayes Search CV Pipeline:")
    print(bayes_search_cv)
    return bayes_search_cv
