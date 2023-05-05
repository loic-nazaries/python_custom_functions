"""Statistical analysis for the Rockwool project - Chewing Gum Analysis.

Definition of the functions customised for the analyses.
"""

# %%
# Call the libraries required
# import glob
import matplotlib.pyplot as plt
import numpy as np
# import openpyxl
import pandas as pd
import pingouin as pg
import re
import seaborn as sns
import statsmodels.api as sm
import statsmodels.stats.multicomp as mc
from mat4py import loadmat
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.formula.api import ols


# ----------------------------------------------------------------------------

# DEFINE THE FUNCTIONS

# %%
def get_mat_file_list(directory_name):
    """Get the list of file from a directory.

    Args:
        directory_name (_type_): _description_

    Returns:
        _type_: _description_
    """
    paths = Path(directory_name).glob("**/*.mat")
    dictionary_names = [str(path) for path in paths]
    return dictionary_names


def load_mat_file(mat_file_name):
    """Load data from '.mat' file as a dictionary and print it.

    Also, remove 'pos_border' column as not present in every file.

    Args:
        mat_file_name (_type_): _description_

    Returns:
        _type_: _description_
    """
    dictionary = loadmat(mat_file_name)
    dictionary.pop("pos_border")  # remove data as sometimes missing value
    return dictionary


def calculate_dictionary_values_mean(
    dictionary,
    dict_key_name,
    new_key_name
):
    """Calculate the mean of a 'dict_key_name' from the raw data dictionary.

    Also, the new key (and values) is appended to the dictionary.

    Args:
        dictionary (_type_): _description_
        dict_key_name (_type_): _description_
        new_key_name (_type_): _description_

    Returns:
        _type_: _description_
    """
    mean_key = np.mean(dictionary[dict_key_name]).round(3)
    dictionary[new_key_name] = mean_key
    return dictionary


def calculate_dictionary_values_median(
    dictionary,
    dict_key_name,
    new_key_name
):
    """Calculate the median of a 'dict_key_name' from the raw data dictionary.

    Also, the new key (and values) is appended to the dictionary.

    Args:
        dictionary (_type_): _description_
        dict_key_name (_type_): _description_
        new_key_name (_type_): _description_

    Returns:
        _type_: _description_
    """
    median_key = np.median(dictionary[dict_key_name]).round(3)
    dictionary[new_key_name] = median_key
    return dictionary


def calculate_dictionary_values_standard_deviation(
    dictionary,
    dict_key_name,
    new_key_name
):
    """Calculate the st.dev of a 'dict_key_name' from the raw data dictionary.

    Also, the new key (and values) is appended to the dictionary.

    Args:
        dictionary (_type_): _description_
        dict_key_name (_type_): _description_
        new_key_name (_type_): _description_

    Returns:
        _type_: _description_
    """
    standard_deviation_key = np.std(dictionary[dict_key_name]).round(3)
    dictionary[new_key_name] = standard_deviation_key
    return dictionary


def remove_key_from_dictionary(dictionary, key_name):
    """Delete a key and its content from a dictionary.

    Args:
        dictionary (_type_): _description_
        key_name (_type_): _description_

    Returns:
        _type_: _description_
    """
    del dictionary[key_name]
    return dictionary


def convert_dictionary_to_dataframe(dictionary_file):
    """Convert a dictionary into a dataframe.

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


def extract_pattern_from_file_name(file_name):
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
    pattern_output = [
        re.split(pattern="_+", string=name)[0][19:] for name in file_name
    ]
    # pattern_output = pattern_output.replace("\\", "")
    # pattern_output = list(pattern_output)
    return pattern_output


def concatenate_raw_dataframes(list_dataframes):
    """Concatenate the raw dataframes after conversion of the dictionary.

    Args:
        list_dataframes (_type_): _description_

    Returns:
        _type_: _description_
    """
    concatenate_dataframes = pd.concat(objs=list_dataframes, ignore_index=True)
    return concatenate_dataframes


def convert_dataframe_to_array(dataframe, column_name):
    """Convert dataframe to array.

    Args:
        dataframe (_type_, optional): _description_. Defaults to dataframe.
        column_name (_type_, optional): _description_. Defaults to column_name.

    Returns:
        _type_: _description_
    """
    array = dataframe[column_name].to_numpy()
    return array


def calculate_array_length(array):
    """Calculate the number of rows in the array.

    Args:
        array (_type_): _description_

    Returns:
        _type_: _description_
    """
    array_length = array.shape[0]
    print(f"The array contains {array_length} rows.")
    return array_length


def calculate_array_row_std(data_array, array_length):
    """Calculate the mean of each of the array's row.

    Args:
        data_array (_type_, optional): _description_. Defaults to data_array.
        array_length (_type_, optional): _description_. Defaults to
        array_length.
    """
    array_row_std = [np.std(data_array[row]) for row in range(array_length)]
    return array_row_std


def convert_array_to_series(array, array_name):
    """Convert the means of the array to a Pandas Series.

    Args:
        array (_type_, optional): _description_. Defaults to array.
        array_name (_type_, optional): _description_. Defaults to array_name.
    """
    array_series = pd.Series(array).rename(array_name)
    return array_series


def concatenate_processed_dataframes(dataframe_list):
    """Concatenate the dataframes from the various processing steps.

    Args:
        dataframe_list (_type_): _description_

    Returns:
        _type_: _description_
    """
    data_concatenation = pd.concat(objs=dataframe_list, axis=1)
    return data_concatenation


def drop_non_needed_data(dataframe, column_list):
    """Drop the following columns from the dataframe.

    "data", "thres", "size", "porosity", "pos_border"

    Args:
        dataframe (_type_): _description_

    Returns:
        _type_: _description_
    """
    dataframe_reduced = dataframe.drop(column_list, axis=1)
    return dataframe_reduced


def describe_dataframe_properties(dataframe):
    """Build a statistical summary of the dataframe.

    Args:
        dataframe (_type_): _description_

    Returns:
        _type_: _description_
    """
    dataframe.info()
    print("\n")  # Add un space between the two tables
    dataframe_description = dataframe.describe().T
    # Kurtosis
    kurt_df = dataframe.kurtosis().rename("kurt")
    # Skewness
    skew_df = dataframe.skew().rename("skew")
    # Missing Data
    nan_df = dataframe.isna().sum().rename("nan")
    # Concatenate statistics into one dataframe
    stats_dataframe = pd.concat(
        objs=[
            dataframe_description,
            kurt_df,
            skew_df,
            nan_df
        ],
        axis=1
    )
    print("\n")
    return stats_dataframe


def get_list_of_unique_values(dataframe, column_name):
    """Get a list of unique values from a dataframe.

    "data", "thres", "size", "porosity", "pos_border"

    Since an array is produced, the output is converted to a list.

    Args:
        dataframe (_type_): _description_

    Returns:
        _type_: _description_
    """
    column_unique_values = dataframe[column_name].unique()
    unique_names = list(column_unique_values)
    return unique_names


def group_columns_by_count(dataframe, by_column_list, column_list):
    """Use the count function to group categories by number of values.

    Args:
        dataframe (_type_): _description_
        column_list (_type_): _description_

    Returns:
        _type_: _description_
    """
    dataframe_groups = dataframe.groupby(
        by=by_column_list,
        axis=0,
        dropna=False,
    )[column_list].count()
    return dataframe_groups


def pivot_to_aggregate(dataframe, index_list, column_list, aggfunc_list):
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
        index=index_list,
        columns=column_list,
        fill_value="",
        aggfunc=aggfunc_list
    )
    return pivot_table


def standardise_features(features):
    """Standardise the features to get them on the same scale.

    This is to be performed BEFORE applying Principal Component Analysis (PCA).

    Args:
        feature_list (_type_): _description_

    Returns:
        _type_: _description_
    """
    # Standardise the features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    return features_scaled


def apply_pca(n_components, features_scaled):
    """Run a Principal Component Analysis (PCA) on scaled data.

    The output is of type array.

    Args:
        n_components (_type_): _description_
        features_scaled (_type_): _description_

    Returns:
        _type_: _description_
    """
    pca = PCA(n_components, random_state=42)
    pca_array = pca.fit_transform(features_scaled)
    return pca, pca_array


def convert_array_to_dataframe(array):
    """Convert an array to a dataframe.

    Args:
        array (_type_): _description_

    Returns:
        _type_: _description_
    """
    dataframe = pd.DataFrame(data=array)
    return dataframe


def explain_pca_variance(pca, pca_components):
    """Build a dataframe of variance explained.

    Args:
        pca (_type_): _description_
        pca_components (_type_): _description_

    Returns:
        _type_: _description_
    """
    variance_explained = pca.explained_variance_ratio_ * 100
    variance_explained_cumulated = np.cumsum(
        pca.explained_variance_ratio_) * 100
    variance_explained_df = pd.DataFrame(
        data=[variance_explained_cumulated, variance_explained],
        columns=pca_components,
        index={"cumulated_percent", "percent variance explained"}
    )
    print("Variance explained by the PCA:")
    return variance_explained_cumulated, variance_explained_df


def apply_anova(dataframe, dependent_variable, independent_variable):
    """Perform Analysis of variance using Ordinary Least Squares (OLS) model.

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
    print("ANOVA Table:")
    return model, anova_table


def apply_unbalanced_anova(
    dataframe,
    dependent_variable,
    group_list,
    effect_size="n2"
):
    """Perform Analysis of variance using Ordinary Least Squares (OLS) model.

    Compared to 'apply_anova()' function, this one uses the 'Pingouin' library
    and can be applied to unbalanced designed.

    Args:
        dataframe (_type_): _description_
        dependent_variable (_type_): _description_
        group_list (_type_): _description_
        effect_size (str, optional): _description_. Defaults to "n2".

    Returns:
        _type_: _description_
    """
    unbalanced_model = pg.anova(
        data=dataframe,
        dv=dependent_variable,
        between=group_list,
        detailed=True,
        effsize=effect_size
    )
    return unbalanced_model


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
    shapiro_wilk

    normality = pg.normality(
        data=data,
        method="normaltest",
        alpha=alpha
    )
    normality.rename(index={0: "normality"}, inplace=True)
    normality

    jarque_bera = pg.normality(
        data=data,
        method="jarque_bera",
        alpha=alpha
    )
    jarque_bera.rename(index={0: "jarque_bera"}, inplace=True)
    jarque_bera

    # Concatenate the tests output
    normality_tests = pd.concat(
        objs=[shapiro_wilk, normality, jarque_bera],
        axis=0,
    )
    normality_tests.rename(
        columns={"W": "statistic", "pval": "p-value", "normal": "normal"},
        inplace=True
    )

    # Print a message depending on the value ('True' or 'False') of the
    # 'jarque_bera' output
    print("Normal Distribution of data:")
    if normality_tests.iloc[2, 2] is False:
        print("The data are NOT normally distributed.")
    elif normality_tests.iloc[2, 2] is True:
        print("The data are normally distributed.")
    return normality_tests


def check_equal_variance_assumption(data, group, alpha=0.05):
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
        # data=dataframe,
        # dv=dependent_variable,
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
        # data=dataframe,
        # dv=dependent_variable,
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

    print("Equal Variance of Data Between Groups:")
    if levene["equal_var"] is False:
        print("The data do NOT have equal variance between groups.")
    elif levene["equal_var"] is True:
        print("The data have equal variance between groups")
    return equal_variance_tests


# Below function NOT working
def run_tukey_post_hoc_test(alpha, ylabel, xlabel, *args):
    """Use Tukey's HSD post-hoc test for multiple comparison between groups.

    It produces a table of significant differences AND a figure of means
    between groups/treatments.

    BUG Function NOT working !

    Args:
        ylabel (_type_): _description_
        xlabel (_type_): _description_
        alpha (float, optional): _description_. Defaults to 0.05.

    Returns:
        _type_: _description_
    """
    comparison = mc.MultiComparison(args)
    tukey = comparison.tukeyhsd(alpha=0.05)
    tukey.summary()

    tukey.plot_simultaneous(
        ylabel=ylabel,
        xlabel=xlabel
    )
    plt.suptitle("Tukey's HSD Post-hoc Test", fontsize=14)
    return tukey


def run_tukey_post_hoc_test_2(dataframe, dependent_variable, group_list):
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

    # TODO change data type back to boolean for 'reject' after concatenation

    # Use
    correction_dataframe = pd.DataFrame(concatenate_arrays)
    correction_dataframe.rename(
        columns={0: "reject_hypothesis", 1: "corrected_p_values"},
        inplace=True
    )
    return correction_dataframe


def draw_histogram(
    dataframe,
    x,
    xlabel,
    ylabel,
    hue=None,
    palette=None,
    binwidth=None,
):
    """Draw an histogram plot of data distribution.

    Args:
        dataframe (_type_): _description_
        x (_type_): _description_
        xlabel (_type_): _description_
        ylabel (_type_): _description_
        hue (_type_, optional): _description_. Defaults to None.
        palette (_type_, optional): _description_. Defaults to None.
        binwidth (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    histogram = sns.histplot(
        data=dataframe,
        x=x,
        kde=True,
        hue=hue,
        palette=palette,
        legend=True,
        binwidth=binwidth
    )
    histogram.set(
        xlabel=xlabel,
        ylabel=ylabel,
    )
    return histogram


def draw_boxplot(
    dataframe,
    x,
    y,
    hue=None,
    palette=None
):
    """Draw a boxplot of data distribution.

    Args:
        dataframe (_type_): _description_
        x (_type_): _description_
        y (_type_): _description_
        hue (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    boxplot = sns.boxplot(
        data=dataframe,
        x=x,
        y=y,
        hue=hue,
        palette=palette,
        orient="h",
    )
    return boxplot


def draw_lineplot(
    dataframe,
    x,
    y,
    ci=None
):
    """Draw a line plot with a confidence interval when provided.

    Args:
        dataframe (_type_): _description_
        x (_type_): _description_
        y (_type_): _description_
        ci (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    lineplot = sns.lineplot(
        data=dataframe,
        x=x,
        y=y,
        # estimator="mean",
        err_style="band",
        ci=95,
    )
    lineplot.set(
        xlabel="Prélèvement N°",
        ylabel="Transmissivité \u03C4 (-)",
    ),
    lineplot.set_ylim(0.50, 0.95),
    lineplot.set_xlim(0, 452),
    return lineplot


def draw_scree_plot(x, y):
    """Draw scree plot following a PCA.

    Args:
        x (_type_): _description_
        y (_type_): _description_

    Returns:
        _type_: _description_
    """
    scree_plot = sns.lineplot(
        x=x,
        y=y,
        color='black',
        linestyle='-',
        linewidth=2,
        marker='o',
        markersize=8,
    )
    return scree_plot


def draw_scatterplot(
    dataframe,
    x,
    y,
    size,
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
        x=x,
        y=y,
        hue=hue,
        palette=palette,
        size=size,
        sizes=(20, 200),
        legend="full"
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
        linewidths=0.05,
        cbar=False,
        cmap="YlGnBu",
    )
    return heatmap


def draw_qq_plot(
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


def save_figure(figure_name):
    """Save a figure as '.png' AND '.svg' file.

    Args:
        figure_name (_type_): _description_
    """
    for extension in (["png", "svg"]):
        plt.savefig(
            fname=f"../images/{figure_name}.{extension}",
            format=extension,
            bbox_inches="tight",
            dpi=300
        )
    return


# %%
def main():
    """Prepare the functions for exporting."""
    get_mat_file_list()
    load_mat_file()
    calculate_dictionary_values_mean()
    calculate_dictionary_values_median()
    calculate_dictionary_values_standard_deviation()
    remove_key_from_dictionary()
    convert_dictionary_to_dataframe()
    extract_pattern_from_file_name()
    concatenate_raw_dataframes()
    convert_dataframe_to_array()
    # calculate_array_length()
    # calculate_array_row_std()
    convert_array_to_series()
    concatenate_processed_dataframes()
    drop_non_needed_data()
    describe_dataframe_properties()
    get_list_of_unique_values()
    group_columns_by_count()
    pivot_to_aggregate()
    standardise_features()
    apply_pca()
    convert_array_to_dataframe()
    explain_pca_variance()
    apply_anova()
    apply_unbalanced_anova()
    check_normality_assumption()
    check_equal_variance_assumption()
    run_tukey_post_hoc_test()
    run_tukey_post_hoc_test_2()
    perform_multicomparison_correction()
    draw_histogram()
    draw_boxplot()
    draw_lineplot()
    draw_scree_plot()
    draw_scatterplot()
    draw_heatmap()
    draw_qq_plot()
    save_figure()


if __name__ == "__main__":
    main()

# %%
