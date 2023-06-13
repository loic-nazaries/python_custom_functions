"""Modelling.

Analysis of the detection parameters for defects/chewing gums.
display a tree classifier with the range of values for each node. then,
create a SHAP analysis to interpret the outcome of the model
MODELLING
The 'fast_ml' is used to split the dataset into train, test and validation sets
in one-go by using the ' train_valid_test_split' function.
Model defect detection test different machine learning algorithms.
TODO Add 'remove_low_variance_features' function instead ?
    Also add 'identify_highly_correlated_features' function ?
TODO Write a main function and sub-functions for the different steps.
"""

# %%
# Suppress unnecessary warnings so that presentation looks clean
from sklearn.base import BaseEstimator, TransformerMixin
import warnings
import copy
import timeit
from pathlib import Path
from pprint import pprint

import custom_functions_romain as cf
# import matplotlib.pyplot as plt
# import joblib
import numpy as np
import pandas as pd
# from scipy.stats import loguniform
import seaborn as sns
# Imports from other modules
# from config import OUTPUT_DIR_FIGURES, OUTPUT_DIRECTORY
# Sampling
# from fast_ml.model_development import train_valid_test_split
from IPython.core.interactiveshell import InteractiveShell
# Assemble pipeline(s)
# from sklearn import set_config
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    # QuadraticDiscriminantAnalysis
)
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
# from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    # RepeatedStratifiedKFold,
    StratifiedKFold,
    # StratifiedShuffleSplit,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import FunctionTransformer
# from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.pipeline import FeatureUnion
from imblearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
# from tqdm import tqdm
# Models
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE  # , ADASYN
from IPython.display import display
# from sklearn.feature_selection import RFECV

# from imblearn.over_sampling import SMOTE
# from imblearn.under_sampling import RandomUnderSampler


InteractiveShell.ast_node_interactivity = "all"

# Set up Pandas and Seaborn options
# np.set_printoptions(threshold=np.inf)  # array printed in full
pd.set_option("display.precision", 2)  # two decimal places
pd.set_option("display.max_columns", None)  # 'None' means unlimited
pd.set_option("display.max_colwidth", None)  # 'None' means unlimited
sns.set_theme(style="whitegrid")
warnings.filterwarnings("ignore")

# Make pipeline visible
# set_config(display="diagram", transform_output="default")


# START OF SCRIPT TIMING

# Measure execution time of the WHOLE analysis
t_0 = timeit.default_timer()


# -----------------------------------------------------------------------------

# %%

# LOAD DATA

# Set-up the default directories
INPUT_DIR = Path("./sources/tagged_defects_fixed_params_2023")
OUTPUT_DIR = Path("./output/tagged_defects_fixed_params_2023")
OUTPUT_DIR_FIG = Path("./figures/tagged_defects_fixed_params_2023")

# Load pickle file data will be extracted from
dataset = cf.load_pickle_file(
    file_name="final_dataset_fixed_params_2023",
    # output_directory=OUTPUT_DIR,
    output_directory=Path(f"../{OUTPUT_DIR}"),
)

# BUG with the validation step when 'healthy' defect type present
# Hence, removed from the dataset for now
dataset = cf.filter_dataframe(
    dataframe=dataset,
    # dataframe=filtered_data_labels_clusters,
    filter_content="defect_type != 'healthy'",
)


# %%
# Step -1: Prepare the model target and features

# Define the target constant for the classification task
TARGET = "defect_type"
# TARGET = "defect_product"

# Separate the target from the rest of the dataset
target = dataset[TARGET]
target

# Show number of items in each class of the target
cf.show_items_per_category(data=target, category_name=TARGET)

# # ! BUG: getting a KeyError message
# NOTE: Fixed with giving a DataFrame instead of a Series
# It was needed for XGBoost
# # Since we will be working with a classification model, fit and transform
# # the target column with a label encoder
target_label_encoded, target_label_list, le = cf.target_label_encoder(
    dataframe=pd.DataFrame(target, columns=['defect_type']),
    target_name="defect_type",
)
# NOTE might be wrong
dataset[TARGET] = target_label_encoded
target = target_label_encoded

# * Create list of defect types manually since function above does not work...
# target_label_list = ["CG", "CGGros", "Melt", "MeltGros"]

# NOTE Unfortunately, the 'LabelEncoder()' class function for target variable
# cannot be added to a Scikit-learn pipeline, hence must be done beforehand.


# -----------------------------------------------------------------------------

# Separate the features from the target
features = dataset.drop(labels=TARGET, axis=1)
feature_list = features.columns.to_list()
print(f"List of features:\n{feature_list}\n")

# Define the features to be used and the others will be filtered out at the
# step 'drop_feature_pipeline'
features_to_keep = [
    # "defect_type",
    # "product",
    # "detection_mode",
    # "defect_product",
    # "product_mode",
    # "KMeans_cluster_labels"
    "length",
    "width",
    "thickness",
    "threshold_fixed",
    "calculated_mean_intensity",
    "calculated_size",
    "calculated_porosity",
    # "pc1",
    # "pc2",
    # "pc3",
    # "pc4",
]
print(f"\nFeatures included in the pipeline:\n{features_to_keep}\n")


# %%
# Step 0: Split the dataset into train, test, AND validation sets

(
    features_train,
    target_train,
    features_valid,
    target_valid,
    features_test,
    target_test
) = cf.train_valid_test_split_fast(
    dataframe=dataset,
    target=TARGET,
    train_size=0.6,
    valid_size=0.2,
    test_size=0.2,
)

print(f"\nFeature Train Set Shape: {features_train.shape}")
print(features_train.head(3))
print(f"\nTarget Train Set Shape: {target_train.shape}")
print(target_train.head(3))
print(f"\nFeature Test Set Shape: {features_test.shape}")
print(f"\nTarget Test Set Shape: {target_test.shape}")

# # Commented for now to speed up the process
# # Compare the data split between the training and testing sets
# cf.compare_sweetviz_eda_report(
#     train_set=features_train,
#     test_set=features_test,
#     eda_report_name="train_vs_test_data_report",
#     # output_directory=OUTPUT_DIR,
#     output_directory=Path("../{OUTPUT_DIR}"),  # for notebooks
# )


# %%
# Step 0.5: Build a 'FunctionTransformer()' to split the dataset into training,
# validate and testing sets

# BUG Below code NOT working when inserted into the final pipeline
train_valid_test_split_transformer = FunctionTransformer(
    func=cf.train_valid_test_split_fast,
    kw_args={
        "train_size": 0.6,
        "valid_size": 0.2,
        "test_size": 0.2,
    },
    # kw_args={"target_train": target_train},
    validate=False,
    # feature_names_out=True
)


# %%
# Step 1: Drop irrelevant columns/features

# remove_nan_transformer = cf.remove_nan_feature_transformer(features=features)

drop_feature_pipeline = cf.drop_feature_pipeline(
    feature_selection=features,
    features_to_keep=features_to_keep,
    # variables_with_nan_values=["KMeans_cluster_labels"],
)


# %%
# Step 2: Preprocess numeric and categorical features

# Impute and encode categorical features
categorical_feature_pipeline = (
    cf.preprocess_categorical_feature_pipeline(encoder="one_hot_encoder")
)

# Impute and scale numeric features
numeric_feature_pipeline = (
    cf.preprocess_numeric_feature_pipeline(scaler="robust_scaler")
    # cf.preprocess_numeric_feature_pipeline(scaler="standard_scaler")
    # cf.preprocess_numeric_feature_pipeline(scaler="min_max_scaler")
)

# Create a column transformer with both numeric and categorical pipelines
preprocess_feature_pipeline = cf.transform_feature_pipeline(
    categorical_feature_pipeline=categorical_feature_pipeline,
    numeric_feature_pipeline=numeric_feature_pipeline,
)


# %%
# Step 3: Sampling to deal with class imbalance of the TARGET only

# SMOTE()
# choice of new samples with KNN over existing samples
# from minority class (no distinction))
resampler = SMOTE(random_state=42)

# ADASYN()
# choice over samples from minority class that are hard to classify
# ! BUG with ADASYN, specifically:
# ! RuntimeError: Not any neigbours belong to the majority class.
# ! This case will induce a NaN case with a division by zero.
# ! ADASYN is not suited for this specific dataset. Use SMOTE instead.
# resampler = ADASYN(random_state=42)


# %%
# Step 4: Building the ensemble classifier

# RandomForest
random_forest_classifier = RandomForestClassifier(
    # n_estimators=500,  # n_estimators=500
    # random_state=42,
    # verbose=1,
    # max_depth=4,
    # max_leaf_nodes=5,
    # # min_samples_leaf=0.05,
    # # min_samples_split=0.05
    n_estimators=100,
    random_state=42,
    verbose=1,
    max_depth=4,
    max_leaf_nodes=5,
    criterion="gini",
    max_features="auto",
    min_samples_leaf=1,
    min_samples_split=2
)

# XGB
xgboost = XGBClassifier(
    tree_method="hist",
    verbosity=0,
    silent=True,
    random_state=42,
    verbose=1,
    # tuned parameters
    max_depth=4,
    learning_rate=0.05,
    gamma=10
)

# Ensemble
ensemble_classifier = VotingClassifier(
    estimators=[
        ("random_forest_classifier", random_forest_classifier),
        ("xgboost", xgboost),
        # ("Quadratic Discriminant Analysis", QuadraticDiscriminantAnalysis()),
        ("Gaussian Naive Bayes", GaussianNB()),
        ("Linear Discriminant Analysis", LinearDiscriminantAnalysis()),
        # ("Random Forest", RandomForestClassifier()),
        ("Decision Tree Classifier", DecisionTreeClassifier(
            min_samples_leaf=20,
            max_depth=3,
            criterion='gini')),
        ("Support Vector Machine", SVC(
            probability=True,
            kernel='linear',
            gamma='auto'))
    ],
    voting="hard",  # getting bad results with "soft" with bad estimators
                    # like Quadratic Discriminant Analysis
)

# # Save model parameters
# rf_classifier_parameters = random_forest_classifier.get_params()
# print(f"\nRandom Forest Base Model Parameters:\n{rf_classifier_parameters}")


# %%
# Step 4.5: Building a PCA and classifier pipeline to run them simultaneously

# Set up the pipeline to run Principal Component Analysis (PCA)
pca_pipeline = Pipeline(
    steps=[
        (
            "pc analysis",
            PCA(n_components=0.95, random_state=42),
        ),
        # (
        #     "svm classifier", SVC(kernel="linear")
        # ),
        # TODO Add CCA function
    ],
    verbose=True
)
print("\nPCA Pipeline Structure:")
pprint(pca_pipeline)

# -----------------------------------------------------------------------------

# # Define the pipeline with the models to test (MUST use a list of tuples)
# models_to_test = [
#     # ("LGBM Classifier", LGBMClassifier()),
#     ("Quadratic Discriminant Analysis", QuadraticDiscriminantAnalysis()),
#     ("Gaussian Naive Bayse", GaussianNB()),
#     ("Linear Discriminant Analysis", LinearDiscriminantAnalysis()),
#     #("Random Forest", RandomForestClassifier()),
#     ("Decision Tree Classifier", DecisionTreeClassifier()),
#     ("Support Vector Machine", SVC()),
# ]
# # cf.test_multiple_model_pipeline(models_to_test=models_to_test)


# Set-up a pipeline to run Random Forest Classifier model (or multiple models)
classifier_pipeline = Pipeline(
    steps=[
        # models_to_test,
        (
            "rf classifier",
            random_forest_classifier
        ),
    ],
    verbose=True
)
print("\nClassifier Pipeline Structure:")
print(classifier_pipeline)

# Save model parameters
classifier_parameters = classifier_pipeline.get_params()
pprint(f"\nRandom Forest Base Model Parameters:\n{classifier_parameters}\n")


# ! # BUG Below code NOT working due to an issue with the estimators that
# ! should have an implementation of fit and transform, however random forest
# ! does not
# Create a column transformer to run in parallel PCA and one (or more)
# classifier (e.g. Random Forest) pipelines
model_transformer = ColumnTransformer(
    transformers=[
        (
            "pca pipeline",
            pca_pipeline,
            make_column_selector(dtype_include="number")
        ),
        (
            "classifier pipeline",
            classifier_pipeline,
            make_column_selector(dtype_include="number")
        ),
    ],
    verbose_feature_names_out=True,  # if True, will display transformers
    remainder="drop",
    n_jobs=-1,
)
print("\nModel Transformer Pipeline Structure ('ColumnTransformer'):")
print(model_transformer)

# %%
# Same as above but using the 'FeatureUnion' class from scikit-learn
model_transformer2 = FeatureUnion(
    transformer_list=[
        ("pca pipeline", pca_pipeline),
        ("classifier pipeline", classifier_pipeline)
    ],
    n_jobs=-1,
    verbose=True
)
print("\nModel Transformer Pipeline Structure ('FeatureUnion'):")
print(model_transformer2)

# -----------------------------------------------------------------------------

# %%
# Alternatively, define classes to fit/transform/predict data for PCA and RF


class PCA_Transformer(BaseEstimator, TransformerMixin):
    """PCA_Transformer _summary_.

    Args:
        BaseEstimator (_type_): _description_
        TransformerMixin (_type_): _description_
    """

    def __init__(self, features):
        """__init__ _summary_.

        Args:
            n_components (float, optional): _description_. Defaults to 0.95.
        """
        self.features = features
        # self.n_components = n_components

    def fit(self, features, y=None, n_components=0.95):
        """Fit _summary_.

        Args:
            features (_type_): _description_

        Returns:
            _type_: _description_
        """
        self.n_components = n_components
        self.pca_transformer = PCA(
            n_components=self.n_components,
            # n_components=0.95,
            random_state=42
        )
        self.pca_transformer.fit(X=self.features)
        # self.pca_transformer.transform(X=self.features)
        return self

    def transform(self, features):
        """Transform _summary_.

        Args:
            features (_type_): _description_

        Returns:
            _type_: _description_
        """
        features_transformed = self.pca_transformer.transform(self.features)
        # print(features_transformed)

        # Get PC axis labels and insert into a dataframe
        pca_components = [
            f"pc{str(col + 1)}"
            for col in range(self.pca_transformer.n_components_)
        ]

        # Convert to a dataframe for later concatenation with predicted values
        features_transformed_df = pd.DataFrame(
            data=features_transformed,
            # data=self.pca_transformer.transform(self.features),
            columns=pca_components,
            # columns=self.pca_transformer.feature_names_in_(self.features),
            index=self.features.index
        )
        print(features_transformed_df)
        return features_transformed_df


class RandomForestTransformer(BaseEstimator, TransformerMixin):
    """RandomForestTransformer _summary_.

    Args:
        BaseEstimator (_type_): _description_
        TransformerMixin (_type_): _description_
    """

    def __init__(self, features):
        """__init__ _summary_.

        Args:
            n_estimators (int, optional): _description_. Defaults to 500.
        """
        self.features = features

    def fit(self, features, target, n_estimators=500):
        """Fit _summary_.

        Args:
            features (_type_): _description_
            target (_type_): _description_

        Returns:
            _type_: _description_
        """
        self.n_estimators = n_estimators
        self.rf_classifier = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=42,
            verbose=1,
            # max_depth=4,
            max_leaf_nodes=5
        )
        self.rf_classifier.fit(X=features, y=target)
        return self

    def transform(self, features):
        """Transform _summary_.

        Args:
            features (_type_): _description_

        Returns:
            _type_: _description_
        """
        # columns = self.rf_classifier.get_feature_names_out(self.features)

        target_pred = self.rf_classifier.predict(X=features)
        features_proba = self.rf_classifier.predict_proba(X=features)
        predictions = np.concatenate(
            arrays=(target_pred, features_proba),
            axis=1
        )
        print(predictions)
        return predictions


# Same as above but using the 'FeatureUnion' class from scikit-learn
model_transformer3 = FeatureUnion(
    transformer_list=[
        # * Below line works well
        # ("pca pipeline", (
        #     PCA_Transformer(features_train.select_dtypes(include="number"))
        # )),
        # ("pca pipeline", pca_pipeline),
        # ! Below line NOT working
        # ! TypeError: concatenate() missing required argument 'seq' (pos 1)
        ("classifier pipeline", (
            RandomForestTransformer(
                features_train.select_dtypes(include="number"))
        )),
        # ("classifier pipeline", classifier_pipeline)
    ],
    n_jobs=-1,
    verbose=True
)
print("\nModel Transformer Pipeline Structure 'TransformerMixin':")
print(model_transformer3)


# %%
# Step 5: Set up the final pipeline

# pipeline = imbPipeline([  # Would crash with Pipeline object from sklearn
# NOTE: this is not a sklearn pipeline but a imblearn pipeline
pipeline = Pipeline(
    steps=[
        # ! # BUG: Below line NOT working
        # ("train-validate-test split", train_valid_test_split_transformer),

        # ("remove_nan_values", remove_nan_feature_transformer),  # BUG
        # ("remove_nan", remove_nan_transformer),  # BUG
        # ("drop features", drop_feature_pipeline),
        # # Pipeline class from imblearn does not support nested class
        # Unpacking the steps of the pipeline is required
        *drop_feature_pipeline.steps,
        ("resampler", resampler),  # SMOTE
        ("transform features", preprocess_feature_pipeline),
        ("ensemble", ensemble_classifier),
        # ensemble_classifier.estimators[5]  # if only one estimator

        # # Below line works well (see Step 7.5)
        # ("pca pipeline", pca_pipeline),

        # ! Below line getting error when applying '.transform()'
        # ! AttributeError: 'RandomForestClassifier' object has no attribute
        # !'transform'
        # * Otherwise OK on '.predict()' (see Step 7.5-bis)
        # ("random_forest_classifier", random_forest_classifier),

        # ! Below line getting error when applying '.transform()'
        # ! AttributeError: 'RandomForestClassifier' object has no attribute
        # ! 'transform'
        # * Otherwise OK on '.predict()'
        # ("classifier pipeline", classifier_pipeline),

        # ! Below line, getting error: 'TypeError: All estimators should
        # ! implement fit and transform, or can be 'drop' or 'passthrough'
        # ! specifiers.'
        # ("run model", model_transformer),

        # ! Below line, getting error: 'TypeError: All estimators should
        # ! implement fit and transform.'
        # ("run model", model_transformer2),

        # ! Below line, getting error: 'TypeError: concatenate() missing
        # ! required argument 'seq' (pos 1)' when working with RandomForest
        # ("run model", model_transformer3),
    ],
    verbose=True
)
# Set the config pipeline output to a (dense, not sparse) Numpy array
# pipeline.set_output(transform="default")
print(f"\nDisplay Analytic Pipeline:\n{pipeline}\n")
pipeline  # for notebooks


# %%
# # Using the 'pca_pipeline' and 'classifier_pipeline' objects
# # ! # BUG: 'ValueError: could not convert string to float: 'Pacific12'
# # ! This is because the two pipelines do not go through the main 'pipeline'
# # Fit and transform PCA on the train data, and transform on the test data
# features_train_pca = pca_pipeline.fit_transform(features_train)
# features_train_pca
# features_test_pca = pca_pipeline.transform(features_test)
# features_test_pca

# # Fit random forest classifier on the train data, & predict on the test data
# classifier_pipeline.fit(features_train, target_train)
# target_pred = classifier_pipeline.predict(features_test)
# target_pred

# # Concatenate the transformed PCA and predicted RF features
# features_train_new = np.concatenate(
#     (features_train_pca, classifier_pipeline.predict_proba(features_train)),
#     axis=1)
# features_train_new
# features_test_new = np.concatenate((features_test_pca, target_pred), axis=1)
# features_test_new


# %%
# Step 7: Fit pipeline model

# # Fit the model/pipeline on the training set
# features_train_transformed = pipeline.fit_transform(
#     X=features_train,
#     y=target_train
# )
# print(type(features_train_transformed))

# # Get PC axis labels and insert into a dataframe
# pca_components = [
#     "pc" + str(col+1) for col in range(features_train_transformed.shape[1])
# ]

# # Convert to a dataframe for later concatenation with predicted values
# features_train_transformed_df = pd.DataFrame(
#     data=features_train_transformed,
#     columns=pca_components,
#     index=features_train.index
# )
# print(features_train_transformed_df)


# %%
# # Step 7.5: Make predictions for the labels of the test set (PCA)

# # Fit the model/pipeline on the train set
# pipeline.fit(X=features_train)

# # Transform the train set
# features_train_transformed = pipeline.transform(X=features_train)
# print(f"Transformed Features Train Set:\n{features_train_transformed}\n")

# # Get PC axis labels and insert into a dataframe
# pca_components = [
#     "pc" + str(col+1) for col in range(features_train_transformed.shape[1])
# ]
# # Convert to a dataframe for later concatenation with (transformed) test data
# features_train_transformed_df = pd.DataFrame(
#     # data=features_train,
#     data=features_train_transformed,
#     columns=pca_components,
#     index=features_train.index
# )

# # Transform the target samples to get PCA scores
# target_pred_pca = pipeline.transform(X=features_test)

# # Convert to a dataframe for later concatenation with predicted values
# target_pred_pca_df = pd.DataFrame(
#     data=target_pred_pca,
#     columns=pca_components,
#     index=target_test.index
# )

# # Concatenate transformed original and test dataframes
# prediction_list_pca = [
#     features_train_transformed_df,
#     target_pred_pca_df
# ]
# predictions_pca_dataframe = pd.concat(
#     objs=prediction_list_pca,
#     axis=0
# )
# print(f"\nTest vs. Predictions:\n{predictions_pca_dataframe}\n")

# -----------------------------------------------------------------------------

# %%

# Step 7.5-bis: Make predictions for the labels of the test set (Random Forest)

# Fit the model/pipeline on the train set
pipeline.fit(X=features_train, y=target_train)

# Predict the target on the test set
target_pred = pipeline.predict(X=features_test)

# Build a dataframe of true/test values vs. prediction values
target_pred_df = pd.DataFrame(
    data=target_pred,
    columns=["predictions"],
    index=features_test.index
)
print(type(target_pred_df))

prediction_list_random_forest = [
    target_test,
    target_pred_df
]
prediction_list_random_forest
predictions_random_forest = pd.concat(
    objs=prediction_list_random_forest,
    axis=1
)
print(f"\nTest vs. Predictions:\n{predictions_random_forest}\n")


# %%
# Step 8: Get modelling scores from cross-validation

cf.calculate_cross_validation_scores(
    model=pipeline,
    features_test=features_test,
    target_test=target_test,
    target_pred=target_pred,
    target_label_list=target_label_list,
    cv=5
)

cf.draw_confusion_matrix_heatmap(
    target_test=le.inverse_transform(target_test),
    target_pred=le.inverse_transform(target_pred),
    target_label_list=target_label_list,
    figure_name="rf_confusion_matrix_predictions",
    # output_directory=OUTPUT_DIR_FIG,
    output_directory=Path(f"../{OUTPUT_DIR_FIG}")  # for notebooks
)

cf.calculate_multiple_cross_validation_scores(
    model=pipeline,
    features=features_train,
    target=target_train,
)


#  %%
# Step 8-bis: Use cross-validation prediction scores for train & test sets

# Results for train set
print("\nCross-Validation Prediction Scores for Train Set")
cf.calculate_cross_validation_prediction_scores(
    model=pipeline,
    features=features_train,
    target=target_train,
    cv=10
)

# Results for test set
print("\nCross-Validation Prediction Scores for Test Set")
cf.calculate_cross_validation_prediction_scores(
    model=pipeline,
    features=features_test,
    target=target_test,
    cv=10
)


# %%
# Step 9: Get feature importance scores for the model

# # Get the name of the features including within the model
# feature_name_list = pipeline[:-1].get_feature_names_out().tolist()

# # NOTE: Only work for Random Forest Classifier, not Voting Classifier
# rf_model_feature_importances = cf.get_feature_importance_scores(
#     # model=random_forest_classifier,
#     # model=pipeline.named_steps['xgboost'],  # for xgboost
#     feature_name_list=feature_name_list,
#     figure_name="rf_feature_importances_predictions",
#     # output_directory=OUTPUT_DIR_FIG,
#     output_directory=Path(f"../{OUTPUT_DIR_FIG}")  # for notebooks
# )


# %%
# print(f"\n{target_test.head() = }\n")

# # BUG 'Reshape your data either using array.reshape(-1, 1) if your data has a
# # single feature or array.reshape(1, -1) if it contains a single sample.'
# encoder = OrdinalEncoder()
# target_test2 = encoder.fit(target_test).reshape(-1, 1)
# print(target_test2)

# # BUG ValueError: could not convert string to float: 'Arctic15'
# # OR  ValueError: Input X contains NaN
# # OR  ValueError: could not convert string to float: 'CGGros'
# rf_model_feature_importances_permutation = (
#     cf.get_feature_importance_scores_permutation(
#         model=random_forest_classifier,
#         features_test=features_test,
#         target_test=target_test,
#         feature_name_list=feature_name_list,
#         figure_name="rf_feature_importances_prediction_permutations",
#         # output_directory=OUTPUT_DIR_FIG,
#         output_directory=Path(f"../{OUTPUT_DIR_FIG}")  # for notebooks
#     )
# )


# # Draw decision tree from the RandomForest model
# cf.draw_random_forest_tree(
#     random_forest_classifier=random_forest_classifier,
#     feature_name_list=feature_name_list,
#     target_label_list=target_label_list,
#     ranked_tree=0,
#     figure_name="random_forest_tree",
#     # output_directory=OUTPUT_DIR_FIG,
#     output_directory=Path(f"../{OUTPUT_DIR_FIG}")  # for notebooks
# )

# %%
# Step 10: Train tree classifier

# # BUG Below NOT working if categorical variables not one-shot encoded ?
# (
#     rf_tree_classifier,
#     rf_predictions_tree_classifier
# ) = cf.train_tree_classifier(
#     features_train=features_train,
#     target_train=target_train,
#     features_test=features_test,
#     target_test=target_test,
#     target_pred=target_pred,
#     index_name="file_name"
# )

# cf.draw_decision_tree(
#     tree_classifier=rf_tree_classifier,
#     feature_name_list=feature_name_list,
#     target_label_list=target_label_list,
#     figure_name="decision_tree",
#     # output_directory=OUTPUT_DIR_FIG,
#     output_directory=Path(f"../{OUTPUT_DIR_FIG}")  # for notebooks

# )

# target_train_count = target_train.value_counts()
# print(f"\nNumber of hits for each defect type:\n{target_train_count}\n")

# cf.show_tree_classifier_feature_importances(
#     tree_classifier=rf_tree_classifier,
#     feature_name_list=feature_name_list,
#     features_train=features_train,
#     target_train=target_train,
#     features_test=features_test,
#     target_test=target_test,
# )


# %%
# Step 11: Explain the model's predictions using SHAP values

# cf.explain_model_with_shap(
#     model=random_forest_classifier,
#     # feature_names=feature_names,
#     # OR
#     feature_name_list=feature_name_list,
#     features_test=features_test,
# )


# %%
# Step XXX: Model comparison

# Finally, we evaluate the models using cross validation. Here we compare the
# models performance in terms of mean_absolute_percentage_error.

# scoring = "neg_mean_absolute_percentage_error"
# n_cv_folds = 3

# pipe.set_params(
#     max_depth=3,
#     max_iter=15,
# )

# model1_result = cross_validate(
#     model1, X, y, cv=n_cv_folds, scoring=scoring
# )
# model2_result = cross_validate(
#     model2, X, y, cv=n_cv_folds, scoring=scoring
# )
# model3_result = cross_validate(
#     model3, X, y, cv=n_cv_folds, scoring=scoring
# )
# model4_result = cross_validate(
#     model4, X, y, cv=n_cv_folds, scoring=scoring
# )


# %%
# Step 12: Hyperparameter tuning of the ensemble classifier

# # Set of optimisation options
parameters = {
    # "ensemble__logistic_regression__solver": [
    #     "newton-cg",
    #     "lbfgs",
    #     "liblinear"
    # ],
    # "ensemble__logistic_regression__penalty": [
    #     "none",
    #     "l1",
    #     "l2",
    #     "elasticnet"
    # ],
    # "ensemble__logistic_regression__C": loguniform(1e-5, 100),
    # "ensemble__random_forest__max_depth": [7, 10, 15, 20],
    # "ensemble__random_forest__min_samples_leaf": [1, 2, 4],
    # "ensemble__random_forest__min_samples_split": [2, 5, 10],
    # "ensemble__random_forest__n_estimators": [300, 400, 500, 600],
    # "ensemble__xgboost__min_child_weight": [10, 15, 20, 25],
    # "ensemble__xgboost__colsample_bytree": [0.8, 0.9, 1],
    # "ensemble__xgboost__n_estimators": [300, 400, 500, 600],
    # "ensemble__xgboost__reg_alpha": [0.5, 0.2, 1],
    # "ensemble__xgboost__reg_lambda": [2, 3, 5],
    "ensemble__xgboost__learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
    "ensemble__xgboost__max_depth": [3, 4, 5],
    "ensemble__xgboost__gamma": [0, 1, 2,  5, 10],
    "ensemble__Gaussian Naive Bayes__var_smoothing":
        [0, 1e-9, 0.01, 0.1, 0.3, 0.5, 0.7, 1],
    "ensemble__Linear Discriminant Analysis__solver": ['svd', 'lsqr', 'eigen'],
    "ensemble__Decision Tree Classifier__criterion": ["gini", "entropy"],
    "ensemble__Decision Tree Classifier__max_depth": [2, 3, 5, 10, 20],
    "ensemble__Decision Tree Classifier__min_samples_leaf":
        [5, 10, 20, 50, 100],
    "ensemble__Support Vector Machine__C": [0.1, 1, 10, 100, 1000],
    "ensemble__Support Vector Machine__gamma":
        ['auto', 'scale', 1, 0.1, 0.01, 0.001, 0.0001],
    "ensemble__Support Vector Machine__kernel": ['rbf', 'linear'],
}

# Apply cross-validation optimisation using K-fold
# rsf = RepeatedStratifiedKFold(random_state=42)
skf = StratifiedKFold(random_state=42, n_splits=5, shuffle=True)

# Model optimisation using 'RandomizedSearchCV()'
random_search_cv = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=parameters,
    scoring="roc_auc",
    verbose=True,
    cv=skf
    # cv=rsf,
)
print(f"\nRandom search CV:\n{random_search_cv}\n")
random_search_cv
# # Is takes a while
# random_search_cv.fit(features_train, target_train)
# target_pred = random_search_cv.predict(features_test)
# print(target_pred)

# # Get best parameters
# ensemble_best_score = random_search_cv.best_score_
# print(f"\nBest Score: {ensemble_best_score}\n")

# ensemble_best_parameters = random_search_cv.best_params_
# print(f"\nBest Parameters:\n{ensemble_best_parameters}\n")


# # Model optimisation using 'GridSearchCV()'
# grid_search_cv = GridSearchCV(
#     estimator=pipeline,
#     param_grid=parameters,
#     scoring="roc_auc",
#     verbose=True,
#     cv = skf
#     # cv=rsf,
# )
# print(f"\nRandom search CV:\n{grid_search_cv}\n")
# grid_search_cv

# grid_search_cv.fit(features_train, target_train)
# target_pred = grid_search_cv.predict(features_test)
# print(target_pred)

# # Get best parameters
# ensemble_best_score = grid_search_cv.best_score_
# print(f"\nBest Score: {ensemble_best_score}\n")

# ensemble_best_parameters = grid_search_cv.best_params_
# print(f"\nBest Parameters:\n{ensemble_best_parameters}\n")


# %%
# Step 12-bis: Create a pipeline step to fine-tune hyper-parameters

param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 10],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}
optimisation_pipeline = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="accuracy",
    # scoring=cf.calculate_multiple_cross_validation_scores(
    #     model=pipeline,
    #     features=features_train,
    #     target=target_train,
    # ),
    cv=5,
    n_jobs=-1,
    verbose=3,
)
print(f"\nDisplay Optimisation Pipeline:\n{optimisation_pipeline}\n")
optimisation_pipeline  # for notebooks

# # BUG
# for _ in tqdm(range(3)):
#     optimisation_pipeline.fit(features_train, target_train)

# # Print the best hyper-parameters and corresponding accuracy score
# print(f"Best parameters:\n{optimisation_pipeline.best_params_}")
# print(f"Accuracy score: {optimisation_pipeline.best_score_:.1%}\n")

# # Make prediction on the validation set
# target_pred = optimisation_pipeline.predict(features_valid)


# %%
# Step 12-tris: Hyperparameter tuning for random forest classifier on its own

# Set of optimisation options
parameters_grid = {
    "random_forest_classifier__n_estimators": [100, 300, 500, 700],
    "random_forest_classifier__max_features": ["auto", "sqrt", "log2"],
    "random_forest_classifier__max_depth": [2, 4, 6, 10, 15],
    "random_forest_classifier__min_samples_leaf": [1, 2, 4],
    "random_forest_classifier__min_samples_split": [2, 5, 10],
    "random_forest_classifier__criterion": ["gini", "entropy"],
}

# # Apply cross-validation optimisation using K-fold
# rsf = RepeatedStratifiedKFold(random_state=42)

# # Model optimisation using 'GridSearchCV()'
# grid_search_cv = GridSearchCV(
#     estimator=pipeline,
#     param_grid=parameters_grid,
#     n_jobs=-1,
#     verbose=3,
#     # cv=rsf,  # commented to save time
# )
# # print(f"\nGrid search CV:\n{grid_search_cv}\n")
# grid_search_cv


# %%
# for _ in tqdm(range(5)):
#     grid_search_cv.fit(X=features_train, y=target_train)

# target_pred_cv = grid_search_cv.predict(X=features_test)

# # Build a dataframe of true/test values vs prediction values
# # target_test_cv_df = pd.DataFrame(data=target, columns=[TARGET])
# target_pred_cv_df = pd.DataFrame(
#     data=target_pred_cv,
#     columns=["predictions"]
# )

# prediction_cv_list = [
#     target_test.reset_index(),  # need to reset the index before concat
#     target_pred_cv_df
# ]
# predictions_cv = pd.concat(
#     objs=prediction_cv_list,
#     axis=1
# ).set_index(keys="file_name")
# print(f"\nTest vs. Predictions:\n{predictions_cv}\n")

# %%
# Step 13: Get best parameters

# random_forest_best_score = grid_search_cv.best_score_
# print(f"\nInternal CV best score = {random_forest_best_score:.1%}\n")

# random_forest_best_parameters = grid_search_cv.best_params_
# print(f"\nBest Parameters:\n{random_forest_best_parameters}\n")

# Manual save of the best parameters when above code not part of
# hyperparameter tuning step
random_forest_best_parameters_dict = {
    "random_forest_classifier__criterion": "gini",
    "random_forest_classifier__max_depth": 4,
    "random_forest_classifier__max_features": "auto",
    "random_forest_classifier__min_samples_leaf": 1,
    "random_forest_classifier__min_samples_split": 2,
    "random_forest_classifier__n_estimators": 100
}

# # NOTE: below, use a list of lists or a list of tuples ?
# random_forest_best_parameters_list = [
#     [key, value] for key, value in random_forest_best_parameters_dict.items()
# ]
# random_forest_best_parameters_list

# random_forest_best_parameters_list2 = list(
#     random_forest_best_parameters_dict.items()
# )
# random_forest_best_parameters_list2


# cv_results = pd.DataFrame(grid_search_cv.cv_results_)
# cv_results = cv_results.sort_values("mean_test_score", ascending=False)
# cv_results_best_5 = cv_results[
#     [
#         "mean_test_score",
#         "std_test_score",
#         "rank_test_score",
#         "param_random_forest_classifier__n_estimators",
#         "param_random_forest_classifier__max_features",
#         "param_random_forest_classifier__max_depth",
#         "param_random_forest_classifier__min_samples_leaf",
#         "param_random_forest_classifier__min_samples_split",
#         "param_random_forest_classifier__criterion",
#     ]
# ].head(5)
# print(f"Top 5 CV results:\n{cv_results_best_5}\n")

# grid_search_cv_score = grid_search_cv.score(X=features, y=target)
# print(
#     f"\nBest random forest from grid search = {grid_search_cv_score:.1%}\n"
# )


# %%
# Step 14: Set up the OPTIMISED pipeline

print("\nOptimising Pipeline...\n")

# # Set a new RF Classifier with the bes parameters following 'GridSearch' step
# random_forest_classifier_optimised = RandomForestClassifier(
#     **random_forest_best_parameters_dict
# )

# ic(features_valid)
print(features_valid)

# )
random_forest_classifier = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    verbose=1,
    max_depth=4,
    max_leaf_nodes=5,
    criterion="gini",
    max_features="auto",
    min_samples_leaf=1,
    min_samples_split=2
)

# NOTE: Why fit on validation set?
# pipeline.fit(X=features_valid, y=target_valid)
pipeline.fit(X=features_train, y=target_train)

# Results for validation set
print("\nCross-Validation Prediction Scores for Validation Set")
cf.calculate_cross_validation_prediction_scores(
    model=pipeline,
    features=features_valid,
    target=target_valid,
    cv=10
)

# ----------------------------------------------------------------------------

# Make predictions for the labels of the validation set
target_valid_pred = pipeline.predict(X=features_valid)

# Build a dataframe of true/validation values vs. prediction values
target_valid_pred_df = pd.DataFrame(
    data=target_valid_pred, columns=["predictions"]
)

prediction_valid_list = [
    target_valid.reset_index(),  # need to reset index before concatenating
    target_valid_pred_df
]
predictions_valid = pd.concat(
    objs=prediction_valid_list,
    axis=1
)
# ).set_index(keys="file_name")
print(f"\nValidation vs. Predictions:\n{predictions_valid}\n")

# Results for validation set
print("\nCross-Validation Prediction Scores for Validation Set")
cf.calculate_cross_validation_prediction_scores(
    model=pipeline,
    features=features_valid,
    target=target_valid,
    cv=10
)

# ----------------------------------------------------------------------------

# Make predictions for the labels of the test set
target_test_pred = pipeline.predict(X=features_test)

# Build a dataframe of true/test values vs. prediction values
target_test_pred_df = pd.DataFrame(
    data=target_test_pred, columns=["predictions"]
)

prediction_test_list = [
    target_test.reset_index(),  # need to reset index before concatenating
    target_test_pred_df
]
predictions_test = pd.concat(
    objs=prediction_test_list,
    axis=1
)
# ).set_index(keys="file_name")
print(f"Test vs. Predictions:\n{predictions_test}\n")

# Results for validation set
print("\nCross-Validation Prediction Scores for Validation Set")
cf.calculate_cross_validation_prediction_scores(
    model=pipeline,
    features=features_test,
    target=target_test,
    cv=10
)


# %%
# # Save the 'features' train/test variables as a reduced dataframe since it is
# # reset automatically once outside the pipeline
# features_train_reduced = features_train.iloc[:, 1:4]
# features_test_reduced = features_test.iloc[:, 1:4]


# %%
# # Step XXX: Explain the model's predictions using SHAP values

# explainer = shap.TreeExplainer(
#     model=random_forest_classifier_optimised,
#     feature_names=feature_names,
# )
# shap_values = explainer.shap_values(X=features_test_reduced)
# # print(f"\n{shap_values = }\n")

# # Display the SHAP summary plot
# shap_summary_plot = shap.summary_plot(
#     shap_values=shap_values,
#     features=feature_names,
# )
# # print(shap_summary_plot)
# shap_summary_plot

# # Visualize the SHAP values for the first sample
# shap_force_plot = shap.force_plot(
#     base_value=explainer.expected_value[0],
#     shap_values=shap_values[0],
#     features=features_test_reduced,
#     feature_names=feature_names,
# )
# # print(shap_force_plot)
# shap_force_plot


# %%
# Step 15: Fit the model and make predictions using VALIDATION set

# First, make a copy of the optimised pipeline and fit it (NOT working !!!)
production_pipeline = copy.deepcopy(pipeline)
# print(f"This is the production pipeline:\n{production_pipeline}\n")
production_pipeline


# %%
# Then, fit the pipeline on the training set
production_pipeline.fit(X=features_train, y=target_train)
# Make predictions for the labels of the test set
target_valid_pred = production_pipeline.predict(X=features_valid)

# Build a dataframe of true/validation values vs. prediction values
target_valid_pred_df = pd.DataFrame(
    data=target_valid_pred, columns=["predictions"]
)

prediction_valid_list = [
    target_valid.reset_index(),  # need to reset index before concatenating
    target_valid_pred_df
]
predictions_valid = pd.concat(
    objs=prediction_valid_list,
    axis=1
)
# ) .set_index(keys="file_name")
print(f"\nValidation vs. Predictions:\n{predictions_valid}\n")

# -----------------------------------------------------------------------------

# Get modelling scores from cross-validation
cf.calculate_cross_validation_scores(
    model=production_pipeline,
    features_test=features_valid,
    target_test=target_valid,
    target_pred=target_valid_pred,
    target_label_list=target_label_list,
    cv=10
)

print(f"\n{target_label_list = }\n")

cf.draw_confusion_matrix_heatmap(
    target_test=le.inverse_transform(target_valid),
    target_pred=le.inverse_transform(target_valid_pred),
    target_label_list=target_label_list,
    figure_name="rf_confusion_matrix_predictions_validation",
    # output_directory=OUTPUT_DIR_FIG,
    output_directory=Path(f"../{OUTPUT_DIR_FIG}")  # for notebooks
)

# cf.calculate_multiple_cross_validation_scores(
#     model=production_pipeline,
#     features=features_train,
#     target=target_train,
# )


# Use cross-validation prediction scores for train & test sets

# Results for train set
print("\nCross-Validation Prediction Scores for Train Set")
cf.calculate_cross_validation_prediction_scores(
    model=production_pipeline,
    features=features_train,
    target=target_train,
    cv=10
)

# Results for validation set
print(
    "\nCross-Validation Prediction Scores for Validation Set"
    "after Optimisation"
)
cf.calculate_cross_validation_prediction_scores(
    model=production_pipeline,
    features=features_valid,
    target=target_valid,
    cv=10
)

# %%
# Step 15-bis: Fit the model and make predictions using TEST set
# First, make a copy of the optimised pipeline and fit it (NOT working !!!)
production_pipeline = copy.deepcopy(pipeline)
# print(f"This is the production pipeline:\n{production_pipeline}\n")
production_pipeline


# %%
# Then, fit the pipeline on the training set
production_pipeline.fit(X=features_train, y=target_train)
# Make predictions for the labels of the test set
target_test_pred = production_pipeline.predict(X=features_test)

# Build a dataframe of true/validation values vs. prediction values
target_test_pred_df = pd.DataFrame(
    data=target_test_pred, columns=["predictions"]
)

prediction_test_list = [
    target_test.reset_index(),  # need to reset index before concatenating
    target_test_pred_df
]
predictions_test = pd.concat(
    objs=prediction_test_list,
    axis=1
)
# ) .set_index(keys="file_name")
print(f"\nValidation vs. Predictions:\n{predictions_test}\n")

# -----------------------------------------------------------------------------

# Get modelling scores from cross-validation
cf.calculate_cross_validation_scores(
    model=production_pipeline,
    features_test=features_test,
    target_test=target_test,
    target_pred=target_test_pred,
    target_label_list=target_label_list,
    cv=10
)

print(f"\n{target_label_list = }\n")

cf.draw_confusion_matrix_heatmap(
    target_test=le.inverse_transform(target_test),
    target_pred=le.inverse_transform(target_test_pred),
    target_label_list=target_label_list,
    figure_name="rf_confusion_matrix_predictions_test",
    # output_directory=OUTPUT_DIR_FIG,
    output_directory=Path(f"../{OUTPUT_DIR_FIG}")  # for notebooks
)

# Results for test set
print(
    "\nCross-Validation Prediction Scores for Test Set "
    "after Optimisation"
)
cf.calculate_cross_validation_prediction_scores(
    model=production_pipeline,
    features=features_test,
    target=target_test,
    cv=10
)


# %%
# Step 16: Retrain the model on the full dataset

production_pipeline.fit(X=features, y=target)


# %%
# Step 17: Save the pipeline

cf.save_pipeline_model(
    pipeline_name=production_pipeline,
    file_name="production_pipeline",
    # output_directory=OUTPUT_DIR,
    output_directory=Path(f"../{OUTPUT_DIR}"),  # for notebooks
)


# -----------------------------------------------------------------------------


# %%
# Summary table for performance of models
# Parameters are hypertuned, unless stated otherwise
# NOTE: Used RandomizedSearchCV to tune,
# but sometimes it gives a different set of params
performance = pd.DataFrame([
    [0.9627, 0.9720, 0.957, 1, 0.931, 1],
    [0.9751, 0.9539, 0.957, 0.917, 1.000, 1.000],
    [0.9592, 0.9720, 0.957, 1.000, 0.931, 1.000],
    [0.9427, 0.9443, 0.957, 1.000, 0.931, 1.000],
    [0.9108, 0.8736, 0.957, 0.917, 0.759, 1.000],
    [0.9563, 0.9330, 0.935, 0.958, 0.966, 1.000],
    [0.9597, 0.9720, 0.957, 1.000, 0.931, 1.000],
    [0.9486, 0.9353, 0.957, 1.000, 0.931, 1.00],
    [0.9527, 0.9507, 0.957, 1.000, 0.931, 1.00]
], columns=["Cross-Validation Accuracy",
            "Recall (mean)",
            # All below are recall
            "CG",
            "CGGros",
            "Melt",
            "MeltGros"
            ], index=["Random Forest Classifier (tuned)",
                      "XGBoost (not tuned)",
                      "XGBoost (tuned)",
                      "Gaussian Naive Bayes (not tuned)",
                      "Linear Discriminant Analysis (tuned)",
                      "Decision Tree Classifier (not tuned)",
                      "Decision Tree Classifier (tuned)",
                      "SVC (not tuned)",
                      "SVC (tuned)"]
).sort_values(
    by=["Recall (mean)",
        "Cross-Validation Accuracy"], ascending=False)

pd.set_option("display.precision", 4)
print(display(performance))
pd.reset_option("display.precision")


# %%
# END OF SCRIPT TIMING

t_1 = timeit.default_timer()
elapsed_time = round((t_1 - t_0), 3)
print(f"\nElapsed time: {elapsed_time:.3f} s")
print(f"Thus, {(elapsed_time / 60):.3f} min\n")

# %%
