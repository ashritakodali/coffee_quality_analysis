from shiny import App, render, ui, reactive

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
from matplotlib.patches import Patch
from scipy import stats
from matplotlib.colors import LinearSegmentedColormap
import plotly.express as px
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression
from statsmodels.graphics.regressionplots import plot_partregress_grid
from statsmodels.graphics.regressionplots import plot_partregress
from statsmodels.stats.outliers_influence import variance_inflation_factor
import plotly.graph_objects as go
from sklearn.linear_model import Lasso, Ridge
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)

# Dataset
df = pd.read_csv('../data/cleaned_data/FINAL_DATA.csv')

# Further cleaning for K-Means
df_kmeans = df.copy()
df_kmeans['Taste'] = df_kmeans[['Flavor', 'Aftertaste', 'Acidity', 'Body', 'Balance']].mean(axis=1)
df_kmeans['Quality Control'] = df_kmeans[['Uniformity', 'Clean.Cup']].mean(axis=1)
df_kmeans['Age'] = 2025 - df_kmeans['Harvest.Year']
df_kmeans['Total Defects'] = df_kmeans['Category.One.Defects'] + df_kmeans['Category.Two.Defects']
df_kmeans = df_kmeans[['Species', 'Taste', 'Aroma', 'Quality Control', 'Sweetness', 'Moisture',
          'Total Defects', 'Age', 'Altitude']]
num_cols = df_kmeans.select_dtypes(include='number').columns
scaler = StandardScaler()
df_kmeans[num_cols] = scaler.fit_transform(df_kmeans[num_cols])

# Further cleaning for KNN
df_knn = df.copy()
df_knn['Altitude'] = pd.to_numeric(df_knn['Altitude'], errors='coerce')
df_knn = df_knn.dropna(subset=['Altitude'])

# Further cleaning for Linear
df_lin = pd.read_csv('../data/cleaned_data/linear.csv')
country_counts = df_lin['country'].value_counts()
threshold = 8
valid_countries = country_counts[country_counts >= threshold].index
df_lin = df_lin[df_lin['country'].isin(valid_countries)]

# cleaning for MLP
df_mlp = df.copy()
df_mlp['total_score'] = df_mlp['Aroma'] + df_mlp['Flavor'] + df_mlp['Aftertaste'] + df_mlp['Acidity'] + df_mlp['Body'] + df_mlp['Balance'] + df_mlp['Uniformity'] + df_mlp['Clean.Cup'] + df_mlp['Sweetness']
df_mlp['market_grade'] = pd.cut(df_mlp['total_score'], bins = [0, 75, 78, 90],
                            labels = ['Normal', 'Premium', 'Specialty'])

# further cleaning for Logistic
df_log = df.copy()

# ------------------------------------------------------------
# UI
# ------------------------------------------------------------
app_ui = ui.page_fluid(
        ui.tags.style("""
            th { text-align: left !important; }
        """),

    ui.page_navbar(
    # ---- Project Overview Tab ----
    ui.nav_panel(
        "Overview",
        ui.h2("Coffee Bean Quality Machine Learning Analysis"),
        ui.p("DS 6021 Final Project"),
        ui.p("Marissa Burton, Hayeon Chung, Maggie Crowner, Asmita Kadam, Ashrita Kodali"),
        ui.p("This project explores coffee bean characteristics and quality measures to evaluate the following research questions:"),
        ui.tags.ul(
            ui.tags.li("What distinct profiles of arabica/robusta coffee beans can we identify using K-Means Clustering?"),
            ui.tags.li("Can we predict total coffee quality scores based on certain characteristics using Linear Regression?"),
            ui.tags.li("How well can we classify coffee beans as arabica or robusta based on their characteristics?"),
            ui.tags.li("How well can we predict the altitude of the coffee bean farms using K-Nearest Neighbors Regression?"),
            ui.tags.li("Are we able to effectively use Multilayer Perceptrons to predict the market grade of coffee beans based on farming and physical attributes?")
        ),
        ui.p("This data was collected from the Coffee Quality Institute, and current data was scraped using code from https://github.com/jldbc/coffee-quality-database/tree/master. The following Shiny app tabs allow users to explore the dataset itself, some interesting findings we discovered, and models that can provide answers to our research questions.")

    ),

    # ---- Dataset Tab ----
    ui.nav_panel(
        "Dataset",
        ui.h3("Data Preview"),
        ui.output_table("df_table")
    ),

    # ---- EDA Tab ----
    ui.nav_panel(
        "EDA",
        ui.h3("Exploratory Data Analysis"),
        ui.h5("Correlation Heatmap"),
        ui.output_plot("corr_heatmap"),
        ui.hr(),
        ui.h5("Variance Inflation Factor (VIF) Plot"),
        ui.layout_sidebar(
            ui.sidebar(
                ui.input_checkbox_group(
                    "vif_vars",
                    "Variables to include:",
                    choices=["aroma", "flavor", "aftertaste", "acidity", "body", "balance"],
                    selected=["aroma", "flavor", "aftertaste", "acidity", "body", "balance"]
                ),
                ui.input_checkbox_group(
                    "vif_lines",
                    "Show threshold lines:",
                    choices={"5": "VIF = 5", "10": "VIF = 10"},
                    selected=["5", "10"]
                ),
            ),
            ui.output_plot("vif_plot")
        ),
        ui.hr(),
        ui.h5("Distribution of Total Quality"),
        ui.output_plot("quality_hist"),
        ui.hr(),
        ui.h5("Total Quality Distribution by Country"),
        ui.layout_sidebar(
            ui.sidebar(
                ui.input_radio_buttons(
                    "country_color_mode",
                    "Color countries by the mode of:",
                    choices=["Processing Method", "Harvest Year"],
                    selected="Processing Method"
                )
            ),
            ui.output_plot("country_quality_plot", height="400px")
        ),
        ui.hr(),
        ui.h5("Sweetness Distribution by Species"),
        ui.output_plot("sweetness_hist", height="400px")
    ),


    # ---- K-Means Clustering Tab ----
    ui.nav_panel(
        "K-Means",
        ui.h3("What distinct profiles of arabica/robusta coffee beans can we identify using K-Means Clustering?"),
        ui.layout_sidebar(
            ui.sidebar(
                ui.input_radio_buttons(
                    "species_filter",
                    "Species:",
                    choices=["Arabica", "Robusta"],
                    selected="Arabica"
                ),
                ui.input_slider(
                    "k_clusters",
                    "Number of Clusters (k):",
                    min=2,
                    max=10,
                    value=3
                )
            ),
            ui.h5("Clustering Plot"),
            ui.output_plot("cluster_plot"),
            ui.h5("Cluster Centroids"),
            ui.output_table("centroid_table"),
            ui.h5("Variable Importance in Clustering"),
            ui.output_table("importance_table")

        ),
        ui.p("Taste: aggregate of Flavor, Aftertaste, Acidity, Body, Balance"),
        ui.p("Quality Control: aggregate of Uniformity, Clean.Cup"),
        ui.p("Age: 2025 - Harvest.Year"),
        ui.p("Total Defects: Category.One.Defects + Category.Two.Defects")
    ),

    # ---- Linear Regression -----
    ui.nav_panel(
        "Linear Reg",
        ui.h3("Can we predict total coffee quality scores based on certain characteristics using Linear Regression?"),
        ui.layout_sidebar(
            ui.sidebar(
                ui.input_checkbox_group(
                    id="predictors",
                    label="Select predictors for model:",
                    choices=["country", "species", "sweetness"],
                    selected=["country", "species", "sweetness"]
                )
            ),
            ui.h5("Model Formula"),
            ui.output_text_verbatim("lin_model_formula"),
            ui.div(
               ui.h5("Model Summary"),
                ui.output_table("lin_model_summary"), 
            ),
            ui.h5("Model Diagnostic Plots"),
            ui.output_plot("lin_actual_pred_plot", height="400px"),
            ui.output_plot("lin_resid_pred_plot", height="400px"),
            ui.output_plot("lin_hist_residuals", height="400px"),
            ui.output_plot("lin_qq_residuals", height="400px")
        )
    ),

    # ---- Logistic Regression -----
    ui.nav_panel(
        "Logistic Reg",
        ui.h3("How well can we classify coffee beans as arabica or robusta based on their characteristics?"),
        ui.layout_sidebar(
            ui.sidebar(
                ui.input_selectize(
                    id="log_pred",
                    label="Select predictors:",
                    choices=[
                        'Country.of.Origin', 'Number.of.Bags', 'Bag.Weight', 'Harvest.Year', 
                        'Grading.Date', 'Processing.Method', 'Aroma', 'Flavor', 'Aftertaste', 
                        'Acidity', 'Body', 'Balance', 'Uniformity', 'Clean.Cup', 'Sweetness', 
                        'Moisture', 'Category.One.Defects', 'Quakers', 'Color', 
                        'Category.Two.Defects', 'Expiration', 'Altitude'
                    ],
                    selected=[
                        'Aroma', 'Flavor', 'Aftertaste', 'Acidity', 'Body', 'Balance', 
                        'Uniformity', 'Clean.Cup', 'Moisture', 'Sweetness'
                    ],
                    multiple=True,
                ),

                ui.input_select(
                    id="log_penalty",
                    label="Penalty Type:",
                    choices=["l2", "l1", None],
                    selected='l2'
                )
            ),

            ui.div(
                ui.h5("Model Formula"),
                ui.output_text_verbatim("log_formula"),
                ui.h5("Model Metrics"),
                ui.output_table("log_summary"),
                ui.h5("Confusion Matrix"),
                ui.output_plot("log_confusion"),
                ui.h5("ROC Curve"),
                ui.output_plot("log_roc"),
                ui.h5("Precision-Recall Curve"),
                ui.output_plot("log_pr"),
                ui.h5("Top Coefficients"),
                ui.output_plot("log_coef_plot")
            )
        )
    ),

    # ---- KNN -----
    ui.nav_panel(
        "KNN",
        ui.h3("How well can we predict the altitude of the coffee bean farms using K-Nearest Neighbors Regression?"),
        ui.layout_sidebar(
            ui.sidebar(
                ui.input_slider("knn_neighbors", 
                    "Number of Neighbors (k):", 
                    min=1, max=20, value=4
                ),
                ui.input_select("knn_weights", 
                    "Weights:", 
                    choices=['uniform', 'distance'],
                    selected='distance'
                ),
                ui.input_select("knn_model", 
                    "Model Type:", 
                    choices=['KNN', 'KNN with PCA']
                ),
                ui.output_ui("pca_slider_ui")
            ),
            ui.div(
                ui.h5("Model Evaluation"),
                ui.output_table("eval_table")
            ),
            ui.h5("KNN Plot"),
            ui.output_plot("knn_2d_plot", height="400px"),

            ui.h5("Model Diagnostic Plots"),

            ui.output_plot("plot_actual_vs_predicted", height="400px"),
            ui.output_plot("plot_residuals_vs_predicted", height="400px"),
            ui.output_plot("plot_hist_residuals", height="400px"),
            ui.output_plot("plot_qq_residuals", height="400px")
        )
    ),

    # ---- MLP -----
    ui.nav_panel(
        "MLP",
        ui.h3("Are we able to effectively use Multilayer Perceptrons to predict the market grade of coffee beans based on farming and physical attributes?"),
        ui.layout_sidebar(
            ui.sidebar(
                ui.input_radio_buttons(
                    "mlp_model",
                    "Model Type:",
                     choices=["Simple MLP", "Complex MLP"],
                    selected="Simple MLP"
                ),
                ui.input_slider("learning_rates",
                    "Learning Rate:",
                    min=0.0001,
                    max=0.01,
                    step=0.0001,
                    value=0.0005
                ),
                ui.input_select(
                    "batch_sizes",
                    "Batch Size:",
                    choices=[16, 32, 64, 128, 256],
                    selected=32
                ),
                ui.input_action_button("train_mlp", "Train MLP")
            ),

            ui.div(
                ui.h5("Model Evaluation"),
                ui.output_table("mlp_eval_table")
                ),

            ui.h5('MLP Plots'),
            ui.output_plot('loss_plot', height='400px'),
            ui.output_plot('mlp_confusion_matrix', height='400px'),
        )
    ),



    title="Coffee Quality Analysis"
))

# ------------------------------------------------------------
# Server
# ------------------------------------------------------------
def server(input, output, session):

    # ----- EDA -----
    # Correlation heatmap 
    @output
    @render.plot
    def corr_heatmap():
        corr = df_lin.loc[:, "aroma":"balance"].corr()

        custom_cmap = LinearSegmentedColormap.from_list(
            "coffee_red", ["#6F4E37", "#C04040"], N=256
        )

        plt.figure(figsize=(8, 5))
        sns.heatmap(corr, annot=True, cmap=custom_cmap)
        plt.title("Correlation Matrix of Scale-Scored Variables")
        plt.tight_layout()
        return plt.gcf()
    
    # VIF
    @output
    @render.plot
    def vif_plot():
        vars_selected = input.vif_vars()
        lines = input.vif_lines()
        X = df_lin[list(vars_selected)].dropna()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        vif_vals = [variance_inflation_factor(X_scaled, i) for i in range(len(vars_selected))]
        vif_df = pd.DataFrame({"variable": vars_selected, "VIF": vif_vals})
        vif_df = vif_df.sort_values("VIF", ascending=False)
        plt.figure(figsize=(8, 6))
        plt.barh(vif_df["variable"], vif_df["VIF"], color="#6F4E37")
        if "5" in lines:
            plt.axvline(5, color="#D4A017", linestyle="--")
        if "10" in lines:
            plt.axvline(10, color="#C04040", linestyle="--")
        plt.xlabel("VIF")
        plt.ylabel("Variable")
        plt.title("Variance Inflation Factors")
        plt.tight_layout()
        return plt.gcf()
    
    # Histogram
    @output
    @render.plot
    def quality_hist():
        plt.figure(figsize=(7, 5))
        sns.histplot(
            df_lin["total_quality"].dropna(),
            bins=10,
            kde=True,
            color="#6F4E37"
        )
        plt.title("Distribution of Overall Quality")
        plt.xlabel("Overall Quality")
        plt.ylabel("Frequency")
        plt.tight_layout()
        return plt.gcf()
    
    # Box plots
    @output
    @render.plot
    def country_quality_plot():
        df = df_lin.copy()
        choice = input.country_color_mode()
        palette = {
            'Washed/Wet': '#6F4E37',
            'Natural/Dry': '#C04040',
            'Honey': '#D4A017',
            'Fermentation': "#D46217",
            'Cherry': "#460606",
            '2020': "#460606",
            '2021': "#D46217",
            '2023': "#D4A017",
            '2024': "#C04040",
            '2025': "#6F4E37"
        }
        if choice == "Processing Method":
            mode_var = df.groupby("country")["processing_method"].agg(lambda x: x.mode().iloc[0])
            df["mode_var"] = df["country"].map(mode_var)
            legend_title = "Processing Method"
            plot_title = "Total Quality by Country (Most Used Processing Method)"
        else:
            df = df.dropna(subset=["harvest_year"])
            df = df[df["harvest_year"] != 2026]
            df["harvest_year"] = df["harvest_year"].astype(str)
            mode_var = df.groupby("country")["harvest_year"].agg(lambda x: x.mode().iloc[0])
            df["mode_var"] = df["country"].map(mode_var)
            legend_title = "Harvest Year"
            plot_title = "Total Quality by Country (Most Popular Harvest Year)"
        plt.figure(figsize=(9, 6))
        sns.boxplot(
            y="country",
            x="total_quality",
            hue="mode_var",
            data=df,
            palette=palette,
            dodge=False
        )
        plt.title(plot_title)
        plt.xlabel("Total Quality")
        plt.ylabel("Country")
        legend_elements = [
            Patch(facecolor=palette[val], label=val)
            for val in sorted(df["mode_var"].unique())
            if val in palette
        ]
        plt.legend(
            handles=legend_elements,
            title=legend_title,
            loc='center left',
            bbox_to_anchor=(1, 0.5)
        )
        plt.tight_layout()
        fig = plt.gcf()
        plt.close()
        return fig
    
    # Sweetness EDA
    @output
    @render.plot
    def sweetness_hist():
        # y_bin: 1 = Arabica, 0 = Robusta
        y_str = df_log["Species"].astype(str).str.strip().str.lower()
        y_bin = y_str.map({"arabica": 1, "robusta": 0})
        plt.figure(figsize=(7,5))
        plt.hist(df_log.loc[y_bin == 1, "Sweetness"].dropna(), bins=30, label="Arabica", color="#6F4E37")
        plt.hist(df_log.loc[y_bin == 0, "Sweetness"].dropna(), bins=30, label="Robusta", color="#C04040")
        plt.title("Sweetness by Species")
        plt.xlabel("Sweetness")
        plt.ylabel("Frequency")
        plt.legend()
        plt.tight_layout()
        fig = plt.gcf()
        plt.close(fig)
        return fig


    
    # K-means clustering reactive data
    @reactive.Calc
    def filtered_df():
        species = input.species_filter()
        df_sub = df_kmeans[df_kmeans["Species"] == species].copy()
        df_sub = df_sub.drop(columns=["Species"])
        return df_sub

    @reactive.Calc
    def kmeans_result():
        k = input.k_clusters()
        scaled = filtered_df()

        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(scaled)

        pca = PCA(n_components=2)
        components = pca.fit_transform(scaled)

        df_clusters = scaled.copy()
        df_clusters["cluster"] = labels
        df_clusters["PC1"] = components[:, 0]
        df_clusters["PC2"] = components[:, 1]

        centroids_original = scaler.inverse_transform(kmeans.cluster_centers_)
        centroids_df = pd.DataFrame(
            centroids_original, columns=scaled.columns
        )
        centroids_df["Cluster"] = range(k)
        cols = ["Cluster"] + [c for c in centroids_df.columns if c != "Cluster"]
        centroids_df = centroids_df[cols]

        centroids_scaled = kmeans.cluster_centers_
        centroids_df_scaled = pd.DataFrame(centroids_scaled, columns=scaled.columns)

        feature_importance = (centroids_df_scaled.max() - centroids_df_scaled.min()).sort_values(ascending=False)
        feature_importance = feature_importance.reset_index()
        feature_importance.columns = ["Variable", "Importance"]

        return df_clusters, centroids_df, feature_importance


    # ----- Display the full dataframe -----
    @output
    @render.table
    def df_table():
        return df   

    # ----- PCA Cluster Plot -----
    @output
    @render.plot
    def cluster_plot():
        df_clusters, _, _ = kmeans_result()

        plt.figure(figsize=(8,4))
        k = input.k_clusters()

        for c in range(k):
            plt.scatter(
                df_clusters[df_clusters["cluster"] == c]["PC1"],
                df_clusters[df_clusters["cluster"] == c]["PC2"],
                label=f"Cluster {c}"
            )

        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title(f"K-Means Clustering ({input.species_filter()}), k={input.k_clusters()}")
        plt.legend()
        plt.tight_layout()
        return plt.gcf()


    # ----- Centroid Table -----
    @output
    @render.table
    def centroid_table():
        _, centroids, _ = kmeans_result()
        return centroids
    
    # ----- Cluster Var Importance Table -----
    @output
    @render.table
    def importance_table():
        _, _, importance = kmeans_result()
        return importance
    
    # ----- Linear Reg reactive data -----
    @reactive.calc
    def run_model():
        preds = input.predictors()
        if len(preds) == 0:
            return None
        formula = "total_quality ~ " + " + ".join(preds)
        model = smf.ols(formula=formula, data=df_lin).fit()
        return model

    # Formula
    @output
    @render.text
    def lin_model_formula():
        result = run_model()
        preds = input.predictors()
        if result is None or len(preds) == 0:
            return "Please select predictors."
        coefs = result.params  # OLS coefficients
        terms = []
        for name, coef in coefs.items():
            coef_str = f"{coef:.3f}"
            if name == "Intercept":
                terms.insert(0, f"{coef_str}")
            else:
                sign = "+" if coef >= 0 else "-"
                terms.append(f" {sign} {abs(coef):.3f} * {name}")
        formula_str = "y = " + " \\\n     ".join(terms)
        return formula_str

    
    # Linear model summary
    @output
    @render.table
    def lin_model_summary():
        result = run_model()
        preds = input.predictors()
        if result is None or len(preds) == 0:
            return pd.DataFrame({"Metric": ["Info"], "Value": ["Please select predictors."]})
        summary_df = pd.DataFrame({
            "Metric": ["Model Type", "Number of Predictors", "R²", "Adjusted R²", "RMSE"],
            "Value": ["OLS", len(preds), round(result.rsquared, 3), round(result.rsquared_adj, 3),
                    round(np.sqrt(mean_squared_error(df_lin["total_quality"], result.fittedvalues)), 3)]
        })
        return summary_df



    # Actual vs Predicted
    @output
    @render.plot
    def lin_actual_pred_plot():
        result = run_model()
        if result is None:
            return plt.figure()
        actual = df_lin["total_quality"]
        predicted = result.fittedvalues
        plt.figure(figsize=(7,5))
        plt.scatter(actual, predicted, alpha=0.6, color="#6F4E37")
        min_val = min(actual.min(), predicted.min())
        max_val = max(actual.max(), predicted.max())
        plt.plot([min_val, max_val], [min_val, max_val], linestyle="--", color='red')
        plt.xlabel("Actual Total Quality")
        plt.ylabel("Predicted Total Quality")
        plt.title("Actual vs Predicted")
        plt.tight_layout()
        fig = plt.gcf()
        plt.close(fig)
        return fig

    # Residuals vs Predicted
    @output
    @render.plot
    def lin_resid_pred_plot():
        result = run_model()
        if result is None:
            return plt.figure()
        residuals = result.resid
        predicted = result.fittedvalues
        plt.figure(figsize=(7,5))
        plt.scatter(predicted, residuals, alpha=0.6, color="#6F4E37")
        plt.axhline(0, linestyle="--", color='red')
        plt.xlabel("Predicted Total Quality")
        plt.ylabel("Residuals (Actual - Predicted)")
        plt.title("Residuals vs Predicted")
        plt.tight_layout()
        fig = plt.gcf()
        plt.close(fig)
        return fig

    # Histogram of Residuals
    @output
    @render.plot
    def lin_hist_residuals():
        result = run_model()
        if result is None:
            return plt.figure()
        residuals = result.resid
        plt.figure(figsize=(7,5))
        plt.hist(residuals, bins=30, edgecolor="black", color="#6F4E37")
        plt.xlabel("Residual")
        plt.ylabel("Frequency")
        plt.title("Histogram of Residuals")
        plt.tight_layout()
        fig = plt.gcf()
        plt.close(fig)
        return fig

    # Q-Q Plot of Residuals
    @output
    @render.plot
    def lin_qq_residuals():
        result = run_model()
        if result is None:
            return plt.figure()
        residuals = result.resid
        plt.figure(figsize=(7,5))
        (osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist="norm")
        plt.scatter(osm, osr, color="#6F4E37", alpha=0.6)
        x = np.linspace(min(osm), max(osm), 100)
        plt.plot(x, slope*x + intercept, linestyle="--", color="red")
        plt.title("Q-Q Plot of Residuals")
        plt.tight_layout()
        fig = plt.gcf()
        plt.close(fig)
        return fig
    
    # ---- Logistic Regression ----
    @reactive.Calc
    def log_model_fit():
        preds = list(input.log_pred())
        penalty = input.log_penalty()
        if len(preds) == 0:
            return None
        y_str = df_log["Species"].astype(str).str.strip().str.lower()
        y_bin = y_str.map({"arabica": 1, "robusta": 0})
        X = df_log[preds].copy()
        y = y_bin.copy()  # 1 = Arabica, 0 = Robusta
        valid_rows = y.notna()
        X = X.loc[valid_rows].reset_index(drop=True)
        y = y.loc[valid_rows].astype(int).reset_index(drop=True)

        numeric_cols_model = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols_model = X.select_dtypes(include=["object"]).columns.tolist()

        numeric_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        categorical_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])
        preprocess = ColumnTransformer([
            ("num", numeric_pipe, numeric_cols_model),
            ("cat", categorical_pipe, categorical_cols_model),
        ])

        # Update Logistic Regression with selected penalty
        if penalty == "none":
            solver = "lbfgs"  # lbfgs ignores penalty if penalty='none'
            log_reg = LogisticRegression(
                penalty="none",
                class_weight="balanced",
                max_iter=1000,
                solver=solver
            )
        elif penalty == "l1":
            log_reg = LogisticRegression(
                penalty="l1",
                class_weight="balanced",
                max_iter=1000,
                solver="saga"
            )
        else:  # l2
            log_reg = LogisticRegression(
                penalty="l2",
                class_weight="balanced",
                max_iter=1000,
                solver="lbfgs"
            )

        clf = Pipeline([
            ("prep", preprocess),
            ("model", log_reg)
        ])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )

        clf.fit(X_train, y_train)
        
        return {
            "clf": clf,
            "X_train": X_train, "X_test": X_test,
            "y_train": y_train, "y_test": y_test
        }


    # ---------------- Formula ----------------
    @output
    @render.text
    def log_formula():
        fit = log_model_fit()
        preds = input.log_pred()
        if fit is None or len(preds) == 0:
            return "Please select predictors."

        clf = fit["clf"]
        coefs = clf.named_steps["model"].coef_[0]
        feature_names = clf.named_steps["prep"].get_feature_names_out()
        terms = []
        for name, coef in zip(feature_names, coefs):
            sign = "+" if coef >= 0 else "-"
            coef_str = f"{abs(coef):.3f}"
            terms.append(f"{sign} {coef_str}*{name}")
        formula = "logit(p) = " + " \\\n     ".join(terms)
        return formula


    # ---------------- Metrics Table ----------------
    @output
    @render.table
    def log_summary():
        fit = log_model_fit()
        if fit is None:
            return pd.DataFrame({"Metric": ["Info"], "Value": ["Please select predictors."]})

        clf = fit["clf"]
        X_test, y_test = fit["X_test"], fit["y_test"]
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:,1]

        report = classification_report(y_test, y_pred, output_dict=True, digits=3)
        auc = roc_auc_score(y_test, y_prob)
        ap = average_precision_score(y_test, y_prob)

        summary_df = pd.DataFrame({
            "Metric": ["Accuracy", "ROC-AUC", "PR-AUC", "Precision (1)", "Recall (1)", "F1 (1)"],
            "Value": [
                round(report["accuracy"], 3),
                round(auc, 3),
                round(ap, 3),
                round(report["1"]["precision"], 3),
                round(report["1"]["recall"], 3),
                round(report["1"]["f1-score"], 3)
            ]
        })
        return summary_df


    # ---------------- Confusion Matrix ----------------
    @output
    @render.plot
    def log_confusion():
        fit = log_model_fit()
        if fit is None:
            return plt.figure()
        clf = fit["clf"]
        X_test, y_test = fit["X_test"], fit["y_test"]
        y_pred = clf.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(5,4))
        plt.imshow(cm, cmap="Blues")
        plt.title("Confusion Matrix")
        plt.xticks([0,1], ["Pred Robusta", "Pred Arabica"])
        plt.yticks([0,1], ["True Robusta", "True Arabica"])
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, cm[i,j], ha="center", va="center", color="red", fontsize=12)
        plt.tight_layout()
        fig = plt.gcf()
        plt.close(fig)
        return fig


    # ---------------- ROC Curve ----------------
    @output
    @render.plot
    def log_roc():
        fit = log_model_fit()
        if fit is None:
            return plt.figure()
        clf = fit["clf"]
        X_test, y_test = fit["X_test"], fit["y_test"]
        y_prob = clf.predict_proba(X_test)[:,1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        
        plt.figure()
        plt.plot(fpr, tpr, color="#6F4E37")
        plt.plot([0,1],[0,1], "--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.tight_layout()
        fig = plt.gcf()
        plt.close(fig)
        return fig


    # ---------------- Precision-Recall Curve ----------------
    @output
    @render.plot
    def log_pr():
        fit = log_model_fit()
        if fit is None:
            return plt.figure()
        clf = fit["clf"]
        X_test, y_test = fit["X_test"], fit["y_test"]
        y_prob = clf.predict_proba(X_test)[:,1]
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        
        plt.figure()
        plt.plot(rec, prec, color="#6F4E37")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.tight_layout()
        fig = plt.gcf()
        plt.close(fig)
        return fig


    # ---------------- Top Coefficients ----------------
    @output
    @render.plot
    def log_coef_plot():
        fit = log_model_fit()
        if fit is None:
            return plt.figure()
        clf = fit["clf"]
        feature_names = clf.named_steps["prep"].get_feature_names_out()
        coefs = clf.named_steps["model"].coef_[0]
        coef_df = pd.DataFrame({"feature": feature_names, "coef": coefs})
        coef_df["abs_coef"] = coef_df["coef"].abs()
        coef_df = coef_df.sort_values("abs_coef", ascending=False).head(15)
        coef_df = coef_df.sort_values("coef")  # for horizontal bar plot

        plt.figure(figsize=(8,6))
        plt.barh(coef_df["feature"], coef_df["coef"], color="#6F4E37")
        plt.xlabel("Coefficient (standardized log-odds)")
        plt.title("Top 15 Features — Logistic Regression")
        plt.tight_layout()
        fig = plt.gcf()
        plt.close(fig)
        return fig
    
    # ----- KNN Evaluation Table -----
    @reactive.Calc
    def knn_eval():
        X = df_knn.drop(columns=['Altitude'])
        y = df_knn['Altitude']
        Xy = pd.concat([X, y], axis=1).dropna()
        X = Xy.drop(columns=['Altitude'])
        y = Xy['Altitude']

        #train/test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()
        numeric_transformer = Pipeline([("scaler", StandardScaler())])
        categorical_transformer = Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore"))])
        preprocessor = ColumnTransformer([
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ])
        model_type = input.knn_model()
        n_neighbors = input.knn_neighbors()
        weights = input.knn_weights()
        if model_type == "KNN":
            pipe = Pipeline([("preprocess", preprocessor), ("knn", KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights))])
        else:
            n_pcs = input.knn_pcs()
            to_dense = FunctionTransformer(lambda x: x.toarray(), accept_sparse=True)
            pipe = Pipeline([
                ("preprocess", preprocessor),
                ("to_dense", to_dense),
                ("pca", PCA(n_components=n_pcs, random_state=42)),
                ("knn", KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights))
            ])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_test_arr = np.array(y_test)
        y_pred_arr = np.array(y_pred)
        residuals = y_test_arr - y_pred_arr
        mse = mean_squared_error(y_test_arr, y_pred_arr)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test_arr, y_pred_arr)
        return {
            "y_true": y_test_arr,
            "y_pred": y_pred_arr,
            "X_test": X_test,
            "residuals": residuals,
            "n_neighbors": n_neighbors,
            "weights": weights,
            "RMSE": round(rmse,3),
            "R^2": round(r2,3)
        }
    
    # Number of PCs slider
    @output
    @render.ui
    def pca_slider_ui():
        if input.knn_model() == "KNN with PCA":
            return ui.input_slider("knn_pcs",
                               "Number of Principal Components:",
                               min=1,
                               max=20,
                               value=15)
        else:
            return None
    
    # KNN 2d Visualization
    @output
    @render.plot
    def knn_2d_plot():
        res = knn_eval() 
        X = df_knn.drop(columns=['Altitude']).dropna()
        X_test = res['X_test']
        numeric_features = X_test.select_dtypes(include=[np.number]).columns.tolist()
        X_num = StandardScaler().fit_transform(X_test[numeric_features])
        pca = PCA(n_components=2, random_state=42)
        X_2d = pca.fit_transform(X_num)
        y_pred = res["y_pred"]
    
        plt.figure(figsize=(8,6))
        plt.scatter(X_2d[:,0], X_2d[:,1], c=y_pred, alpha=0.7)
        plt.colorbar(label="Predicted Altitude")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("KNN Predictions in 2D PCA Space")
        plt.tight_layout()
        fig = plt.gcf()
        plt.close(fig)
        return fig

    # Render table
    @output
    @render.table
    def eval_table():
        res = knn_eval()
        return pd.DataFrame({
            "Metric": ["n_neighbors", "weights", "RMSE", "R²"],
            "Value": [res["n_neighbors"],
                      res["weights"],
                      res["RMSE"],
                      res["R^2"]]
        })
    
    # KNN Plots
    @output
    @render.plot
    def plot_actual_vs_predicted():
        res = knn_eval()
        y_true = res["y_true"]
        y_pred = res["y_pred"]
        plt.figure(figsize=(7,5))
        plt.scatter(y_true, y_pred, alpha=0.6, color="#6F4E37")
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], linestyle="--", color='red')
        plt.xlabel("Actual Altitude")
        plt.ylabel("Predicted Altitude")
        plt.title("Actual vs Predicted Altitude (KNN)")
        plt.tight_layout()
        fig = plt.gcf()
        plt.close(fig)
        return fig

    @output
    @render.plot
    def plot_residuals_vs_predicted():
        res = knn_eval()
        y_pred = res["y_pred"]
        residuals = res["residuals"]
        plt.figure(figsize=(7,5))
        plt.scatter(y_pred, residuals, alpha=0.6, color="#6F4E37")
        plt.axhline(0, linestyle="--", color='red')
        plt.xlabel("Predicted Altitude")
        plt.ylabel("Residuals (Actual - Predicted)")
        plt.title("Residuals vs Predicted Values")
        plt.tight_layout()
        fig = plt.gcf()
        plt.close(fig)
        return fig

    @output
    @render.plot
    def plot_hist_residuals():
        res = knn_eval()
        residuals = res["residuals"]
        plt.figure(figsize=(7,5))
        plt.hist(residuals, bins=30, edgecolor="black", color="#6F4E37")
        plt.xlabel("Residual")
        plt.ylabel("Frequency")
        plt.title("Histogram of Residuals")
        plt.tight_layout()
        fig = plt.gcf()
        plt.close(fig)
        return fig

    @output
    @render.plot
    def plot_qq_residuals():
        res = knn_eval()
        residuals = res["residuals"]
        plt.figure(figsize=(7,5))
        (osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist="norm")
        plt.scatter(osm, osr, color="#6F4E37", alpha=0.6)
        x = np.linspace(min(osm), max(osm), 100)
        plt.plot(x, slope*x + intercept, linestyle="--", color="red")
        plt.title("Q-Q Plot of Residuals")
        plt.tight_layout()
        fig = plt.gcf()
        plt.close(fig)
        return fig

    # MLP model
    @reactive.Calc
    def mlp_eval():
        # Tell Shiny that this reactive depends on input values
        input.train_mlp()
        
        model_type = input.mlp_model()
        lr = input.learning_rates()
        batch = int(input.batch_sizes())

        if df_mlp is None or df_mlp.empty:
            return None
        # Preprocessing
        X = df_mlp[['Country.of.Origin', 'Number.of.Bags', 'Bag.Weight', 'Harvest.Year',
                'Processing.Method', 'Moisture', 'Category.One.Defects', 'Quakers',
                'Color', 'Category.Two.Defects', 'Species', 'Altitude']]
        y = df_mlp['market_grade']

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, shuffle=True, random_state=6021
        )

        # Numeric/categorical handling
        numeric_columns = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_columns = X.select_dtypes(include=['object']).columns

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numeric_columns),
                ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_columns)
            ]
        )
        preprocessor.fit(X_train)
        X_train_processed = preprocessor.transform(X_train)
        X_test_processed = preprocessor.transform(X_test)

        labeler = LabelEncoder()
        y_train_encoded = labeler.fit_transform(y_train)
        y_test_encoded = labeler.transform(y_test)

        # Build model
        num_classes = len(np.unique(y_train))
        tf.random.set_seed(6021)
        np.random.seed(6021)
        random.seed(6021)

        if model_type == "Simple MLP":
            model = keras.Sequential([
                keras.layers.Dense(16, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
                keras.layers.Dropout(0.3),
                keras.layers.Dense(num_classes, activation='softmax')
            ])
        else:  # Complex MLP
            model = keras.Sequential([
                keras.layers.Dense(256, activation='relu'),
                keras.layers.BatchNormalization(),
                keras.layers.Dropout(0.3),
                keras.layers.Dense(128, activation='relu'),
                keras.layers.BatchNormalization(),
                keras.layers.Dropout(0.3),
                keras.layers.Dense(64, activation='relu'),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(num_classes, activation='softmax')
            ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        early_stop = keras.callbacks.EarlyStopping(
            patience=20,
            restore_best_weights=True,
            monitor='val_loss'
        )

        # Class weights
        cw = compute_class_weight("balanced", classes=np.unique(y_train_encoded), y=y_train_encoded)
        class_weight = dict(enumerate(cw))

        history = model.fit(
            X_train_processed, y_train_encoded,
            validation_split=0.2,
            epochs=100,
            batch_size=batch,
            callbacks=[early_stop],
            verbose=0,
            class_weight=class_weight
        )

        y_pred_probs = model.predict(X_test_processed)
        y_pred_classes = np.argmax(y_pred_probs, axis=1)

        return {
            "model": model,
            "history": history,
            "X_test": X_test_processed,
            "y_test": y_test_encoded,
            "y_pred_classes": y_pred_classes,
            "labeler": labeler
        }


    @output
    @render.plot
    def loss_plot():
        eval_res = mlp_eval()
        if eval_res is None:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.text(0.5, 0.5, 'Click "Train MLP" to generate plot', 
                    ha='center', va='center', fontsize=14)
            ax.axis('off')
            return fig
        history = eval_res["history"]

        train_loss = history.history['loss']
        val_loss = history.history['val_loss']

        plt.figure(figsize=(8, 5))
        plt.plot(train_loss, label='Training Loss', color='#4B2E2B')
        plt.plot(val_loss, label='Validation Loss', color='#C8A27A')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training & Validation Loss Curve')
        plt.legend()
        plt.grid(True)
        fig = plt.gcf()
        plt.close(fig)
        return fig

    @output
    @render.plot
    def mlp_confusion_matrix():
        eval_res = mlp_eval()
        if eval_res is None:
            return None

        model = eval_res["model"]
        X_test = eval_res["X_test"]
        y_test = eval_res["y_test"]
        y_pred_classes = eval_res["y_pred_classes"]
        labeler = eval_res["labeler"]

        y_pred_labels = labeler.inverse_transform(y_pred_classes)
        y_true_labels = labeler.inverse_transform(y_test)

        brown_cmap = LinearSegmentedColormap.from_list(
            "brown_gradient",
            ["#f3e9dd", "#c19a6b", "#8b5a2b", "#5c3317", "#2a1c12"]
        )

        cm = confusion_matrix(y_true_labels, y_pred_labels, labels=labeler.classes_)
        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt='d', cmap=brown_cmap,
                    xticklabels=labeler.classes_, yticklabels=labeler.classes_)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        fig = plt.gcf()
        plt.close(fig)
        return fig

    @output
    @render.table
    def mlp_eval_table():
        eval_res = mlp_eval()
        y_test = eval_res["y_test"]
        y_pred_classes = eval_res["y_pred_classes"]
        labeler = eval_res["labeler"]
        
        # Encode labels to original names
        y_pred_labels = labeler.inverse_transform(y_pred_classes)
        y_true_labels = labeler.inverse_transform(y_test)

        # Classification report as dict
        report_dict = classification_report(
            y_true_labels,
            y_pred_labels,
            target_names=labeler.classes_,
            output_dict=True
        )
        report_df = pd.DataFrame(report_dict).T

        # Add overall test accuracy
        test_accuracy = accuracy_score(y_true_labels, y_pred_labels)
        report_df.loc["Test Accuracy"] = {
            "precision": "",
            "recall": "",
            "f1-score": "",
            "support": test_accuracy
        }

        return report_df




# ------------------------------------------------------------
# App
# ------------------------------------------------------------
app = App(app_ui, server)
