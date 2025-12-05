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
from statsmodels.stats.outliers_influence import variance_inflation_factor
import plotly.graph_objects as go
from sklearn.linear_model import Lasso, Ridge

# Dataset
df = pd.read_csv("cleaned_data/FINAL_DATA.csv")

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
df_lin = pd.read_csv('cleaned_data/linear.csv')
country_counts = df_lin['country'].value_counts()
threshold = 8
valid_countries = country_counts[country_counts >= threshold].index
df_lin = df_lin[df_lin['country'].isin(valid_countries)]


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
        ui.p("Marissa Burton, Hayeon Chung, Maggie Crowner, Asmita Kadam, Ashrita Kodali")
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

        )
    ),

    # ---- Linear Regression -----
    ui.nav_panel(
        "Linear Reg",
        ui.h3("Linear Regression"),
        ui.p("placeholder")
    ),

    # ---- Logistic Regression -----
    ui.nav_panel(
        "Logistic Reg",
        ui.h3("How well can we classify coffee beans as arabica or robusta based on their characteristics?"),
        ui.p("placeholder")
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
        ui.h3("MLP"),
        ui.p("placeholder")
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



# ------------------------------------------------------------
# App
# ------------------------------------------------------------
app = App(app_ui, server)
