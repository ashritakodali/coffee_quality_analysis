from shiny import App, render, ui, reactive
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Dataset
df = pd.read_csv("cleaned_data/FINAL_DATA.csv")

# Further cleaning for Maggie EDA (kmeans)
df_mc = df.copy()
df_mc['Defective'] = np.where(
    (df_mc['Category.One.Defects'] > 0) | (df_mc['Category.Two.Defects'] > 0),
    "Defective",
    "Not defective"
)
df_mc['Country.of.Origin'] = df_mc['Country.of.Origin'].replace('Tanzania, United Republic Of', 'Tanzania')
df_mc['Country.of.Origin'] = df_mc['Country.of.Origin'].replace('Papua New Guinea', 'PN Guinea')

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
        "Project Overview",
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
        ui.h3("EDA"),
        ui.layout_sidebar(
            ui.sidebar(
                ui.input_select(
                    "xvar",
                    "Select X-axis Variable:",
                    choices=["Color", "Processing.Method", "Country.of.Origin"],
                    selected="Color"
                )
            ),
            ui.output_plot("countplot")
        )
    ),

    # ---- K-Means Clustering Tab ----
    ui.nav_panel(
        "K-Means Clustering",
        ui.h3("K-Means Clustering"),
        ui.layout_sidebar(
            ui.sidebar(
                ui.input_radio_buttons(
                    "species_filter",
                    "Select Species:",
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

    title="DS 6021 Final Project"
))

# ------------------------------------------------------------
# Server
# ------------------------------------------------------------
def server(input, output, session):

    # EDA reactive data - by defective status
    @reactive.Calc
    def selected_var():
        return input.xvar()
    
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

    # ----- EDA countplot -----
    @output
    @render.plot
    def countplot():
        plt.figure(figsize=(8,4))
        sns.countplot(
            x=selected_var(),
            hue="Defective",
            data=df_mc,
            palette=["#6F4E37", "#C04040"]
        )
        plt.title(f"Distribution of {selected_var()} by Defective Status")
        plt.xlabel(selected_var())
        plt.ylabel("Count")
        if selected_var() == "Country.of.Origin":
            plt.xticks(rotation=90, fontsize=8)
        plt.legend(title="Defective")
        plt.tight_layout()
        return plt.gcf()
    
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


# ------------------------------------------------------------
# App
# ------------------------------------------------------------
app = App(app_ui, server)
