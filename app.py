from shiny import App, render, ui, reactive
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Dataset
df = pd.read_csv("cleaned_data/FINAL_DATA.csv")
df['Defective'] = np.where(
    (df['Category.One.Defects'] > 0) | (df['Category.Two.Defects'] > 0),
    "Defective",
    "Not defective"
)

# --------------------------- UI ---------------------------
app_ui = ui.page_navbar(
    ui.nav_panel(
        "Text Tab",
        ui.h2("Welcome!"),
        ui.p("This is the text-only tab.")
    ),
    ui.nav_panel(
        "Visualization",
        ui.layout_sidebar(
            ui.sidebar(
                ui.input_select(
                    "xvar",
                    "Select Parameter:",
                    choices=["Color", "Processing.Method"],
                    selected="Color"
                )
            ),
            ui.output_plot("countplot")
        )
    ),
    title="Two-Tab Shiny App"
)

# ------------------------------------------------------------
# Server
# ------------------------------------------------------------
def server(input, output, session):

    @reactive.Calc
    def selected_var():
        return input.xvar()

    @output
    @render.plot
    def countplot():
        plt.figure(figsize=(8,5))
        sns.countplot(
            x=selected_var(),
            hue="Defective",
            data=df,
            palette=["#6F4E37", "#C04040"]
        )
        plt.title(f"Distribution of {selected_var()} by Defective Status")
        plt.xlabel(selected_var())
        plt.ylabel("Count")
        plt.legend(title="Defective")
        plt.tight_layout()
        return plt.gcf()

# ------------------------------------------------------------
# App object
# ------------------------------------------------------------
app = App(app_ui, server)