# Coffee Quality Analysis (PUT MORE CREATIVE TITLE)
#### Marissa Burton, Hayeon Chung, Maggie Crowner, Asmita Kadam, Ashrita Kodali 

## Repository Contents
This repository contains all of the necessary files and scripts for conducting a coffee quality analysis on different coffee beans. The analysis aims to determine whether different modeling techniques, such as regression, classification,  unsupervised learning and deep learning can be applied to physical, farming, and taste attributes to analyze coffee beans better. 

## 1. Software and Platform

### Programming Languages and Software
- **Python**: For data preprocessing, model assumption testing, developing and evaluating all of the models, developing Shiny App dashboard
- **R**: For data preprocessing and cleaning

### Main Python Packages
- `selenium` – communicating with web-browser and automating webscraping
- `webdriver_manager` – connecting to chrome web browser
- `bs4` – webscraping individual links
- `time` – spacing out each step while automating the webscraping process
- `os` – getting and updating working directory
- `dotenv` – loading in username and passwords hidden in a .env file
- `random` – setting seeds for reproducible results
- `pandas` – cleaning data, manipulating data, organizing results
- `numpy` – cleaning data, manipulating data, organizing results
- `plotly` – conducting exploratory data analysis, visualizing data and modeling results 
- `seaborn` – conducting exploratory data analysis, visualizing data and modeling results 
- `matplotlib` – conducting exploratory data analysis, visualizing data and modeling results 
- `sklearn` – standardizing data, creating model pipelines/models, analyzing model performance
- `scipy` – performing linear algebra calculations
- `statsmodels`-- creating various different regression models
- `tensorflow` – creating MLP model architectures 
- `shiny` – developing shiny application dashboard


### Main R Packages
- `stringr` – modifying/cleaning text related columns
- `dplyr` – cleaning, simplifying, and organizing data
- `digest` – applying hash functions to R objects


### Platform Compatibility
Developed and tested on Mac, but should work on Windows and Linux with appropriate installations.

## 2. Project Folder Structure

```
📂 Coffee Quality Analysis
 ├── 📂 data/  
 ├──── 📂 raw_in_progress_data/
 │   │   ├── arabica_data_cleaned.csv
 │   │   ├── clean_coffee_data.R
 │   │   ├── clean_more.ipynb
 │   │   ├── df_arabica_final.csv
 │   │   ├── df_robusta_final.csv
 │   │   ├── robusta_data_cleaned.csv
 │   │   ├── web_scraper.ipynb
 ├──── 📂 cleaned_data/
 │   │   ├── FINAL_DATA.csv
 │   │   ├── linear.csv
 │
 ├── 📂 eda/  
 │   ├── ML_Project_KNN_EDA.ipynb
 │   ├── linear_regression_eda.ipynb
 │   ├── mlp_eda.ipynb
 │  
 ├── 📂 modeling/  
 │   ├── ML_Project_Clustering.ipynb
 │   ├── ML_Project_KNN_Modeling.ipynb
 │   ├── final_linear_regression.ipynb
 │   ├── linear_regression.ipynb
 │   ├── logistic_reg.ipynb
 │   ├── mlp_model_FINAL.ipynb
 │  
 ├── app.py 
 ├── requirements.txt
 ├── README.md   
```

## 3. Instructions for Reproducing Results

### Stage 1: Data Preparation
- Create an account with the [Coffee Quality Institute](https://www.coffeeinstitute.org/).
- Download the files in the data folder of this repository
- Open up VSCode and import the necessary packages to webscrape the website. Use pip install to add any packages that are not already installed
- Run `web_scraper.ipynb` in order to scrape the data from the website. If the webscraper breaks in the middle, rerun that chunk of code and change the page where it broke. 
- Once finished, there should be two csv files (one for the Arabica beans and one for the robusta beans)
- Run `clean_more.ipynb` in order to organize the delete and delete any unnecessary information and columns
- Run `clean_coffee_data.R` in order to clean the text columns and perform the final required cleaning sets
- In the end, there should be one csv file that contains the data for both the Arabica and Robusta coffee beans
- The data cleaning scripts were adapted from this [Github Repository](https://github.com/jldbc/coffee-quality-database/tree/master). Changes were made to some of the files since some packages have deprecated since then.
- The dataset used for this analysis contains records from October 2024 - November 2025.

### Stage 2: Exploratory Data Analysis
- Understand what the data looks like by doing an exploratory data analysis.
- Download the files in the eda folder of this repository
- Open up VSCode and import the necessary packages to webscrape the website. Use pip install to add any packages that are not already installed
- Run files `ML_Project_KNN_EDA.ipynb`, `linear_regression_eda.ipynb`, and `mlp_eda.ipynb` in order to understand the various transformations and decisions that were made during the model building process

### Stage 3: Model Building & Evaluation
- Download the files in the modeling folder of this repository
- Using the data set, 5 different models were created: Linear Regression Model, Logistic Regression Model, KNN Model, a K-Means Analysis, and a Multilayer Perceptron Model
- Open up VSCode and import the necessary packages to build the models. Use pip install to add any packages that are not already installed
- Run the following scripts in order to create these 5 models: `final_linear_regression.ipynb`, `logistic_reg.ipynb`,`ML_Project_KNN_modeling.ipynb`, `ML_Project_Clustering.ipynb`, and `mlp_model_FINAL.ipynb`
- After running these scripts, you will be able to see to analyze the 5 different research questions and see how the models perform

### Stage 4: Shiny App Development
- In order to compile all of the results, a Shiny dashboard will be made
- Download the file `app.py` that is found in the main branch of this repository
- Run `app.py` in order to create the Shiny Dashboard
- Upload the app to [ShinyApps.io](ShinyApps.io)  in order to deploy the dashboard
- The final [dashboard](https://maggiecrowner.shinyapps.io/coffee_quality_app/) should look like this

