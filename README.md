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
- Open up VSCode and import the necessary packages to webscrape the website. Use pip install to add any packages that are not already installed in the laptop.
- Run `web_scraper.ipynb` in order to scrape the data from the website. If the webscraper breaks in the middle, rerun that chunk of code and change the page where it broke. 
- Once finished, there should be two csv files (one for the Arabica beans and one for the robusta beans)
- Run `clean_more.ipynb` in order to organize the delete and delete any unnecessary information and columns
- Run `clean_coffee_data.R` in order to clean the text columns and perform the final required cleaning sets
- In the end, there should be one csv file that contains the data for both the Arabica and Robusta coffee beans
- The data cleaning scripts were adepted from this [Github Repository](https://github.com/jldbc/coffee-quality-database/tree/master). Changes were made to some of the files since some packages have deprecated since then.
- The dataset used for this analysis containes records from October 2024 - Novemeber 2025. 
