Task 2 - Exploratory Data Analysis (EDA)


 Objective
Understand the House Prices dataset using statistics and visualizations to discover patterns, trends, correlations, and anomalies before building ML models.

 Tools & Libraries Used
ToolPurposePython 3.xProgramming languagePandasData loading & statisticsNumPyNumerical operationsMatplotlibStatic visualizationsSeabornStatistical plotsPlotlyInteractive visualizations

 Dataset
Ames Housing Dataset (train.csv)
PropertyValueRows1460Columns81TargetSalePriceSourceKaggle House Prices Competition

 EDA Steps Performed
✅ Step 1 — Load & Explore Dataset

Loaded train.csv using Pandas
Checked shape, column names, data types

✅ Step 2 — Summary Statistics

Generated mean, median, std, min, max
Identified missing values per column
Analyzed both numerical and categorical features

✅ Step 3 — Histograms

Plotted distributions of 8 key numerical features
Identified skewed vs normally distributed features

✅ Step 4 — Boxplots

Visualized spread and outliers in 6 key features
Identified extreme outliers in LotArea and GrLivArea

✅ Step 5 — Correlation Heatmap

Built full correlation matrix of top 15 features
Found OverallQual (0.79) as strongest predictor of SalePrice

✅ Step 6 — Pairplot

Visualized relationships between 5 key features
Confirmed linear relationships with SalePrice

✅ Step 7 — Categorical Analysis

Boxplots of SalePrice vs OverallQual, Neighborhood, HouseStyle
Found Neighborhood and Quality strongly affect price

✅ Step 8 — Skewness Analysis

Measured skewness of all numerical features
Applied log1p transform on SalePrice to reduce skew

✅ Step 9 — Interactive Plotly Chart

Created interactive scatter plot (GrLivArea vs SalePrice)
Color-coded by Overall Quality, saved as HTML

✅ Step 10 — Key Insights Summary

Summarized all findings and patterns discovered


 Output Files
FileDescriptionhistograms.pngDistribution of key numerical featuresboxplots.pngOutlier detection via boxplotscorrelation_heatmap.pngFeature correlation matrixpairplot.pngPairwise feature relationshipscategorical_analysis.pngCategorical features vs SalePriceskewness_analysis.pngBefore vs after log transforminteractive_scatter.htmlInteractive Plotly chart (open in browser)

 Repository Structure
task2-eda-analysis/
│
├── eda_analysis.py              # Main Python EDA script
├── train.csv                    # Original dataset
├── histograms.png               # Histogram plots
├── boxplots.png                 # Boxplot visualizations
├── correlation_heatmap.png      # Correlation heatmap
├── pairplot.png                 # Pairplot
├── categorical_analysis.png     # Categorical analysis
├── skewness_analysis.png        # Skewness before/after
├── interactive_scatter.html     # Interactive Plotly chart
└── README.md                    # Project documentation

 How to Run
1. Install dependencies
bashpip install pandas numpy matplotlib seaborn plotly
2. Place train.csv in the same folder
3. Run the script
bashpython eda_analysis.py

 Key Findings
FindingDetailStrongest predictorOverallQual (corr = 0.79)SalePrice distributionRight-skewed, log transform neededBiggest outliersLotArea, GrLivAreaMost impactful categoryNeighborhoodMissing data issueAlley, PoolQC > 80% missing
