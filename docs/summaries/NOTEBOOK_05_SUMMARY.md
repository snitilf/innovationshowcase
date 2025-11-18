# Comprehensive Summary: Data Preparation for Machine Learning Model Training

## Introduction and Purpose

This analysis integrates governance indicators, economic indicators, and sentiment scores into a unified dataset for machine learning model training. The core methodological contribution addresses circular reasoning: governance indicators were used to create corruption risk labels, so they are excluded from the predictive feature set. Instead, the model must learn to predict risk labels using only economic and sentiment indicators, testing whether these measures can function as leading indicators that signal corruption risk before governance metrics reflect institutional weaknesses.

The analysis transforms the expanded labeled dataset (266 country-year observations across 19 countries from 2010-2023) into a structure suitable for supervised machine learning, where the model learns patterns from economic and sentiment indicators to predict governance-based risk labels.

## The Circular Reasoning Problem and Solution

Circular reasoning occurs when a model uses the same information to both create labels and make predictions. Since governance indicators determine corruption risk labels through the threshold-based system (4+ indicators below thresholds = high-risk), using these same indicators as predictive features would cause the model to simply memorize the labeling rule rather than discover whether other indicators can predict corruption risk.

**The Separation Strategy**: Governance indicators serve two distinct roles: (1) they determine the target variable (corruption risk labels) and are retained for validation, but (2) they are explicitly excluded from the predictive feature set. Five economic indicators and one sentiment score serve as predictive features. This separation creates a rigorous test: if economic conditions and sentiment can successfully predict governance-based risk labels, it demonstrates that these measures capture early warning signals that precede institutional deterioration.

## Data Integration and Missing Data Handling

### Governance Indicators

The six World Bank governance indicators provide complete coverage across all 266 observations. These indicators are used to create risk labels but excluded from predictive features. They show clear separation: high-risk countries average -0.62 to 0.14, while low-risk countries average 0.82 to 1.67.

### Economic Indicators

Five economic indicators serve as predictive features: GDP Growth Rate, External Debt as Percentage of GNI, Government Expenditure as Percentage of GDP, Foreign Direct Investment Inflows as Percentage of GDP, and Poverty Headcount Ratio at $2.15 per Day.

**Missing Data Imputation**: Economic indicators show 26.9% missing values overall. The analysis employs a two-step imputation strategy: (1) forward-filling within countries (using the most recent previous year's value), and (2) median imputation for remaining gaps (using the median value across all countries). After imputation, all economic indicators show complete coverage.

Economic indicators show substantial variation: GDP growth ranges from -28.8% to 19.7%, external debt from 8.0% to 420.6% of GNI, and poverty rates from 0% to 81.6%. This variation provides diverse patterns for the model to learn from.

### Sentiment Scores

Sentiment scores from corruption-related news articles provide 234 country-years (88% coverage) from Guardian API (2010-2016) and GDELT API (2017-2023). Missing sentiment scores (32 country-years, 12%) are filled with neutral values (0.0). Low-risk countries show more negative sentiment (mean -0.1004) than high-risk countries (mean -0.0694), reflecting the transparency pattern where free press enables corruption exposure.

## Feature Matrix and Target Variable

**Feature Matrix (X)**: Contains six predictive features (five economic indicators, one sentiment score) across 266 country-year observations. Each row represents one country-year, each column represents one predictive feature. All features have complete coverage after imputation.

**Target Variable (y)**: Contains corruption risk labels (0 = low-risk, 1 = high-risk) created using governance indicators. The model only sees the feature matrix when making predictions. Distribution: 112 low-risk (42.1%), 154 high-risk (57.9%).

## Train-Test Split

A stratified 80/20 train-test split divides the dataset: 80% for training (212 samples), 20% for testing (54 samples). The stratified approach maintains the same proportion of high-risk and low-risk cases in both sets.

**Training Set**: 212 samples (58.0% high-risk, 42.0% low-risk)
**Testing Set**: 54 samples (57.4% high-risk, 42.6% low-risk)

This split prevents overfitting by evaluating whether the model learns generalizable patterns rather than country-specific memorization.

## Case Study Validation

The prepared dataset maintains alignment with documented corruption cases:

- **Malaysia 1MDB (2013-2015)**: High-risk (corruption_risk = 1), sentiment -0.1772
- **Mozambique Hidden Debt (2013-2016)**: High-risk (corruption_risk = 1), sentiment 0.0030
- **Canada (Control)**: Low-risk (corruption_risk = 0) throughout all years

## Dataset Export Structure

The final dataset is exported in four formats:

1. **Final Training Data** (`final_training_data.csv`): Complete dataset with all variables for validation
2. **Training Set** (`train_set.csv`): 212 samples with only predictive features and target variable
3. **Testing Set** (`test_set.csv`): 54 samples with only predictive features and target variable
4. **Feature Names** (`feature_names.txt`): Text file listing the six predictive features in order

## Methodological Contributions

The separation of labeling features (governance indicators) from predictive features (economic and sentiment indicators) creates a rigorous test of whether economic conditions and public sentiment can predict corruption risk before governance metrics reflect institutional weaknesses. The analysis establishes a reproducible data preparation pipeline: data integration, missing data imputation, feature selection, and train-test split.

## Conclusion

This analysis successfully integrates three data sources into a unified dataset for machine learning model training. The core methodological contribution—separating labeling features from predictive features—ensures the model learns meaningful patterns rather than memorizing the labeling rule. The prepared dataset contains 266 country-year observations with complete coverage for all predictive features after imputation. The stratified train-test split (212/54 samples) enables evaluation on unseen data, testing generalizability across countries and time periods. This dataset provides the foundation for developing the Global Trust Engine as a data-driven early warning system for corruption risk in development contexts.

