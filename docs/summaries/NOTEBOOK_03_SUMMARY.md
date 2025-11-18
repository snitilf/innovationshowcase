# Dataset Expansion for Machine Learning Model Training

## Introduction and Purpose

This analysis expands the baseline dataset from 3 countries to 19 countries, increasing observations from 42 to 266 country-year observations. The expansion provides sufficient sample size for training a decision tree classifier (266 observations) and introduces diversity in governance patterns across different institutional contexts, economic development levels, and geographic regions.

## Country Selection Strategy

The expansion adds 16 countries to the original 3-country baseline (Canada, Malaysia, Mozambique), selected to span the full spectrum of governance quality:

**High-Risk Countries**: Angola, Venezuela, Zimbabwe, Iraq, Ukraine - countries with documented governance weaknesses or corruption vulnerabilities.

**Medium-Risk Countries**: Brazil, South Africa, India, Philippines - countries with mixed governance profiles.

**Low-Risk Control Countries**: Norway, Denmark, Singapore, Australia, New Zealand, Switzerland, Germany - countries with consistently strong governance institutions.

This selection ensures the model learns generalizable patterns across diverse contexts rather than memorizing country-specific patterns.

## Data Collection and Quality Assessment

Data collection follows the same methodology as the baseline analysis, retrieving six governance indicators and five economic indicators for all 19 countries across 2010-2023 using the World Bank API.

### Data Completeness Results

**Governance Indicators**: All six Worldwide Governance Indicators show 100% completeness across all 19 countries and all years (266 observations). This complete coverage is essential for the labeling methodology, which requires all six indicators to calculate threshold-based flags.

**Economic Indicators**: Variable completeness reflects country-specific reporting differences. External Debt shows 52.2% missing data, Poverty Headcount Ratio shows 57.1% missing data, Government Expenditure shows 18.3% missing data, while GDP Growth and Foreign Direct Investment show low missing data (~4%). These missing values do not affect risk labeling, which depends solely on governance indicators. Economic indicators will be handled through imputation strategies during model training.

## Labeling Methodology Application

The same threshold-based labeling methodology from notebook 02 is applied uniformly across all 19 countries. The methodology creates binary flags for each of six governance indicators when scores fall below established thresholds, and a country-year observation is labeled as high-risk (1) if it receives 4 or more flags; otherwise, it is labeled as low-risk (0).

## Validation Results

The labeling methodology maintains perfect validation when applied to the expanded dataset: Malaysia's 1MDB period (2013-2015) correctly labeled as high-risk (3/3 years), Mozambique's hidden debt crisis (2013-2016) correctly labeled as high-risk (4/4 years), and Canada's control case correctly labeled as low-risk throughout (0/14 high-risk years).

## Key Findings

### Label Distribution

The final labeled dataset contains 266 country-year observations. Of these, 112 (42.1%) are classified as low-risk and 154 (57.9%) are classified as high-risk. This balanced distribution provides adequate representation for machine learning model training.

### Governance Patterns

Analysis reveals clear separation across risk categories: high-risk countries show consistently low scores below thresholds across all dimensions, medium-risk countries show mixed profiles, and low-risk countries show consistently high scores above thresholds.

### Temporal Patterns

Some countries show consistent risk labels throughout the study period (Canada consistently low-risk, Mozambique consistently high-risk), while others show variation across years (Malaysia transitions between risk categories), reflecting changes in governance quality over time.

## Technical Contributions

### Dataset Scale

The expansion from 42 to 266 observations provides sufficient sample size for training a decision tree classifier while maintaining generalizability. The balanced class distribution (57.9% high-risk, 42.1% low-risk) prevents class imbalance issues.

### Dataset Structure

The final dataset contains 266 country-year observations with 23 variables: 6 governance indicators, 5 economic indicators, 6 binary flags, total flag count, risk category classification, and the corruption risk label (target variable). This structure provides both predictive features and the target variable needed for supervised machine learning model training.

## Dataset Export

The final dataset is exported as `corruption_data_expanded_labeled.csv` containing 266 country-year observations with 23 variables. This dataset provides the foundation for machine learning model training with both predictive features and the target variable.
