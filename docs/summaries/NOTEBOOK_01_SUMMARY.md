# Summary: World Bank API Exploration and Baseline Dataset Establishment

## Introduction and Purpose

This analysis establishes the foundational dataset for the Global Trust Engine by collecting standardized governance and economic indicators from the World Bank API. The dataset contains six Worldwide Governance Indicators (WGI) and five economic indicators for three case study countries across a 14-year period (2010-2023), enabling subsequent analysis to test whether measurable indicators can distinguish between high-risk and low-risk environments for corruption.

## Case Study Selection

Three countries were selected to represent distinct governance profiles: **Canada** (control country with consistently strong governance institutions), **Malaysia** (middle-income country with documented 1MDB corruption scandal, 2015), and **Mozambique** (lower-income country with documented hidden debt crisis, 2013-2016). This selection creates a comparative framework spanning the spectrum from strong governance to moderate and weak governance with documented corruption.

## Data Collection Methodology

Data was retrieved from the World Bank API using standardized country codes (CAN, MYS, MOZ) and indicator codes. Two categories of indicators were collected:

### Governance Indicators: Worldwide Governance Indicators (WGI)

Six WGI indicators measuring institutional quality on a standardized scale (-2.5 to +2.5, higher scores indicate better governance):
- Voice and Accountability (VA.EST)
- Political Stability and Absence of Violence (PV.EST)
- Government Effectiveness (GE.EST)
- Regulatory Quality (RQ.EST)
- Rule of Law (RL.EST)
- Control of Corruption (CC.EST)

### Economic Indicators

Five economic indicators providing context on financial conditions:
- External Debt as Percentage of GNI (DT.DOD.DECT.GN.ZS)
- Annual GDP Growth Rate (NY.GDP.MKTP.KD.ZG)
- Government Expenditure as Percentage of GDP (GC.XPN.TOTL.GD.ZS)
- Foreign Direct Investment Inflows as Percentage of GDP (BX.KLT.DINV.WD.GD.ZS)
- Poverty Headcount Ratio at $2.15 per Day (SI.POV.DDAY)

## Temporal Scope: 2010-2023

The 14-year timeframe captures three phases relative to documented corruption cases: pre-scandal period (2010-2012), scandal period (2013-2016 for Mozambique, 2013-2015 for Malaysia), and post-scandal period (2017-2023). Year 2024 was excluded due to incomplete governance data availability at the time of collection.

## Data Quality Assessment

The data collection process yielded a dataset of 42 country-year observations (14 years × 3 countries) with 13 variables.

### Governance Indicators: Complete Coverage

All six Worldwide Governance Indicators show 100% data completeness across all countries and years. This complete coverage is essential because governance indicators form the foundation for subsequent risk labeling.

### Economic Indicators: Variable Completeness

Economic indicators show variable levels of data completeness:
- GDP Growth Rate: 100% completeness (0 missing values)
- Foreign Direct Investment Inflows: 100% completeness (0 missing values)
- Government Expenditure: 94.48% completeness (4 missing values, 9.52% missing)
- External Debt: 33.33% completeness (28 missing values, 66.67% missing)
- Poverty Headcount Ratio: 47.62% completeness (22 missing values, 52.38% missing)

The variable completeness reflects differences in data collection practices and country-specific reporting capabilities. The complete coverage of governance indicators ensures reliable risk labeling in subsequent analysis, while the variable coverage of economic indicators will require missing data handling strategies (forward-filling within countries or median imputation) in downstream analysis.

## Baseline Governance Analysis: Key Findings

Comparative analysis of governance scores reveals clear and consistent patterns. Average governance scores across the entire study period (2010-2023) show stark differences: Canada demonstrates consistently high scores across all six dimensions (averages ranging from 1.04 to 1.81), Malaysia shows moderate scores (averages ranging from 0.14 to 0.96, with Control of Corruption averaging 0.19), and Mozambique shows the lowest scores (averages ranging from -0.62 to -0.78, with Control of Corruption averaging -0.73).

Temporal analysis at key time points (2013 pre-scandal baseline, 2018 post-scandal period, 2023 most recent data) reveals that governance scores showed clear distinctions between countries even before corruption was exposed, suggesting predictive value. Governance scores align with the severity of documented corruption cases, and governance weaknesses can persist or worsen following corruption scandals. Canada maintains consistently high scores across all time periods, validating its selection as a control case.

## Technical Implementation

The data collection process uses the `wbdata` Python library to retrieve indicators from the World Bank API. The technical implementation involves:

1. **API Query Setup**: Using ISO country codes (CAN, MYS, MOZ) and World Bank indicator codes to construct API queries with date range specification (2010-01-01 to 2023-12-31).

2. **Data Retrieval**: The `wbdata.get_dataframe()` function retrieves all indicators for specified countries and date range, with `parse_dates=False` to preserve year values as strings for consistent formatting.

3. **Data Transformation**: Converting index columns to regular columns using `reset_index()`, renaming columns (date → Year, country → Country), and reordering columns with Country and Year first.

4. **Data Sorting**: Sorting chronologically by country and year using `sort_values(by=['Country', 'Year'])` for time series analysis.

5. **Data Export**: Exporting the cleaned dataset as `corruption_data_baseline.csv` to `data/raw/` directory for downstream analysis.

## Dataset Structure and Output

The final baseline dataset contains 42 country-year observations with 13 variables:
- **Identifiers**: Country, Year
- **Governance Indicators**: Voice_Accountability, Political_Stability, Government_Effectiveness, Regulatory_Quality, Rule_of_Law, Control_of_Corruption
- **Economic Indicators**: External_Debt_perc_GNI, GDP_Growth_annual_perc, Govt_Expenditure_perc_GDP, FDI_Inflows_perc_GDP, Poverty_Headcount_Ratio

The dataset is exported to `data/raw/corruption_data_baseline.csv` and serves as the foundation for subsequent analysis phases, including risk labeling, dataset expansion, and machine learning model training.

## Conclusion

This analysis establishes a baseline dataset that enables subsequent machine learning model development. The data collection strategy yields a dataset with clear patterns: governance indicators show consistent and meaningful differences between high-risk and low-risk environments, with the control country demonstrating consistently high scores, the moderate-risk country showing moderate scores, and the high-risk country showing consistently low scores. The complete coverage of governance indicators ensures reliable risk labeling in subsequent analysis, while the collection of economic indicators provides additional signals that may serve as leading indicators for corruption risk.
