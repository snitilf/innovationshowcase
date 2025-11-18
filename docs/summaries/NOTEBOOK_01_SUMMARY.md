# Comprehensive Summary: World Bank API Exploration and Baseline Dataset Establishment

## Introduction and Purpose

This analysis establishes the foundational dataset for the Global Trust Engine by collecting standardized governance and economic indicators from the World Bank API. The core technical contribution is the systematic retrieval and organization of six Worldwide Governance Indicators (WGI) and five economic indicators for three strategically selected case study countries across a 14-year time period (2010-2023). This baseline dataset enables subsequent analysis to test whether measurable indicators can reliably distinguish between high-risk and low-risk environments for corruption.

## Case Study Selection

Three countries were selected to represent distinct governance profiles and corruption histories:

**Canada** serves as the control country with consistently strong governance institutions throughout the study period. As a high-income country with well-established democratic institutions, Canada provides a baseline demonstrating what strong governance looks like across all six dimensions.

**Malaysia** represents a middle-income country with a documented major corruption scandal: the 1MDB (1Malaysia Development Berhad) scandal that came to light in 2015, involving the theft of approximately $4.5 billion USD from a state development fund. Malaysia demonstrates that corruption can occur even in countries with moderate governance scores.

**Mozambique** represents a lower-income country with a documented corruption case: the hidden debt crisis (2013-2016) involving $2 billion USD in illicit loans intended for maritime security and fishing industry development, which were diverted by corrupt government officials. Mozambique demonstrates how structural vulnerabilities enable large-scale corruption in development contexts.

The selection creates a comparative framework spanning the spectrum from strong governance (Canada) to moderate governance with documented corruption (Malaysia) to weak governance with documented corruption (Mozambique).

## Data Collection Methodology

The data collection employs two complementary categories of indicators retrieved from the World Bank API using standardized country codes and indicator codes.

### Governance Indicators: Worldwide Governance Indicators (WGI)

Six Worldwide Governance Indicators were collected, each measuring a distinct dimension of institutional quality on a standardized scale (typically -2.5 to +2.5, with higher scores indicating better governance):

- **Voice and Accountability (VA.EST)**: Measures citizen participation in selecting government, freedom of expression, freedom of association, and free media. Higher scores indicate greater citizen participation and accountability mechanisms.

- **Political Stability and Absence of Violence (PV.EST)**: Measures perceptions of the likelihood that government will be destabilized by unconstitutional or violent means. Higher scores indicate greater political stability.

- **Government Effectiveness (GE.EST)**: Measures quality of public services, civil service independence, policy formulation and implementation quality, and government credibility. Higher scores indicate more effective government institutions.

- **Regulatory Quality (RQ.EST)**: Measures ability of government to formulate and implement sound policies and regulations that permit and promote private sector development. Higher scores indicate better regulatory frameworks.

- **Rule of Law (RL.EST)**: Measures extent to which agents have confidence in and abide by rules of society, including contract enforcement, property rights, police, and courts. Higher scores indicate stronger legal systems.

- **Control of Corruption (CC.EST)**: Measures extent to which public power is exercised for private gain, including both petty and grand forms of corruption. Higher scores indicate better corruption control.

All six indicators are standardized measures based on hundreds of underlying data sources including surveys, risk rating agencies, and non-governmental organization evaluations.

### Economic Indicators: Contextual Vulnerability Measures

Five economic indicators were collected to provide context on financial conditions that may signal vulnerability to corruption:

- **External Debt as Percentage of GNI (DT.DOD.DECT.GN.ZS)**: Total external debt relative to national income. High debt burdens can create pressure on governments and signal financial stress.

- **Annual GDP Growth Rate (NY.GDP.MKTP.KD.ZG)**: Year-over-year percentage change in gross domestic product. Economic performance provides context on conditions during the study period.

- **Government Expenditure as Percentage of GDP (GC.XPN.TOTL.GD.ZS)**: Total government spending relative to economic size. Patterns in spending may signal vulnerability or capacity constraints.

- **Foreign Direct Investment Inflows as Percentage of GDP (BX.KLT.DINV.WD.GD.ZS)**: Investment from foreign entities relative to economic size. Investment patterns may signal governance concerns or attractiveness to legitimate investors.

- **Poverty Headcount Ratio at $2.15 per Day (SI.POV.DDAY)**: Percentage of population living below international poverty line. High poverty rates may indicate corruption impact or economic vulnerability.

## Temporal Scope: 2010-2023

The 14-year timeframe was strategically selected to capture three distinct phases relative to documented corruption cases:

**Pre-Scandal Period (2010-2012)**: Captures governance and economic conditions before documented corruption cases emerged, enabling analysis of whether indicators showed early warning signals.

**Scandal Period (2013-2016 for Mozambique, 2013-2015 for Malaysia)**: Captures years during which documented corruption cases occurred, enabling analysis of whether indicators deteriorated during active corruption.

**Post-Scandal Period (2017-2023)**: Captures years following corruption exposure, enabling analysis of whether indicators improved as a result of institutional responses and reforms.

The temporal dimension tests whether governance indicators serve as early warning signals that deteriorate before corruption is exposed, or only reflect problems retrospectively. Year 2024 was excluded due to incomplete governance data availability at the time of collection.

## Data Quality Assessment

The data collection process yielded a dataset of 42 country-year observations (14 years × 3 countries) with 13 variables (Country, Year, 6 governance indicators, 5 economic indicators).

### Governance Indicators: Complete Coverage

All six Worldwide Governance Indicators show 100% data completeness across all countries and years. This complete coverage is essential because governance indicators form the foundation for subsequent risk labeling. The consistent availability ensures that governance-based analysis is not compromised by missing data.

### Economic Indicators: Variable Completeness

Economic indicators show variable levels of data completeness:

- **GDP Growth Rate**: 100% completeness (0 missing values)
- **Foreign Direct Investment Inflows**: 100% completeness (0 missing values)
- **Government Expenditure**: 94.48% completeness (4 missing values, 9.52% missing)
- **External Debt**: 33.33% completeness (28 missing values, 66.67% missing)
- **Poverty Headcount Ratio**: 47.62% completeness (22 missing values, 52.38% missing)

The variable completeness reflects differences in data collection practices and country-specific reporting capabilities. External debt and poverty data show substantial missing values because these indicators require specialized surveys or reporting that may not be conducted annually in all countries.

### Implications for Analysis

The complete coverage of governance indicators ensures reliable risk labeling in subsequent analysis. The variable coverage of economic indicators will require missing data handling strategies (forward-filling within countries or median imputation) in downstream analysis, but still provides substantial data for comparative analysis.

## Baseline Governance Analysis: Comparative Findings

The comparative analysis of governance scores reveals clear and consistent patterns that validate the case study selection strategy and support the theoretical framework.

### Average Governance Scores: 2010-2023

Analysis of average governance scores across the entire study period shows stark differences:

**Canada** demonstrates consistently high scores across all six dimensions, with averages ranging from 1.04 (Political Stability) to 1.81 (Control of Corruption). These scores place Canada in the top tier of global governance rankings, reflecting strong democratic institutions, effective government services, robust regulatory frameworks, strong rule of law, and effective corruption control.

**Malaysia** shows moderate scores substantially lower than Canada but higher than Mozambique. Averages range from 0.14 (Political Stability) to 0.96 (Government Effectiveness), with Control of Corruption averaging 0.19. These scores reflect Malaysia's position as a middle-income country with some functional governance institutions but notable weaknesses, particularly in corruption control.

**Mozambique** shows the lowest scores across all dimensions, with averages ranging from -0.62 (Political Stability) to -0.78 (Government Effectiveness), and Control of Corruption averaging -0.73. These negative scores indicate governance quality below the global average, reflecting weak institutions, limited accountability, and poor corruption control.

The comparative pattern is consistent: Canada (control) shows highest scores, Malaysia (moderate corruption case) shows moderate scores, and Mozambique (severe corruption case) shows lowest scores. This pattern validates that governance indicators distinguish between high-risk and low-risk environments.

### Temporal Analysis: Governance Scores at Key Time Points

Analysis of governance scores at three critical time points (2013 pre-scandal baseline, 2018 post-scandal period, 2023 most recent data) reveals important temporal patterns:

**2013 (Pre-Scandal Baseline)**: Governance scores already showed clear distinctions. Canada maintained high scores (ranging from 1.06 to 1.88), Malaysia showed moderate scores (ranging from 0.05 to 0.99), and Mozambique showed low scores (ranging from -0.23 to -0.82). For Mozambique, 2013 represents the year when the hidden debt crisis began, yet governance scores were already low (Control of Corruption -0.60, Rule of Law -0.82), suggesting governance weaknesses were present when corruption was initiated.

**2018 (Post-Scandal Period)**: Canada maintained consistently high scores (ranging from 0.96 to 1.79). Malaysia showed some improvement in certain dimensions (Government Effectiveness increased to 1.05, Rule of Law to 0.53) but Control of Corruption remained low at 0.30. Mozambique showed further deterioration (Political Stability declined to -0.83, Rule of Law to -1.07, Control of Corruption to -0.81), suggesting the corruption crisis may have further weakened governance institutions.

**2023 (Most Recent Data)**: Canada maintained high scores (ranging from 0.82 to 1.67). Malaysia showed continued moderate scores with Control of Corruption remaining low at 0.30. Mozambique showed continued low scores (Political Stability -1.27, Rule of Law -1.03, Control of Corruption -0.83), suggesting governance weaknesses persist years after corruption exposure.

### Key Discoveries from Temporal Analysis

The temporal analysis reveals several important discoveries:

1. **Early Warning Signals**: Governance scores showed clear distinctions between countries even before corruption was exposed, suggesting predictive value.

2. **Alignment with Corruption Severity**: Governance scores align with the severity of documented corruption cases—Mozambique (more severe case, weaker institutions) shows consistently lower scores than Malaysia (moderate case, stronger institutions).

3. **Persistence of Weaknesses**: Governance scores do not necessarily improve immediately after corruption is exposed. Both Malaysia and Mozambique show that governance weaknesses can persist or worsen following corruption scandals.

4. **Control Country Stability**: Canada maintains consistently high scores across all time periods, demonstrating stability in strong governance institutions and validating its selection as a control case.

## Technical Implementation

The data collection process uses the `wbdata` Python library to retrieve indicators from the World Bank API. The technical implementation involves:

1. **Country Code Specification**: Using ISO country codes (CAN, MYS, MOZ) to retrieve data for the three case study countries.

2. **Indicator Code Mapping**: Mapping World Bank indicator codes to descriptive variable names for clarity in analysis.

3. **Date Range Specification**: Specifying the date range (2010-01-01 to 2023-12-31) to retrieve annual data across the study period.

4. **Data Formatting**: Converting index columns to regular columns, renaming columns for clarity (date → Year, country → Country), reordering columns with Country and Year first, and sorting chronologically by country and year for time series analysis.

5. **Data Export**: Exporting the cleaned dataset as `corruption_data_baseline.csv` for downstream analysis.

## Dataset Structure and Output

The final baseline dataset contains 42 country-year observations with 13 variables:
- **Identifiers**: Country, Year
- **Governance Indicators**: Voice_Accountability, Political_Stability, Government_Effectiveness, Regulatory_Quality, Rule_of_Law, Control_of_Corruption
- **Economic Indicators**: External_Debt_perc_GNI, GDP_Growth_annual_perc, Govt_Expenditure_perc_GDP, FDI_Inflows_perc_GDP, Poverty_Headcount_Ratio

The dataset is exported to `data/raw/corruption_data_baseline.csv` and serves as the foundation for subsequent analysis phases, including risk labeling, dataset expansion, and machine learning model training.

## Methodological Contributions

This baseline dataset establishment makes several important methodological contributions:

1. **Foundation for Risk Labeling**: Complete coverage of governance indicators enables creation of governance-based risk labels in subsequent analysis.

2. **Validation of Case Study Selection**: Clear differences in governance scores between control country and countries with documented corruption validate the case study selection strategy.

3. **Establishment of Temporal Baseline**: The 2010-2023 timeframe enables analysis of governance patterns before, during, and after documented corruption cases.

4. **Integration of Governance and Economic Indicators**: Collection of both indicator types establishes foundation for testing whether economic conditions can predict governance-based risk labels.

5. **Data Quality Documentation**: Comprehensive assessment documents missing data patterns that inform subsequent analysis decisions.

## Conclusion

This foundational analysis successfully establishes a baseline dataset that enables subsequent machine learning model development. The data collection strategy yields a dataset with clear patterns that align with documented corruption cases and theoretical predictions. The comparative analysis reveals that governance indicators show consistent and meaningful differences between high-risk and low-risk environments, with the control country demonstrating consistently high scores, the moderate-risk country showing moderate scores, and the high-risk country showing consistently low scores.

The temporal analysis suggests that governance indicators may have predictive value, showing weaknesses before corruption is exposed, though they do not necessarily improve immediately after corruption is discovered. The complete coverage of governance indicators ensures reliable risk labeling in subsequent analysis, while the collection of economic indicators provides additional signals that may serve as leading indicators for corruption risk.

This baseline dataset provides the foundation for developing a machine learning model that can identify high-risk environments before corruption occurs, ultimately supporting the protection of development funds and the improvement of aid effectiveness. The methodological contributions create a robust foundation for the Global Trust Engine's development as a data-driven early warning system for corruption risk in development contexts.
