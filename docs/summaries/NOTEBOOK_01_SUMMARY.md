# Comprehensive Summary: World Bank API Exploration and Baseline Dataset Establishment

## Introduction and Purpose

This foundational analysis establishes the initial dataset for the Global Trust Engine, a machine learning model designed to detect corruption risk in development aid contexts. The notebook serves as the critical first step in building a data-driven early warning system that can identify countries at high risk for corruption before scandals occur. Rather than relying on retrospective analysis of corruption after it has been exposed, this approach seeks to identify measurable indicators that signal vulnerability to corruption before funds are diverted.

The primary objective of this notebook is to collect standardized, internationally comparable data on governance quality and economic conditions for a carefully selected set of case study countries. This baseline dataset enables subsequent analysis to test whether measurable indicators can reliably distinguish between high-risk and low-risk environments, ultimately supporting the development of a predictive model that can protect development funds and improve aid effectiveness.

## Theoretical Foundation

The data collection strategy is grounded in established research on the structural conditions that enable corruption. Corruption is fundamentally a governance issue that thrives in environments characterized by limited accountability, weak enforcement systems, and institutional weaknesses. The United Nations Development Programme identifies these structural vulnerabilities as the root causes that create environments where the rewards of corruption outweigh the risks.

The World Bank's Worldwide Governance Indicators (WGI) provide standardized, internationally comparable measures that directly capture these structural vulnerabilities. These indicators are not measures of corruption itself, but rather measures of the institutional quality and accountability mechanisms that either prevent or enable corruption. When these indicators show low scores, they signal the presence of structural weaknesses—such as limited citizen participation, weak rule of law, or poor government effectiveness—that create environments where corruption can flourish.

The theoretical framework suggests that these governance indicators should show deterioration before, during, and after documented corruption cases. By collecting data across a time period that spans pre-scandal, scandal, and post-scandal periods, this analysis can examine whether governance indicators serve as reliable signals of corruption risk, or whether they only reflect problems after they have already emerged.

## Case Study Selection

The analysis focuses on three strategically selected countries that represent distinct governance profiles and corruption histories:

**Canada** serves as the control country, selected because it maintains consistently strong governance institutions throughout the study period. As a high-income country with well-established democratic institutions, strong rule of law, and transparent governance systems, Canada provides a baseline against which to compare countries with documented corruption cases. Canada's governance indicators consistently rank among the highest globally, making it an ideal control case that demonstrates what strong governance looks like across all six dimensions.

**Malaysia** represents a middle-income country with a documented major corruption scandal: the 1MDB (1Malaysia Development Berhad) scandal that came to light in 2015. This case involved the theft of approximately $4.5 billion USD from a state development fund, orchestrated by high-level government officials including the former Prime Minister. Malaysia provides an important case study because it demonstrates that corruption can occur even in countries with moderate governance scores—not just in countries with the weakest institutions. The 1MDB scandal is particularly significant because it involved sophisticated money laundering through international financial systems, showing how corruption can occur even when some governance mechanisms appear functional.

**Mozambique** represents a lower-income country with a documented corruption case: the hidden debt crisis that unfolded from 2013 to 2016. This case involved the illicit use of $2 billion USD in loans intended for maritime security and fishing industry development, which were instead diverted by corrupt government officials. The former Finance Minister received $7 million USD in bribes to facilitate the scheme, and over $200 million USD was illegally diverted for personal benefit. Mozambique provides a case study of corruption in a context with weaker governance institutions, demonstrating how structural vulnerabilities enable large-scale corruption in development contexts.

The selection of these three countries creates a comparative framework that spans the spectrum from strong governance (Canada) to moderate governance with documented corruption (Malaysia) to weak governance with documented corruption (Mozambique). This design enables analysis of whether governance indicators can distinguish between these different risk profiles.

## Data Collection Methodology

The data collection strategy employs two complementary categories of indicators: governance indicators and economic indicators. This dual approach recognizes that corruption risk is not solely a function of governance quality, but also reflects economic conditions that may create vulnerabilities or incentives for corrupt behavior.

### Governance Indicators: The Worldwide Governance Indicators (WGI)

Six Worldwide Governance Indicators were collected, each measuring a distinct dimension of institutional quality:

**Voice and Accountability** measures the extent to which citizens can participate in selecting their government, as well as freedom of expression, freedom of association, and free media. This indicator captures whether citizens have mechanisms to hold leaders accountable and whether corruption can be exposed through free press and public discourse. Higher scores indicate greater citizen participation and accountability mechanisms.

**Political Stability and Absence of Violence** measures perceptions of the likelihood that the government will be destabilized or overthrown by unconstitutional or violent means. This includes political terrorism and politically motivated violence. This indicator captures whether the political system is stable enough to maintain consistent governance standards, or whether instability creates opportunities for corruption during transitions or crises.

**Government Effectiveness** measures the quality of public services, the quality of the civil service and its independence from political pressures, the quality of policy formulation and implementation, and the credibility of the government's commitment to such policies. This indicator captures whether government institutions function effectively and transparently, or whether weak implementation creates opportunities for corruption.

**Regulatory Quality** measures the ability of the government to formulate and implement sound policies and regulations that permit and promote private sector development. This indicator captures whether regulatory frameworks are well-designed and consistently enforced, or whether weak or inconsistent regulation creates opportunities for corruption.

**Rule of Law** measures the extent to which agents have confidence in and abide by the rules of society, particularly the quality of contract enforcement, property rights, the police, and the courts, as well as the likelihood of crime and violence. This indicator captures whether legal systems function effectively to prevent and punish corruption, or whether weak enforcement enables corrupt behavior.

**Control of Corruption** measures the extent to which public power is exercised for private gain, including both petty and grand forms of corruption, as well as capture of the state by elites and private interests. This indicator directly measures perceptions of corruption control, making it the most directly relevant to the research question.

All six indicators are standardized on a scale that typically ranges from approximately -2.5 to +2.5, with higher scores indicating better governance. These scores are based on hundreds of underlying data sources including surveys of households and firms, assessments by commercial risk rating agencies, and evaluations by non-governmental organizations.

### Economic Indicators: Contextual Vulnerability Measures

Five economic indicators were collected to provide context on financial conditions that may signal vulnerability to corruption:

**External Debt as Percentage of Gross National Income (GNI)** measures the total external debt of a country relative to its national income. High external debt burdens can create pressure on governments to divert funds, and debt crises can create opportunities for corruption during restructuring or emergency lending. This indicator helps identify countries under financial stress that may be vulnerable to corruption.

**Annual GDP Growth Rate** measures the year-over-year percentage change in gross domestic product, which represents the total value of goods and services produced in a country. Low or negative growth rates can indicate economic distress that may create incentives for corruption, while high growth rates may mask underlying governance problems. This indicator provides context on economic performance during the study period.

**Government Expenditure as Percentage of GDP** measures total government spending relative to the size of the economy. This indicator helps identify patterns in government spending that may signal vulnerability—for example, unusually high spending during periods of weak governance may indicate opportunities for corruption, while very low spending may indicate capacity constraints that enable corruption.

**Foreign Direct Investment (FDI) Inflows as Percentage of GDP** measures investment from foreign entities into the country relative to economic size. This indicator captures whether countries are attracting legitimate investment, which may be associated with stronger governance, or whether investment patterns suggest vulnerabilities. Low or volatile FDI may signal governance concerns that deter legitimate investors.

**Poverty Headcount Ratio at $2.15 per Day** measures the percentage of the population living below the international poverty line. High poverty rates may indicate that corruption has diverted resources from development, or may signal economic conditions that create incentives for corruption. This indicator provides a measure of economic vulnerability and development outcomes.

These economic indicators complement the governance indicators by providing measurable signals of financial stress, economic performance, and development outcomes that may either result from corruption or create conditions that enable it.

## Temporal Scope: The 2010-2023 Timeframe

The analysis covers the period from 2010 to 2023, a 14-year span that was strategically selected to capture three distinct phases relative to the documented corruption cases:

**Pre-Scandal Period (2010-2012 for Malaysia, 2010-2012 for Mozambique)**: This period captures governance and economic conditions before the documented corruption cases emerged. For Malaysia, this represents the years leading up to the 1MDB scandal that came to light in 2015. For Mozambique, this represents the years before the hidden debt crisis began in 2013. This baseline period enables analysis of whether governance indicators showed early warning signals before corruption was exposed.

**Scandal Period (2013-2016 for Mozambique, 2013-2015 for Malaysia)**: This period captures the years during which the documented corruption cases occurred. For Mozambique, the hidden debt crisis unfolded from 2013 to 2016, with state-owned companies defaulting on over $700 million USD in loans. For Malaysia, the 1MDB scandal came to light in 2015, though the corruption itself occurred over several years. This period enables analysis of whether governance indicators deteriorated during active corruption.

**Post-Scandal Period (2017-2023)**: This period captures the years following the exposure of corruption cases, enabling analysis of whether governance indicators improved as a result of institutional responses, legal proceedings, and reforms. For both countries, this period includes years when legal actions were taken against corrupt officials and when governance reforms may have been implemented.

The temporal dimension is critical because it tests a key hypothesis: can governance indicators serve as early warning signals that deteriorate before corruption is exposed, or do they only reflect problems retrospectively? By examining patterns across pre-scandal, scandal, and post-scandal periods, the analysis can determine whether these indicators have predictive value for identifying corruption risk.

The year 2024 was excluded from the analysis due to incomplete governance data availability at the time of data collection. Governance indicators are typically published with a lag, and 2024 data may not have been fully available when this baseline dataset was established.

## Data Quality Assessment

The data collection process yielded a dataset of 42 country-year observations (14 years × 3 countries) with 13 variables (Country, Year, 6 governance indicators, 5 economic indicators, plus the Country and Year identifiers). The data quality assessment reveals important patterns in data completeness that inform subsequent analysis decisions.

### Governance Indicators: Complete Coverage

All six Worldwide Governance Indicators show 100% data completeness across all countries and years. This complete coverage is expected because the World Bank prioritizes comprehensive reporting of governance indicators, recognizing their importance for international development analysis. The consistent availability of governance data across all case study countries and all years in the study period ensures that governance-based analysis is not compromised by missing data.

This complete coverage is particularly important because governance indicators form the foundation for subsequent risk labeling. The ability to calculate governance-based risk labels for every country-year observation depends on having complete governance data, which this baseline dataset provides.

### Economic Indicators: Variable Completeness

The economic indicators show variable levels of data completeness, reflecting differences in data collection practices and country-specific reporting capabilities:

**GDP Growth Rate** shows 100% completeness, indicating that all countries consistently report this fundamental economic indicator. This complete coverage enables reliable analysis of economic performance patterns.

**Foreign Direct Investment Inflows** shows 100% completeness, indicating consistent reporting of investment flows across all countries and years.

**Government Expenditure** shows 94.48% completeness (4 missing values out of 42 observations, representing 9.52% of the dataset). The missing values are concentrated in specific country-year combinations, likely reflecting reporting delays or methodological changes in specific years.

**External Debt** shows only 33.33% completeness (28 missing values out of 42 observations, representing 66.67% of the dataset). This substantial missing data reflects the fact that external debt reporting varies significantly across countries and years. Some countries may not report external debt data consistently, or may use different methodologies that make data unavailable for certain periods. Canada, in particular, shows missing external debt data for most years, which may reflect differences in how high-income countries report debt statistics compared to developing countries.

**Poverty Headcount Ratio** shows 47.62% completeness (22 missing values out of 42 observations, representing 52.38% of the dataset). This missing data pattern reflects the fact that poverty measurement requires household surveys that are not conducted annually in all countries. Poverty data is typically available every few years rather than annually, and some countries may have gaps in survey coverage.

### Implications for Analysis

The variable completeness of economic indicators has important implications for subsequent analysis. While governance indicators provide complete coverage for risk labeling, economic indicators will require careful handling of missing data in downstream analysis. The missing data patterns suggest that some economic indicators may be more reliable than others for predictive modeling, and that missing data strategies (such as forward-filling within countries or median imputation) will be necessary.

However, the complete coverage of governance indicators ensures that the foundational risk labels can be calculated reliably, and the partial coverage of economic indicators still provides substantial data for analysis. The dataset contains sufficient information to proceed with comparative analysis and risk assessment.

## Baseline Governance Analysis: Comparative Findings

The comparative analysis of governance scores across the three case study countries reveals clear and consistent patterns that validate the theoretical framework and support the case study selection strategy.

### Average Governance Scores: 2010-2023

The analysis of average governance scores across the entire study period reveals stark differences between the control country and the countries with documented corruption cases:

**Canada** demonstrates consistently high governance scores across all six dimensions, with averages ranging from 1.04 (Political Stability) to 1.81 (Control of Corruption). These scores place Canada in the top tier of global governance rankings, reflecting strong democratic institutions, effective government services, robust regulatory frameworks, strong rule of law, and effective corruption control mechanisms. The consistently high scores across all dimensions indicate that Canada maintains strong governance as an integrated system, with no single dimension showing weakness.

**Malaysia** shows moderate governance scores that are substantially lower than Canada but higher than Mozambique. Average scores range from 0.14 (Political Stability) to 0.96 (Government Effectiveness), with Control of Corruption averaging 0.19. These scores reflect Malaysia's position as a middle-income country with some functional governance institutions but with notable weaknesses, particularly in corruption control and political stability. The moderate scores across most dimensions, combined with low scores on Control of Corruption, align with the documented 1MDB scandal, suggesting that governance weaknesses enabled the corruption that occurred.

**Mozambique** shows the lowest governance scores across all dimensions, with averages ranging from -0.62 (Political Stability) to -0.78 (Government Effectiveness), and Control of Corruption averaging -0.73. These negative scores indicate governance quality below the global average, reflecting weak institutions, limited accountability, and poor corruption control. The consistently low scores across all dimensions, particularly the very low Control of Corruption score, align with the documented hidden debt crisis, suggesting that structural governance weaknesses created an environment where large-scale corruption could occur.

The comparative pattern is consistent and clear: Canada (control country) shows the highest scores, Malaysia (moderate corruption case) shows moderate scores, and Mozambique (severe corruption case) shows the lowest scores. This pattern validates the theoretical framework's prediction that governance indicators should distinguish between high-risk and low-risk environments.

### Temporal Analysis: Governance Scores at Key Time Points

The analysis of governance scores at three critical time points—2013 (pre-scandal baseline), 2018 (post-scandal period), and 2023 (most recent data)—reveals important temporal patterns:

**2013: Pre-Scandal Baseline**

In 2013, before the documented corruption cases fully emerged, governance scores already showed clear distinctions between countries. Canada maintained high scores across all dimensions (ranging from 1.06 to 1.88), while Malaysia showed moderate scores (ranging from 0.05 to 0.99) and Mozambique showed low scores (ranging from -0.23 to -0.82). This pattern suggests that governance indicators may have had predictive value, showing weaknesses before corruption was exposed.

For Mozambique specifically, 2013 represents the year when the hidden debt crisis began, yet governance scores were already low. Control of Corruption was -0.60, Rule of Law was -0.82, and Government Effectiveness was -0.64. These low scores suggest that governance weaknesses were already present when the corruption scheme was initiated, supporting the hypothesis that governance indicators can signal vulnerability before corruption occurs.

**2018: Post-Scandal Period**

By 2018, several years after the corruption cases were exposed, governance scores show interesting patterns. Canada maintained consistently high scores (ranging from 0.96 to 1.79), demonstrating stability in strong governance institutions.

Malaysia showed some improvement in certain dimensions by 2018, with Government Effectiveness increasing to 1.05 (from 0.99 in 2013) and Rule of Law increasing to 0.53 (from 0.34 in 2013). However, Control of Corruption remained low at 0.30 (similar to 0.33 in 2013), suggesting that while some governance dimensions may have improved following the 1MDB scandal exposure, corruption control mechanisms remained weak.

Mozambique showed further deterioration by 2018, with Political Stability declining to -0.83 (from -0.23 in 2013), Rule of Law declining to -1.07 (from -0.82 in 2013), and Control of Corruption declining to -0.81 (from -0.60 in 2013). This pattern suggests that the hidden debt crisis and its aftermath may have further weakened governance institutions, rather than triggering improvements.

**2023: Most Recent Data**

By 2023, the most recent available data shows continued patterns. Canada maintained high scores (ranging from 0.82 to 1.67), though some dimensions showed slight declines from earlier periods, possibly reflecting global challenges or measurement variations.

Malaysia showed continued moderate scores, with some dimensions showing improvement (Voice and Accountability increased to 0.09, up from negative values in earlier periods) while others remained stable. Control of Corruption remained low at 0.30, suggesting persistent challenges in corruption control despite legal actions taken against corrupt officials.

Mozambique showed continued low scores, with Political Stability declining further to -1.27 and Rule of Law remaining very low at -1.03. Control of Corruption remained low at -0.83, suggesting that governance weaknesses persist years after the corruption case was exposed.

### Key Discoveries from Temporal Analysis

The temporal analysis reveals several important discoveries:

First, governance scores showed clear distinctions between countries even before corruption was exposed. The 2013 baseline scores already distinguished Canada (high), Malaysia (moderate), and Mozambique (low), suggesting that governance indicators may have predictive value for identifying vulnerability before corruption occurs.

Second, the pattern of governance scores aligns with the severity of documented corruption cases. Mozambique, which experienced a more severe corruption case in a context of weaker institutions, shows consistently lower scores than Malaysia, which experienced corruption in a context of somewhat stronger (though still moderate) institutions.

Third, governance scores do not necessarily improve immediately after corruption is exposed. Both Malaysia and Mozambique show that governance weaknesses can persist or even worsen following corruption scandals, suggesting that exposing corruption does not automatically lead to institutional strengthening.

Fourth, the control country (Canada) maintains consistently high scores across all time periods, demonstrating stability in strong governance institutions and validating its selection as a control case.

## Methodological Contributions

This baseline dataset establishment makes several important methodological contributions to the larger research project:

### Foundation for Risk Labeling

The complete coverage of governance indicators across all countries and years enables the creation of governance-based risk labels in subsequent analysis. These labels will classify each country-year observation as high-risk or low-risk based on governance indicator thresholds, providing the target variable for machine learning model training.

### Validation of Case Study Selection

The clear and consistent differences in governance scores between the control country and the countries with documented corruption cases validate the case study selection strategy. The pattern of scores aligns with theoretical predictions and documented corruption histories, confirming that these countries represent distinct risk profiles suitable for comparative analysis.

### Establishment of Temporal Baseline

The 2010-2023 timeframe establishes a temporal baseline that enables analysis of governance patterns before, during, and after documented corruption cases. This temporal dimension is critical for testing whether governance indicators can serve as early warning signals, or whether they only reflect problems retrospectively.

### Integration of Governance and Economic Indicators

The collection of both governance and economic indicators establishes a foundation for testing whether economic conditions can predict governance-based risk labels. This approach tests the "leading indicator" hypothesis—that economic conditions may deteriorate before governance metrics reflect institutional weaknesses, making economic indicators potentially more useful for early warning systems.

### Data Quality Documentation

The comprehensive data quality assessment documents patterns of missing data that inform subsequent analysis decisions. The complete coverage of governance indicators ensures reliable risk labeling, while the variable coverage of economic indicators informs missing data handling strategies.

## Conclusion

This foundational analysis successfully establishes a baseline dataset that enables subsequent machine learning model development. The data collection strategy, grounded in established theoretical frameworks on corruption and governance, yields a dataset with clear patterns that align with documented corruption cases and theoretical predictions.

The comparative analysis reveals that governance indicators show consistent and meaningful differences between high-risk and low-risk environments, with the control country (Canada) demonstrating consistently high scores, the moderate-risk country (Malaysia) showing moderate scores, and the high-risk country (Mozambique) showing consistently low scores. The temporal analysis suggests that governance indicators may have predictive value, showing weaknesses before corruption is exposed, though they do not necessarily improve immediately after corruption is discovered.

The complete coverage of governance indicators ensures reliable risk labeling in subsequent analysis, while the collection of economic indicators provides additional signals that may serve as leading indicators for corruption risk. This baseline dataset provides the foundation for developing a machine learning model that can identify high-risk environments before corruption occurs, ultimately supporting the protection of development funds and the improvement of aid effectiveness.

The methodological contributions of this analysis—including the validation of case study selection, the establishment of temporal baselines, and the integration of governance and economic indicators—create a robust foundation for the Global Trust Engine's development as a data-driven early warning system for corruption risk in development contexts.

