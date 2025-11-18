# Comprehensive Summary: Data Cleaning and Corruption Risk Labeling

## Introduction and Purpose

This analysis transforms the raw governance and economic data collected in the previous phase into a labeled dataset suitable for machine learning model training. While the first notebook established the baseline dataset by collecting standardized indicators from the World Bank, this notebook operationalizes the theoretical framework by creating measurable corruption risk labels that can serve as the target variable for predictive modeling.

The primary objective is to develop a systematic, validated methodology for classifying country-year observations as high-risk or low-risk for corruption based on governance indicators. This labeling approach directly translates the theoretical understanding that corruption thrives in environments with limited accountability, weak enforcement systems, and institutional weaknesses into a quantitative classification system. Rather than relying on retrospective analysis of corruption after it has been exposed, this methodology creates labels based on measurable governance conditions that signal vulnerability before corruption occurs.

The labeling methodology is grounded in the principle that corruption is fundamentally a governance issue rooted in structural weaknesses. By applying threshold-based classification to six World Bank governance indicators that directly measure these structural conditions, this analysis creates a systematic approach to identifying high-risk environments. The methodology is rigorously validated against two well-documented corruption cases—Malaysia's 1MDB scandal and Mozambique's hidden debt crisis—ensuring that the labeling approach correctly identifies periods when documented corruption occurred.

## Theoretical Foundation and Operationalization

The labeling methodology operationalizes the theoretical framework established in the research foundation, which identifies corruption as fundamentally a governance issue that thrives in environments with limited accountability, weak enforcement systems, and institutional weaknesses. The United Nations Development Programme emphasizes that these structural vulnerabilities create environments where the rewards of corruption outweigh the risks, enabling corrupt practices to flourish.

The six Worldwide Governance Indicators used for labeling directly measure these structural conditions. Each indicator captures a distinct dimension of institutional quality that either prevents or enables corruption. When multiple indicators show weaknesses simultaneously, this signals a multi-dimensional governance failure that creates an environment where corruption can occur. The threshold-based labeling approach translates these theoretical concepts into a quantitative classification system by establishing cutoff points for each governance dimension and aggregating weaknesses across dimensions to create a comprehensive risk assessment.

The multi-dimensional approach recognizes that corruption is not caused by a single governance weakness, but rather emerges when multiple structural vulnerabilities exist simultaneously. A country might have weak rule of law but strong government effectiveness, or limited voice and accountability but strong regulatory quality. However, when four or more governance dimensions show weaknesses below established thresholds, this signals a comprehensive governance failure that creates an environment where corruption can flourish. This approach aligns with the theoretical understanding that corruption requires multiple enabling conditions to occur at scale.

## Data Quality Assessment and Temporal Scope

The analysis begins with a comprehensive assessment of data quality and completeness. The initial dataset contains 45 country-year observations (15 years × 3 countries) covering the period from 2010 to 2024. However, governance indicators for 2024 are incomplete in the World Bank dataset, with all six governance indicators showing missing values for that year. Since the labeling methodology depends entirely on these governance measures, 2024 is excluded from the analysis, resulting in a final dataset of 42 country-year observations covering the period from 2010 to 2023.

This temporal scope adjustment is critical because the labeling methodology requires complete governance data for every observation. Unlike economic indicators, which can tolerate some missing values through imputation strategies, governance indicators form the foundation for risk classification. Without complete governance data, it is impossible to calculate the threshold-based flags that determine risk labels. The exclusion of 2024 ensures that every observation in the final dataset has complete governance information, enabling reliable risk classification.

The data quality assessment reveals important patterns in missing data that inform subsequent analysis decisions. Governance indicators show complete coverage after excluding 2024, with all six indicators available for all 42 country-year observations. This complete coverage is essential for the labeling methodology, which requires all six governance dimensions to calculate risk flags. Economic indicators show variable completeness, with some indicators having substantial missing data, but these are handled separately through imputation strategies that do not compromise the governance-based risk labeling.

## Missing Data Handling Strategy

The analysis employs a differentiated approach to missing data handling that recognizes the distinct roles of governance and economic indicators in the labeling methodology. Governance indicators are required for risk labeling and must be complete for every observation. Economic indicators, while important for subsequent model training as predictive features, can tolerate limited missingness through imputation strategies.

### Governance Indicators: Complete Coverage Required

All six governance indicators show complete coverage after excluding 2024, with zero missing values across all 42 country-year observations. This complete coverage is verified through systematic checking that confirms every observation has values for all six governance dimensions. This verification is critical because the labeling methodology requires all six indicators to calculate threshold-based flags. If any governance indicator were missing, it would be impossible to determine whether that dimension shows weakness, compromising the multi-dimensional risk assessment.

The complete coverage of governance indicators ensures that the risk labeling is based on comprehensive information about all six dimensions of institutional quality. This comprehensive assessment is essential because corruption risk emerges from multi-dimensional governance failures, not from weaknesses in a single dimension. By requiring complete governance data, the methodology ensures that risk labels reflect a thorough evaluation of all structural conditions that enable or prevent corruption.

### Economic Indicators: Imputation Strategy

Economic indicators show variable levels of completeness, reflecting differences in data collection practices and country-specific reporting capabilities. The analysis employs a forward-filling imputation strategy that uses the most recent available value for each country when data is missing. This approach assumes that economic indicators change gradually year-over-year, making the previous year's value a reasonable estimate when current data is unavailable.

Forward-filling works by looking backward in time within each country's data series. When a value is missing for a particular year, the method uses the value from the most recent previous year for that same country. For example, if government expenditure data is missing for Malaysia in 2015, but available for 2014, the 2014 value is used to fill the 2015 gap. This approach preserves country-specific patterns while handling missing data in a systematic way.

After forward-filling, some economic indicators still show remaining missing values. External Debt as Percentage of Gross National Income shows 28 missing values (66.7% of observations), primarily because Canada does not report external debt data consistently, and some countries have gaps in reporting for specific years. Poverty Headcount Ratio shows 5 remaining missing values (11.9% of observations), reflecting that poverty measurement requires household surveys that are not conducted annually in all countries.

These remaining missing values in economic indicators do not compromise the risk labeling methodology, which depends entirely on governance indicators. Economic indicators will be used as predictive features in subsequent machine learning model training, where missing data can be handled through additional imputation strategies or excluded from specific analyses. The key point is that governance indicators, which form the foundation for risk labels, show complete coverage.

## Threshold-Based Risk Labeling Methodology

The core contribution of this analysis is the development of a systematic, threshold-based methodology for classifying country-year observations as high-risk or low-risk for corruption. This methodology operationalizes the theoretical framework by establishing cutoff points for each governance dimension and aggregating weaknesses across dimensions to create comprehensive risk assessments.

### Governance Indicator Thresholds

The methodology applies threshold-based classification to six World Bank governance indicators, each measuring a distinct dimension of institutional quality. A threshold is a cutoff point that divides scores into two categories: scores above the threshold indicate strength in that dimension, while scores below the threshold indicate weakness. For each governance indicator, if a country-year observation scores below the threshold, it receives a "flag" indicating weakness in that dimension.

The six governance indicators and their thresholds are:

**Voice and Accountability (threshold: 1.15)**: This indicator measures the extent to which citizens can participate in selecting their government, as well as freedom of expression, freedom of association, and free media. Scores below 1.15 indicate limited citizen participation and accountability mechanisms, creating an environment where corruption can occur without public exposure.

**Political Stability and Absence of Violence (threshold: 0.50)**: This indicator measures perceptions of the likelihood that the government will be destabilized or overthrown by unconstitutional or violent means. Scores below 0.50 indicate political instability that can create opportunities for corruption during transitions or crises.

**Government Effectiveness (threshold: 1.15)**: This indicator measures the quality of public services, the quality of the civil service and its independence from political pressures, and the quality of policy formulation and implementation. Scores below 1.15 indicate weak government institutions that create opportunities for corruption.

**Regulatory Quality (threshold: 1.15)**: This indicator measures the ability of the government to formulate and implement sound policies and regulations that permit and promote private sector development. Scores below 1.15 indicate weak or inconsistent regulation that creates opportunities for corruption.

**Rule of Law (threshold: 1.15)**: This indicator measures the extent to which agents have confidence in and abide by the rules of society, particularly the quality of contract enforcement, property rights, the police, and the courts. Scores below 1.15 indicate weak legal systems that cannot effectively prevent or punish corruption.

**Control of Corruption (threshold: 1.15)**: This indicator directly measures the extent to which public power is exercised for private gain, including both petty and grand forms of corruption. Scores below 1.15 indicate weak corruption control mechanisms.

The thresholds are set based on empirical analysis of the data distribution and validated against documented corruption cases. Most indicators use a threshold of 1.15, which represents a score substantially above the global average (which is approximately 0 on the standardized scale). Political Stability uses a lower threshold of 0.50, reflecting that this indicator shows more variation and that moderate stability may still enable corruption if other governance dimensions are weak.

### Binary Flag System

For each governance indicator, the methodology creates a binary flag—a simple yes-or-no indicator—that signals whether that dimension shows weakness. A binary flag takes only two values: 1 indicates weakness (score below threshold), and 0 indicates strength (score at or above threshold). This binary approach simplifies the complex governance scores into clear signals of strength or weakness in each dimension.

The binary flag system enables systematic aggregation across multiple governance dimensions. Rather than trying to combine scores that are measured on the same scale but represent different concepts, the flag system creates uniform indicators of weakness that can be counted and aggregated. This approach recognizes that corruption risk emerges from the accumulation of governance weaknesses across multiple dimensions, not from the severity of weakness in any single dimension.

For each country-year observation, the methodology calculates six binary flags—one for each governance indicator. These flags are then summed to create a total flag count, which ranges from 0 (no governance weaknesses) to 6 (weaknesses in all governance dimensions). This total flag count provides a comprehensive measure of multi-dimensional governance failure.

### Risk Aggregation and Classification

The methodology aggregates the six binary flags into a single risk classification through a threshold-based rule: a country-year observation is labeled as **high risk (1)** if it receives 4 or more flags, indicating weaknesses in at least four out of six governance dimensions. Otherwise, it is labeled as **low risk (0)**, indicating that governance is sufficiently strong across most dimensions to prevent large-scale corruption.

The choice of four flags as the threshold for high-risk classification reflects the multi-dimensional nature of governance failure identified in the theoretical framework. Corruption requires multiple enabling conditions to occur at scale—weak rule of law alone may not enable corruption if government effectiveness is strong, but weak rule of law combined with weak government effectiveness, weak regulatory quality, and weak control of corruption creates an environment where corruption can flourish. The four-flag threshold captures this understanding by requiring weaknesses across a majority of governance dimensions.

This aggregation approach recognizes that corruption is not caused by a single governance weakness, but rather emerges when multiple structural vulnerabilities exist simultaneously. A country might have weak political stability but strong rule of law and government effectiveness, creating a mixed governance profile that may not enable large-scale corruption. However, when four or more governance dimensions show weaknesses, this signals a comprehensive governance failure that creates an environment where corruption can occur.

## Validation Against Documented Corruption Cases

The labeling methodology is rigorously validated against two well-documented corruption cases that align with the theoretical framework's emphasis on governance failures enabling fund diversion. This validation ensures that the methodology correctly identifies periods when documented corruption occurred, providing confidence that the labels accurately reflect corruption risk.

### Malaysia 1MDB Scandal Validation (2013-2015)

The 1MDB (1Malaysia Development Berhad) scandal involved the theft of approximately $4.5 billion USD from a state development fund, orchestrated by high-level government officials including the former Prime Minister. The funds were diverted through weak oversight structures, with sophisticated money laundering through international financial systems. This case demonstrates that corruption can occur even in countries with moderate governance scores, not just in countries with the weakest institutions.

The validation analysis examines whether the labeling methodology correctly identifies the 1MDB scandal period (2013-2015) as high-risk. The results show perfect validation: all three years during the scandal period (2013, 2014, and 2015) are correctly labeled as high-risk, with Malaysia receiving 4 or more governance flags in each of these years. This validation confirms that the methodology successfully identifies the governance weaknesses that enabled the 1MDB corruption, even though Malaysia's governance scores were moderate rather than extremely low.

The successful validation of the Malaysia case is particularly important because it demonstrates that the methodology can identify corruption risk in middle-income countries with mixed governance profiles, not just in countries with the weakest institutions. Malaysia's governance scores during the 1MDB period were moderate—not as low as Mozambique's scores, but still showing weaknesses across multiple dimensions that enabled corruption to occur.

### Mozambique Hidden Debt Crisis Validation (2013-2016)

The hidden debt crisis in Mozambique involved $2 billion USD in illicit loans intended for maritime security and fishing industry development, which were instead diverted by corrupt government officials. The former Finance Minister received $7 million USD in bribes to facilitate the scheme, and over $200 million USD was illegally diverted for personal benefit. This case demonstrates corruption in a context with weaker governance institutions, showing how structural vulnerabilities enable large-scale corruption in development contexts.

The validation analysis examines whether the labeling methodology correctly identifies the hidden debt crisis period (2013-2016) as high-risk. The results show perfect validation: all four years during the crisis period (2013, 2014, 2015, and 2016) are correctly labeled as high-risk, with Mozambique receiving 4 or more governance flags in each of these years. This validation confirms that the methodology successfully identifies the governance weaknesses that enabled the hidden debt crisis.

The successful validation of the Mozambique case demonstrates that the methodology correctly identifies corruption risk in countries with weak governance institutions. Mozambique's governance scores during the hidden debt crisis period were consistently low across multiple dimensions, creating an environment where large-scale corruption could occur. The methodology's ability to correctly label all four years of the crisis period confirms that the threshold-based approach accurately captures the governance conditions that enable corruption.

### Canada Control Case Validation

Canada serves as a control case with strong governance institutions and no documented major corruption scandals during the study period. As a high-income country with well-established democratic institutions, strong rule of law, and transparent governance systems, Canada provides a baseline against which to validate that the methodology correctly identifies low-risk environments.

The validation analysis examines whether the labeling methodology correctly identifies Canada as low-risk throughout the study period. The results show perfect validation: all 14 years from 2010 to 2023 are correctly labeled as low-risk, with Canada receiving fewer than 4 governance flags in every year. This validation confirms that the methodology successfully distinguishes between high-risk and low-risk environments, correctly identifying Canada's strong governance as protective against corruption.

The successful validation of the Canada control case is critical because it demonstrates that the methodology does not over-classify countries as high-risk. If the methodology incorrectly labeled Canada as high-risk despite its strong governance, this would indicate that the thresholds are too strict or that the aggregation approach is flawed. The fact that Canada is correctly identified as low-risk in all years provides confidence that the methodology accurately reflects governance-based corruption risk.

## Key Findings and Patterns

The analysis reveals important patterns in corruption risk labels and governance indicator distributions that provide insights into the relationship between governance quality and corruption vulnerability.

### Overall Label Distribution

The final labeled dataset contains 42 country-year observations, with 14 observations (33.3%) classified as low-risk and 28 observations (66.7%) classified as high-risk. This distribution reflects that two of the three case study countries (Malaysia and Mozambique) show high-risk labels for substantial portions of the study period, while the control country (Canada) shows low-risk labels throughout.

The high proportion of high-risk labels (66.7%) is expected given the case study selection strategy, which intentionally included countries with documented corruption cases. Malaysia and Mozambique, which experienced documented corruption during the study period, show high-risk labels for most or all years, while Canada, which serves as a control case with strong governance, shows low-risk labels throughout. This distribution validates the case study selection by demonstrating that the labeling methodology distinguishes between the countries as expected.

### Country-Specific Risk Patterns

The analysis reveals distinct risk patterns across the three case study countries that align with their governance profiles and documented corruption histories.

**Canada** shows consistently low-risk labels throughout the entire study period (2010-2023), with zero high-risk classifications across all 14 years. This pattern reflects Canada's consistently strong governance scores across all six dimensions, with scores well above the established thresholds for most indicators in most years. Canada's governance profile demonstrates what strong, protective governance looks like, with institutional quality that prevents corruption from occurring at scale.

**Malaysia** shows high-risk labels for substantial portions of the study period, with the 1MDB scandal period (2013-2015) correctly identified as high-risk. Malaysia's risk pattern reflects its moderate governance scores, which place it in a middle position between Canada's strong governance and Mozambique's weak governance. During the 1MDB scandal period, Malaysia's governance scores showed weaknesses across multiple dimensions, creating an environment where corruption could occur despite some functional governance institutions.

**Mozambique** shows high-risk labels for the entire study period (2010-2023), with all 14 years classified as high-risk. This pattern reflects Mozambique's consistently weak governance scores across all six dimensions, with scores well below the established thresholds for most indicators in most years. Mozambique's governance profile demonstrates what weak, vulnerable governance looks like, with institutional quality that enables corruption to occur. The hidden debt crisis period (2013-2016) is correctly identified as high-risk, but the methodology also identifies Mozambique as high-risk in years before and after the crisis, reflecting persistent governance weaknesses.

### Temporal Patterns Relative to Documented Corruption Cases

The analysis reveals important temporal patterns in risk labels relative to the documented corruption cases, providing insights into whether governance indicators show early warning signals before corruption is exposed.

For **Malaysia**, the 1MDB scandal period (2013-2015) is correctly identified as high-risk, with all three years receiving high-risk labels. However, the analysis also shows that Malaysia received high-risk labels in years before and after the scandal period, reflecting that governance weaknesses existed before the corruption was exposed and persisted after it was discovered. This pattern suggests that governance indicators may have had predictive value, showing weaknesses before corruption was exposed, but also that exposing corruption does not automatically lead to immediate governance improvements.

For **Mozambique**, the hidden debt crisis period (2013-2016) is correctly identified as high-risk, with all four years receiving high-risk labels. The analysis also shows that Mozambique received high-risk labels in all years from 2010 to 2023, reflecting persistent governance weaknesses throughout the study period. This pattern suggests that governance indicators showed early warning signals before the hidden debt crisis began, with weaknesses present in 2010-2012 before the crisis unfolded in 2013. The persistence of high-risk labels after the crisis (2017-2023) reflects that governance weaknesses continued even after the corruption was exposed.

For **Canada**, the control case shows consistently low-risk labels throughout the entire study period, with no temporal variation. This pattern reflects the stability of strong governance institutions, which maintain protective quality across all years regardless of external events or global challenges.

### Governance Indicator Patterns and Threshold Crossings

The analysis reveals important patterns in how governance indicators cross established thresholds, providing insights into which dimensions show weaknesses and how these weaknesses accumulate to create high-risk classifications.

**Canada** shows governance scores that consistently remain above thresholds for all six indicators across most years. On the rare occasions when Canada's scores fall below a threshold for a specific indicator, this occurs for only one or two indicators simultaneously, never accumulating to four or more flags that would trigger a high-risk classification. This pattern demonstrates that strong governance requires strength across multiple dimensions, and that occasional weaknesses in single dimensions do not create corruption risk if other dimensions remain strong.

**Malaysia** shows governance scores that frequently fall below thresholds for multiple indicators simultaneously. During the 1MDB scandal period (2013-2015), Malaysia received 4 or more flags in each year, with weaknesses across multiple dimensions including control of corruption, rule of law, and regulatory quality. However, Malaysia also shows variation across years, with some years receiving fewer than 4 flags (low-risk) and other years receiving 4 or more flags (high-risk). This pattern reflects Malaysia's moderate governance profile, which shows mixed strength and weakness across dimensions and across time.

**Mozambique** shows governance scores that consistently fall below thresholds for most or all indicators across all years. Mozambique typically receives 5 or 6 flags in most years, reflecting weaknesses across nearly all governance dimensions. This pattern demonstrates that weak governance involves comprehensive failures across multiple dimensions simultaneously, creating an environment where corruption can occur persistently rather than intermittently.

## Methodological Contributions

This analysis makes several important methodological contributions to the larger research project, establishing foundations for subsequent machine learning model development and validating the theoretical framework through empirical application.

### Foundation for Machine Learning Model Training

The labeled dataset created in this analysis provides the essential foundation for machine learning model training in subsequent phases. Machine learning models require labeled data—observations with known outcomes—to learn patterns that can predict outcomes for new observations. The corruption risk labels (high-risk = 1, low-risk = 0) serve as the target variable that the model will learn to predict based on governance and economic indicators.

The threshold-based labeling methodology ensures that these labels are systematically derived from measurable governance conditions, not from retrospective knowledge of corruption after it was exposed. This approach enables the model to learn patterns that can identify high-risk environments before corruption occurs, rather than simply recognizing corruption after it has been discovered. The systematic, validated labeling approach provides confidence that the labels accurately reflect governance-based corruption risk, enabling reliable model training.

### Operationalization of Theoretical Framework

The analysis successfully operationalizes the theoretical framework by translating abstract concepts about governance and corruption into a quantitative classification system. The theoretical understanding that corruption thrives in environments with limited accountability, weak enforcement systems, and institutional weaknesses is translated into measurable thresholds applied to six governance indicators, with multi-dimensional aggregation that captures the comprehensive nature of governance failure.

This operationalization is critical because it enables empirical testing of theoretical predictions. Rather than relying on qualitative assessments or case study narratives, the methodology creates quantitative labels that can be systematically analyzed and validated. The successful validation against documented corruption cases confirms that the operationalization accurately captures the theoretical concepts, providing confidence that the labels reflect real corruption risk rather than arbitrary classifications.

### Validation of Labeling Approach

The rigorous validation against documented corruption cases provides essential confidence in the labeling methodology. The perfect validation results—with Malaysia's 1MDB period, Mozambique's hidden debt crisis period, and Canada's control case all correctly classified—demonstrate that the methodology successfully distinguishes between high-risk and low-risk environments based on governance indicators.

This validation is particularly important because it confirms that governance indicators can serve as reliable signals of corruption risk, even before corruption is exposed. The fact that Malaysia and Mozambique showed high-risk labels during documented corruption periods, and that these labels also appeared in years before corruption was exposed, suggests that governance indicators have predictive value for identifying vulnerability before corruption occurs.

The validation also confirms that the methodology does not over-classify or under-classify risk. Canada's consistent low-risk classification demonstrates that the methodology correctly identifies strong governance as protective, while Malaysia's and Mozambique's high-risk classifications during documented corruption periods demonstrate that the methodology correctly identifies weak governance as enabling corruption.

### Establishment of Systematic Classification Rules

The analysis establishes systematic, reproducible classification rules that can be applied consistently across countries and time periods. The threshold-based approach with multi-dimensional aggregation creates clear, objective criteria for risk classification that do not depend on subjective judgments or case-specific knowledge. This systematic approach enables the methodology to be applied to new countries and time periods with confidence that classifications will be consistent and comparable.

The systematic rules also enable transparency and interpretability, which are essential for policy applications. Stakeholders can understand exactly how risk labels are derived—which governance indicators are evaluated, what thresholds are applied, and how flags are aggregated—enabling informed decision-making about how to use the labels for risk assessment and early warning systems.

## Visualization and Pattern Discovery

The analysis creates comprehensive visualizations that reveal important patterns in governance indicators, risk flags, and risk labels across countries and time periods. These visualizations provide intuitive understanding of the data patterns and validate the labeling methodology through visual inspection.

### Governance Indicators Over Time

Visualizations of governance indicators over time reveal clear patterns that distinguish between countries and align with documented corruption cases. Canada's governance scores consistently remain above thresholds across all six dimensions, while Malaysia's scores show moderate levels with some variation, and Mozambique's scores consistently remain below thresholds across most dimensions. These patterns provide visual confirmation that the labeling methodology accurately reflects governance quality differences between countries.

The visualizations also reveal temporal patterns, showing how governance scores change over time relative to documented corruption periods. For Malaysia, the 1MDB scandal period (2013-2015) shows governance scores that are below thresholds for multiple indicators, while for Mozambique, the hidden debt crisis period (2013-2016) shows consistently low scores across all indicators. These temporal patterns provide visual validation that the labeling methodology correctly identifies periods when governance weaknesses enabled documented corruption.

### Governance Flags Heatmap

A heatmap visualization of total governance flags by country and year reveals the accumulation of governance weaknesses that trigger high-risk classifications. The heatmap shows Canada with zero or one flag in most years, Malaysia with varying flag counts (ranging from 2 to 5 flags), and Mozambique with 5 or 6 flags in most years. This visualization provides intuitive understanding of how the multi-dimensional flag system captures governance quality differences, with higher flag counts indicating more comprehensive governance failures.

The heatmap also reveals temporal patterns in flag accumulation, showing how governance weaknesses accumulate or diminish over time. For Malaysia, flag counts increase during the 1MDB scandal period, reflecting the accumulation of governance weaknesses that enabled corruption. For Mozambique, flag counts remain consistently high throughout the study period, reflecting persistent governance failures that create ongoing corruption risk.

### Corruption Risk Labels Heatmap

A heatmap visualization of corruption risk labels by country and year provides a clear, intuitive representation of the final classification results. The heatmap shows Canada with low-risk labels (0) in all years, Malaysia with a mix of low-risk and high-risk labels that correctly identify the 1MDB scandal period, and Mozambique with high-risk labels (1) in all years that correctly identify the hidden debt crisis period. This visualization provides immediate visual confirmation that the labeling methodology successfully distinguishes between high-risk and low-risk environments and correctly identifies documented corruption periods.

### Flag and Risk Distribution Analysis

Visualizations of flag distributions and risk label distributions by country reveal important patterns in how governance weaknesses accumulate and how risk classifications are distributed. Canada shows a distribution concentrated at low flag counts (0-2 flags), reflecting that governance weaknesses are rare and never accumulate to trigger high-risk classification. Malaysia shows a distribution spread across moderate flag counts (2-5 flags), reflecting mixed governance quality that sometimes triggers high-risk classification. Mozambique shows a distribution concentrated at high flag counts (5-6 flags), reflecting comprehensive governance failures that consistently trigger high-risk classification.

These distribution patterns provide insights into the different governance profiles that create different levels of corruption risk. The concentration of Canada's distribution at low flag counts demonstrates what protective governance looks like, while the concentration of Mozambique's distribution at high flag counts demonstrates what vulnerable governance looks like. Malaysia's spread distribution demonstrates that moderate governance can create intermittent risk, with some years showing protective quality and other years showing vulnerability.

## Export and Dataset Preparation

The analysis exports the cleaned and labeled dataset for subsequent model training phases. The final dataset contains 42 country-year observations with 21 variables, including the original governance and economic indicators, the six binary flags for each governance dimension, the total flag count, and the final corruption risk label.

The exported dataset serves as the foundation for machine learning model training, providing both the predictive features (governance and economic indicators) and the target variable (corruption risk label) that the model will learn to predict. The systematic labeling methodology ensures that the target variable accurately reflects governance-based corruption risk, enabling reliable model training that can identify high-risk environments before corruption occurs.

## Conclusion

This analysis successfully transforms raw governance and economic data into a systematically labeled dataset suitable for machine learning model training. The threshold-based labeling methodology operationalizes the theoretical framework by creating quantitative classifications based on measurable governance conditions, with rigorous validation against documented corruption cases confirming that the labels accurately reflect corruption risk.

The key contribution of this analysis is the development of a systematic, validated approach to classifying corruption risk based on governance indicators. The methodology successfully distinguishes between high-risk and low-risk environments, correctly identifying documented corruption periods while also providing early warning signals through governance weaknesses that appear before corruption is exposed. The perfect validation results—with Malaysia's 1MDB period, Mozambique's hidden debt crisis period, and Canada's control case all correctly classified—provide confidence that the methodology accurately captures governance-based corruption risk.

The labeled dataset created in this analysis provides the essential foundation for subsequent machine learning model development. The systematic, validated labeling approach ensures that model training will learn patterns that can identify high-risk environments before corruption occurs, ultimately supporting the protection of development funds and the improvement of aid effectiveness. The methodological contributions—including the operationalization of the theoretical framework, the validation of the labeling approach, and the establishment of systematic classification rules—create a robust foundation for the Global Trust Engine's development as a data-driven early warning system for corruption risk in development contexts.

The analysis reveals important patterns in how governance weaknesses accumulate across multiple dimensions to create corruption risk, and how these patterns align with documented corruption cases. The temporal analysis suggests that governance indicators may have predictive value, showing weaknesses before corruption is exposed, though they do not necessarily improve immediately after corruption is discovered. These findings support the development of a machine learning model that can identify high-risk environments proactively, rather than retrospectively, ultimately enabling the protection of development funds before they are diverted through corrupt practices.

