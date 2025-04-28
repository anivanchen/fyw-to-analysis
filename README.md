# Test-Optional Impact Study

### Project Overview
This project aims to analyze the effects of test-optional admissions policies on various educational outcomes in U.S. higher education. The project looks at the period around 2020 when many institutions adopted test-optional policies in response to the COVID-19 pandemic.

### Data
The analysis uses College Scorecard data spanning from 1996 to 2023, tracking institution admissions policies and various outcome metrics over time. The dataset includes:

- Admissions test requirements (ADMCON7)
- Demographic indicators (PCTPELL, UGDS_BLACK, UGDS_HISP, etc.)
- Academic outcomes (retention rates, withdrawal rates by income level, graduation rates)
- Financial metrics (student debt, percentage of students with loans)

### Analysis Components

- Data Processing
- Extracts relevant variables from raw College Scorecard data files
- Tracks changes in test requirement policies (ADMCON7) over time
- Creates a longitudinal dataset of selected institutions

### Descriptive Analysis

- Time series visualizations of admissions policy changes
- Distribution plots comparing test-optional vs. test-required institutions
- Institutional type analysis separating public, private non-profit, and for-profit institutions

#### Sample Graph

![Pell Percentage Split by Institution Type](sample-imgs/descriptive_institutional_data.png)

### Panel Analysis
- Difference-in-difference models using PanelOLS with entity and time fixed effects
- Multicollinearity diagnostics via variance inflation factor analysis
- Residual diagnostics including heteroskedasticity and normality tests
- Visualization of trends and treatment effects

#### Sample Graph

![Pell Percentage Trend Over Time](sample-imgs/trend_plot.png)

![Pell Percentage Standardized Panel Residuals](sample-imgs/standardized_residuals_plot.png)

#### Sample Output

```
                          PanelOLS Estimation Summary                           
================================================================================
Dep. Variable:                PCTPELL   R-squared:                        0.0076
Estimator:                   PanelOLS   R-squared (Between):              0.1632
No. Observations:               16317   R-squared (Within):              -0.0447
Date:                Sun, Apr 27 2025   R-squared (Overall):              0.1630
Time:                        22:29:25   Log-likelihood                 2.957e+04
Cov. Estimator:             Clustered                                           
                                        F-statistic:                      38.490
Entities:                        1126   P-value                           0.0000
Avg Obs:                       14.491   Distribution:                 F(3,15174)
Min Obs:                       1.0000                                           
Max Obs:                       15.000   F-statistic (robust):             6.4615
                                        P-value                           0.0002
Time periods:                      15   Distribution:                 F(3,15174)
Avg Obs:                       1087.8                                           
Min Obs:                       1077.0                                           
Max Obs:                       1106.0                                           
                                                                                
                              Parameter Estimates                               
================================================================================
              Parameter  Std. Err.     T-stat    P-value    Lower CI    Upper CI
--------------------------------------------------------------------------------
Treated_Post     0.0121     0.0053     2.2619     0.0237      0.0016      0.0225
UGDS           1.26e-06  1.018e-06     1.2380     0.2157  -7.351e-07   3.256e-06
COSTT4_A      7.756e-07  2.474e-07     3.1352     0.0017   2.907e-07    1.26e-06
================================================================================

F-test for Poolability: 106.63
P-value: 0.0000
Distribution: F(1139,15174)

Included effects: Entity, Time
DiD Coefficient (Treated_Post): 0.0121
P-value: 0.0237
Clustered Standard Error: 0.0053
```

```
           variable       VIF
0   C(YEAR)[T.2022]       inf
1   C(YEAR)[T.2023]       inf
2   C(YEAR)[T.2020]       inf
3          PostYear       inf
4   C(YEAR)[T.2021]       inf
5      Treated_Post  8.384447
6      TreatedGroup  7.145883
7          COSTT4_A  6.266131
8              UGDS  1.604556
9   C(YEAR)[T.2019]  1.567301
10  C(YEAR)[T.2018]  1.559829
11  C(YEAR)[T.2017]  1.546086
12  C(YEAR)[T.2016]  1.530682
13  C(YEAR)[T.2015]  1.516542
14  C(YEAR)[T.2014]  1.502341
15  C(YEAR)[T.2013]  1.488801
16  C(YEAR)[T.2012]  1.469565
17  C(YEAR)[T.2011]  1.456006
18  C(YEAR)[T.2010]  1.442636

VIF > 5 = mild concern
VIF > 10 = serious multicollinearity
```