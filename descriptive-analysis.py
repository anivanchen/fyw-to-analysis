import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from scipy.stats import norm
from scipy import stats
import os

os.makedirs('outputs', exist_ok=True)

df = pd.read_csv('filtered_college_data.csv')
df['YEAR'] = pd.to_numeric(df['YEAR'])

i = 0
for col in df.columns:
    if col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            original_dtype = df[col].dtype
            df[col] = pd.to_numeric(df[col], errors='coerce')
            print(f" ({i}) - Converted '{col}' from {original_dtype} to {df[col].dtype}")
        else:
             print(f" ({i}) - '{col}' is already numeric ({df[col].dtype}), skipping conversion.")
    else:
        print(f" ({i}) - Warning: Column '{col}' not found in DataFrame.")
    i+=1

print("\n===== Missing Vals Distribution =====")
print(df.isnull().sum())
# df = df.dropna() # this would be too aggressive

print("\n===== Distribution of ADMCON7 over time =====")
print(df.groupby('YEAR')['ADMCON7'].value_counts(normalize=True).unstack())

'''
"ADMCON7 indicates if test scores are required, recommended,
considered but not required, or neither required nor recommended."
1: Required
2: Recommended
3: Neither Required nor Recommended
4: Do Not Know
5: Test-Optional
'''
admcon7_map = {1: 'Required', 2: 'Recommended', 3: 'Neither', 4: 'Do Not Know', 5: 'Test Optional', -1: 'Other', -2: 'Other'}
df['TOS_CATEGORY'] = df['ADMCON7'].map(admcon7_map)

print("\n===== Distribution of Grouped ADMCON7 over time =====")
distribution_data = df.groupby('YEAR')['TOS_CATEGORY'].value_counts(normalize=True).unstack()
print(distribution_data)

distribution_data = distribution_data.fillna(0)

ax1 = distribution_data.plot(kind='area', stacked=True, figsize=(12, 6))

plt.get_current_fig_manager().set_window_title('Admission Type Distribution Over Time')
ax1.set_title('Proportion of Institutions by Admission Type (ADMCON7) Over Time')
ax1.set_xlabel('Year')
ax1.set_ylabel('Proportion of Institutions')
ax1.axvline(x=2020, color='black', linestyle='--', label='Approximate Policy Shift')
ax1.set_ylim(0, 1)
ax1.legend(title='Admission Type', loc='lower left')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.xticks(distribution_data.index)
plt.tight_layout()

plot_filename = f'outputs/admit_type_dist.png'
plt.savefig(plot_filename, dpi = 200)
plt.close()
print(f"Plots saved to {plot_filename}")

# Identify institutions that became Test-Optional during the period
institutions = df['UNITID'].unique()
treated_institutions = []
policy_change_years = {}

pre_period_years = range(1996, 2020) # Years before widespread TO adoption
post_period_years = range(2020, 2024) # Years during widespread TO adoption

for unitid in institutions:
    inst_data = df[df['UNITID'] == unitid].sort_values('YEAR')

    pre_policy_status_modes = inst_data[inst_data['YEAR'].isin(pre_period_years)]['ADMCON7'].mode()
    post_policy_status_modes = inst_data[inst_data['YEAR'].isin(post_period_years)]['ADMCON7'].mode()

    if not pre_policy_status_modes.empty and not post_policy_status_modes.empty:
        # Check if it was NOT Test-Optional/Neither (5, 3) in the pre-period AND IS Test-Optional/Neither (5, 3) in the post-period
        # Checking if any mode in the pre-period was 1 or 2, and any mode in the post-period was 5 or 3
        if (pre_policy_status_modes.isin([1, 2]).any()) and (post_policy_status_modes.isin([5, 3])).any():
             treated_institutions.append(unitid)

outcomes = ['PCTPELL', 'UGDS_BLACK', 'UGDS_HISP', 'UGDS_WHITE', 'UGDS_ASIAN', 'ADM_RATE', 'SAT_AVG', 'PCTFLOAN', 'DEBT_MDN', 'RET_FT4',
            'WDRAW_ORIG_YR2_RT', 'LO_INC_WDRAW_ORIG_YR2_RT', 'MD_INC_WDRAW_ORIG_YR2_RT', 'HI_INC_WDRAW_ORIG_YR2_RT', 'PELL_WDRAW_ORIG_YR2_RT', 'NOPELL_WDRAW_ORIG_YR2_RT',
            'WDRAW_ORIG_YR4_RT', 'LO_INC_WDRAW_ORIG_YR4_RT', 'MD_INC_WDRAW_ORIG_YR4_RT', 'HI_INC_WDRAW_ORIG_YR4_RT', 'PELL_WDRAW_ORIG_YR4_RT', 'NOPELL_WDRAW_ORIG_YR4_RT',
            'C100_4', 'C150_4', 'C150_4_PELL', 'C150_4_NOLOANNOPELL', 'C150_4_LOANNOPELL'
            ]

for outcome in outcomes:

    os.makedirs(f'outputs/{outcome}', exist_ok=True)

    # =========================
    # === All Data Analysis ===
    # =========================

    df_filtered = df[df[outcome].notna()]

    valid_years = df_filtered.groupby('YEAR')[outcome].count()
    valid_years = valid_years[valid_years > 0].index

    df_filtered = df[df['YEAR'].isin(valid_years)]

    grouped = df.groupby('YEAR')[f'{outcome}']

    adm_rate_by_year_mean = grouped.mean()
    adm_rate_by_year_median = grouped.median()
    adm_rate_by_year_counts = grouped.count()
    adm_rate_by_year_stddev = grouped.std()

    adm_rate_clean = df[f'{outcome}'].dropna()
    adm_rate_mean_val = adm_rate_clean.mean()
    adm_rate_median_val = adm_rate_clean.median()

    se = adm_rate_by_year_stddev / np.sqrt(adm_rate_by_year_counts)

    # 95% confidence interval (approximate using z = 1.96)
    ci = 1.96 * se

    fig, ax = plt.subplots(1, 3, figsize=(24, 8), sharex=False)

    plt.get_current_fig_manager().set_window_title(f'{outcome} All Data Analysis')
    ax[0].plot(adm_rate_by_year_mean.index, adm_rate_by_year_mean.values, marker='o', linestyle='-')
    ax[0].errorbar(adm_rate_by_year_mean.index, adm_rate_by_year_mean.values, yerr=ci, fmt='o-', capsize=5, label='Mean ADM_RATE ± 95% CI')

    ax[0].plot(adm_rate_by_year_median.index, adm_rate_by_year_median.values, marker='o', linestyle='-')
    ax[0].errorbar(adm_rate_by_year_median.index, adm_rate_by_year_median.values, yerr=ci, fmt='o-', capsize=5, label='Median ADM_RATE ± 95% CI')

    ax[0].axvline(x=2020, color='black', linestyle='--', label='Approximate Policy Shift')

    ax[0].set_title(f'{outcome} by Year')
    ax[0].set_xlabel('Year')
    ax[0].set_ylabel(f'{outcome}')
    ax[0].grid(True)
    ax[0].legend()

    ax[1].hist(df[f'{outcome}'].dropna(), bins=20, color='skyblue', edgecolor='black')
    ax[1].axvline(adm_rate_mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {adm_rate_mean_val:.2f}')
    ax[1].axvline(adm_rate_median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {adm_rate_median_val:.2f}')

    ax[1].set_title(f'Distribution of {outcome}')
    ax[1].set_xlabel(f'{outcome}')
    ax[1].set_ylabel('Frequency')
    ax[1].grid(True)
    ax[1].legend()

    filtered_df = df[df[f'{outcome}'].notna()]
    year_counts = filtered_df['YEAR'].value_counts()
    valid_years = year_counts[year_counts > 0].sort_index

    filtered_df = filtered_df[filtered_df['YEAR'].isin(valid_years)]

    sns.boxplot(data=filtered_df, x='YEAR', y=f'{outcome}', ax=ax[2])
    ax[2].set_title(f'{outcome} Distribution by Year')
    ax[2].set_xlabel('Year')
    ax[2].set_ylabel(f'{outcome}')
    ax[2].grid(True)
    ax[2].set_xticklabels(ax[2].get_xticklabels(), rotation=45)

    plt.tight_layout()
    plot_filename = f'outputs/{outcome}/descriptive_all_data.png'
    plt.savefig(plot_filename, dpi = 200)
    plt.close()
    print(f"Plots saved to {plot_filename}")

    # ===================================
    # === Institutional Type Outcomes ===
    # ===================================

    # === Regression by Institution Type ===
    fig, ax = plt.subplots(1, 3, figsize=(24, 8), sharex=False)
    plt.get_current_fig_manager().set_window_title(f'{outcome} By Institution Analysis')

    control_mapping = {
        1: 'Public',
        2: 'Private Nonprofit',
        3: 'Private For-Profit'
    }

    df['CONTROL_LABEL'] = df['CONTROL'].map(control_mapping)

    grouped_by_type = df.groupby(['CONTROL_LABEL', 'YEAR'])[f'{outcome}'].mean().unstack()

    model_dict = {}
    for institution_type in ['Public', 'Private Nonprofit', 'Private For-Profit']:
        inst_data = df[df['CONTROL_LABEL'] == institution_type]

        grouped = inst_data.groupby('YEAR')[f'{outcome}'].mean()

        grouped = grouped.dropna()

        if len(grouped) > 1:
            X = grouped.index.values.reshape(-1, 1)
            
            lr_model = LinearRegression()

            model = lr_model.fit(X, grouped.values)

            model_dict[institution_type] = {
                'model': model,
                'predictions': model.predict(X),
                'slope': model.coef_[0],
                'intercept': model.intercept_
            }

            slope = model_dict[institution_type]['slope']
            intercept = model_dict[institution_type]['intercept']

            ax[0].plot(grouped.index, grouped.values, label=f'{institution_type} Institutions', marker='o')
            ax[0].plot(grouped.index, model_dict[institution_type]['predictions'], label=f'{institution_type} Trendline\n{slope:.4f}x + {intercept:.4f}', linestyle='--')
        else:
            print(f"Not enough data to fit a model for {institution_type}.")

    ax[0].axvline(x=2020, color='black', linestyle='--', label='Approximate Policy Shift')

    ax[0].set_title(f'{outcome} Trends by Institution Type')
    ax[0].set_xlabel('Year')
    ax[0].set_ylabel(f'{outcome}')
    ax[0].legend(loc='best')
    ax[0].grid(True)

    # --- Plot 2: Histogram Comparison by Institution Type ---
    # Plot distributions of the outcome by institution type
    sns.histplot(df[df['CONTROL_LABEL'] == 'Public'][f'{outcome}'], kde=True, label='Public', color='blue', stat='density', bins=30, ax=ax[1])
    sns.histplot(df[df['CONTROL_LABEL'] == 'Private Nonprofit'][f'{outcome}'], kde=True, label='Private Nonprofit', color='green', stat='density', bins=30, ax=ax[1])
    sns.histplot(df[df['CONTROL_LABEL'] == 'Private For-Profit'][f'{outcome}'], kde=True, label='Private For-Profit', color='orange', stat='density', bins=30, ax=ax[1])

    # Add normal distributions
    for label, color in [('Public', 'blue'), ('Private Nonprofit', 'green'), ('Private For-Profit', 'orange')]:
        data = df[df['CONTROL_LABEL'] == label][f'{outcome}'].dropna()

        mean = np.mean(data)
        std = np.std(data)

        x = np.linspace(data.min(), data.max(), 100)
        y = norm.pdf(x, mean, std)

        ax[1].plot(x, y, label=f'{label} Norm Dist', color=color, linestyle='--')

    ax[1].set_title(f'Distribution of {outcome} by Institution Type')
    ax[1].set_xlabel(f'{outcome}')
    ax[1].set_ylabel('Density')
    ax[1].legend(title='Institution Type')
    ax[1].grid(True)

    # --- Plot 3: Boxplot Comparison by Institution Type ---
    sns.boxplot(data=df, x='CONTROL_LABEL', y=f'{outcome}', ax=ax[2])
    ax[2].set_title(f'{outcome} Distribution by Institution Type')
    ax[2].set_xlabel('Institution Type')
    ax[2].set_ylabel(f'{outcome}')
    ax[2].grid(True)

    plt.tight_layout()
    plot_filename = f'outputs/{outcome}/descriptive_institutional_data.png'
    plt.savefig(plot_filename, dpi = 200)
    plt.close()
    print(f"Plots saved to {plot_filename}")

    # =================================================
    # === TO vs Non-TO trends/regression difference ===
    # =================================================

    fig, ax = plt.subplots(1, 3, figsize=(24, 8), sharex=False)
    plt.get_current_fig_manager().set_window_title(f'{outcome} To vs Non-TO Analysis')

    test_optional_institutions = df_filtered[df_filtered['UNITID'].isin(treated_institutions)]

    test_optional_grouped = test_optional_institutions.groupby('YEAR')[f'{outcome}'].mean()
    non_test_optional_grouped = df_filtered[~df_filtered['UNITID'].isin(treated_institutions)].groupby('YEAR')[f'{outcome}'].mean()

    X_test_optional = test_optional_grouped.index.values.reshape(-1, 1)
    X_non_test_optional = non_test_optional_grouped.index.values.reshape(-1, 1)

    test_lr_model = LinearRegression()
    non_test_lr_model = LinearRegression()

    lr_model_test_optional = test_lr_model.fit(X_test_optional, test_optional_grouped.values)
    lr_model_non_test_optional = non_test_lr_model.fit(X_non_test_optional, non_test_optional_grouped.values)

    test_optional_trend = lr_model_test_optional.predict(X_test_optional)
    non_test_optional_trend = lr_model_non_test_optional.predict(X_non_test_optional)

    slope_test_optional = lr_model_test_optional.coef_[0]
    intercept_test_optional = lr_model_test_optional.intercept_

    slope_non_test_optional = lr_model_non_test_optional.coef_[0]
    intercept_non_test_optional = lr_model_non_test_optional.intercept_

    # --- Plot 1: Trends of Test-Optional vs Non-Test-Optional Institutions ---
    # Plot the original grouped data for both test-optional and non-test-optional
    ax[0].plot(test_optional_grouped.index, test_optional_grouped.values, label='Test-Optional Institutions', marker='o', color='blue')
    ax[0].plot(non_test_optional_grouped.index, non_test_optional_grouped.values, label='Non-Test-Optional Institutions', marker='o', color='orange')

    ax[0].plot(test_optional_grouped.index, test_optional_trend, label=f'Test-Optional Trendline\n{slope_test_optional:.4f}x + {intercept_test_optional:.4f}', color='red', linestyle='--')
    ax[0].plot(non_test_optional_grouped.index, non_test_optional_trend, label=f'Non-Test-Optional Trendline\n{slope_non_test_optional:.4f}x + {intercept_non_test_optional:.4f}', color='green', linestyle='--')

    ax[0].axvline(x=2020, color='black', linestyle='--', label='Approximate Policy Shift')

    ax[0].set_title(f'{outcome} Trends: TO vs Non-TO Institutions')
    ax[0].set_xlabel('Year')
    ax[0].set_ylabel(f'{outcome}')
    ax[0].legend(loc='best')
    ax[0].grid(True)

    # --- Plot 2: TO vs Non-TO Histogram Difference ---
    sns.histplot(df[df['UNITID'].isin(treated_institutions)][outcome], kde=True, label='Test-Optional', color='blue', stat='density', bins=30, ax=ax[1])
    sns.histplot(df[~df['UNITID'].isin(treated_institutions)][outcome], kde=True, label='Non-Test-Optional', color='orange', stat='density', bins=30, ax=ax[1])

    # Add normal distributions for TO and Non-TO
    for group, color, label in [('Test-Optional', 'blue', 'Test-Optional'), ('Non-Test-Optional', 'orange', 'Non-Test-Optional')]:
        if group == 'Test-Optional':
            data = df[df['UNITID'].isin(treated_institutions)][outcome].dropna()
        else:
            data = df[~df['UNITID'].isin(treated_institutions)][outcome].dropna()

        mean = np.mean(data)
        std = np.std(data)

        x = np.linspace(data.min(), data.max(), 100)
        y = norm.pdf(x, mean, std)

        ax[1].plot(x, y, label=f'{label} Norm Dist', color=color, linestyle='--')

    ax[1].set_title(f'Distribution of {outcome} for TO vs Non-TO Institutions')
    ax[1].set_xlabel(f'{outcome}')
    ax[1].set_ylabel('Density')
    ax[1].legend(title='Admission Status')
    ax[1].grid(True)

    # --- Plot 3: Boxplot for Test-Optional vs Non-Test-Optional SAT Distribution ---

    palette = {
        "Test Optional": "blue", 
        "Required": "orange", 
        "Recommended": "green", 
        "Neither": "red", 
        "Do Not Know": "purple",
        "Other": "gray"
    }

    sns.boxplot(x='TOS_CATEGORY', y=outcome, data=df, ax=ax[2], palette=palette)
    ax[2].set_title(f'{outcome} Distribution by TO Status')
    ax[2].set_xlabel('Test-Optional Status')
    ax[2].set_ylabel(f'{outcome}')
    ax[2].grid(True)

    plt.tight_layout()
    plot_filename = f'outputs/{outcome}/descriptive_to_v_nto_data.png'
    plt.savefig(plot_filename, dpi = 200)
    plt.close()
    print(f"Plots saved to {plot_filename}")

    # ========================================
    # === Difference in summary statistics ===
    # ========================================

    # Save the regression results to a text file
    results_filename = f'outputs/{outcome}/sum_stats_t_test_results.txt'
    with open(results_filename, 'w') as f:

        # Filter data for pre/post 2020 for test-optional institutions
        pre_2020_test_optional = test_optional_institutions[test_optional_institutions['YEAR'] < 2020]
        post_2020_test_optional = test_optional_institutions[test_optional_institutions['YEAR'] >= 2020]

        pre_2020_mean = pre_2020_test_optional[outcome].mean()
        post_2020_mean = post_2020_test_optional[outcome].mean()

        t_stat, p_value = stats.ttest_ind(pre_2020_test_optional[outcome], post_2020_test_optional[outcome], nan_policy='omit')

        f.write('Pre 2020 Test Optional\n')
        f.write(pre_2020_test_optional[outcome].describe().to_string())
        f.write('\nPost 2020 Test Optional\n')
        f.write(post_2020_test_optional[outcome].describe().to_string())
        f.write('\n')
        f.write(f"\nPre-2020 Mean: {pre_2020_mean}\n")
        f.write(f"Post-2020 Mean: {post_2020_mean}\n")
        f.write('The null hypothesis is that there is no statistically significant difference between pre and post 2020 means.\n')
        f.write('The alpha is 0.05.\n')
        f.write(f"T-statistic: {t_stat}\n")
        f.write(f"P-value: {p_value}\n")

        if p_value < 0.05:
            f.write("The difference in means is statistically significant (p < 0.05).\n")
        else:
            f.write("The difference in means is not statistically significant (p >= 0.05).\n")