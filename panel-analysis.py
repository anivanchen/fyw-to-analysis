import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import shapiro
from statsmodels.stats.diagnostic import het_breuschpagan
import os
from linearmodels import PanelOLS

def calculate_vif(df, formula, sort):
    try:
        model_data = smf.ols(formula, data=df).data
    except Exception as e:
        print(f"Error parsing formula or accessing data: {e}")
        return None

    # Get processed numerical matrix of independent variables
    exog_data = model_data.exog

    # Get names corresponding to the columns in exog_data
    exog_names = model_data.xnames

    exog_to_vif = exog_data
    exog_names_to_vif = list(exog_names)

    if 'Intercept' in exog_names_to_vif:
        intercept_index = exog_names_to_vif.index('Intercept')
        exog_to_vif = exog_data[:, np.arange(exog_data.shape[1]) != intercept_index]
        exog_names_to_vif.pop(intercept_index)

    if exog_to_vif.shape[1] == 0:
        print("No predictor variables found after handling intercept.")
        return pd.DataFrame(columns=['variable', 'VIF'])

    vif_results = pd.DataFrame()
    vif_results["variable"] = exog_names_to_vif
    vif_results["VIF"] = [variance_inflation_factor(exog_to_vif, i) for i in range(exog_to_vif.shape[1])]

    if (sort):
        vif_results = vif_results.sort_values(by="VIF", ascending=False).reset_index(drop=True)
        if (vif_results["VIF"] > 5).any():
            print("Warning: Some variables have VIF > 5 (potential multicollinearity).")

    return vif_results

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

'''
"ADMCON7 indicates if test scores are required, recommended,
considered but not required, or neither required nor recommended."
1: Required
2: Recommended
3: Neither Required nor Recommended
4: Do Not Know
5: Test-Optional
'''
admcon7_map = {1: 'Required', 2: 'Recommended', 3: 'Neither', 4: 'Do Not Know', 5: 'Test_Optional'}
df['TOS_CATEGORY'] = df['ADMCON7'].map(admcon7_map)

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

# Identifies if a institution had adopted a test-optional or test-blind policy after 2020. (1); (0) if not.
df['TreatedGroup'] = df['UNITID'].apply(lambda x: 1 if x in treated_institutions else 0)

# Identifies the time period after the treatment has taken effect. (1) is post-2020, (0) is pre-2020.
df['PostYear'] = df['YEAR'].apply(lambda x: 1 if x >= 2020 else 0)

# Create the Interaction Term for DiD
df['Treated_Post'] = df['TreatedGroup'] * df['PostYear']

treated_counts = df['TreatedGroup'].value_counts()
print("\n=== Treatment Group Counts ===")
print(treated_counts)
print(f"Treatment group percentage: {(treated_counts.get(1, 0) / len(df)) * 100:.2f}%")

treated_post_counts = df['Treated_Post'].value_counts()
print("\n=== Treated × Post Period Counts ===")
print(treated_post_counts)
print(f"Treated × Post observations percentage: {(treated_post_counts.get(1, 0) / len(df)) * 100:.2f}%")

# Cross-tabulation to see the full treatment/period matrix
print("\n=== Cross-tabulation of Treatment and Period ===")
print(pd.crosstab(df['TreatedGroup'], df['PostYear'], margins=True, margins_name="Total"))

treated_institutions_count = df[df['TreatedGroup'] == 1]['UNITID'].nunique()
control_institutions_count = df[df['TreatedGroup'] == 0]['UNITID'].nunique()
print(f"\nNumber of unique institutions in treatment group: {treated_institutions_count}")
print(f"Number of unique institutions in control group: {control_institutions_count}")

# --- Prepare Data for PanelOLS ---
# PanelOLS requires a MultiIndex: [entity_id, time_id]
df = df.set_index(['UNITID', 'YEAR'])

outcomes = ['PCTPELL', 'UGDS_BLACK', 'UGDS_HISP', 'UGDS_WHITE', 'UGDS_ASIAN', 'ADM_RATE', 'SAT_AVG', 'PCTFLOAN', 'DEBT_MDN', 'RET_FT4',
            'WDRAW_ORIG_YR2_RT', 'LO_INC_WDRAW_ORIG_YR2_RT', 'MD_INC_WDRAW_ORIG_YR2_RT', 'HI_INC_WDRAW_ORIG_YR2_RT', 'PELL_WDRAW_ORIG_YR2_RT', 'NOPELL_WDRAW_ORIG_YR2_RT',
            'WDRAW_ORIG_YR4_RT', 'LO_INC_WDRAW_ORIG_YR4_RT', 'MD_INC_WDRAW_ORIG_YR4_RT', 'HI_INC_WDRAW_ORIG_YR4_RT', 'PELL_WDRAW_ORIG_YR4_RT', 'NOPELL_WDRAW_ORIG_YR4_RT',
            'C100_4', 'C150_4', 'C150_4_PELL', 'C150_4_NOLOANNOPELL', 'C150_4_LOANNOPELL'
            ]

for outcome in outcomes: 

    os.makedirs(f'outputs/{outcome}', exist_ok=True)

    time_varying_controls = ['UGDS', 'COSTT4_A']

    formula = f"{outcome} ~ Treated_Post"
    if time_varying_controls: formula += " + " + " + ".join(time_varying_controls)
    formula = formula + ' + EntityEffects + TimeEffects'

    print("\nPanelOLS Formula:")
    print(formula)

    print("Columns in df before PanelOLS call:", df.columns)
    print("Sample data (first 5 rows) before PanelOLS call:\n", df[['TreatedGroup', 'PostYear', 'Treated_Post', outcome] + time_varying_controls].head()) # Check relevant columns

    # --- Run the Panel DiD Regression ---
    print(f"\n=== Panel DiD Results for {outcome} ===")

    # Use PanelOLS from linearmodels
    model = PanelOLS.from_formula(formula=formula, data=df)

    # Fit the model with clustered standard errors at the institution level
    results = model.fit(cov_type='clustered', cluster_entity=True)

    print(results)

    # --- Interpretation of Panel DiD Results (using linearmodels output) ---
    # Look at the 'Treated_Post' row in the results table.
    # The 'Coef.' column is your DiD estimate.
    # The 'P>|t|' column tells you its statistical significance.

    print(f"\n=== Interpretation of Treated_Post for {outcome} ===")
    if 'Treated_Post' in results.params:
        did_coef = results.params['Treated_Post']
        did_pvalue = results.pvalues['Treated_Post']
        std_err = results.std_errors['Treated_Post']

        print(f"DiD Coefficient (Treated_Post): {did_coef:.4f}")
        print(f"P-value: {did_pvalue:.4f}")
        print(f"Clustered Standard Error: {std_err:.4f}")

        interpretation = f"The estimated impact of adopting a test-optional policy (compared to remaining test-required) on the {outcome} is a change of {did_coef:.4f} percentage points in the post-2020 period, relative to the control group, after controlling for institution fixed effects, year fixed effects, and {', '.join(time_varying_controls if time_varying_controls else ['other time-varying factors'])}."

        print("\nInterpretation:")
        print(interpretation)

        if did_pvalue < 0.05:
            print(f"This effect is statistically significant at the 5% level (p = {did_pvalue:.4f}).")
            if did_coef < 0:
                print(f"Specifically, being in the treated group is associated with a statistically significant decrease in {outcome} in the post-2020 period relative to the control group's change.")
            else:
                print(f"Specifically, being in the treated group is associated with a statistically significant increase in {outcome} in the post-2020 period relative to the control group's change.")
        else:
            print(f"This effect is NOT statistically significant at the 5% level (p = {did_pvalue:.4f}).")
            print(f"There is no statistically significant evidence that the change in {outcome} for the treated group differed from the change for the control group in the post-2020 period, after controlling for other factors.")

        results_filename = f'outputs/{outcome}/did_results.txt'
        with open(results_filename, 'w') as f:
            f.write(str(results))
            f.write(f"\nDiD Coefficient (Treated_Post): {did_coef:.4f}\n")
            f.write(f"P-value: {did_pvalue:.4f}\n")
            f.write(f"Clustered Standard Error: {std_err:.4f}\n")

            interpretation = f"The estimated impact of adopting a test-optional policy (compared to remaining test-required) on the {outcome} is a change of {did_coef:.4f} percentage points in the post-2020 period, relative to the control group, after controlling for institution fixed effects, year fixed effects, and {', '.join(time_varying_controls if time_varying_controls else ['other time-varying factors'])}."

            f.write("\nInterpretation:")
            f.write(interpretation)

            if did_pvalue < 0.05:
                f.write(f"\nThis effect is statistically significant at the 5% level (p = {did_pvalue:.4f}).")
                if did_coef < 0:
                    f.write(f"\nSpecifically, being in the treated group is associated with a statistically significant decrease in {outcome} in the post-2020 period relative to the control group's change.")
                else:
                    f.write(f"\nSpecifically, being in the treated group is associated with a statistically significant increase in {outcome} in the post-2020 period relative to the control group's change.")
            else:
                f.write(f"\nThis effect is NOT statistically significant at the 5% level (p = {did_pvalue:.4f}).")
                f.write(f"\nThere is no statistically significant evidence that the change in {outcome} for the treated group differed from the change for the control group in the post-2020 period, after controlling for other factors.")

        print(f"\nPanelOLS results saved to {results_filename}")

    else:
        print("The 'Treated_Post' term was not found in the model results. Check formula and variable names.")


    # --- VIF Calculation ---

    print(f"\n=== VIFs (Pooled OLS Proxy for Panel Predictors) ===")

    pooled_vif_formula = f"{outcome} ~ TreatedGroup + PostYear + Treated_Post"
    if time_varying_controls:
        pooled_vif_formula += " + " + " + ".join(time_varying_controls)
    pooled_vif_formula += " + C(YEAR)"

    try:
        df_reset = df.reset_index()
        vif_vars = [outcome, 'TreatedGroup', 'PostYear', 'Treated_Post'] + time_varying_controls + ['YEAR']
        df_for_vif = df_reset[vif_vars].dropna()

        if df_for_vif.empty:
            print("DataFrame is empty after dropping NaNs for VIF calculation. Cannot calculate VIFs.")
        else:
            vif_results_did = calculate_vif(df_for_vif, pooled_vif_formula, True)

            results_filename = f'outputs/{outcome}/vif_results.txt'
            with open(results_filename, 'w') as f:
                f.write(vif_results_did.to_string())
                f.write('\n\nVIF > 5 = mild concern')
                f.write('\nVIF > 10 = serious multicollinearity')
                f.write("\nNote: High/Infinite VIFs for TreatedGroup, PostYear, Treated_Post are expected in this pooled context.")
                f.write("\nThe Panel FE estimator handles this structural collinearity differently.")
                f.write("\nFocus interpretation on VIFs of other included covariates (e.g., UGDS).")
            
            print(vif_results_did)
            print('\n\nVIF > 5 = mild concern')
            print('\nVIF > 10 = serious multicollinearity')
            print("\nNote: High/Infinite VIFs for TreatedGroup, PostYear, Treated_Post are expected in this pooled context.")
            print("\nThe Panel FE estimator handles this structural collinearity differently.")
            print("\nFocus interpretation on VIFs of other included covariates (e.g., UGDS).")

    except Exception as e:
        print(f"Error calculating VIF: {e}")

    # --- Plotting the Time Trends (with Overall Average) ---

    plot_df = df.reset_index()[['YEAR', outcome, 'TreatedGroup']].copy()

    # Calculate the overall average trend across all institutions for each year
    overall_trend_df = plot_df.groupby('YEAR')[outcome].mean().reset_index()
    overall_trend_df['TreatedGroup'] = 'Overall Average'

    # Rename the existing TreatedGroup values for clarity in the legend
    plot_df['TreatedGroup'] = plot_df['TreatedGroup'].map({0: 'Control Group', 1: 'Treated Group'})

    combined_plot_df = pd.concat([plot_df, overall_trend_df], ignore_index=True)

    group_order = ['Control Group', 'Treated Group', 'Overall Average']

    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    vline_handle = ax.axvline(x=2020, color='red', linestyle='--', label='Approximate Policy Shift')

    sns.lineplot(data=combined_plot_df,
                x='YEAR',
                y=outcome,
                hue='TreatedGroup', # Use the combined category column for color
                hue_order=group_order, # Apply the specified order
                marker='o',
                ax=ax)

    # Customize the plot title and labels
    plt.title(f'{outcome} Over Time by Group and Overall Average') # Updated Title
    plt.xlabel('Year')
    plt.ylabel(outcome) # Use outcome variable name
    plt.grid(True)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, title='Group / Event')

    trend_plot_filename = f'outputs/{outcome}/trend_plot.png'
    plt.savefig(trend_plot_filename, dpi = 200)
    plt.close()
    print(f"Trend plot saved to {trend_plot_filename}")

    # --- Residual Diagnostics (Normality, Heteroskedasticity, Outliers) ---

    print('\n === Panel Residual Diagnostics ===')

    if results: # Check if model fitting was successful
        panel_residuals = results.resids
        fitted_values_panel = results.fitted_values # Use fitted values from panel model

        # --- Plot Residuals vs Fitted ---
        try:
            plt.scatter(fitted_values_panel, panel_residuals, alpha=0.5)
            plt.axhline(0, color='red', linestyle='--')
            plt.title('Panel Residuals vs Fitted Values')
            plt.xlabel('Fitted Values')
            plt.ylabel('Residuals')
            plt.grid(True)
            plt.tight_layout()
            resid_vs_fitted_plot = f'outputs/{outcome}/residuals_vs_fitted_plot.png'
            plt.savefig(resid_vs_fitted_plot, dpi = 200)
            plt.close()
            print(f"Residuals vs Fitted plot saved to {resid_vs_fitted_plot}")

        except Exception as e:
            print(f"Could not plot Panel Residuals vs Fitted: {e}")

        # --- Plot Standardized Residuals ---
        try:
            # Standardize residuals (simple standardization)
            panel_resid_std = np.std(panel_residuals.dropna())
            if panel_resid_std > 0: # Avoid division by zero if all residuals are the same
                standardized_panel_residuals = panel_residuals / panel_resid_std

                plt.figure(figsize=(10, 6))
                plt.scatter(range(len(standardized_panel_residuals)), standardized_panel_residuals, alpha=0.5)
                plt.axhline(3, color='red', linestyle='--', label='+3 Std. Dev.')
                plt.axhline(-3, color='red', linestyle='--', label='-3 Std. Dev.')
                plt.title("Standardized Panel Residuals (Outliers marked)")
                plt.xlabel("Observation Index")
                plt.ylabel("Standardized Residuals")
                plt.grid(True)
                plt.legend()
                plt.tight_layout()

                standardized_plot_filename = f'outputs/{outcome}/standardized_residuals_plot.png'
                plt.savefig(standardized_plot_filename, dpi = 200)
                plt.close()
                print(f"Standardized residuals plot saved to {standardized_plot_filename}")

            else:
                print("\nCannot plot Standardized Residuals: Residual standard deviation is zero.")


        except Exception as e:
            print(f"Could not plot Standardized Panel Residuals: {e}")

        # --- Shapiro-Wilk Test (for Normality) ---
        print('\n --- Shapiro-Wilk Test on Panel Residuals ---')
        try:
            residuals_for_test = panel_residuals.dropna()
            if len(residuals_for_test) > 5000:
                print("Shapiro-Wilk test is not recommended for sample sizes > 5000.")
                print("Consider alternative tests or focus on visual inspection.")
            elif len(residuals_for_test) < 3:
                print("Not enough non-missing residuals to perform Shapiro-Wilk test.")
            else:
                stat, p_value_shapiro = shapiro(residuals_for_test)
                print(f"Shapiro-Wilk Test Statistic: {stat:.4f}, p-value: {p_value_shapiro:.4f}")
                if p_value_shapiro > 0.05:
                    print(f"p = {p_value_shapiro:.4f}. Fail to reject null (residuals might be normal - weak evidence).")
                else:
                    print(f"p = {p_value_shapiro:.4f}. Reject null (residuals are likely NOT normally distributed).")
                print("Note: Interpretation of normality tests on Panel FE residuals requires caution.")

        except Exception as e:
            print(f"Could not perform Shapiro-Wilk test: {e}")


    else:
        print("\nPanelOLS model fitting failed, cannot perform residual diagnostics.")