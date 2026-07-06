import numpy as np
import pandas as pd
from scipy import stats


def analyze_text_lengths(df: pd.DataFrame, text_column: str, label_column: str):
    if text_column not in df.columns or label_column not in df.columns:
        raise KeyError("Specified columns do not exist in the DataFrame.")

    df_analysis = df.copy()
    df_analysis['word_count'] = df_analysis[text_column].astype(str).apply(lambda x: len(x.split()))

    print("\n=== Text Length Distribution by Class ===")
    summary = df_analysis.groupby(label_column)['word_count'].agg(['mean', 'std', 'min', 'max', 'count'])
    print(summary)

    unique_labels = df_analysis[label_column].unique()
    if len(unique_labels) == 2:
        group1 = df_analysis[df_analysis[label_column] == unique_labels[0]]['word_count']
        group2 = df_analysis[df_analysis[label_column] == unique_labels[1]]['word_count']

        t_stat, p_val = stats.ttest_ind(group1, group2, equal_var=False)
        print("\n--- Two-Sample T-Test (Word Count Differences) ---")
        print(f"T-statistic: {t_stat:.4f}")
        print(f"P-value: {p_val:.4f}")
        if p_val < 0.05:
            print("Conclusion: The difference in feedback text length between groups is statistically significant (p < 0.05).")
        else:
            print("Conclusion: No statistically significant difference in text length found between groups (p >= 0.05).")

    return summary


def compute_mcnemar_test(y_true, preds_model_a, preds_model_b):
    a_correct = (preds_model_a == y_true)
    b_correct = (preds_model_b == y_true)

    yes_yes = np.sum(a_correct & b_correct)
    yes_no = np.sum(a_correct & ~b_correct)
    no_yes = np.sum(~a_correct & b_correct)
    no_no = np.sum(~a_correct & ~b_correct)

    print("\n=== McNemar's Test Contingency Table ===")
    print(f"Both Correct (a+/b+): {yes_yes}")
    print(f"Model A Correct / Model B Wrong (a+/b-): {yes_no}")
    print(f"Model A Wrong / Model B Correct (a-/b+): {no_yes}")
    print(f"Both Wrong (a-/b-): {no_no}")

    if (yes_no + no_yes) == 0:
        print("Models made identical errors. McNemar's test cannot compute unique variance.")
        return 1.0

    stat = (abs(yes_no - no_yes) - 1) ** 2 / (yes_no + no_yes)
    p_value = stats.chi2.sf(stat, df=1)

    print("\n--- Significance Test Analysis ---")
    print(f"Chi-Squared Statistic: {stat:.4f}")
    print(f"P-value: {p_value:.4f}")
    if p_value < 0.05:
        print("Conclusion: The difference in performance between the two models is statistically significant (p < 0.05).")
    else:
        print("Conclusion: The performance difference could be due to random chance (p >= 0.05).")

    return p_value