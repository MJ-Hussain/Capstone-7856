import numpy as np
import os
import pandas as pd
import re
from summary_extract import read_model_file, extract_model_info, process_summary_files

def score_parameter_significance(model_info, weight=0.25):
    """
    Score the significance of model parameters.
    Lower p-values get higher scores.
    
    Parameters:
    -----------
    model_info : dict
        Dictionary containing model information
    weight : float
        Weight for this score component
        
    Returns:
    --------
    float
        Weighted score for parameter significance
    """
    if 'optimal_parameters' not in model_info or not model_info['optimal_parameters']:
        return 0
    
    score = 0
    n_params = len(model_info['optimal_parameters'])
    
    for param_name, values in model_info['optimal_parameters'].items():
        # Skip mean parameter (mu) - focus on volatility parameters
        if param_name == 'mu':
            n_params -= 1
            continue
            
        p_value = values['p_value']
        
        # Score based on p-value significance levels
        if p_value < 0.01:  # Highly significant
            param_score = 1.0
        elif p_value < 0.05:  # Significant
            param_score = 0.75
        elif p_value < 0.1:  # Marginally significant
            param_score = 0.5
        else:  # Not significant
            param_score = 0.25
        
        score += param_score
    
    # Normalize by number of parameters
    if n_params > 0:
        score /= n_params
    
    return score * weight

def score_information_criteria(model_info, weight=0.20):
    """
    Score based on information criteria (AIC, BIC).
    Scores are relative to baseline values since these metrics
    are on different scales for different datasets.
    
    Parameters:
    -----------
    model_info : dict
        Dictionary containing model information
    weight : float
        Weight for this score component
        
    Returns:
    --------
    float
        Weighted score for information criteria
    """
    # Check if information criteria exist in the model info
    if not all(k in model_info for k in ['AIC', 'BIC', 'loglikelihood']):
        return 0
    
    # For AIC and BIC, lower is better
    # Convert to a 0-1 scale (this implementation assumes we know the range of values)
    # In a real application, you might want to normalize these across all models
    
    # Simple scaling - assuming values below 3.0 are good for AIC/BIC
    # Adjust these thresholds based on your specific data
    aic_score = max(0, 1 - (model_info['AIC'] / 5.0))
    bic_score = max(0, 1 - (model_info['BIC'] / 5.0))
    
    # For log-likelihood, higher (less negative) is better
    # Since log-likelihood scale depends on data, just use a placeholder formula
    # In practice, you'd normalize this across all models
    ll_score = 0.5  # Placeholder - real implementation would compare across models
    
    # Combine scores with internal weights
    combined_score = (0.4 * aic_score) + (0.4 * bic_score) + (0.2 * ll_score)
    
    return combined_score * weight

def score_residual_diagnostics(model_info, weight=0.25):
    """
    Score based on residual diagnostic tests:
    - Ljung-Box on residuals
    - Ljung-Box on squared residuals
    - ARCH LM tests
    
    Higher p-values (indicating no evidence against null hypothesis
    of no serial correlation or ARCH effects) get higher scores.
    
    Parameters:
    -----------
    model_info : dict
        Dictionary containing model information
    weight : float
        Weight for this score component
        
    Returns:
    --------
    float
        Weighted score for residual diagnostics
    """
    score = 0
    components = 0
    
    # Ljung-Box test on standardized residuals (want high p-values > 0.05)
    if 'ljung_box_residuals' in model_info and model_info['ljung_box_residuals']:
        lb_res_score = 0
        for lag, values in model_info['ljung_box_residuals'].items():
            p_value = values['p_value']
            # Higher p-values are better (no serial correlation)
            lb_res_score += min(1.0, p_value / 0.05) if p_value < 0.05 else 1.0
        
        lb_res_score /= len(model_info['ljung_box_residuals'])
        score += lb_res_score
        components += 1
    
    # Ljung-Box test on standardized squared residuals (want high p-values > 0.05)
    if 'ljung_box_squared_residuals' in model_info and model_info['ljung_box_squared_residuals']:
        lb_sq_res_score = 0
        for lag, values in model_info['ljung_box_squared_residuals'].items():
            p_value = values['p_value']
            # Higher p-values are better (no heteroskedasticity)
            lb_sq_res_score += min(1.0, p_value / 0.05) if p_value < 0.05 else 1.0
        
        lb_sq_res_score /= len(model_info['ljung_box_squared_residuals'])
        score += lb_sq_res_score
        components += 1
    
    # ARCH LM tests (want high p-values > 0.05)
    if 'arch_lm_tests' in model_info and model_info['arch_lm_tests']:
        arch_lm_score = 0
        for lag, values in model_info['arch_lm_tests'].items():
            p_value = values['p_value']
            # Higher p-values are better (no ARCH effects)
            arch_lm_score += min(1.0, p_value / 0.05) if p_value < 0.05 else 1.0
        
        arch_lm_score /= len(model_info['arch_lm_tests'])
        score += arch_lm_score
        components += 1
    
    # Normalize by number of components
    if components > 0:
        score /= components
    
    return score * weight

def score_stability(model_info, weight=0.15):
    """
    Score based on Nyblom stability test.
    Lower test statistics (relative to critical values) get higher scores.
    
    Parameters:
    -----------
    model_info : dict
        Dictionary containing model information
    weight : float
        Weight for this score component
        
    Returns:
    --------
    float
        Weighted score for stability
    """
    if 'nyblom_stability' not in model_info or 'nyblom_critical_values' not in model_info:
        return 0
    
    # Extract joint statistic and its critical value at 5%
    if 'joint_statistic' not in model_info['nyblom_stability'] or 'joint' not in model_info['nyblom_critical_values']:
        return 0
    
    joint_stat = model_info['nyblom_stability']['joint_statistic']
    critical_val_5pct = model_info['nyblom_critical_values']['joint']['5%']
    
    # Score based on how much below critical value the statistic is
    if joint_stat <= critical_val_5pct:
        # Below critical value is good (stable model)
        score = 1.0 - (joint_stat / critical_val_5pct)
    else:
        # Above critical value means unstable model
        # Higher values get exponentially lower scores
        score = max(0, np.exp(-(joint_stat - critical_val_5pct)))
    
    # Limit score between 0 and 1
    score = max(0, min(1, score))
    
    return score * weight

def score_sign_bias(model_info, weight=0.10):
    """
    Score based on sign bias tests.
    Higher p-values (indicating no asymmetric effects) get higher scores.
    
    Parameters:
    -----------
    model_info : dict
        Dictionary containing model information
    weight : float
        Weight for this score component
        
    Returns:
    --------
    float
        Weighted score for sign bias tests
    """
    if 'sign_bias_test' not in model_info or not model_info['sign_bias_test']:
        return 0
    
    score = 0
    n_tests = len(model_info['sign_bias_test'])
    
    for test_name, values in model_info['sign_bias_test'].items():
        p_value = values['prob']
        
        # Higher p-values are better (no sign bias)
        if p_value >= 0.05:
            test_score = 1.0
        else:
            # Partial score for borderline cases
            test_score = p_value / 0.05
        
        score += test_score
    
    # Normalize by number of tests
    score /= n_tests
    
    return score * weight

def score_goodness_of_fit(model_info, weight=0.05):
    """
    Score based on Pearson goodness-of-fit test.
    Higher p-values get higher scores.
    
    Parameters:
    -----------
    model_info : dict
        Dictionary containing model information
    weight : float
        Weight for this score component
        
    Returns:
    --------
    float
        Weighted score for goodness-of-fit
    """
    if 'pearson_gof' not in model_info or not model_info['pearson_gof']:
        return 0
    
    score = 0
    n_groups = len(model_info['pearson_gof'])
    
    for group, values in model_info['pearson_gof'].items():
        p_value = values['p_value']
        
        # Higher p-values are better for goodness-of-fit
        if p_value >= 0.05:
            group_score = 1.0
        else:
            # Partial score for borderline cases
            group_score = p_value / 0.05
        
        score += group_score
    
    # Normalize by number of groups
    score /= n_groups
    
    return score * weight

def calculate_overall_score(model_info):
    """
    Calculate the overall score for a GARCH model based on multiple criteria.
    
    Parameters:
    -----------
    model_info : dict
        Dictionary containing model information extracted from summary
    
    Returns:
    --------
    float
        Overall score between 0 and 1
    dict
        Component scores
    """
    # Calculate component scores
    param_score = score_parameter_significance(model_info)
    info_criteria_score = score_information_criteria(model_info)
    resid_diag_score = score_residual_diagnostics(model_info)
    stability_score = score_stability(model_info)
    sign_bias_score = score_sign_bias(model_info)
    gof_score = score_goodness_of_fit(model_info)
    
    # Calculate overall score
    overall_score = param_score + info_criteria_score + resid_diag_score + stability_score + sign_bias_score + gof_score
    
    # Store component scores
    component_scores = {
        'parameter_significance': param_score,
        'information_criteria': info_criteria_score,
        'residual_diagnostics': resid_diag_score,
        'stability': stability_score,
        'sign_bias': sign_bias_score,
        'goodness_of_fit': gof_score,
        'overall': overall_score
    }
    
    return overall_score, component_scores

def extract_model_details_from_filename(filename):
    """
    Extract series number and model number from filename.
    
    Parameters:
    -----------
    filename : str
        Filename like 'series_3_model_144_summary.txt'
    
    Returns:
    --------
    tuple
        (series_number, model_number)
    """
    pattern = r'series_(\d+)_model_(\d+)_summary.txt'
    match = re.match(pattern, filename)
    if match:
        series_num = int(match.group(1))
        model_num = int(match.group(2))
        return series_num, model_num
    return None, None

def model_info_to_dataframe(model_name, model_info, component_scores):
    """
    Convert model information and scores to a pandas DataFrame row.
    
    Parameters:
    -----------
    model_name : str
        Name of the model
    model_info : dict
        Dictionary containing model information
    component_scores : dict
        Dictionary containing component scores
    
    Returns:
    --------
    dict
        Dictionary representing a row in the DataFrame
    """
    # Extract series and model number from filename
    series_num, model_num = extract_model_details_from_filename(model_name)
    
    row = {
        'series': series_num if series_num else 'Unknown',
        'model': model_num if model_num else 'Unknown',
        'garch_type': model_info.get('garch_type', ''),
        'garch_p': model_info.get('garch_p', ''),
        'garch_q': model_info.get('garch_q', ''),
        'arma_p': model_info.get('arma_p', ''),
        'arma_q': model_info.get('arma_q', ''),
        'distribution': model_info.get('distribution', ''),
        'param_significance': component_scores['parameter_significance'],
        'info_criteria': component_scores['information_criteria'],
        'residual_diagnostics': component_scores['residual_diagnostics'],
        'stability': component_scores['stability'],
        'sign_bias': component_scores['sign_bias'],
        'goodness_of_fit': component_scores['goodness_of_fit'],
        'total_score': component_scores['overall'],
        'AIC': model_info.get('AIC', ''),
        'BIC': model_info.get('BIC', ''),
        'loglikelihood': model_info.get('loglikelihood', '')
    }
    
    return row

def evaluate_models_df(models_info):
    """
    Evaluate multiple models and return a DataFrame with all model information and scores.
    
    Parameters:
    -----------
    models_info : dict
        Dictionary with model names as keys and model info as values
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing all model information and scores
    """
    rows = []
    
    for model_name, model_info in models_info.items():
        overall_score, component_scores = calculate_overall_score(model_info)
        row = model_info_to_dataframe(model_name, model_info, component_scores)
        rows.append(row)
    
    # Convert to DataFrame
    df = pd.DataFrame(rows)
    
    # Sort by overall score in descending order
    df = df.sort_values('overall_score', ascending=False)
    
    return df

def get_top_models_df(df, top_n=3):
    """
    Return the top N models from a DataFrame.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing model information and scores
    top_n : int
        Number of top models to return
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing the top N models
    """
    return df.head(top_n)

def get_model_strengths(df):
    """
    Identify key strengths of each model in the DataFrame.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing model information and scores
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added column for model strengths
    """
    # Score component columns
    score_cols = [
        'param_significance', 'info_criteria', 'residual_diagnostics',
        'stability', 'sign_bias', 'goodness_of_fit'
    ]
    
    # Map column names to more readable names
    col_names = {
        'param_significance': 'Parameter significance',
        'info_criteria': 'Information criteria',
        'residual_diagnostics': 'Residual diagnostics',
        'stability': 'Parameter stability',
        'sign_bias': 'Symmetric effects',
        'goodness_of_fit': 'Distribution fit'
    }
    
    strengths = []
    
    for _, row in df.iterrows():
        # Get the top 3 score components
        scores = {col: row[col] for col in score_cols}
        top_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
        
        # Create a string of top strengths
        strength_str = ", ".join([f"{col_names[col]} ({score:.3f})" for col, score in top_scores])
        strengths.append(strength_str)
    
    # Add strengths column to DataFrame
    result_df = df.copy()
    result_df['key_strengths'] = strengths
    
    return result_df

def get_model_summary_df(models_info):
    """
    Create a summary DataFrame with key model information.
    
    Parameters:
    -----------
    models_info : dict
        Dictionary with model names as keys and model info as values
    
    Returns:
    --------
    pandas.DataFrame
        Summary DataFrame with key model information
    """
    rows = []
    
    for model_name, model_info in models_info.items():
        overall_score, component_scores = calculate_overall_score(model_info)
        row = model_info_to_dataframe(model_name, model_info, component_scores)
        rows.append(row)
    
    # Convert to DataFrame
    df = pd.DataFrame(rows)
    
    # Sort by total_score in descending order
    df = df.sort_values('total_score', ascending=False)
    
    # Add model strengths
    df = get_model_strengths(df)
    
    # Define column order
    columns = [
        'series', 'garch_type', 'garch_p', 'garch_q', 
        'arma_p', 'arma_q', 'distribution',
        'param_significance', 'info_criteria', 'residual_diagnostics',
        'stability', 'sign_bias', 'goodness_of_fit',
        'total_score', 'key_strengths', 'AIC', 'BIC', 'loglikelihood'
    ]
    
    # Select only columns that exist in the DataFrame
    existing_cols = [col for col in columns if col in df.columns]
    
    # Return the DataFrame with columns in the desired order
    return df[existing_cols]

def get_top_models_by_series(df, top_n=3):
    """
    Get top N models for each series.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with model information
    top_n : int
        Number of top models to return per series
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing top models for each series
    """
    # Group by series and get top models
    top_models = df.groupby('series').apply(lambda x: x.nlargest(top_n, 'total_score')).reset_index(drop=True)
    return top_models

def evaluate_models(models_info, top_n=3):
    """
    Evaluate multiple models and return the top N models.
    
    Parameters:
    -----------
    models_info : dict
        Dictionary with model names as keys and model info as values
    top_n : int
        Number of top models to return
    
    Returns:
    --------
    list
        List of tuples (model_name, score, component_scores) for top N models
    """
    scores = []
    
    for model_name, model_info in models_info.items():
        overall_score, component_scores = calculate_overall_score(model_info)
        scores.append((model_name, overall_score, component_scores))
    
    # Sort by overall score in descending order
    scores.sort(key=lambda x: x[1], reverse=True)
    
    # Return top N models
    return scores[:top_n]

def normalize_scores_across_models(models_info):
    """
    Normalize information criteria scores across all models.
    This improves comparison between models.
    
    Parameters:
    -----------
    models_info : dict
        Dictionary with model names as keys and model info as values
    
    Returns:
    --------
    dict
        The same dictionary with normalized AIC and BIC values
    """
    # Extract AIC and BIC values
    aic_values = []
    bic_values = []
    ll_values = []
    
    for model_name, model_info in models_info.items():
        if 'AIC' in model_info:
            aic_values.append(model_info['AIC'])
        if 'BIC' in model_info:
            bic_values.append(model_info['BIC'])
        if 'loglikelihood' in model_info:
            ll_values.append(model_info['loglikelihood'])
    
    # Calculate min and max values
    if aic_values:
        min_aic, max_aic = min(aic_values), max(aic_values)
        aic_range = max_aic - min_aic
    
    if bic_values:
        min_bic, max_bic = min(bic_values), max(bic_values)
        bic_range = max_bic - min_bic
    
    if ll_values:
        min_ll, max_ll = min(ll_values), max(ll_values)
        ll_range = max_ll - min_ll
    
    # Normalize values in place
    for model_name, model_info in models_info.items():
        if 'AIC' in model_info and aic_range > 0:
            model_info['AIC_normalized'] = 1 - ((model_info['AIC'] - min_aic) / aic_range)
        
        if 'BIC' in model_info and bic_range > 0:
            model_info['BIC_normalized'] = 1 - ((model_info['BIC'] - min_bic) / bic_range)
            
        if 'loglikelihood' in model_info and ll_range > 0:
            model_info['loglikelihood_normalized'] = (model_info['loglikelihood'] - min_ll) / ll_range
    
    return models_info

# Example usage
if __name__ == "__main__":
    # Define base directory using absolute path
    base_dir = os.path.join(os.path.dirname(__file__), "model_summaries")
    
    # Process all summary files in the directory
    models_info = process_summary_files(base_dir)
    
    if models_info:
        # Create summary DataFrame
        all_models_df = get_model_summary_df(models_info)
        
        print("All Models Summary:")
        pd.set_option('display.max_columns', None)  # Show all columns
        pd.set_option('display.width', 1000)       # Wide display
        print(all_models_df.to_string())
        
        # Get top models for each series
        top_models_df = get_top_models_by_series(all_models_df, top_n=3)
        
        print("\nTop Models by Series:")
        print(top_models_df.to_string())
        
        # Save DataFrames to CSV
        output_dir = os.path.dirname(base_dir)
        all_models_df.to_csv(os.path.join(output_dir, "all_models_evaluation.csv"), index=False)
        top_models_df.to_csv(os.path.join(output_dir, "top_models_by_series.csv"), index=False)
        
        print(f"\nResults saved to {output_dir}")
