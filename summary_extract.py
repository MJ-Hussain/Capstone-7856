import re
import os

def read_model_file(file_path):
    """
    Read a model summary file and return its content.
    
    Parameters:
    -----------
    file_path : str
        Path to the model summary text file
    
    Returns:
    --------
    str or None
        The content of the file, or None if an error occurs
    """
    try:
        with open(file_path, 'r') as file:
            content = file.read()
        return content
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return None
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None

def extract_model_info(content):
    """
    Extract key information from GARCH model summary content.
    
    Parameters:
    -----------
    content : str
        Content of the model summary text file
    
    Returns:
    --------
    dict
        Dictionary containing model type, order, ARMA order, distribution, and diagnostic tests
    """
    try:
        # Initialize dictionary to store extracted information
        model_info = {}
        
        # Extract GARCH model type (e.g., sGARCH, eGARCH, etc.)
        garch_match = re.search(r"GARCH Model\s*:\s*(\w+)\((\d+),(\d+)\)", content)
        if garch_match:
            model_info['garch_type'] = garch_match.group(1)
            model_info['garch_p'] = int(garch_match.group(2))
            model_info['garch_q'] = int(garch_match.group(3))
        
        # Extract Mean Model (ARFIMA order)
        arma_match = re.search(r"Mean Model\s*:\s*ARFIMA\((\d+),(\d+),(\d+)\)", content)
        if arma_match:
            model_info['arma_p'] = int(arma_match.group(1))
            model_info['arma_d'] = int(arma_match.group(2))
            model_info['arma_q'] = int(arma_match.group(3))
        
        # Extract Distribution
        dist_match = re.search(r"Distribution\s*:\s*(\w+)", content)
        if dist_match:
            model_info['distribution'] = dist_match.group(1)
            
        # Extract LogLikelihood value
        ll_match = re.search(r"LogLikelihood\s*:\s*([-\d.]+)", content)
        if ll_match:
            model_info['loglikelihood'] = float(ll_match.group(1))
            
        # Extract information criteria
        aic_match = re.search(r"Akaike\s*([-\d.]+)", content)
        if aic_match:
            model_info['AIC'] = float(aic_match.group(1))
            
        bic_match = re.search(r"Bayes\s*([-\d.]+)", content)
        if bic_match:
            model_info['BIC'] = float(bic_match.group(1))
        
        # Extract Optimal Parameters
        model_info['optimal_parameters'] = {}
        
        # Find the Optimal Parameters section
        opt_params_match = re.search(r"Optimal Parameters\s*\n-+\s*\n(.*?)(?:\n\n|\nRobust)", content, re.DOTALL)
        if opt_params_match:
            param_section = opt_params_match.group(1)
            param_lines = [line.strip() for line in param_section.split('\n') if line.strip()]
            
            # Skip the header line
            for line in param_lines[1:]:
                parts = re.split(r'\s+', line.strip())
                if len(parts) >= 5:
                    param_name = parts[0]
                    param_estimate = float(parts[1])
                    param_std_error = float(parts[2])
                    param_t_value = float(parts[3])
                    param_p_value = float(parts[4])
                    
                    model_info['optimal_parameters'][param_name] = {
                        'estimate': param_estimate,
                        'std_error': param_std_error,
                        't_value': param_t_value,
                        'p_value': param_p_value
                    }
        
        # Extract Robust Standard Errors
        model_info['robust_std_errors'] = {}
        
        # Find the Robust Standard Errors section
        robust_match = re.search(r"Robust Standard Errors:\s*\n(.*?)(?:\n\n|\nLogLikelihood)", content, re.DOTALL)
        if robust_match:
            robust_section = robust_match.group(1)
            robust_lines = [line.strip() for line in robust_section.split('\n') if line.strip()]
            
            # Skip the header line
            for line in robust_lines[1:]:
                parts = re.split(r'\s+', line.strip())
                if len(parts) >= 5:
                    param_name = parts[0]
                    param_estimate = float(parts[1])
                    param_std_error = float(parts[2])
                    param_t_value = float(parts[3])
                    param_p_value = float(parts[4])
                    
                    model_info['robust_std_errors'][param_name] = {
                        'estimate': param_estimate,
                        'std_error': param_std_error,
                        't_value': param_t_value,
                        'p_value': param_p_value
                    }
        
        # Extract Weighted Ljung-Box Test on Standardized Residuals
        model_info['ljung_box_residuals'] = {}
        
        ljung_box_match = re.search(r"Weighted Ljung-Box Test on Standardized Residuals\s*\n-+\s*\n(.*?)(?:\n\n|\nWeighted Ljung-Box Test on Standardized Squared Residuals)", content, re.DOTALL)
        if ljung_box_match:
            ljung_section = ljung_box_match.group(1)
            ljung_lines = [line.strip() for line in ljung_section.split('\n') if line.strip()]
            
            # Skip header lines and H0 line
            test_lines = [line for line in ljung_lines if re.match(r'Lag\[\d+', line)]
            for line in test_lines:
                parts = re.split(r'\s+', line.strip())
                if len(parts) >= 3:
                    lag_name = parts[0]
                    statistic = float(parts[1])
                    p_value = float(parts[2])
                    
                    model_info['ljung_box_residuals'][lag_name] = {
                        'statistic': statistic,
                        'p_value': p_value
                    }
        
        # Extract Weighted Ljung-Box Test on Standardized Squared Residuals
        model_info['ljung_box_squared_residuals'] = {}
        
        ljung_box_squared_match = re.search(r"Weighted Ljung-Box Test on Standardized Squared Residuals\s*\n-+\s*\n(.*?)(?:\n\n|\nWeighted ARCH LM Tests)", content, re.DOTALL)
        if ljung_box_squared_match:
            ljung_squared_section = ljung_box_squared_match.group(1)
            ljung_squared_lines = [line.strip() for line in ljung_squared_section.split('\n') if line.strip()]
            
            # Skip header line and d.o.f line
            test_lines = [line for line in ljung_squared_lines if re.match(r'Lag\[\d+', line)]
            for line in test_lines:
                parts = re.split(r'\s+', line.strip())
                if len(parts) >= 3:
                    lag_name = parts[0]
                    statistic = float(parts[1])
                    p_value = float(parts[2])
                    
                    model_info['ljung_box_squared_residuals'][lag_name] = {
                        'statistic': statistic,
                        'p_value': p_value
                    }
        
        # Extract ARCH LM Tests
        model_info['arch_lm_tests'] = {}
        
        arch_lm_match = re.search(r"Weighted ARCH LM Tests\s*\n-+\s*\n(.*?)(?:\n\n|\nNyblom stability test)", content, re.DOTALL)
        if arch_lm_match:
            arch_lm_section = arch_lm_match.group(1)
            arch_lm_lines = [line.strip() for line in arch_lm_section.split('\n') if line.strip()]
            
            # Skip header line
            for line in arch_lm_lines[1:]:
                parts = re.split(r'\s+', line.strip())
                if len(parts) >= 5 and parts[0] == "ARCH":
                    lag_name = parts[1]
                    statistic = float(parts[2])
                    shape = float(parts[3])
                    scale = float(parts[4])
                    p_value = float(parts[5])
                    
                    model_info['arch_lm_tests'][lag_name] = {
                        'statistic': statistic,
                        'shape': shape,
                        'scale': scale,
                        'p_value': p_value
                    }
        
        # Extract Nyblom stability test
        model_info['nyblom_stability'] = {}
        
        nyblom_match = re.search(r"Nyblom stability test\s*\n-+\s*\n(.*?)(?:\n\n|\nAsymptotic Critical Values)", content, re.DOTALL)
        if nyblom_match:
            nyblom_section = nyblom_match.group(1)
            nyblom_lines = [line.strip() for line in nyblom_section.split('\n') if line.strip()]
            
            # Extract Joint Statistic
            joint_stat_line = nyblom_lines[0]
            joint_stat_match = re.search(r"Joint Statistic:\s*([\d.]+)", joint_stat_line)
            if joint_stat_match:
                model_info['nyblom_stability']['joint_statistic'] = float(joint_stat_match.group(1))
            
            # Extract Individual Statistics
            individual_stats = {}
            for line in nyblom_lines[2:]:  # Skip "Joint Statistic" and "Individual Statistics" lines
                parts = re.split(r'\s+', line.strip())
                if len(parts) >= 2:
                    param_name = parts[0]
                    param_value = float(parts[1])
                    individual_stats[param_name] = param_value
            
            if individual_stats:
                model_info['nyblom_stability']['individual_statistics'] = individual_stats
        
        # Extract Asymptotic Critical Values for Nyblom test
        model_info['nyblom_critical_values'] = {}
        
        nyblom_crit_match = re.search(r"Asymptotic Critical Values \(10% 5% 1%\)\s*\n(.*?)(?:\n\n|\nSign Bias Test)", content, re.DOTALL)
        if nyblom_crit_match:
            nyblom_crit_section = nyblom_crit_match.group(1)
            nyblom_crit_lines = [line.strip() for line in nyblom_crit_section.split('\n') if line.strip()]
            
            # Extract Joint Statistic critical values
            joint_crit_line = nyblom_crit_lines[0]
            joint_crit_match = re.search(r"Joint Statistic:\s*([\d.]+) ([\d.]+) ([\d.]+)", joint_crit_line)
            if joint_crit_match:
                model_info['nyblom_critical_values']['joint'] = {
                    '10%': float(joint_crit_match.group(1)),
                    '5%': float(joint_crit_match.group(2)),
                    '1%': float(joint_crit_match.group(3))
                }
            
            # Extract Individual Statistic critical values
            ind_crit_line = nyblom_crit_lines[1]
            ind_crit_match = re.search(r"Individual Statistic:\s*([\d.]+) ([\d.]+) ([\d.]+)", ind_crit_line)
            if ind_crit_match:
                model_info['nyblom_critical_values']['individual'] = {
                    '10%': float(ind_crit_match.group(1)),
                    '5%': float(ind_crit_match.group(2)),
                    '1%': float(ind_crit_match.group(3))
                }
                
        # Extract Sign Bias Test - Using a completely different, simpler approach
        model_info['sign_bias_test'] = {}
        
        sign_bias_match = re.search(r"Sign Bias Test\s*\n-+\s*\n(.*?)(?:\n\n|\nAdjusted Pearson)", content, re.DOTALL)
        if sign_bias_match:
            sign_bias_section = sign_bias_match.group(1)
            
            # Use regex to directly find patterns for each test
            # Look for specific test names and capture their values
            
            # Sign Bias
            sign_bias = re.search(r"Sign Bias\s+([-\d.]+)\s+([-\d.eE]+)", sign_bias_section)
            if sign_bias:
                try:
                    model_info['sign_bias_test']['Sign Bias'] = {
                        't_value': float(sign_bias.group(1)),
                        'prob': float(sign_bias.group(2)),
                        'significance': ''  # We'll skip significance to simplify
                    }
                except ValueError:
                    pass  # Skip if conversion fails
            
            # Negative Sign Bias
            neg_sign_bias = re.search(r"Negative Sign Bias\s+([-\d.]+)\s+([-\d.eE]+)", sign_bias_section)
            if neg_sign_bias:
                try:
                    model_info['sign_bias_test']['Negative Sign Bias'] = {
                        't_value': float(neg_sign_bias.group(1)),
                        'prob': float(neg_sign_bias.group(2)),
                        'significance': ''
                    }
                except ValueError:
                    pass
            
            # Positive Sign Bias
            pos_sign_bias = re.search(r"Positive Sign Bias\s+([-\d.]+)\s+([-\d.eE]+)", sign_bias_section)
            if pos_sign_bias:
                try:
                    model_info['sign_bias_test']['Positive Sign Bias'] = {
                        't_value': float(pos_sign_bias.group(1)),
                        'prob': float(pos_sign_bias.group(2)),
                        'significance': ''
                    }
                except ValueError:
                    pass
            
            # Joint Effect/Test
            joint_effect = re.search(r"Joint Effect\s+([-\d.]+)\s+([-\d.eE]+)", sign_bias_section)
            if not joint_effect:
                joint_effect = re.search(r"Joint Test\s+([-\d.]+)\s+([-\d.eE]+)", sign_bias_section)
            
            if joint_effect:
                try:
                    model_info['sign_bias_test']['Joint Effect'] = {
                        't_value': float(joint_effect.group(1)),
                        'prob': float(joint_effect.group(2)),
                        'significance': ''
                    }
                except ValueError:
                    pass
        
        # Extract Adjusted Pearson Goodness-of-Fit Test
        model_info['pearson_gof'] = {}
        
        pearson_match = re.search(r"Adjusted Pearson Goodness-of-Fit Test:\s*\n-+\s*\n(.*?)(?:\n\n|\nElapsed time)", content, re.DOTALL)
        if pearson_match:
            pearson_section = pearson_match.group(1)
            pearson_lines = [line.strip() for line in pearson_section.split('\n') if line.strip()]
            
            # Skip header line
            for line in pearson_lines[1:]:
                parts = re.split(r'\s+', line.strip())
                if len(parts) >= 4:
                    group = parts[0]
                    group_size = int(parts[1])
                    statistic = float(parts[2])
                    p_value = float(parts[3])
                    
                    model_info['pearson_gof'][group] = {
                        'group_size': group_size,
                        'statistic': statistic,
                        'p_value': p_value
                    }
        
        return model_info
    
    except Exception as e:
        print(f"Error extracting model information: {e}")
        return None

def process_summary_files(directory):
    """
    Process multiple model summary files from a directory.
    
    Parameters:
    -----------
    directory : str
        Path to directory containing model summary files
    
    Returns:
    --------
    dict
        Dictionary with filenames as keys and model info as values
    """
    results = {}
    
    if not os.path.isdir(directory):
        print(f"Error: Directory {directory} not found.")
        return results
    
    # Get all txt files in the directory
    txt_files = [f for f in os.listdir(directory) if f.endswith('_summary.txt')]
    
    for filename in txt_files:
        file_path = os.path.join(directory, filename)
        content = read_model_file(file_path)
        
        if content:
            model_info = extract_model_info(content)
            if model_info:
                results[filename] = model_info
    
    return results

# Example usage
if __name__ == "__main__":
    # Define base directory using absolute path
    base_dir = os.path.join(os.path.dirname(__file__), "model_summaries")
    
    # Single file example
    file_path = os.path.join(base_dir, "series_3_model_144_summary.txt")
    content = read_model_file(file_path)
    
    if content:
        model_info = extract_model_info(content)
        if model_info:
            print("Single File Example:")
            print(f"File: {os.path.basename(file_path)}")
            
            # Print basic model information
            print("\nBasic Model Information:")
            for key in ['garch_type', 'garch_p', 'garch_q', 'arma_p', 'arma_d', 'arma_q', 'distribution']:
                if key in model_info:
                    print(f"  {key}: {model_info[key]}")
            
            # Print optimal parameters
            if 'optimal_parameters' in model_info and model_info['optimal_parameters']:
                print("\nOptimal Parameters:")
                for param, values in model_info['optimal_parameters'].items():
                    print(f"  {param}:")
                    print(f"    Estimate: {values['estimate']}")
                    print(f"    P-value: {values['p_value']}")
            
            # Print robust standard errors
            if 'robust_std_errors' in model_info and model_info['robust_std_errors']:
                print("\nRobust Standard Errors:")
                for param, values in model_info['robust_std_errors'].items():
                    print(f"  {param}:")
                    print(f"    Estimate: {values['estimate']}")
                    print(f"    P-value: {values['p_value']}")
                    
            # Print Ljung-Box Test results for standardized residuals
            if 'ljung_box_residuals' in model_info and model_info['ljung_box_residuals']:
                print("\nLjung-Box Test on Standardized Residuals:")
                for lag, values in model_info['ljung_box_residuals'].items():
                    print(f"  {lag}:")
                    print(f"    Statistic: {values['statistic']}")
                    print(f"    P-value: {values['p_value']}")
            
            # Print Ljung-Box Test results for standardized squared residuals
            if 'ljung_box_squared_residuals' in model_info and model_info['ljung_box_squared_residuals']:
                print("\nLjung-Box Test on Standardized Squared Residuals:")
                for lag, values in model_info['ljung_box_squared_residuals'].items():
                    print(f"  {lag}:")
                    print(f"    Statistic: {values['statistic']}")
                    print(f"    P-value: {values['p_value']}")
                    
            # Print ARCH LM Test results
            if 'arch_lm_tests' in model_info and model_info['arch_lm_tests']:
                print("\nARCH LM Tests:")
                for lag, values in model_info['arch_lm_tests'].items():
                    print(f"  {lag}:")
                    print(f"    Statistic: {values['statistic']}")
                    print(f"    P-value: {values['p_value']}")
            
            # Print information criteria
            print("\nInformation Criteria:")
            for key in ['loglikelihood', 'AIC', 'BIC']:
                if key in model_info:
                    print(f"  {key}: {model_info[key]}")
            
            # Print Nyblom stability test results
            if 'nyblom_stability' in model_info:
                print("\nNyblom Stability Test:")
                if 'joint_statistic' in model_info['nyblom_stability']:
                    print(f"  Joint Statistic: {model_info['nyblom_stability']['joint_statistic']}")
                
                if 'individual_statistics' in model_info['nyblom_stability']:
                    print("  Individual Statistics:")
                    for param, value in model_info['nyblom_stability']['individual_statistics'].items():
                        print(f"    {param}: {value}")
                
                if 'nyblom_critical_values' in model_info:
                    print("  Critical Values:")
                    if 'joint' in model_info['nyblom_critical_values']:
                        print("    Joint Statistic:")
                        for level, value in model_info['nyblom_critical_values']['joint'].items():
                            print(f"      {level}: {value}")
                    
                    if 'individual' in model_info['nyblom_critical_values']:
                        print("    Individual Statistic:")
                        for level, value in model_info['nyblom_critical_values']['individual'].items():
                            print(f"      {level}: {value}")
            
            # Print Sign Bias Test results
            if 'sign_bias_test' in model_info and model_info['sign_bias_test']:
                print("\nSign Bias Test:")
                for test, values in model_info['sign_bias_test'].items():
                    print(f"  {test}:")
                    print(f"    t-value: {values['t_value']}")
                    print(f"    p-value: {values['prob']}")
                    if values['significance']:
                        print(f"    Significance: {values['significance']}")
            
            # Print Adjusted Pearson Goodness-of-Fit Test results
            if 'pearson_gof' in model_info and model_info['pearson_gof']:
                print("\nAdjusted Pearson Goodness-of-Fit Test:")
                for group, values in model_info['pearson_gof'].items():
                    print(f"  Group {group} (size {values['group_size']}):")
                    print(f"    Statistic: {values['statistic']}")
                    print(f"    p-value: {values['p_value']}")
