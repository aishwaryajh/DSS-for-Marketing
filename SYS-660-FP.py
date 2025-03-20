import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Read the Excel file
df = pd.read_excel('smm.xlsx')
#print(df)
#print(df.columns)
# Function to validate age group input
def validate_age_group(age_group):
    age_groups = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
    if age_group in age_groups:
        return True
    else:
        return False

# Function to validate content preference input
def validate_content_preference(content_preference):
    try:
        content_preference = int(content_preference)
        if content_preference >= 1 and content_preference <= 10:
            return True
        else:
            return False
    except ValueError:
        return False

# Function to validate budget input
def validate_budget(budget):
    try:
        budget = float(budget)
        if budget >= 3.21:
            return True
        else:
            return False
    except ValueError:
        return False

# Ask for target audience age group
age_group = input("Enter the target audience age group (e.g., 18-24, 25-34, 35-44, 45-54, 55-64, 65+): ")
while not validate_age_group(age_group):
    print("Invalid age group.")
    age_group = input("Enter the target audience age group (e.g., 18-24, 25-34, 35-44, 45-54, 55-64, 65+): ")

# Ask for content type preference
content_preference = int(input("Enter the number of ad formats preferred from 1-10: "))
while not validate_content_preference(content_preference):
    print("Invalid content preference.")
    content_preference = int(input("Enter the number of ad formats preferred from 1-10: "))

# Ask for budget
budget = float(input("Enter the budget per mile starting from 3.21 (in USD): "))
while not validate_budget(budget):
    print("Invalid budget. Budget must be a numeric value greater than or equal to 3.21 USD.")
    budget = float(input("Enter the budget per mile starting from 3.21 (in USD): "))
df['no._of_ad_format_available'] = df['no._of_ad_format_available'].astype(int)
df['cpm'] = df['cpm'].astype(float)
filtered_data = df[(df['age_group'] == age_group) &
                     (df['no._of_ad_format_available'] >= content_preference) &
                     (df['cpm'] <= budget)]
filtered_data_to_print = filtered_data.drop(columns=['age_group'])
print("Available platforms based on your preferences for social media marketing:")
print(filtered_data_to_print)
filtered_df = filtered_data.copy()

# Function to validate preference ranking input and check for uniqueness
def validate_preference_rank(rank, used_ranks):
    try:
        rank = int(rank)
        if 1 <= rank <= 3 and rank not in used_ranks:
            return True
        else:
            return False
    except ValueError:
        return False

print("\nPlease rank your preferences for the following criteria from 1-3 (1 highest, 3 lowest):")
preferences = {}
used_ranks = []

attributes = {
    'reach': 'Maximizing brand engagement',
    'avg_conversion_rate': 'Maximizing profits',
    'cpm': 'Minimizing cost'
}

for attr, description in attributes.items():
    rank = input(f" {description}: ")
    while not validate_preference_rank(rank, used_ranks):
        print("Invalid input or rank already used. Please enter a unique rank from 1-3.")
        rank = input(f" {description}: ")
    preferences[attr] = int(rank)
    used_ranks.append(int(rank))

# Define swing weights based on user preferences
swing_weights = {
    'reach': 50 if preferences.get('reach', 0) == 1 else (25 if preferences.get('reach', 0) == 2 else 15),
    'avg_conversion_rate': 50 if preferences.get('avg_conversion_rate', 0) == 1
    else (25 if preferences.get('avg_conversion_rate', 0) == 2 else 15),
    'cpm': 50 if preferences.get('cpm', 0) == 1 else (25 if preferences.get('cpm', 0) == 2 else 15),
    'reliability': 5,
    'no._of_ad_format_available': 5
}

print("User preferences and swing weights:")
#print(preferences)
#print(swing_weights)


normalized_weights = {}
total_swing = sum(swing_weights.values())
#print(total_swing)
# Normalize the swing weights
normalized_weights = {k: v / total_swing for k, v in swing_weights.items()}
#print(normalized_weights)

# Define worst and best values for each attribute
worst_values = {
    'reach': 2,
    'avg_conversion_rate': 1.10,
    'cpm': 14.9,
    'reliability': 75 ,
    'no._of_ad_format_available': 6
}
best_values = {
    'reach':66.9,
    'avg_conversion_rate':12 ,
    'cpm': 3.21,
    'reliability': 95 ,
    'no._of_ad_format_available': 10
}

# Define the function to calculate MAU
# Define utility functions for each attribute
def reach_utility(reach):
    return (reach - worst_values['reach']) / (best_values['reach'] - worst_values['reach'])

def avg_conversion_rate_utility(avg_conversion_rate):
    return (avg_conversion_rate - worst_values['avg_conversion_rate']) / (best_values['avg_conversion_rate'] - worst_values['avg_conversion_rate'])

def cpm_utility(cpm):
    return (worst_values['cpm'] - cpm) / (worst_values['cpm'] - best_values['cpm'])

def reliability_utility(reliability):
    return (reliability - worst_values['reliability']) / (best_values['reliability'] - worst_values['reliability'])

def ad_formats_utility(ad_formats):
    return (ad_formats - worst_values['no._of_ad_format_available']) / (best_values['no._of_ad_format_available'] - worst_values['no._of_ad_format_available'])

# Define the function to calculate MAU
def calculate_mau(row):
    utilities = {
        'reach': reach_utility(row['reach']),
        'avg_conversion_rate': avg_conversion_rate_utility(row['avg_conversion_rate']),
        'cpm': cpm_utility(row['cpm']),
        'reliability': reliability_utility(row['reliability']),
        'no._of_ad_format_available': ad_formats_utility(row['no._of_ad_format_available'])
    }
    total_utility = sum(utilities[attr] * swing_weights[attr] for attr in utilities.keys())
    return total_utility

# Calculate MAU for each platform in filtered_data
filtered_df['MAU'] = filtered_df.apply(calculate_mau, axis=1)/100
# Round the values to 4 decimal places
filtered_df['MAU'] = filtered_df['MAU'].round(4)
#print(filtered_df[['platform', 'MAU']])


# Monte Carlo simulation
num_simulations = 1000
simulation_results = {}

for index, platform in filtered_df.iterrows():
    # Generate random samples for each attribute
    reach_samples = np.random.normal(platform["reach"], 10, num_simulations)
    avg_conversion_rate_samples = np.random.normal(platform["avg_conversion_rate"], 2, num_simulations)
    cpm_samples = np.random.normal(platform["cpm"], 1, num_simulations)
    reliability_samples = np.random.normal(platform["reliability"], 5, num_simulations)
    ad_formats_samples = np.random.normal(platform["no._of_ad_format_available"], 2, num_simulations)

    # Calculate utilities for each sample
    utilities = (
        reach_utility(reach_samples) * normalized_weights["reach"] +
        avg_conversion_rate_utility(avg_conversion_rate_samples) * normalized_weights["avg_conversion_rate"] +
        cpm_utility(cpm_samples) * normalized_weights["cpm"] +
        reliability_utility(reliability_samples) * normalized_weights["reliability"] +
        ad_formats_utility(ad_formats_samples) * normalized_weights["no._of_ad_format_available"]
    )

    # Store simulation results for the current platform
    simulation_results[platform["platform"]] = utilities

# Calculate the mean utility for each platform from the Monte Carlo simulation
mean_utilities = {platform: np.mean(utilities).round(4) for platform, utilities in simulation_results.items()}

# Print the mean utility for each platform
print("\nMean Utility for Each Platform:")
for platform, mean_utility in mean_utilities.items():
    print(f"{platform}: {mean_utility}")

# Identify the platform with the highest mean utility
best_platform = max(mean_utilities, key=mean_utilities.get)

# Print the best platform
# Print the best platform with bold, red color, and all caps for Facebook
print(f"The best platform based on your preferences is: \033[1m\033[91m{best_platform.upper()}\033[0m")

# Sensitivity Analysis for Reach
# Ask the user to input the platform for sensitivity analysis
#selected_platform = input("Enter the platform for sensitivity analysis: ")

# Set a flag for the loop
continue_analysis = True

# Set a flag for the loop
continue_analysis = True

# Loop for sensitivity analysis
while continue_analysis:
    # Sensitivity Analysis for Reach
    # Ask the user to input the platform for sensitivity analysis
    selected_platform = input("Enter the platform for sensitivity analysis (type 'done' to exit): ")

    # Check if the user wants to exit
    if selected_platform.lower() == 'done':
        continue_analysis = False
        break

    # Check if the selected platform exists in the filtered data
    if selected_platform not in filtered_data['platform'].unique():
        print("Invalid platform. Please select a platform from the available options.")
        continue

    # Create a new DataFrame to store modified values
    new_df_reach = pd.DataFrame(columns=['platform', 'reach', 'MAU'])

    # Initialize an empty list for MAU values
    mau_values = []

    # Iterate through each platform to determine MAU values
    for platform in filtered_df['platform']:
        if platform == selected_platform:
            # Calculate MAU values for different reach values for the selected platform
            reach_values = np.linspace(0, 100, num=10)
            for reach in reach_values:
                # Create a copy of the filtered_df to avoid modifying the original DataFrame
                temp_df = filtered_df.copy()
                # Modify the reach value for the selected platform
                temp_df.loc[temp_df['platform'] == selected_platform, 'reach'] = reach
                # Recalculate MAU
                temp_df['MAU'] = temp_df.apply(calculate_mau, axis=1) / 100
                # Store the MAU value for the current reach value
                mau_values.extend(temp_df.loc[temp_df['platform'] == selected_platform, 'MAU'])
            # Add the platform, reach, and MAU values to the new DataFrame
            new_df_reach = pd.concat([new_df_reach, pd.DataFrame(
                {'platform': [selected_platform] * 10, 'reach': reach_values, 'MAU': mau_values})], ignore_index=True)
        else:
            # Get the constant MAU value for the current platform
            constant_mau_value = filtered_df.loc[filtered_df['platform'] == platform, 'MAU'].values[0]
            # Add the constant MAU value for the current platform to the new DataFrame
            new_df_reach = pd.concat([new_df_reach, pd.DataFrame(
                {'platform': [platform] * 10, 'reach': np.linspace(0, 100, num=10), 'MAU': [constant_mau_value] * 10})],
                                     ignore_index=True)

    #print(new_df_reach)
    # Sensitivity Analysis for Avg Conversion Rate
    # Create a new DataFrame to store the MAU values for each platform for avg_conversion_rate
    new_df_avg_conversion_rate = pd.DataFrame(columns=['platform', 'avg_conversion_rate', 'MAU'])

    # Iterate through each platform to determine MAU values
    for platform in filtered_df['platform']:
        if platform == selected_platform:
            # Calculate MAU values for different avg_conversion_rate values for the selected platform
            avg_conversion_rate_values = np.linspace(1, 13, num=10)
            mau_values_ac = []
            for avg_conversion_rate in avg_conversion_rate_values:
                # Modify the avg_conversion_rate value for the selected platform
                filtered_df_copy = filtered_df.copy()
                filtered_df_copy.loc[
                    filtered_df_copy['platform'] == selected_platform, 'avg_conversion_rate'] = avg_conversion_rate
                # Recalculate MAU
                filtered_df_copy['MAU'] = filtered_df_copy.apply(calculate_mau, axis=1) / 100
                # Store the MAU value for the current avg_conversion_rate value
                mau_values_ac.extend(filtered_df_copy.loc[filtered_df_copy['platform'] == selected_platform, 'MAU'])
            # Add the platform, avg_conversion_rate, and MAU values to the new DataFrame
            new_df_avg_conversion_rate = pd.concat([new_df_avg_conversion_rate, pd.DataFrame(
                {'platform': [selected_platform] * 10, 'avg_conversion_rate': avg_conversion_rate_values,
                 'MAU': mau_values_ac})], ignore_index=True)
        else:
            # Get the constant MAU value for the current platform
            constant_mau_value_ac = filtered_df.loc[filtered_df['platform'] == platform, 'MAU'].values[0]
            # Add the constant MAU value for the current platform to the new DataFrame
            new_df_avg_conversion_rate = pd.concat([new_df_avg_conversion_rate, pd.DataFrame(
                {'platform': [platform] * 10, 'avg_conversion_rate': np.linspace(1, 13, num=10),
                 'MAU': [constant_mau_value_ac] * 10})], ignore_index=True)

    #print(new_df_avg_conversion_rate)

    # Sensitivity Analysis for CPM
    # Create a new DataFrame to store the MAU values for each platform for cpm
    new_df_cpm = pd.DataFrame(columns=['platform', 'cpm', 'MAU'])

    # Iterate through each platform to determine MAU values
    for platform in filtered_df['platform']:
        if platform == selected_platform:
            # Calculate MAU values for different cpm values for the selected platform
            cpm_values = np.linspace(3.21, 15, num=10)  # Assuming the range of cpm
            mau_values_cpm = []
            for cpm in cpm_values:
                # Modify the cpm value for the selected platform
                filtered_df_copy = filtered_df.copy()
                filtered_df_copy.loc[filtered_df_copy['platform'] == selected_platform, 'cpm'] = cpm
                # Recalculate MAU
                filtered_df_copy['MAU'] = filtered_df_copy.apply(calculate_mau, axis=1) / 100
                # Store the MAU value for the current cpm value
                mau_values_cpm.extend(filtered_df_copy.loc[filtered_df_copy['platform'] == selected_platform, 'MAU'])
            # Add the platform, cpm, and MAU values to the new DataFrame
            new_df_cpm = pd.concat([new_df_cpm, pd.DataFrame(
                {'platform': [selected_platform] * 10, 'cpm': cpm_values, 'MAU': mau_values_cpm})], ignore_index=True)
        else:
            # Get the constant MAU value for the current platform
            constant_mau_value_cpm = filtered_df.loc[filtered_df['platform'] == platform, 'MAU'].values[0]
            # Add the constant MAU value for the current platform to the new DataFrame
            new_df_cpm = pd.concat([new_df_cpm, pd.DataFrame(
                {'platform': [platform] * 10, 'cpm': np.linspace(3.21, 15, num=10),
                 'MAU': [constant_mau_value_cpm] * 10})], ignore_index=True)

    #print(new_df_cpm)
    # Plotting
    # Set the style of the plot
    sns.set_style('darkgrid')

    # Create subplots
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))

    # Sensitivity analysis for reach
    axes[0].set_title(f'Sensitivity to {selected_platform} Audience Reach')
    for platform in new_df_reach['platform'].unique():
        platform_data = new_df_reach[new_df_reach['platform'] == platform]
        axes[0].plot(platform_data['reach'], platform_data['MAU'], label=platform)
    axes[0].set_xlabel('Reach (in millions)')
    axes[0].set_ylabel('Value')
    axes[0].legend()
    axes[0].set_ylim(0.1, 1.0)
    axes[0].set_yticks(np.arange(0.1, 1.1, 0.1))

    # Sensitivity analysis for avg_conversion_rate
    axes[1].set_title(f'Sensitivity to {selected_platform} Average Conversion Rate')
    for platform in new_df_avg_conversion_rate['platform'].unique():
        platform_data = new_df_avg_conversion_rate[new_df_avg_conversion_rate['platform'] == platform]
        axes[1].plot(platform_data['avg_conversion_rate'], platform_data['MAU'], label=platform)
    axes[1].set_xlabel('Average Conversion Rate (%)')
    axes[1].set_ylabel('Value')
    axes[1].legend()
    axes[1].set_ylim(0.1, 1.0)
    axes[1].set_yticks(np.arange(0.1, 1.1, 0.1))

    # Sensitivity analysis for cpm
    axes[2].set_title(f'Sensitivity to {selected_platform} CPM (Cost per Mile)')
    for platform in new_df_cpm['platform'].unique():
        platform_data = new_df_cpm[new_df_cpm['platform'] == platform]
        axes[2].plot(platform_data['cpm'], platform_data['MAU'], label=platform)
    axes[2].set_xlabel('CPM (in $)')
    axes[2].set_ylabel('Value')
    axes[2].legend()
    axes[2].set_ylim(0.1, 1.0)
    axes[2].set_yticks(np.arange(0.1, 1.1, 0.1))

    # Show the plot
    plt.tight_layout()
    plt.show()
