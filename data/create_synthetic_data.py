import os
import numpy as np
import pandas as pd

def generate_synthetic_data(num_records=2000, seed=42):
    """
    Generates a realistic synthetic dataset for ambulance dispatch delay risk prediction
    with complex non-linear feature interactions to showcase the power of deep learning.
    
    Parameters:
        num_records (int): Number of emergency cases to generate.
        seed (int): Random seed for reproducibility.
        
    Returns:
        pd.DataFrame: Synthetic dataset with 13 input features and 1 target column.
    """
    np.random.seed(seed)
    
    # Categorical variables categories
    priority_levels = ['Low', 'Medium', 'High', 'Critical']
    incident_types = ['Cardiac Arrest', 'Traffic Accident', 'Respiratory Emergency', 'Fall/Injury', 'Stroke', 'Fire/Hazmat']
    times_of_day = ['Morning', 'Afternoon', 'Evening', 'Night']
    weather_conditions = ['Clear', 'Rainy', 'Snowy', 'Foggy']
    days_of_week = ['Weekday', 'Weekend']
    traffic_densities = ['Low', 'Medium', 'High']
    dispatch_zones = ['Zone A', 'Zone B', 'Zone C', 'Zone D', 'Zone E']
    
    # Generate random features
    case_ids = [f"CASE_{1000 + i}" for i in range(num_records)]
    
    priority_level = np.random.choice(priority_levels, size=num_records, p=[0.2, 0.4, 0.3, 0.1])
    incident_type = np.random.choice(incident_types, size=num_records, p=[0.15, 0.25, 0.2, 0.2, 0.1, 0.1])
    time_of_day = np.random.choice(times_of_day, size=num_records, p=[0.25, 0.35, 0.25, 0.15])
    weather = np.random.choice(weather_conditions, size=num_records, p=[0.6, 0.2, 0.1, 0.1])
    day_of_week = np.random.choice(days_of_week, size=num_records, p=[0.7, 0.3])
    traffic = np.random.choice(traffic_densities, size=num_records, p=[0.3, 0.5, 0.2])
    zone = np.random.choice(dispatch_zones, size=num_records, p=[0.25, 0.2, 0.2, 0.15, 0.2])
    
    distance_to_scene = np.random.uniform(0.5, 15.0, size=num_records)
    crew_experience = np.random.randint(1, 21, size=num_records)
    ambulance_age = np.random.randint(1, 11, size=num_records)
    temperature = np.random.uniform(10.0, 100.0, size=num_records)
    
    # Zone-based delay rates: Zone D has highest delays, Zone A has lowest
    zone_delay_mapping = {'Zone A': 0.08, 'Zone B': 0.15, 'Zone C': 0.18, 'Zone D': 0.32, 'Zone E': 0.22}
    historical_zone_delay_rate = np.array([zone_delay_mapping[z] for z in zone])
    
    caller_stress_score = np.random.randint(1, 11, size=num_records)
    
    # Complex non-linear interaction terms which linear models cannot capture
    interaction_term = (
        # Interaction 1: Severe Weather + High Traffic + Distant Scene
        ((weather == 'Snowy') & (traffic == 'High') & (distance_to_scene > 6.0)).astype(float) * 2.8
        + ((weather == 'Foggy') & (traffic == 'High') & (distance_to_scene > 5.0)).astype(float) * 1.8
        
        # Interaction 2: Nighttime + Long distance + Critical Priority
        + ((time_of_day == 'Night') & (distance_to_scene > 8.0) & (priority_level == 'Critical')).astype(float) * 2.2
        
        # Interaction 3: Low experience crew + high caller stress on a Weekend
        + ((crew_experience < 5) & (caller_stress_score > 7) & (day_of_week == 'Weekend')).astype(float) * 2.0
        
        # Interaction 4: High traffic in Zone D
        + ((traffic == 'High') & (zone == 'Zone D')).astype(float) * 1.6
    )
    
    # Base mathematical log-odds formula
    log_odds = (
        -2.4
        + 0.25 * distance_to_scene
        - 0.10 * crew_experience
        + 0.06 * ambulance_age
        + 0.12 * caller_stress_score
        + 2.20 * historical_zone_delay_rate
        + np.where(priority_level == 'Critical', 1.4, 0.0)
        + np.where(priority_level == 'High', 0.7, 0.0)
        + np.where(priority_level == 'Low', -0.5, 0.0)
        + np.where(traffic == 'High', 1.1, 0.0)
        + np.where(traffic == 'Medium', 0.4, 0.0)
        + np.where(weather == 'Snowy', 0.8, 0.0)
        + np.where(weather == 'Rainy', 0.4, 0.0)
        + np.where(time_of_day == 'Night', 0.7, 0.0)
        + interaction_term
    )
    
    # Sigmoid function to obtain probabilities
    probabilities = 1.0 / (1.0 + np.exp(-log_odds))
    
    # Binary delay risk
    delay_risk = (np.random.rand(num_records) < probabilities).astype(int)
    
    # Build dataframe
    df = pd.DataFrame({
        'case_id': case_ids,
        'priority_level': priority_level,
        'incident_type': incident_type,
        'time_of_day': time_of_day,
        'weather_conditions': weather,
        'day_of_week': day_of_week,
        'traffic_density': traffic,
        'dispatch_zone': zone,
        'distance_to_scene': np.round(distance_to_scene, 2),
        'crew_experience_years': crew_experience,
        'ambulance_age_years': ambulance_age,
        'temperature': np.round(temperature, 1),
        'historical_zone_delay_rate': np.round(historical_zone_delay_rate, 3),
        'caller_stress_score': caller_stress_score,
        'delay_risk': delay_risk
    })
    
    return df

if __name__ == '__main__':
    print("Generating advanced synthetic emergency response data...")
    data_dir = "/home/priya_paul/.venv/repos/Ambulance_Dispatch_Delay_Risk_Prediction/data"
    os.makedirs(data_dir, exist_ok=True)
    
    df_synthetic = generate_synthetic_data(2000, seed=42)
    file_path = os.path.join(data_dir, "emergency_response_data.csv")
    df_synthetic.to_csv(file_path, index=False)
    
    print(f"Dataset generated successfully and saved to {file_path}")
    print(f"Total records: {len(df_synthetic)}")
    print(f"Delay risk ratio:\n{df_synthetic['delay_risk'].value_counts(normalize=True)}")
