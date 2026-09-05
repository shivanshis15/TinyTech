import pandas as pd

# Define Study 1: Effect of Nanoparticle Radius
study1_data = {
    'Material': ['Au']*5 + ['Ag']*5,
    'Radius (nm)': [20, 40, 60, 80, 100, 20, 40, 60, 80, 100],
    'Host n': [1.0]*10,
    'Resonance (nm)': [507, 518, 526, 555, 595, 360, 379, 415, 370, 390],
    'Qext at resonance': [0.9278, 2.6338, 4.8124, 5.2180, 4.5332, 13.5752, 11.4687, 6.7238, 7.1174, 5.9928]
}

# Define Study 2: Effect of Host-Medium Refractive Index
study2_data = {
    'Material': ['Au']*5 + ['Ag']*5,
    'Radius (nm)': [40]*10,
    'Host n': [1.00, 1.33, 1.45, 1.52, 1.60, 1.00, 1.33, 1.45, 1.52, 1.60],
    'Resonance (nm)': [518, 549, 570, 582, 595, 379, 444, 473, 491, 511],
    'Qext at resonance': [2.6338, 6.4696, 7.5946, 8.2151, 8.6255, 11.4687, 9.8635, 9.3446, 9.1886, 9.0240]
}

# Create DataFrames
df_study1 = pd.DataFrame(study1_data)
df_study2 = pd.DataFrame(study2_data)

# Print Formatted Tables to Terminal
print("=" * 65)
print("STUDY 1: EFFECT OF NANOPARTICLE RADIUS")
print("=" * 65)
print(df_study1.to_string(index=False))

print("\n" + "=" * 65)
print("STUDY 2: EFFECT OF HOST-MEDIUM REFRACTIVE INDEX")
print("=" * 65)
print(df_study2.to_string(index=False))

# Export to CSV Files
df_study1.to_csv('Study1_Radius_Effect.csv', index=False)
df_study2.to_csv('Study2_Host_Medium_Effect.csv', index=False)

print("\nCSVs generated successfully: 'Study1_Radius_Effect.csv' & 'Study2_Host_Medium_Effect.csv'")