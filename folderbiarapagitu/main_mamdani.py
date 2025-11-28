import pandas as pd
from fuzzy_mamdani import GridEarlyWarning_Mamdani
from bayesmodule import GridBayesEngine



df = pd.read_csv("PowerLoad_Dataset.csv") 
    
#Preprocess
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df['Hour'] = df['Timestamp'].dt.hour
    
#load Fuzzy Mamdani and Bayes Engine
mamdani= GridEarlyWarning_Mamdani()
bayes = GridBayesEngine()
    
print(f"{'TIMESTAMP':<20} | {'TSUKAMOTO%':<10} | {'STATUS'}")

    

for i in range(40):
    row = df.iloc[i]
        
        # Inputs
    t_val = float(row['Hour'])
    d_val = float(row['DayOfWeek']) - 1.0 # Hari  1-7 ke 0-6 (senin=0)
    r_val = float(row['Precipitation_mm'])
        
        #Nilai Fuzzy
    score_m = mamdani.evaluate(t_val, d_val, r_val)
        
        #Probabilitas Bayes
    prop_m = bayes.get_failure_probability(score_m)
        

    if prop_m >= 0.8:
            status = "CRITICAL"
    elif prop_m >= 0.3:
            status = "WARNING"
    else:
            status = "NORMAL"
        
    print(f"{str(row['Timestamp']):<20} | {prop_m:.1%}    | {status}")
