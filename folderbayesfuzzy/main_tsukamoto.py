import pandas as pd
from bayes_module import bayes_prob_overload
from fuzzy_tsukamoto import tsukamoto_decision

df = pd.read_csv("../PowerLoad_Dataset.csv")
df['Hour'] = pd.to_datetime(df['Timestamp']).dt.hour

for i in range(10):
    row = df.iloc[i]
    hour = row['Hour']
    day = row['DayOfWeek']
    precip = row['Precipitation_mm']

    prob = bayes_prob_overload(hour, day, precip)
    adj = tsukamoto_decision(prob)

    print(f"Hour={hour} Day={day} Precip={precip:.2f} "
          f"=> Prob={prob:.2f}, Adjustment={adj:.2f}")