import numpy as np
import skfuzzy as fuzz

# --- MAMDANI SYSTEM ---
class GridEarlyWarning_Mamdani:
    def __init__(self):
        self._setup_domains()
        self._setup_membership_functions()
        
    def _setup_domains(self):
        self.x_time = np.arange(0, 24.1, 0.1)
        self.x_day = np.arange(0, 7, 0.1) 
        self.x_weather = np.arange(0, 10.1, 0.1)
        self.x_risk = np.arange(0, 101, 1)

    def _setup_membership_functions(self):
        # Time
        self.time_night = fuzz.trapmf(self.x_time, [0, 0, 6, 8]) 
        self.time_morning = fuzz.trimf(self.x_time, [7, 9, 11])
        self.time_afternoon = fuzz.trapmf(self.x_time, [10, 12, 17, 19]) #siang
        self.time_evening = fuzz.trapmf(self.x_time, [18, 20, 24, 24])

        # Day
        self.day_weekday = fuzz.trapmf(self.x_day, [0, 0, 4, 5])
        self.day_weekend = fuzz.trapmf(self.x_day, [4, 5, 6, 6])

        # Weather
        self.weather_clear = fuzz.sigmf(self.x_weather, 0.5, -5)
        self.weather_heavy = fuzz.sigmf(self.x_weather, 0.5, 5)

        # Risk Output
        self.risk_low = fuzz.trimf(self.x_risk, [0, 0, 50])
        self.risk_med = fuzz.trimf(self.x_risk, [25, 50, 75])
        self.risk_high = fuzz.trimf(self.x_risk, [50, 100, 100])

    def evaluate(self, time_input, day_input, rain_input):
        # Fuzzification
        mu_morning = fuzz.interp_membership(self.x_time, self.time_morning, time_input)
        mu_afternoon = fuzz.interp_membership(self.x_time, self.time_afternoon, time_input)
        mu_night = fuzz.interp_membership(self.x_time, self.time_night, time_input)
        mu_evening = fuzz.interp_membership(self.x_time, self.time_evening, time_input)
        
        mu_weekday = fuzz.interp_membership(self.x_day, self.day_weekday, day_input)
        mu_weekend = fuzz.interp_membership(self.x_day, self.day_weekend, day_input)
        
        mu_heavy = fuzz.interp_membership(self.x_weather, self.weather_heavy, rain_input)
        mu_clear = fuzz.interp_membership(self.x_weather, self.weather_clear, rain_input)

        # Rules
        # High
        r1 = np.fmin(mu_morning, mu_weekday)
        r2 = np.fmin(mu_morning, mu_heavy)
        r3 = np.fmin(mu_afternoon, np.fmin(mu_weekday, mu_heavy))
        active_high = np.fmax(r1, np.fmax(r2, r3))
        
        # Medium
        r4 = np.fmin(mu_morning, np.fmin(mu_weekend, mu_clear))
        r5 = np.fmin(mu_afternoon, np.fmin(mu_weekday, mu_clear))
        r6 = np.fmin(mu_afternoon, np.fmin(mu_weekend, mu_heavy))
        active_med = np.fmax(r4, np.fmax(r5, r6))
        
        # Low
        r7 = mu_night
        r8 = mu_evening
        r9 = np.fmin(mu_afternoon, np.fmin(mu_weekend, mu_clear))
        active_low = np.fmax(r7, np.fmax(r8, r9))
        
        # Defuzifikasi (Centroid)
        out_high = np.fmin(active_high, self.risk_high)
        out_med = np.fmin(active_med, self.risk_med)
        out_low = np.fmin(active_low, self.risk_low)
        
        aggregated = np.fmax(out_low, np.fmax(out_med, out_high))
        
        if np.sum(aggregated) == 0: return 0.0
        return fuzz.defuzz(self.x_risk, aggregated, 'centroid')