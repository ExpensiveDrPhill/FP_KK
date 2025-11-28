import numpy as np
import skfuzzy as fuzz


class TsukamotoFuzzySystem:
    def __init__(self):
        self._setup_domains()
        self._setup_membership_functions()

    def _setup_domains(self):
        self.x_time = np.arange(0, 24.1, 0.1)
        self.x_day = np.arange(0, 7, 0.1) 
        self.x_weather = np.arange(0, 10.1, 0.1)
        self.x_risk = np.arange(0, 101, 1)

    def _setup_membership_functions(self):
        # Uses same definitions as Mamdani for Inputs
        self.time_night = fuzz.trapmf(self.x_time, [0, 0, 6, 8]) 
        self.time_morning = fuzz.trimf(self.x_time, [7, 9, 11])
        self.time_afternoon = fuzz.trapmf(self.x_time, [10, 12, 17, 19])#siang  
        self.time_evening = fuzz.trapmf(self.x_time, [18, 20, 24, 24])

        self.day_weekday = fuzz.trapmf(self.x_day, [0, 0, 4, 5])
        self.day_weekend = fuzz.trapmf(self.x_day, [4, 5, 6, 6])

        self.weather_clear = fuzz.sigmf(self.x_weather, 0.5, -5)
        self.weather_heavy = fuzz.sigmf(self.x_weather, 0.5, 5)

    # Tsukamoto Inverse Functions
    def _inv_low(self, alpha): return 50 - (alpha * 50)
    def _inv_med(self, alpha): return 50.0 
    def _inv_high(self, alpha): return (alpha * 50) + 50

    def evaluate(self, time_input, day_input, rain_input):
        # Fuzzification (Same as Mamdani)
        mu_morning = fuzz.interp_membership(self.x_time, self.time_morning, time_input)
        mu_afternoon = fuzz.interp_membership(self.x_time, self.time_afternoon, time_input)
        mu_night = fuzz.interp_membership(self.x_time, self.time_night, time_input)
        mu_evening = fuzz.interp_membership(self.x_time, self.time_evening, time_input)
        
        mu_weekday = fuzz.interp_membership(self.x_day, self.day_weekday, day_input)
        mu_weekend = fuzz.interp_membership(self.x_day, self.day_weekend, day_input)
        
        mu_heavy = fuzz.interp_membership(self.x_weather, self.weather_heavy, rain_input)
        mu_clear = fuzz.interp_membership(self.x_weather, self.weather_clear, rain_input)

        # Rule Strength Calculation (Same as Mamdani)
        r1 = np.fmin(mu_morning, mu_weekday)
        r2 = np.fmin(mu_morning, mu_heavy)
        r3 = np.fmin(mu_afternoon, np.fmin(mu_weekday, mu_heavy))
        
        r4 = np.fmin(mu_morning, np.fmin(mu_weekend, mu_clear))
        r5 = np.fmin(mu_afternoon, np.fmin(mu_weekday, mu_clear))
        r6 = np.fmin(mu_afternoon, np.fmin(mu_weekend, mu_heavy))
        
        r7 = mu_night
        r8 = mu_evening
        r9 = np.fmin(mu_afternoon, np.fmin(mu_weekend, mu_clear))

        # Tsukamoto Aggregation
        alpha_list, z_list = [], []

        for alpha in [r1, r2, r3]:
            if alpha > 0:
                alpha_list.append(alpha)
                z_list.append(self._inv_high(alpha))
        
        for alpha in [r4, r5, r6]:
            if alpha > 0:
                alpha_list.append(alpha)
                z_list.append(self._inv_med(alpha))
        
        for alpha in [r7, r8, r9]:
            if alpha > 0:
                alpha_list.append(alpha)
                z_list.append(self._inv_low(alpha))

        alpha_arr = np.array(alpha_list)
        z_arr = np.array(z_list)

        if np.sum(alpha_arr) == 0: return 0.0
        return np.sum(alpha_arr * z_arr) / np.sum(alpha_arr)