def classify_time(hour):
    if 0 <= hour <= 7:
        return "malam"
    elif 8 <= hour <= 10:
        return "pagi"
    elif 11 <= hour <= 18:
        return "siang"
    else:
        return "sore"


def classify_day(day_of_week):
    # 1–5 = kerja, 6–7 = weekend
    if 1 <= day_of_week <= 5:
        return "jam_kerja"
    else:
        return "weekend"


def classify_weather(precip):
    return "badai" if precip > 0.5 else "cerah"


def bayes_prob_overload(hour, day_of_week, precip):
    waktu = classify_time(hour)
    day   = classify_day(day_of_week)
    cuaca = classify_weather(precip)

    # --- rules yang kamu tulis ---

    # Tinggi
    if waktu == "pagi" and day == "jam_kerja":
        level = "tinggi"
    elif waktu == "pagi" and cuaca == "badai":
        level = "tinggi"
    elif waktu == "siang" and day == "jam_kerja" and cuaca == "badai":
        level = "tinggi"

    # Sedang
    elif waktu == "pagi" and day == "weekend" and cuaca == "cerah":
        level = "sedang"
    elif waktu == "siang" and day == "jam_kerja" and cuaca == "cerah":
        level = "sedang"
    elif waktu == "siang" and day == "weekend" and cuaca == "badai":
        level = "sedang"

    # Rendah
    elif waktu == "malam" or waktu == "sore":
        level = "rendah"
    elif waktu == "siang" and day == "weekend" and cuaca == "cerah":
        level = "rendah"

    # fallback kalau tidak ada rule yang kena
    else:
        level = "sedang"

    # mapping level → angka probabilitas
    if level == "tinggi":
        return 0.8
    elif level == "sedang":
        return 0.5
    else:
        return 0.2
