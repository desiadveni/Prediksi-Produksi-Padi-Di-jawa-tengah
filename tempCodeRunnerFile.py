
def load_data():
    df = pd.read_csv("dataset_terintegrasi.csv")
    with open("wilayah_stats.json") as f:
        stats = json.load(f)
    with open("wilayah_defaults.json") as f:
        defaults = json.load(f)
    return df, stats, defaults