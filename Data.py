import pandas as pd
import seaborn as sns
import numpy as np
import warnings
import matplotlib.pyplot as plt
from scipy import stats

warnings.filterwarnings("ignore")

crimes_dataset = pd.read_csv("data/Crimes_Dataset.csv")
suspect_dataset = pd.read_csv("data/Suspects_Dataset.csv")


# Alle Zeichenfolgen in den DataFrames in Kleinbuchstaben umwandeln
crimes_df = crimes_dataset.applymap(lambda x: x.lower() if isinstance(x, str) else x)
suspects_df = suspect_dataset.applymap(lambda x: x.lower() if isinstance(x, str) else x)

# Filterungsprozess
crimes_filtered_evidence = crimes_dataset.dropna(subset=['Evidence Found']) # Loescht alle Eintraege in Evidence Found, die keinen Eintrag haben
village_day_df = village_df[village_df['Time of Day'] == 'day'] # Loescht alle Eintraege in Time of Day, die nicht 'day' sind
suspects_filtered = suspects_df[~suspects_df.isin(['sunlight']).any(axis=1)] # Löschen aller Zeilen, in denen "sunlight" vorkommt
# Anzahl der Zeilen ermitteln, in denen "sunlight" vorkommt
sunlight_count = suspects_filtered.isin(['sunlight']).any(axis=1).sum()

# Behalten nur der Zeilen, in denen die Region "village" ist
village_df = crimes_df[crimes_df['Region'] == 'village']
crime_weapon = village_df[village_df['Crime Weapon'] == 'knife']
print(crime_weapon)
#print(suspects_filtered)




