# Imports
import pandas as pd
import seaborn as sns
import numpy as np
import warnings
import matplotlib.pyplot as plt
from scipy import stats

warnings.filterwarnings("ignore")

# Daten werden eingelesen
crimes_dataset = pd.read_csv("data/Crimes_Dataset.csv")
suspect_dataset = pd.read_csv("data/Suspects_Dataset.csv")

# Aenderungen für Crimes_Dataset
# Alle Zeichenfolgen in den DataFrame in Kleinbuchstaben umwandeln
crimes_df = crimes_dataset.applymap(lambda x: x.lower() if isinstance(x, str) else x)

# Filterungsprozess für Crimes_Dataset
crimes_evidence = crimes_dataset.dropna(subset=['Evidence Found'])  # Loescht alle Einträge in Evidence Found, die keinen Eintrag haben
crimes_evidence_day = crimes_evidence[crimes_evidence['Time of Day'] == 'day']  # Speichert nur die Werte, in denen in der Kategorie "Time of Day" == 'day' sind

# Behalten nur der Zeilen, in denen die Region "village" ist
crimes_evidence_day_region = crimes_evidence_day[crimes_evidence_day['Region'] == 'village']
crimes_evidence_day_region_weapon = crimes_evidence_day_region[crimes_evidence_day_region['Crime Weapon'] == 'knife']



# Aenderungen fuer Suspects_Dataset
# Alle Zeichenfolgen in den DataFrame in Kleinbuchstaben umwandeln
suspects_df = suspect_dataset.applymap(lambda x: x.lower() if isinstance(x, str) else x)

# Filterungsprozess fuer Suspects_Dataset
suspects_sunlight_ = suspects_df[~suspects_df.isin(['sunlight']).any(axis=1)]  # Loeschen aller Zeilen, in denen "sunlight" vorkommt





