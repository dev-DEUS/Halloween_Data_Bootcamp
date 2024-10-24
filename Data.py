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
# Filterungsprozess für Crimes_Dataset - alle Filterungen in einer Zeile mit &-Operator
crimes_filtered = crimes_dataset[
    crimes_dataset['Evidence Found'].notna() &
    (crimes_dataset['Time of Day'] == 'day') &
    (crimes_dataset['Region'] == 'village') &
    (crimes_dataset['Crime Weapon'] == 'knife') &
    (crimes_dataset['Evidence Found'] == 'bones') &
    (crimes_dataset['Crime Type'] == 'kidnapping')
]

print(crimes_filtered)
# Aenderungen fuer Suspects_Dataset
# Alle Zeichenfolgen in den DataFrame in Kleinbuchstaben umwandeln
suspects_df = suspect_dataset.applymap(lambda x: x.lower() if isinstance(x, str) else x)

# Filterungsprozess fuer Suspects_Dataset
suspects_sunlight = suspects_df[~suspects_df.isin(['sunlight']).any(axis=1)]  # Loeschen aller Zeilen, in denen "sunlight" vorkommt
# Filtern des suspects_sunlight nach dem Monster "Witch"
witch_suspects = suspects_sunlight[suspects_sunlight['Monster'] == 'witch']
# Filtern des witch_suspects nach "Witch" welche eine kriminelle Historie haben
witch_suspects_criminalhistory = witch_suspects[witch_suspects['Criminal record'] == 'yes']
# Filtern der Hexen, die stark und schnell sind
potential_witch = witch_suspects_criminalhistory[
    (witch_suspects_criminalhistory['Strength Level'] > 5) &
    (witch_suspects_criminalhistory['Speed Level'] > 30) &
    (witch_suspects_criminalhistory['Favorite Food'] == 'humans') &
    (witch_suspects_criminalhistory['Age'] >= 300) &
    (witch_suspects_criminalhistory['Age'] <= 500)
]

potential_witch.to_csv('potential_witch_dataset.csv', index=False)

# Filter für Verbrechen, die am 30.10.2024 stattfanden
crime_on_specific_date = crimes_dataset[crimes_dataset['Date'] == '2024-10-30']

# Ausgabe der Verbrechen, die an diesem Datum stattfanden
crime_on_specific_date.to_csv('Verbrechen_am_30102024.csv', index=False)