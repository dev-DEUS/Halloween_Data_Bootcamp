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



# Aenderungen fuer Suspects_Dataset
# Alle Zeichenfolgen in den DataFrame in Kleinbuchstaben umwandeln
suspects_df = suspect_dataset.applymap(lambda x: x.lower() if isinstance(x, str) else x)

# Filterungsprozess fuer Suspects_Dataset
suspects_sunlight = suspects_df[~suspects_df.isin(['sunlight']).any(axis=1)]  # Loeschen aller Zeilen, in denen "sunlight" vorkommt
# Filtern des suspects_sunlight nach dem Monster "Witch"
witch_suspects = suspects_sunlight[suspects_sunlight['Monster'] == 'witch']
# Filtern des witch_suspects nach "Witch" welche eine kriminelle Historie haben
witch_suspects_criminalhistory = witch_suspects[witch_suspects['Criminal record'] == 'yes']


# Plot: Verteilung der Altersgruppen von Hexen mit krimineller Vorgeschichte
plt.figure(figsize=(10, 6))
sns.histplot(witch_suspects_criminalhistory['Age'], bins=20, kde=True)
plt.title('Verteilung des Alters von Hexen mit krimineller Vorgeschichte')
plt.xlabel('Alter')
plt.ylabel('Anzahl')
plt.show()

# Plot: Stärkenlevel der Hexen mit krimineller Vorgeschichte
plt.figure(figsize=(10, 6))
sns.countplot(data=witch_suspects_criminalhistory, x='Strength Level')
plt.title('Stärkenlevel der Hexen mit krimineller Vorgeschichte')
plt.xlabel('Stärkenlevel')
plt.ylabel('Anzahl')
plt.show()

# Plot: Allergien der Hexen mit krimineller Vorgeschichte
plt.figure(figsize=(10, 6))
sns.countplot(data=witch_suspects_criminalhistory, x='Allergy')
plt.title('Allergien der Hexen mit krimineller Vorgeschichte')
plt.xlabel('Allergie')
plt.ylabel('Anzahl')
plt.xticks(rotation=45)
plt.show()



