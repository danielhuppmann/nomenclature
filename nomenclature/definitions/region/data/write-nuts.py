from pathlib import Path
import pandas as pd
import yaml
import pyam

this = Path(__file__).name

# load mapping of iso2 to common country names
import yaml
with open('../countries.yaml', 'r') as stream:
    country_mapping = yaml.load(stream, Loader=yaml.FullLoader)

iso2_mapping = dict(
    [(country_mapping[c]['iso2'], c) for c in country_mapping]
    # add alternative iso2 codes used by the European Commission to the mapping
    + [(country_mapping[c]['iso2_alt'], c) for c in country_mapping
       if 'iso2_alt' in country_mapping[c]]
)

# descriptions of the three NUTS region levels
descriptions = [
    (1, 'major socio-economic regions'),
    (2, 'basic regions for the application of regional policies'),
    (3, 'small regions for specific diagnoses')
]

# read data file
df = pd.read_excel('NUTS2016-NUTS2021.xlsx', sheet_name='NUTS & SR 2021')

# open the three files and write headers
files = []
for (n, text) in descriptions:
    file = open(f'../nuts{n}.yaml', 'w')
    file.write(f'# This file was created using the script `data/{this}`\n')
    file.write('# DO NOT ALTER THIS FILE MANUALLY!\n\n')
    file.write(f'# List of NUTS-{n} regions: {text}\n')
    files.append(file)

# iterate over dataframe and parse codes and names
country, _n1, _n2 = None, None, None
for i, row in df.iterrows():

    # treat country information translate iso2 to common country names
    if pyam.isstr(row['Country']):
        country = iso2_mapping[row['Code 2021']]
        continue

    # treat NUTS-1 level
    if pyam.isstr(row['NUTS level 1']):
        _n1 = row['Code 2021']
        level, _id, name = 1, _n1, row['NUTS level 1']

    # treat NUTS-2 level
    if pyam.isstr(row['NUTS level 2']):
        _n2 = row['Code 2021']
        level, _id, name = 2, _n2, row['NUTS level 2']

    # treat NUTS-3 level
    if pyam.isstr(row['NUTS level 3']):
        level, _id, name = 3, row['Code 2021'], row['NUTS level 3']

    # write to respective file
    file = files[level - 1]
    file.write(f'\n{_id}:\n')
    file.write(f'   name: {name}\n')
    file.write(f'   country: {country}\n')
    if level > 1:
        file.write(f'   nuts1: {_n1}\n')
    if level > 2:
        file.write(f'   nuts2: {_n2}\n')

for file in files:
    file.close()
