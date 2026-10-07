import pandas as pd

datasets = [
    ('UNSW-NB15',       'data/UNSW_NB15_combined.csv',  'attack_cat'),
    ('CSE-CIC-IDS2018', 'data/CIC_IDS2018_combined.csv', 'Label'),
    ('CIC-IOT2023',     'data/CICIOT23_combined.csv',    'label'),
]

for name, path, col in datasets:
    try:
        df = pd.read_csv(path, usecols=[col])
        vc = df[col].value_counts().sort_values(ascending=False)
        total = len(df)
        print(f'\n=== {name} === Total rows: {total:,} | Classes: {len(vc)}')
        print(f'  {"Class":<35} {"Count":>10}  {"% of total":>10}')
        print(f'  {"-"*58}')
        for cls, cnt in vc.items():
            pct = cnt/total*100
            print(f'  {str(cls):<35} {cnt:>10,}  {pct:>9.2f}%')
    except Exception as e:
        print(f'Error reading {name}: {e}')
