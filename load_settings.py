import pandas as pd

n_exps = {'a': 9, 'b': 5, 'c': 5, 'd': 4, 'e': 4, 'f': 4}
ALL_KEYS = []
for n in 'abcdef':
    ALL_KEYS += [n + str(i) for i in range(1, n_exps[n]+1)]


def load_settings(keys):
    if type(keys) is str:
        keys = ALL_KEYS if keys == 'all' else [keys]

    settings = {}
    for k in keys:
        df = pd.read_excel('processing_settings.xlsx', sheet_name=k, index_col=0)

        settings[k] = {"id": k}
        for key, row in df.iterrows():
            settings[k][key] = row.Value
        if type(settings[k]['hcrop']) == str:
            settings[k]["hcrop"] = [int(val) for val in settings[k]["hcrop"][1:-1].split(',')]
            settings[k]["vcrop"] = [int(val) for val in settings[k]["vcrop"][1:-1].split(',')]
    if len(settings) == 1:
        return settings[keys[0]]
    return settings

