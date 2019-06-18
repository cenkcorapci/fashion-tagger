def generate_get_data_set(df):
    ds = []
    for _, row in df.iterrows():
        labels = []
        labels.append(row['gender'])
        labels.append(row['masterCategory'])
        labels.append(row['subCategory'])
        labels.append(row['articleType'])
        labels.append(row['baseColour'])
        labels.append(row['season'])
        labels.append(row['usage'])
        ds.append([row['image'], labels])
    return ds


def get_target_list(df):
    target_list = []
    target_list += df.gender.unique().tolist()
    target_list += df.masterCategory.unique().tolist()
    target_list += df.subCategory.unique().tolist()
    target_list += df.articleType.unique().tolist()
    target_list += df.baseColour.unique().tolist()
    target_list += df.season.unique().tolist()
    target_list += df.usage.unique().tolist()
    return sorted(list(set(target_list)))
