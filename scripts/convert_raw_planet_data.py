import os
import shutil
import pandas as pd


if __name__ == '__main__':
    src_dir = '../data/raw/planet'
    dst_dir = '../data/datasets/planet/train'

    shutil.copytree(os.path.join(src_dir, 'train-jpg'), os.path.join(dst_dir, 'images'))

    train_df = pd.read_csv(os.path.join(src_dir, 'train_classes.csv'))

    unique_tags = set()

    for temp_list in train_df.tags.apply(lambda x: x.split(' ')):
        unique_tags.update(temp_list)

    for temp_tag in unique_tags:
        train_df[temp_tag] = train_df.tags.apply(lambda x: temp_tag in x)

    unique_tags = list(unique_tags)
    # Переставляем метки в порядке написания в README
    train_df = train_df[[
        'image_name',
        'clear',
        'partly_cloudy',
        'cloudy',
        'haze',
        'agriculture',
        'cultivation',
        'bare_ground',
        'conventional_mine',
        'artisinal_mine',
        'primary',
        'blooming',
        'selective_logging',
        'blow_down',
        'slash_burn',
        'habitation',
        'water',
        'road',
    ]]

    train_df.to_csv(os.path.join(dst_dir, 'labels.csv'))
