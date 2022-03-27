import pandas as pd
import joblib
from utils import clean_data


def save_label_dicts(df):
    # remove rows from the DataFrame which do not have corresponding images
    df = clean_data(df)
    # we will use the `gender`, `masterCategory`. and `subCategory` labels
    # mapping `gender` to numerical values
    cat_list_gender = df['gender'].unique()
    # 5 unique categories for gender
    num_list_gender = {cat:i for i, cat in enumerate(cat_list_gender)}
    # mapping `masterCategory` to numerical values
    cat_list_master = df['masterCategory'].unique()
    # 7 unique categories for `masterCategory`
    num_list_master = {cat:i for i, cat in enumerate(cat_list_master)}
    # mapping `subCategory` to numerical values
    cat_list_sub = df['subCategory'].unique()
    # 45 unique categories for `subCategory`
    num_list_sub = {cat:i for i, cat in enumerate(cat_list_sub)}
    joblib.dump(num_list_gender, '../input/num_list_gender.pkl')
    joblib.dump(num_list_master, '../input/num_list_master.pkl')
    joblib.dump(num_list_sub, '../input/num_list_sub.pkl')
df = pd.read_csv('/styles.csv', usecols=[0, 1, 2, 3, 4, 4, 5, 6, 9])
save_label_dicts(df)


