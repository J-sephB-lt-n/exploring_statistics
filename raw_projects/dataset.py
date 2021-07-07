import pandas as pd
import tensorflow as tf


data_dir = "../data/task_2.csv"
label = "overall"


def get_ds_from_df(df_tmp, cols, label, batch_size=1024):
    dataset = (
        tf.data.Dataset.from_tensor_slices(
            (df_tmp[cols].to_dict("list"), df_tmp[label].values)
        )
        .shuffle(len(df_tmp))
        .repeat(1)
        .prefetch(1)
        .batch(batch_size)
    )
    return dataset


def get_test_and_train(cols, df=None):
    ratings = pd.read_csv(data_dir) if df is None else df
    train_ds = ratings[~ratings["valid"]]
    train_ds = get_ds_from_df(train_ds, cols, label)

    valid_ds = ratings[ratings["valid"]]
    valid_ds = get_ds_from_df(valid_ds, cols, label)

    return train_ds, valid_ds
