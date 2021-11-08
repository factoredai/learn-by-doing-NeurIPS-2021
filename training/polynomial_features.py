import glob
import pickle
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures, StandardScaler

DictDataset = Dict[str, pd.DataFrame]


def get_state_columns(df: pd.DataFrame) -> List[str]:
    """Get names of columns related to the state of the system.

    Parameters
    ----------
    df : pd.DataFrame
        Raw input data.

    Returns
    -------
    List[str]
        List of state column names.
    """
    state_columns = [
        col for col in df.columns if col if col.startswith(("X", "Y", "d"))
    ]
    return state_columns


def get_input_columns(df: pd.DataFrame) -> List[str]:
    """Get names of columns related to the inputs given to the system.

    Parameters
    ----------
    df : pd.DataFrame
        Raw input data.

    Returns
    -------
    List[str]
        List of input column names.
    """
    input_columns = [col for col in df.columns if col if col.startswith("U")]
    return input_columns


def sample_keys_train_val(
    dict_dataset: DictDataset, percentage_val=0.2
) -> Tuple[DictDataset, DictDataset, np.ndarray]:
    """Returns data split by train and validation trajectories.

    Parameters
    ----------
    dict_dataset : DictDataset
        Dataset with all trajectories.
    percentage_val : float, optional
        Percentage of trajectories to use in validation, by default 0.2

    Returns
    -------
    Tuple[DictDataset, DictDataset, np.ndarray]
        Train and validation dataset and keys of trajectories defined for validation.
    """
    list_of_keys = list(dict_dataset.keys())
    validation_keys = np.random.choice(
        list_of_keys, size=int(len(dict_dataset) * percentage_val), replace=False
    )
    train_dict = {}
    validation_dict = {}

    for key in list_of_keys:
        if key in validation_keys:
            validation_dict[key] = dict_dataset[key]
        else:
            train_dict[key] = dict_dataset[key]
    return train_dict, validation_dict, validation_keys


def convert_dict_to_np_dataset(
    dict_dataset: DictDataset, include_delta=False
) -> Tuple[np.ndarray, np.ndarray]:
    """Converts dictionary of datasets into numpy arrays used for training the model.

    Parameters
    ----------
    dict_dataset : DictDataset
        Datasets organized by trajectory
    include_delta : bool, optional
        Whether to include the delta of positions between steps, by default False

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Trajectory data joined in a numpy array.
    """
    X = []
    y = []
    for _, traj_df in dict_dataset.items():
        if include_delta:
            X.append(
                np.concatenate(
                    [
                        traj_df.iloc[:-1][state_columns].to_numpy(),
                        traj_df.iloc[1:][next_positions].to_numpy(),
                        traj_df.iloc[1:][next_positions].to_numpy()
                        - traj_df.iloc[:-1][next_positions].to_numpy(),
                        traj_df.iloc[:-1][control_input_columns].to_numpy(),
                    ],
                    axis=1,
                )
            )
        else:
            X.append(
                np.concatenate(
                    [
                        traj_df.iloc[:-1][state_columns].to_numpy(),
                        traj_df.iloc[1:][next_positions].to_numpy(),
                        traj_df.iloc[:-1][control_input_columns].to_numpy(),
                    ],
                    axis=1,
                )
            )
        y.append(traj_df.iloc[1:][state_columns].to_numpy())

    X_arr = np.concatenate(X, axis=0)
    y_arr = np.concatenate(y, axis=0)
    return X_arr, y_arr


if __name__ == "__main__":
    systems_path = "./training_trajectories/systems"

    with open(systems_path, "r") as f:
        systems_list = [i.strip() for i in f.readlines()]

    for system_idx in range(len(systems_list)):
        data = {}
        for file in glob.glob(
            f"./training_trajectories/{systems_list[system_idx]}_*.csv"
        ):
            data[file[-6:-4]] = pd.read_csv(file)

        # Getting the first trajectory to explore columns
        traj1 = data["00"]
        state_columns = get_state_columns(traj1)
        control_input_columns = get_input_columns(traj1)
        next_positions = ["X", "Y"]
        INCLUDE_DELTA = True

        n_inputs = len(state_columns) + len(next_positions)
        if INCLUDE_DELTA:
            n_inputs += 2
        n_outputs = len(control_input_columns)

        # Splitting the dicts
        train_dict, validation_dict, validation_keys = sample_keys_train_val(data)
        X_train, y_train = convert_dict_to_np_dataset(
            train_dict, include_delta=INCLUDE_DELTA
        )
        X_test, y_test = convert_dict_to_np_dataset(
            validation_dict, include_delta=INCLUDE_DELTA
        )

        # Convert x train and x test to cuadratic
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("poly", PolynomialFeatures(2)),
                ("model", LinearRegression()),
            ]
        )

        # Get the rotation matrix and its inverse for the input transformation:
        R, _, Rinv = np.linalg.svd(np.cov(y_train, rowvar=False))

        pipe = TransformedTargetRegressor(
            regressor=pipe, transformer=MinMaxScaler((-1, 1), clip=True)
        )

        print(R.shape, y_train.shape)
        y_train = np.matmul(y_train, R)
        model = pipe.fit(X_train, y_train)

        y_test_pred = np.matmul(model.predict(X_test), Rinv)
        y_train_pred = model.predict(X_train)

        model_save_path = (
            f"../models/model_scikitquadratic_clipping_{systems_list[system_idx]}"
        )
        r_save_path = (
            f"../models/model_scikitquadratic_clipping_R_{systems_list[system_idx]}"
        )

        with open(model_save_path, "wb") as f:
            pickle.dump(model, f)

        with open(r_save_path, "wb") as f:
            pickle.dump(Rinv, f)

        print(
            f"MAE for system {systems_list[system_idx]} is :",
            np.mean(np.abs(y_test_pred.flatten() - y_test.flatten())),
        )
