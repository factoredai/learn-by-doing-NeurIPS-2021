import joblib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def train_linear_controller(
    system: str,
    training_data: Path,
    mode: str = 'submission'
) -> dict:
    """
    Trains a linear controller for the given `system` using the data stored in
    the `training_data` path employing the system model / controller approach
    and returns the controller parameters in a dictionary.

    The `mode` parameter may be "testing" or "submission" and controls whether
    a test split is used and test performances reported.
    """
    # Verify that the `mode` argument is valid:
    assert mode in ('testing', 'submission'), (
        'mode can only be "testing" or "submission"'
    )

    # Initialize and fill a list that stores one dataframe for each trajectory:
    dataframes = []
    for file in training_data.glob(f'{system}_*.csv'):
        dataframes.append(pd.read_csv(file))

    # Get state and input variables:
    state_vars = [
        col for col in dataframes[0].columns
        if col.startswith(('X', 'Y', 'd'))
    ]
    n_states = len(state_vars)  # number of state variables
    input_vars = [
        col for col in dataframes[0].columns if col.startswith('U')
    ]
    n_inputs = len(input_vars)  # number of inputs

    # Add bias and targets to the dataframes:

    for i, df in enumerate(dataframes):
        new_df = df.iloc[:-1].copy()

        new_df['bias'] = np.full(len(new_df), 1)

        new_df[[f'new_{var}' for var in state_vars]] =\
            df.iloc[1:][state_vars].reset_index(drop=True)

        dataframes[i] = new_df

    # Concatenate all the dataframes together:
    df = pd.concat(dataframes, axis=0, ignore_index=True)

    if mode == 'testing':
        # Define the training and test sets:
        X_train, X_test, Y_train, Y_test = train_test_split(
            df.loc[:, state_vars + input_vars + ['bias']],
            df.loc[:, [f'new_{var}' for var in state_vars]],
            test_size=0.2,
            random_state=42,
        )
    elif mode == 'submission':
        # Use all data (do not split) to train models used for submission:
        X_train = df.loc[:, state_vars + input_vars + ['bias']]
        Y_train = df.loc[:, [f'new_{var}' for var in state_vars]]

    # Get the rotation matrix and its inverse for the input transformation:
    R, _, Rinv = np.linalg.svd(X_train[input_vars].cov().to_numpy())

    X_train.loc[:, input_vars] = X_train[input_vars].to_numpy() @ R

    # Get the min-max scaler for the input transformation:
    mms = MinMaxScaler(feature_range=(-1, 1))
    mms.fit(X_train[input_vars])

    X_train.loc[:, input_vars] = mms.transform(X_train[input_vars])

    # Find a model matrix with a linear regression:
    M = np.linalg.lstsq(
        X_train.to_numpy(), Y_train.to_numpy(), rcond=None)[0]

    if mode == 'testing':
        # Report test MAE:
        mae = np.mean(np.abs(X_test.to_numpy() @ M - Y_test.to_numpy()))
        print(f'Test MAE for {system} system: {mae}\n')

    # System matrix in usual Control Theory notation such that the system
    # can be expressed as x(k + 1) = A x(k) + B u(k) + d:
    A = M[:n_states].T
    B = M[n_states:n_states + n_inputs].T
    d = M[n_states + n_inputs]

    # Matrix that determines the control law:
    Binv = np.linalg.pinv(B[:2])

    # Return the controller parameters:
    return {
        'n_states': n_states,
        'n_inputs': n_inputs,
        'scaler': mms,
        'R': R,
        'A': A,
        'B': B,
        'd': d,
        'Binv': Binv,
    }


def train_linear_controllers_for_bumblebees(mode: str) -> None:
    """
    Trains the linear controllers for the bumblebee systems following the
    system model / controller approach and saves the controllers' parameters to
    a file.

    The `mode` parameter may be "testing" or "submission" and controls whether
    a test split is used and test performances reported.
    """
    # Specify the training data path and where the controller data will be
    # stored:
    training_data = Path('training_trajectories')
    controller_data = Path('models')

    # Read system names:
    with open(training_data / 'systems', 'r') as f:
        systems = f.read().splitlines()

    # Create a dictionary with information about the controllers:
    controllers_params = {
        system: train_linear_controller(system, training_data, mode)
        for system in systems if system.endswith('bumblebee')
    }

    # Save the controllers' parameters:
    joblib.dump(
        controllers_params,
        controller_data / 'linear_controllers_params_bumblebee.joblib'
    )


if __name__ == '__main__':
    MODE = 'submission'
    train_linear_controllers_for_bumblebees(MODE)
