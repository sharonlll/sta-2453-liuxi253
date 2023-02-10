import pandas as pd
import argparse
import sys

# Load whatever imports you need, but make sure to add them to the requirements.txt file.

# My imports
import torch
import torch.nn.functional as F
import torch.nn as nn


def riskfuel_test(df: pd.DataFrame) -> float:
    """
    Riskfuel Testing Function
    by <student-name>: <member_1> <member_2> .... <member_k>

    arguments: pandas DataFrame type with the following columns.. ['S','K','T','r','sigma','value'] all are of type float32
    ouputs: mean absolute error (float)

    Once you have finished model training/developemnt you must save the model within the repo and load it in using this function.

    You are free to import any python packages you desire but you must add them to the requirements.txt file.

    This function must do the following:
        - Successfully load your own model.
        - Take in model dataframe consisting of (N x 6) float32's.
        - Take the (N x 5) columns regarding the inputs to the pricer ['S','K','T','r','sigma'] and have your model price them.
        - Return the Mean  Absolute Error of the model.

    Do not put the analytic pricer as part of your network.
    Do not do any trickery with column switching as part of your answer.

    These will be checked by hand, any gaslighting will result in automatic disqualification.

    The following example has been made available to you.
    """

    # DEFINITIONS.
    student_name = "Xiaoran Liu"  # adjust this.
    full_name = "Xiaoran Liu" # adjust this.
    student_id_name = "1005313377"

    print(f"\n\n ============ Evaluating Student: {student_name} ========================= ")
    print(" Full name:")
    print("", full_name)
    print(" Student ID:")
    print("", student_id_name)
    print(" ================================================================ \n")

    # ===============   Example Code  ===============

    # My model uses PyTorch but you can use whatever package you like,
    # as long you write code to load it and effectively calculate the mean absolute aggregate error.

    # LOAD MODEL
    mm = PutNet()
    mm.load_state_dict(torch.load("simple-model.pt"))
    mm.eval()  # evaluation mode

    # EVALUATE MODEL

    # Acquire inputs/outputs
    x = torch.Tensor(df[["S", "K", "T", "r", "sigma"]].to_numpy())
    y = torch.Tensor(df[["value"]].to_numpy())

    # Pass data through model
    y_hat = mm(x)

    # Calculate mean squared error
    result = F.mse_loss(y_hat, y)

    # Return performance metric; must be of type float
    return result.item()


# A SIMPLE MODEL.
class PutNet(nn.Module):
    """
    Example of model Neural Network that could be trained price model put option.
    TODO: modify me!
    """

    def __init__(self) -> None:
        super(PutNet, self).__init__()
        self.l1 = nn.Linear(5, 23)
        self.l2 = nn.Linear(23, 23)
        self.l3 = nn.Linear(23, 11)
        self.out = nn.Linear(11, 1)
        # self.l1 = nn.Linear(5, 13)
        # self.l2 = nn.Linear(13, 26)
        # self.l3 = nn.Linear(26, 52)
        # self.l4 = nn.Linear(52, 26)
        # self.l5 = nn.Linear(26, 13)
        # self.l6 = nn.Linear(13, 5)
        # self.out = nn.Linear(5, 1)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = self.out(x)
        # x = F.relu(self.l1(x))
        # x = F.relu(self.l2(x))
        # x = F.relu(self.l3(x))
        # x = F.relu(self.l4(x))
        # x = F.relu(self.l5(x))
        # x = F.relu(self.l6(x))
        # x = self.out(x)
        return x



def get_parser():
    """Parses the command line for the dataframe file name"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_frame_name", type=str)
    return parser


def main(args):
    """Parses arguments and evaluates model performance"""

    # Parse arguments.
    parser = get_parser()
    args = parser.parse_args(args)

    # Load DataFrame and pass through riskfuel_test function.
    df = pd.read_csv(args.data_frame_name)
    performance_metric = riskfuel_test(df)

    # Must pass this assertion
    assert isinstance(performance_metric, float)

    print(f" MODEL PERFORMANCE: {performance_metric} \n\n")


if __name__ == "__main__":
    main(sys.argv[1:])
