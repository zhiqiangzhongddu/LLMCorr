import numpy as np
import re

import torch


def extract_numbers(input_string):
    # Define a regular expression pattern to match numbers with optional positive/negative signs
    pattern = r'-?\d+\.?\d*'  # This pattern matches integers and floating-point numbers with optional signs

    # Use re.findall to find all occurrences of the pattern in the input string
    numbers = re.findall(pattern, input_string)

    # Convert the strings to actual numbers (float or int)
    numbers = [float(num) if '.' in num else int(num) for num in numbers]

    return numbers


def evaluate(y_true, y_pred_list, split_idx, task, evaluator, target="all"):
    if target.lower() == "all":
        result_train_list = []
        result_valid_list = []
        result_test_list = []

        for y_pred in y_pred_list:
            y_true = y_true.view(-1, 1)
            if "classification" in task:
                if y_pred.dim() == 2:
                    if y_pred.size(1) == 2:
                        y_pred = torch.argmax(y_pred, -1).view(-1, 1)
                    else:
                        pass
                else:
                    y_pred = y_pred.view(-1, 1)
            else:
                y_pred = y_pred.view(-1, 1)

            input_dict_train = {
                "y_true": y_true[split_idx["train"]],
                "y_pred": y_pred[split_idx["train"]]
            }
            input_dict_valid = {
                "y_true": y_true[split_idx["valid"]],
                "y_pred": y_pred[split_idx["valid"]]
            }
            input_dict_test = {
                "y_true": y_true[split_idx["test"]],
                "y_pred": y_pred[split_idx["test"]]
            }

            result_dict_train = evaluator.eval(input_dict_train)
            result_dict_valid = evaluator.eval(input_dict_valid)
            result_dict_test = evaluator.eval(input_dict_test)
            assert len(result_dict_train.keys()) == 1
            key = list(result_dict_train.keys())[0]

            result_train_list.append(result_dict_train[key])
            result_valid_list.append(result_dict_valid[key])
            result_test_list.append(result_dict_test[key])

        print("Train {}: {:.4f}; Valid {}: {:.4f}; Test {}: {:.4f}".format(
            key, np.mean(result_train_list),
            key, np.mean(result_valid_list),
            key, np.mean(result_test_list)
        ))
        print("& {:.4f} & {:.4f}".format(
            np.mean(result_valid_list),
            np.mean(result_test_list),
        ))
    elif target.lower() == "valid_test":
        result_valid_list = []
        result_test_list = []

        for y_pred in y_pred_list:
            y_true = y_true.view(-1, 1)
            if "classification" in task:
                if y_pred.dim() == 2:
                    if y_pred.size(1) == 2:
                        y_pred = torch.argmax(y_pred, -1).view(-1, 1)
                    else:
                        pass
                else:
                    y_pred = y_pred.view(-1, 1)
            else:
                y_pred = y_pred.view(-1, 1)

            input_dict_valid = {
                "y_true": y_true[split_idx["valid"]],
                "y_pred": y_pred[:len(split_idx["valid"])]
            }
            input_dict_test = {
                "y_true": y_true[split_idx["test"]],
                "y_pred": y_pred[len(split_idx["valid"]):]
            }

            result_dict_valid = evaluator.eval(input_dict_valid)
            result_dict_test = evaluator.eval(input_dict_test)
            assert len(result_dict_valid.keys()) == 1
            key = list(result_dict_valid.keys())[0]

            result_valid_list.append(result_dict_valid[key])
            result_test_list.append(result_dict_test[key])

        print("Valid {}: {:.4f}; Test {}: {:.4f}".format(
            key, np.mean(result_valid_list),
            key, np.mean(result_test_list)
        ))
        print("& {:.4f} & {:.4f}".format(
            np.mean(result_valid_list),
            np.mean(result_test_list),
        ))
    elif target.lower() == "valid":
        result_valid_list = []

        for y_pred in y_pred_list:
            y_true = y_true.view(-1, 1)
            if "classification" in task:
                if y_pred.dim() == 2:
                    if y_pred.size(1) == 2:
                        y_pred = torch.argmax(y_pred, -1).view(-1, 1)
                    else:
                        pass
                else:
                    y_pred = y_pred.view(-1, 1)
            else:
                y_pred = y_pred.view(-1, 1)

            input_dict_test = {
                "y_true": y_true[split_idx["valid"]],
                "y_pred": y_pred
            }

            result_dict_test = evaluator.eval(input_dict_test)
            assert len(result_dict_test.keys()) == 1
            key = list(result_dict_test.keys())[0]

            result_valid_list.append(result_dict_test[key])

        print("Valid {}: {:.4f}".format(
            key, np.mean(result_valid_list)
        ))
        print("& {:.4f}".format(
            np.mean(result_valid_list),
        ))
    elif target.lower() == "test":
        result_test_list = []

        for y_pred in y_pred_list:
            y_true = y_true.view(-1, 1)
            if "classification" in task:
                if y_pred.dim() == 2:
                    if y_pred.size(1) == 2:
                        y_pred = torch.argmax(y_pred, -1).view(-1, 1)
                    else:
                        pass
                else:
                    y_pred = y_pred.view(-1, 1)
            else:
                y_pred = y_pred.view(-1, 1)

            input_dict_test = {
                "y_true": y_true[split_idx["test"]],
                "y_pred": y_pred
            }

            result_dict_test = evaluator.eval(input_dict_test)
            assert len(result_dict_test.keys()) == 1
            key = list(result_dict_test.keys())[0]

            result_test_list.append(result_dict_test[key])

        print("Test {}: {:.4f}".format(
            key, np.mean(result_test_list)
        ))
        print("& {:.4f}".format(
            np.mean(result_test_list),
        ))
    else:
        raise ValueError("{} is not a valid target".format(target))
