import copy
import torch

from code.data_utils.dataset import DatasetLoader
from code.data_utils.utils import (load_caption, load_description,
                                   save_message, load_gnn_predictions)
from code.message_template import (cls_explainer_template, reg_explainer_template,
                                   task_list_first_person, task_list_third_person)
from code.config import cfg, update_cfg
from code.utils import set_seed


def generate_ep_message_cls(
        indices, template, task,
        smiles, preds, labels
):
    list_message = []
    for index in indices:
        pred_label = "True" if preds[index] > 0.5 else "False"
        true_label = "True" if labels[index] > 0.5 else "False"
        possibility = preds[index] if preds[index] > 0.5 \
            else 1 - preds[index]
        message = copy.deepcopy(template)
        message[0]["content"] = message[0]["content"].format(task)
        message[1]["content"] = message[1]["content"].format(
            index, smiles[index],
            index, pred_label, possibility,
            index, true_label, index
        )
        list_message.append(message)

    return list_message


def generate_ep_message_reg(
        indices, template, task,
        smiles, preds, labels
):
    list_message = []
    for index in indices:
        message = copy.deepcopy(template)
        message[0]["content"] = message[0]["content"].format(task)
        message[1]["content"] = message[1]["content"].format(
            index, smiles[index],
            index, preds[index],
            index, labels[index], index
        )
        list_message.append(message)

    return list_message


def generate_epd_message_cls(
        indices, template, task,
        smiles, description, preds, labels
):
    list_message = []
    for index in indices:
        pred_label = "True" if preds[index] > 0.5 else "False"
        true_label = "True" if labels[index] > 0.5 else "False"
        possibility = preds[index] if preds[index] > 0.5 \
            else 1 - preds[index]
        message = copy.deepcopy(template)
        message[0]["content"] = message[0]["content"].format(task)
        message[1]["content"] = message[1]["content"].format(
            index, smiles[index], description[index],
            index, pred_label, possibility,
            index, true_label, index
        )
        list_message.append(message)

    return list_message


def generate_epd_message_reg(
        indices, template, task,
        smiles, description, preds, labels
):
    list_message = []
    for index in indices:
        message = copy.deepcopy(template)
        message[0]["content"] = message[0]["content"].format(task)
        message[1]["content"] = message[1]["content"].format(
            index, smiles[index], description[index],
            index, preds[index],
            index, labels[index], index
        )
        list_message.append(message)

    return list_message


def main(cfg):
    set_seed(cfg.seed)

    # Preprocess data
    dataloader = DatasetLoader(name=cfg.dataset, text='raw')
    dataset, smiles = dataloader.dataset, dataloader.text

    caption = load_caption(dataset_name=cfg.dataset, demo_test=False)
    description = load_description(dataset_name=cfg.dataset, demo_test=cfg.demo_test)

    task = task_list_third_person[cfg.dataset]

    predictions = torch.sigmoid(load_gnn_predictions(
        dataset_name=cfg.dataset, gnn_model_name=cfg.gnn.model.name,
        feature=cfg.data.feature, lm_model_name=cfg.lm.model.name, seed=cfg.seed
    )).squeeze().numpy()
    labels = dataset.y.squeeze().numpy()

    split_idx = dataset.get_idx_split()
    valid_indices = split_idx['valid']

    if 'classification' in dataset.task_type:
        template_set = cls_explainer_template

        template_name = "EP"
        print("Generating {}...".format(template_name))
        list_message = generate_ep_message_cls(
            indices=valid_indices, template=template_set[template_name],
            task=task,
            smiles=smiles, preds=predictions, labels=labels
        )
        save_message(
            dataset_name=cfg.dataset, list_message=list_message,
            message_type=template_name, gnn_model=cfg.gnn.model.name,
            seed=cfg.seed, demo_test=cfg.demo_test
        )

        template_name = "EPD"
        print("Generating {}...".format(template_name))
        list_message = generate_epd_message_cls(
            indices=valid_indices, template=template_set[template_name],
            task=task, smiles=smiles, description=description,
            preds=predictions, labels=labels
        )
        save_message(
            dataset_name=cfg.dataset, list_message=list_message,
            message_type=template_name, gnn_model=cfg.gnn.model.name,
            seed=cfg.seed, demo_test=cfg.demo_test
        )

        template_name = "EPC"
        print("Generating {}...".format(template_name))
        list_message = generate_epd_message_cls(
            indices=valid_indices, template=template_set["EPD"],
            task=task, smiles=smiles, description=caption,
            preds=predictions, labels=labels
        )
        save_message(
            dataset_name=cfg.dataset, list_message=list_message,
            message_type=template_name, gnn_model=cfg.gnn.model.name,
            seed=cfg.seed, demo_test=cfg.demo_test
        )

    else:
        template_set = reg_explainer_template

        template_name = "EP"
        print("Generating {}...".format(template_name))
        list_message = generate_ep_message_reg(
            indices=valid_indices, template=template_set[template_name],
            task=task, smiles=smiles, preds=predictions, labels=labels
        )
        save_message(
            dataset_name=cfg.dataset, list_message=list_message,
            message_type=template_name, gnn_model=cfg.gnn.model.name,
            seed=cfg.seed, demo_test=cfg.demo_test
        )

        template_name = "EPD"
        print("Generating {}...".format(template_name))
        list_message = generate_epd_message_reg(
            indices=valid_indices, template=template_set[template_name],
            task=task, smiles=smiles, description=description,
            preds=predictions, labels=labels
        )
        save_message(
            dataset_name=cfg.dataset, list_message=list_message,
            message_type=template_name, gnn_model=cfg.gnn.model.name,
            seed=cfg.seed, demo_test=cfg.demo_test
        )

        template_name = "EPC"
        print("Generating {}...".format(template_name))
        list_message = generate_epd_message_reg(
            indices=valid_indices, template=template_set["EPD"],
            task=task, smiles=smiles, description=caption,
            preds=predictions, labels=labels
        )
        save_message(
            dataset_name=cfg.dataset, list_message=list_message,
            message_type=template_name, gnn_model=cfg.gnn.model.name,
            seed=cfg.seed, demo_test=cfg.demo_test
        )


if __name__ == "__main__":
    cfg = update_cfg(cfg)
    main(cfg)
