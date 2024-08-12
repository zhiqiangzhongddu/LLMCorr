import numpy as np
import copy

import torch

from code.data_utils.dataset import DatasetLoader
from code.data_utils.utils import (load_caption, load_description,
                                   save_message,
                                   load_gnn_predictions)
from code.message_template import (cls_corrector_message_template,
                                   reg_corrector_message_template,
                                   task_list_first_person, task_list_third_person)
from code.config import cfg, update_cfg
from code.utils import set_seed


def generate_corrector_fs_message_cls(
        message_type, template_set, smiles, valid_indices, indices,
        task_first_p, task_third_p, preds, labels
):

    fs = len(valid_indices) if message_type.split('-')[1] == "all" \
        else int(message_type.split('-')[1])
    template = template_set["CorrFS"]
    knowledge_template = template_set["CorrFS_knowledge"]

    list_message = []
    for idx in indices:
        smi = smiles[idx]
        knowledge = []

        _index = np.random.choice(valid_indices, fs)

        for knowledge_id in range(fs):
            _knowledge = copy.deepcopy(knowledge_template)
            pred_label = "{}".format(task_third_p) if preds[_index[knowledge_id]] > 0.5 \
                else "cannot {}".format(task_first_p)
            true_label = "True" if labels[_index[knowledge_id]] > 0.5 else "False"
            possibility = preds[_index[knowledge_id]] if preds[_index[knowledge_id]] > 0.5 \
                else 1 - preds[_index[knowledge_id]]
            _knowledge[0]["content"] = _knowledge[0]["content"].format(
                _index[knowledge_id], smiles[_index[knowledge_id]],
                _index[knowledge_id], pred_label, possibility,
                _index[knowledge_id], task_third_p
            )
            _knowledge[1]["content"] = _knowledge[1]["content"].format(true_label)
            knowledge += _knowledge

        _message = copy.deepcopy(template)
        # _message[0]["content"] = _message[0]["content"].format(task_third_p)
        pred_label = "{}".format(task_third_p) if preds[idx] > 0.5 \
            else "cannot {}".format(task_first_p)
        possibility = preds[idx] if preds[idx] > 0.5 else 1 - preds[idx]
        _message[1]["content"] = _message[1]["content"].format(
            idx, smi,
            idx, pred_label, task_first_p, possibility,
            idx, task_third_p
        )

        message = [_message[0]]
        message += knowledge
        message += [_message[1]]

        list_message.append(message)

    return list_message


# def generate_corrector_fsd_message_cls(
#         message_type, template_set, smiles, description,
#         valid_indices, indices, task, preds, labels
# ):
#
#     fs = len(valid_indices) if message_type.split('-')[1] == "all" \
#         else int(message_type.split('-')[1])
#     template = template_set["CorrFSD"]
#     knowledge_template = template_set["CorrFSD_knowledge"]
#
#     list_message = []
#     for idx in indices:
#         smi = smiles[idx]
#         des = description[idx]
#         knowledge = []
#
#         _index = np.random.choice(valid_indices, fs)
#
#         for knowledge_id in range(fs):
#             _knowledge = copy.deepcopy(knowledge_template)
#             pred_label = "True" if preds[_index[knowledge_id]] > 0.5 else "False"
#             true_label = "True" if labels[_index[knowledge_id]] > 0.5 else "False"
#             possibility = preds[_index[knowledge_id]] if preds[_index[knowledge_id]] > 0.5 \
#                 else 1 - preds[_index[knowledge_id]]
#             _knowledge[0]["content"] = _knowledge[0]["content"].format(
#                 _index[knowledge_id], smiles[_index[knowledge_id]],
#                 pred_label, round(possibility, 4), description[_index[knowledge_id]]
#             )
#             _knowledge[1]["content"] = _knowledge[1]["content"].format(true_label)
#             knowledge += _knowledge
#
#         _message = copy.deepcopy(template)
#         _message[0]["content"] = _message[0]["content"].format(task)
#         pred_label = "True" if preds[idx] > 0.5 else "False"
#         possibility = preds[idx] if preds[idx] > 0.5 else 1 - preds[idx]
#         _message[1]["content"] = _message[1]["content"].format(
#             idx, smi, pred_label, round(possibility, 4), des
#         )
#
#         message = [_message[0]]
#         message += knowledge
#         message += [_message[1]]
#
#         list_message.append(message)
#
#     return list_message


def generate_corrector_fs_message_reg(
        message_type, template_set, smiles,
        valid_indices, indices, task, preds, labels
):

    fs = len(valid_indices) if message_type.split('-')[1] == "all" \
        else int(message_type.split('-')[1])
    template = template_set["CorrFS"]
    knowledge_template = template_set["CorrFS_knowledge"]

    list_message = []
    for idx in indices:
        smi = smiles[idx]
        knowledge = []

        _index = np.random.choice(valid_indices, fs)

        for knowledge_id in range(fs):
            _knowledge = copy.deepcopy(knowledge_template)
            _knowledge[0]["content"] = _knowledge[0]["content"].format(
                _index[knowledge_id], smiles[_index[knowledge_id]], preds[_index[knowledge_id]]
            )
            _knowledge[1]["content"] = _knowledge[1]["content"].format(labels[_index[knowledge_id]])
            knowledge += _knowledge

        _message = copy.deepcopy(template)
        _message[0]["content"] = _message[0]["content"].format(task)
        _message[1]["content"] = _message[1]["content"].format(idx, smi, preds[idx])

        message = [_message[0]]
        message += knowledge
        message += [_message[1]]

        list_message.append(message)

    return list_message


# def generate_corrector_fsd_message_reg(
#         message_type, template_set, smiles, description,
#         valid_indices, indices, task, preds, labels
# ):
#
#     fs = len(valid_indices) if message_type.split('-')[1] == "all" \
#         else int(message_type.split('-')[1])
#     template = template_set["CorrFSD"]
#     knowledge_template = template_set["CorrFSD_knowledge"]
#
#     list_message = []
#     for idx in indices:
#         smi = smiles[idx]
#         des = description[idx]
#         knowledge = []
#
#         _index = np.random.choice(valid_indices, fs)
#
#         for knowledge_id in range(fs):
#             _knowledge = copy.deepcopy(knowledge_template)
#             _knowledge[0]["content"] = _knowledge[0]["content"].format(
#                 _index[knowledge_id], smiles[_index[knowledge_id]],
#                 preds[_index[knowledge_id]], description[_index[knowledge_id]]
#             )
#             _knowledge[1]["content"] = _knowledge[1]["content"].format(labels[_index[knowledge_id]])
#             knowledge += _knowledge
#
#         _message = copy.deepcopy(template)
#         _message[0]["content"] = _message[0]["content"].format(task)
#         _message[1]["content"] = _message[1]["content"].format(idx, smi, preds[idx], des)
#
#         message = [_message[0]]
#         message += knowledge
#         message += [_message[1]]
#
#         list_message.append(message)
#
#     return list_message


def main(cfg):
    set_seed(cfg.seed)

    # Preprocess data
    dataloader = DatasetLoader(name=cfg.dataset, text='raw')
    dataset, smiles = dataloader.dataset, dataloader.text

    caption = load_caption(dataset_name=cfg.dataset, demo_test=False)
    description = load_description(dataset_name=cfg.dataset, demo_test=cfg.demo_test)

    task_first_p = task_list_first_person[cfg.dataset]
    task_third_p = task_list_third_person[cfg.dataset]

    predictions = torch.sigmoid(load_gnn_predictions(
        dataset_name=cfg.dataset, gnn_model_name=cfg.gnn.model.name,
        feature=cfg.data.feature, lm_model_name=cfg.lm.model.name, seed=cfg.seed
    )).squeeze().numpy()
    labels = dataset.y.squeeze().numpy()

    split_idx = dataset.get_idx_split()
    valid_indices = split_idx['valid']

    if 'classification' in dataset.task_type:
        template_set = cls_corrector_message_template

        for template_name in [
            "CorrFS-10", "CorrFS-20", "CorrFS-30", "CorrFS-50", "CorrFS-100", "CorrFS-all"
        ]:
            print("Generating {}...".format(template_name))
            list_message = generate_corrector_fs_message_cls(
                message_type=template_name, template_set=template_set,
                smiles=smiles, valid_indices=valid_indices, indices=range(dataset.y.size(0)),
                task_first_p=task_first_p, task_third_p=task_third_p,
                preds=predictions, labels=labels
            )
            save_message(
                dataset_name=cfg.dataset, list_message=list_message,
                message_type=template_name, gnn_model=cfg.gnn.model.name,
                seed=cfg.seed, demo_test=cfg.demo_test
            )

        # for template_name in [
        #     "CorrFSC-10", "CorrFSC-20", "CorrFSC-30", "CorrFSC-50", "CorrFSC-100", "CorrFSC-all"
        # ]:
        #     print("Generating {}...".format(template_name))
        #     list_message = generate_corrector_fsd_message_cls(
        #         message_type=template_name, template_set=template_set,
        #         smiles=smiles, description=caption,
        #         valid_indices=valid_indices, indices=range(dataset.y.size(0)),
        #         task=task, preds=predictions, labels=labels
        #     )
        #     save_message(
        #         dataset_name=cfg.dataset, list_message=list_message,
        #         message_type=template_name, gnn_model=cfg.gnn.model.name,
        #         seed=cfg.seed, demo_test=cfg.demo_test
        #     )

    else:
        template_set = reg_corrector_message_template

        for template_name in [
            "CorrFS-10", "CorrFS-20", "CorrFS-30", "CorrFS-50", "CorrFS-100", "CorrFS-all"
        ]:
            print("Generating {}...".format(template_name))
            list_message = generate_corrector_fs_message_reg(
                message_type=template_name, template_set=template_set,
                smiles=smiles, valid_indices=valid_indices, indices=range(dataset.y.size(0)),
                preds=predictions, task=task, labels=labels
            )
            save_message(
                dataset_name=cfg.dataset, list_message=list_message,
                message_type=template_name, gnn_model=cfg.gnn.model.name,
                seed=cfg.seed, demo_test=cfg.demo_test
            )

        # for template_name in [
        #     "CorrFSC-10", "CorrFSC-20", "CorrFSC-30", "CorrFSC-50", "CorrFSC-100", "CorrFSC-all"
        # ]:
        #     print("Generating {}...".format(template_name))
        #     list_message = generate_corrector_fsd_message_reg(
        #         message_type=template_name, template_set=template_set,
        #         smiles=smiles, description=caption,
        #         valid_indices=valid_indices, indices=range(dataset.y.size(0)),
        #         task=task, preds=predictions, labels=labels
        #     )
        #     save_message(
        #         dataset_name=cfg.dataset, list_message=list_message,
        #         message_type=template_name, gnn_model=cfg.gnn.model.name,
        #         seed=cfg.seed, demo_test=cfg.demo_test
        #     )


if __name__ == "__main__":
    cfg = update_cfg(cfg)
    main(cfg)
