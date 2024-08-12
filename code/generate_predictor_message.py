import numpy as np
import copy
import torch

from code.data_utils.dataset import DatasetLoader
from code.data_utils.utils import load_caption, load_description, save_message
from code.message_template import (cls_predictor_message_template,
                                   reg_predictor_message_template,
                                   task_list_third_person)
from code.config import cfg, update_cfg
from code.utils import set_seed


def generate_fs_message_cls(
        message_type, template_set, smiles, task, index_pos, index_neg
):
    fs = int(message_type.split('-')[1])
    template = template_set["FS"]
    knowledge_pos_template = template_set["FS_knowledge_pos"]
    knowledge_neg_template = template_set["FS_knowledge_neg"]

    list_message = []
    for idx, smi in enumerate(smiles):
        knowledge = []

        _pos = np.random.choice(index_pos, fs)
        _neg = np.random.choice(index_neg, fs)
        while idx in _pos:
            _pos = np.random.choice(index_pos, fs)
        while idx in _neg:
            _neg = np.random.choice(index_neg, fs)

        for knowledge_id in range(fs):
            _knowledge_pos = copy.deepcopy(knowledge_pos_template)
            _knowledge_pos[0]["content"] = _knowledge_pos[0]["content"].format(
                smiles[_pos[knowledge_id]], task
            )
            knowledge += _knowledge_pos

            _knowledge_neg = copy.deepcopy(knowledge_neg_template)
            _knowledge_neg[0]["content"] = _knowledge_neg[0]["content"].format(
                smiles[_neg[knowledge_id]], task
            )
            knowledge += _knowledge_neg

        _message = copy.deepcopy(template)
        _message[1]["content"] = _message[1]["content"].format(smi, task)

        message = [_message[0]]
        message += knowledge
        message += [_message[1]]

        list_message.append(message)

    return list_message


def generate_fsd_message_cls(
        message_type, template_set, smiles, description,
        task, index_pos, index_neg
):
    fs = int(message_type.split('-')[1])
    template = template_set["FSD"]
    knowledge_pos_template = template_set["FSD_knowledge_pos"]
    knowledge_neg_template = template_set["FSD_knowledge_neg"]

    list_message = []
    for idx, (smi, des) in enumerate(zip(smiles, description)):
        knowledge = []

        _pos = np.random.choice(index_pos, fs)
        _neg = np.random.choice(index_neg, fs)
        while idx in _pos:
            _pos = np.random.choice(index_pos, fs)
        while idx in _neg:
            _neg = np.random.choice(index_neg, fs)

        for knowledge_id in range(fs):
            _knowledge_pos = copy.deepcopy(knowledge_pos_template)
            _knowledge_pos[0]["content"] = _knowledge_pos[0]["content"].format(
                smiles[_pos[knowledge_id]], task, description[_pos[knowledge_id]]
            )
            knowledge += _knowledge_pos

            _knowledge_neg = copy.deepcopy(knowledge_neg_template)
            _knowledge_neg[0]["content"] = _knowledge_neg[0]["content"].format(
                smiles[_neg[knowledge_id]], task, description[_neg[knowledge_id]]
            )
            knowledge += _knowledge_neg

        _message = copy.deepcopy(template)
        _message[1]["content"] = _message[1]["content"].format(smi, task, des)

        message = [_message[0]]
        message += knowledge
        message += [_message[1]]

        list_message.append(message)

    return list_message


def generate_fs_message_reg(message_type, template_set, smiles, task, label):
    fs = int(message_type.split('-')[1])
    template = template_set["FS"]
    knowledge_template = template_set["FS_knowledge"]

    list_message = []
    for idx, smi in enumerate(smiles):
        knowledge = []

        _index = np.random.choice(range(label.size(0)), fs)
        while idx in _index:
            _index = np.random.choice(range(label.size(0)), fs)

        for knowledge_id in range(fs):
            _knowledge = copy.deepcopy(knowledge_template)
            _knowledge[0]["content"] = _knowledge[0]["content"].format(
                task, smiles[_index[knowledge_id]]
            )
            _knowledge[1]["content"] = _knowledge[1]["content"].format(
                label[_index[knowledge_id]].item()
            )
            knowledge += _knowledge

        _message = copy.deepcopy(template)
        _message[1]["content"] = _message[1]["content"].format(task, smi)

        message = [_message[0]]
        message += knowledge
        message += [_message[1]]

        list_message.append(message)

    return list_message


def generate_fsd_message_reg(
        message_type, template_set, smiles, description, task, label
):
    fs = int(message_type.split('-')[1])
    template = template_set["FSD"]
    knowledge_template = template_set["FSD_knowledge"]

    list_message = []
    for idx, (smi, des) in enumerate(zip(smiles, description)):
        knowledge = []

        _index = np.random.choice(range(label.size(0)), fs)
        while idx in _index:
            _index = np.random.choice(range(label.size(0)), fs)

        for knowledge_id in range(fs):
            _knowledge = copy.deepcopy(knowledge_template)
            _knowledge[0]["content"] = _knowledge[0]["content"].format(
                task, smiles[_index[knowledge_id]], description[_index[knowledge_id]]
            )
            _knowledge[1]["content"] = _knowledge[1]["content"].format(
                label[_index[knowledge_id]].item()
            )
            knowledge += _knowledge

        _message = copy.deepcopy(template)
        _message[1]["content"] = _message[1]["content"].format(task, smi, des)

        message = [_message[0]]
        message += knowledge
        message += [_message[1]]

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

    split_idx = dataset.get_idx_split()
    index_pos = np.intersect1d(split_idx['train'], torch.where(dataset.y == 1)[0])
    index_neg = np.intersect1d(split_idx['train'], torch.where(dataset.y == 0)[0])

    if 'classification' in dataset.task_type:
        template_set = cls_predictor_message_template
    else:
        template_set = reg_predictor_message_template

    message_type = "IF"
    print('Generating {}'.format(message_type))
    template = template_set[message_type]
    list_message = []
    for smi in smiles:
        message = copy.deepcopy(template)
        message[1]["content"] = message[1]["content"].format(smi, task)
        list_message.append(message)
    save_message(
        dataset_name=cfg.dataset, list_message=list_message,
        message_type=message_type, demo_test=cfg.demo_test
    )

    message_type = "IFD"
    print('Generating {}'.format(message_type))
    template = template_set[message_type]
    list_message = []
    for smi, des in zip(smiles, description):
        message = copy.deepcopy(template)
        message[1]["content"] = message[1]["content"].format(smi, des, task)
        list_message.append(message)
    save_message(
        dataset_name=cfg.dataset, list_message=list_message,
        message_type=message_type, demo_test=cfg.demo_test
    )

    message_type = "IFC"
    print('Generating {}'.format(message_type))
    template = template_set["IFD"]
    list_message = []
    for smi, cap in zip(smiles, caption):
        message = copy.deepcopy(template)
        message[1]["content"] = message[1]["content"].format(smi, cap, task)
        list_message.append(message)
    save_message(
        dataset_name=cfg.dataset, list_message=list_message,
        message_type=message_type, demo_test=cfg.demo_test
    )

    message_type = "IP"
    print('Generating {}'.format(message_type))
    template = template_set[message_type]
    list_message = []
    for smi in smiles:
        message = copy.deepcopy(template)
        if 'classification' in dataset.task_type:
            message[1]["content"] = message[1]["content"].format(smi, task)
        else:
            message[1]["content"] = message[1]["content"].format(task, smi)
        list_message.append(message)
    save_message(
        dataset_name=cfg.dataset, list_message=list_message,
        message_type=message_type, demo_test=cfg.demo_test
    )

    message_type = "IPD"
    print('Generating {}'.format(message_type))
    template = template_set[message_type]
    list_message = []
    for smi, des in zip(smiles, description):
        message = copy.deepcopy(template)
        if 'classification' in dataset.task_type:
            message[1]["content"] = message[1]["content"].format(smi, task, des)
        else:
            message[1]["content"] = message[1]["content"].format(task, smi, des)
        list_message.append(message)
    save_message(
        dataset_name=cfg.dataset, list_message=list_message,
        message_type=message_type, demo_test=cfg.demo_test
    )

    message_type = "IPC"
    print('Generating {}'.format(message_type))
    template = template_set["IPD"]
    list_message = []
    for smi, cap in zip(smiles, caption):
        message = copy.deepcopy(template)
        if 'classification' in dataset.task_type:
            message[1]["content"] = message[1]["content"].format(smi, task, cap)
        else:
            message[1]["content"] = message[1]["content"].format(task, smi, cap)
        list_message.append(message)
    save_message(
        dataset_name=cfg.dataset, list_message=list_message,
        message_type=message_type, demo_test=cfg.demo_test
    )

    message_type = "IE"
    print('Generating {}'.format(message_type))
    template = template_set[message_type]
    list_message = []
    for smi in smiles:
        message = copy.deepcopy(template)
        if 'classification' in dataset.task_type:
            message[1]["content"] = message[1]["content"].format(smi, task)
        else:
            message[1]["content"] = message[1]["content"].format(task, smi)
        list_message.append(message)
    save_message(
        dataset_name=cfg.dataset, list_message=list_message,
        message_type=message_type, demo_test=cfg.demo_test
    )

    message_type = "IED"
    print('Generating {}'.format(message_type))
    template = template_set[message_type]
    list_message = []
    for smi, des in zip(smiles, description):
        message = copy.deepcopy(template)
        if 'classification' in dataset.task_type:
            message[1]["content"] = message[1]["content"].format(smi, task, des)
        else:
            message[1]["content"] = message[1]["content"].format(task, smi, des)
        list_message.append(message)
    save_message(
        dataset_name=cfg.dataset, list_message=list_message,
        message_type=message_type, demo_test=cfg.demo_test
    )

    message_type = "IEC"
    print('Generating {}'.format(message_type))
    template = template_set["IED"]
    list_message = []
    for smi, cap in zip(smiles, caption):
        message = copy.deepcopy(template)
        if 'classification' in dataset.task_type:
            message[1]["content"] = message[1]["content"].format(smi, task, cap)
        else:
            message[1]["content"] = message[1]["content"].format(task, smi, cap)
        list_message.append(message)
    save_message(
        dataset_name=cfg.dataset, list_message=list_message,
        message_type=message_type, demo_test=cfg.demo_test
    )

    message_type = "FS-1"
    print('Generating {}'.format(message_type))
    if 'classification' in dataset.task_type:
        list_message = generate_fs_message_cls(
            message_type=message_type, template_set=template_set,
            smiles=smiles, task=task, index_pos=index_pos, index_neg=index_neg
        )
    else:
        list_message = generate_fs_message_reg(
            message_type=message_type, template_set=template_set,
            smiles=smiles, task=task, label=dataset.y
        )
    save_message(
        dataset_name=cfg.dataset, list_message=list_message,
        message_type=message_type, demo_test=cfg.demo_test
    )

    message_type = "FS-2"
    print('Generating {}'.format(message_type))
    if 'classification' in dataset.task_type:
        list_message = generate_fs_message_cls(
            message_type=message_type, template_set=template_set,
            smiles=smiles, task=task, index_pos=index_pos, index_neg=index_neg
        )
    else:
        list_message = generate_fs_message_reg(
            message_type=message_type, template_set=template_set,
            smiles=smiles, task=task, label=dataset.y
        )
    save_message(
        dataset_name=cfg.dataset, list_message=list_message,
        message_type=message_type, demo_test=cfg.demo_test
    )

    message_type = "FS-3"
    print('Generating {}'.format(message_type))
    if 'classification' in dataset.task_type:
        list_message = generate_fs_message_cls(
            message_type=message_type, template_set=template_set,
            smiles=smiles, task=task, index_pos=index_pos, index_neg=index_neg
        )
    else:
        list_message = generate_fs_message_reg(
            message_type=message_type, template_set=template_set,
            smiles=smiles, task=task, label=dataset.y
        )
    save_message(
        dataset_name=cfg.dataset, list_message=list_message,
        message_type=message_type, demo_test=cfg.demo_test
    )

    message_type = "FSD-1"
    print('Generating {}'.format(message_type))
    if 'classification' in dataset.task_type:
        list_message = generate_fsd_message_cls(
            message_type=message_type, template_set=template_set,
            smiles=smiles, description=description,
            task=task, index_pos=index_pos, index_neg=index_neg
        )
    else:
        list_message = generate_fsd_message_reg(
            message_type=message_type, template_set=template_set,
            smiles=smiles, description=description, task=task, label=dataset.y
        )
    save_message(
        dataset_name=cfg.dataset, list_message=list_message,
        message_type=message_type, demo_test=cfg.demo_test
    )

    message_type = "FSD-2"
    print('Generating {}'.format(message_type))
    if 'classification' in dataset.task_type:
        list_message = generate_fsd_message_cls(
            message_type=message_type, template_set=template_set,
            smiles=smiles, description=description,
            task=task, index_pos=index_pos, index_neg=index_neg
        )
    else:
        list_message = generate_fsd_message_reg(
            message_type=message_type, template_set=template_set,
            smiles=smiles, description=description, task=task, label=dataset.y
        )
    save_message(
        dataset_name=cfg.dataset, list_message=list_message,
        message_type=message_type, demo_test=cfg.demo_test
    )

    message_type = "FSD-3"
    print('Generating {}'.format(message_type))
    if 'classification' in dataset.task_type:
        list_message = generate_fsd_message_cls(
            message_type=message_type, template_set=template_set,
            smiles=smiles, description=description,
            task=task, index_pos=index_pos, index_neg=index_neg
        )
    else:
        list_message = generate_fsd_message_reg(
            message_type=message_type, template_set=template_set,
            smiles=smiles, description=description, task=task, label=dataset.y
        )
    save_message(
        dataset_name=cfg.dataset, list_message=list_message,
        message_type=message_type, demo_test=cfg.demo_test
    )

    message_type = "FSC-1"
    print('Generating {}'.format(message_type))
    if 'classification' in dataset.task_type:
        list_message = generate_fsd_message_cls(
            message_type="FSD-1", template_set=template_set,
            smiles=smiles, description=caption,
            task=task, index_pos=index_pos, index_neg=index_neg
        )
    else:
        list_message = generate_fsd_message_reg(
            message_type="FSD-1", template_set=template_set,
            smiles=smiles, description=caption, task=task, label=dataset.y
        )
    save_message(
        dataset_name=cfg.dataset, list_message=list_message,
        message_type=message_type, demo_test=cfg.demo_test
    )

    message_type = "FSC-2"
    print('Generating {}'.format(message_type))
    if 'classification' in dataset.task_type:
        list_message = generate_fsd_message_cls(
            message_type="FSD-2", template_set=template_set,
            smiles=smiles, description=caption,
            task=task, index_pos=index_pos, index_neg=index_neg
        )
    else:
        list_message = generate_fsd_message_reg(
            message_type="FSD-2", template_set=template_set,
            smiles=smiles, description=caption, task=task, label=dataset.y
        )
    save_message(
        dataset_name=cfg.dataset, list_message=list_message,
        message_type=message_type, demo_test=cfg.demo_test
    )

    message_type = "FSC-3"
    print('Generating {}'.format(message_type))
    if 'classification' in dataset.task_type:
        list_message = generate_fsd_message_cls(
            message_type="FSD-3", template_set=template_set,
            smiles=smiles, description=caption,
            task=task, index_pos=index_pos, index_neg=index_neg
        )
    else:
        list_message = generate_fsd_message_reg(
            message_type="FSD-3", template_set=template_set,
            smiles=smiles, description=caption, task=task, label=dataset.y
        )
    save_message(
        dataset_name=cfg.dataset, list_message=list_message,
        message_type=message_type, demo_test=cfg.demo_test
    )


if __name__ == "__main__":
    cfg = update_cfg(cfg)
    main(cfg)
