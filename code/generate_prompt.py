import numpy as np
import torch

from code.data_utils.dataset import DatasetLoader
from code.data_utils.utils import load_caption
from code.prompt_template import molhiv_prompt_template, molbace_prompt_template
from code.config import cfg, update_cfg
from code.utils import set_seed
from code.data_utils.utils import save_prompt


def generate_fs_prompt(prompt_type, template_set, smiles, index_pos, index_neg):
    fs = int(prompt_type.split('-')[1])
    template = template_set[prompt_type.split('-')[0]]
    knowledge_pos_template = template_set["FS_knowledge_pos"]
    knowledge_neg_template = template_set["FS_knowledge_neg"]

    list_prompt = []
    for idx, s in enumerate(smiles):
        knowledge = ""
        _pos = np.random.choice(index_pos, fs)
        _neg = np.random.choice(index_neg, fs)
        while idx in _pos:
            _pos = np.random.choice(index_pos, fs)
        while idx in _neg:
            _neg = np.random.choice(index_neg, fs)

        for knowledge_id in range(fs):
            knowledge += knowledge_pos_template.format(smiles[_pos[knowledge_id]]) + "\n"
            if knowledge_id < fs - 1:
                knowledge += knowledge_neg_template.format(smiles[_neg[knowledge_id]]) + "\n"
            else:
                knowledge += knowledge_neg_template.format(smiles[_neg[knowledge_id]])

        list_prompt.append(template.format(knowledge, s))
    return list_prompt


def generate_fsc_prompt(prompt_type, template_set, smiles, caption, index_pos, index_neg):
    fs = int(prompt_type.split('-')[1])
    template = template_set[prompt_type.split('-')[0]]
    knowledge_pos_template = template_set["FSC_knowledge_pos"]
    knowledge_neg_template = template_set["FSC_knowledge_neg"]

    list_prompt = []
    for idx, (s, c) in enumerate(zip(smiles, caption)):
        knowledge = ""
        _pos = np.random.choice(index_pos, fs)
        _neg = np.random.choice(index_neg, fs)
        while idx in _pos:
            _pos = np.random.choice(index_pos, fs)
        while idx in _neg:
            _neg = np.random.choice(index_neg, fs)

        for knowledge_id in range(fs):
            knowledge += knowledge_pos_template.format(
                smiles[_pos[knowledge_id]], caption[_pos[knowledge_id]]
            ) + "\n"
            if knowledge_id < fs - 1:
                knowledge += knowledge_neg_template.format(
                    smiles[_neg[knowledge_id]], caption[_neg[knowledge_id]]
                ) + "\n"
            else:
                knowledge += knowledge_neg_template.format(
                    smiles[_neg[knowledge_id]], caption[_neg[knowledge_id]]
                )

        list_prompt.append(template.format(knowledge, s, c))
    return list_prompt


def main(cfg):
    set_seed(cfg.seed)

    # Preprocess data
    dataloader = DatasetLoader(name=cfg.dataset, text='raw')
    dataset, smiles = dataloader.dataset, dataloader.text

    caption = load_caption(dataset_name=cfg.dataset, demo_test=False)

    split_idx = dataset.get_idx_split()
    index_pos = np.intersect1d(split_idx['train'], torch.where(dataset.y == 1)[0])
    index_neg = np.intersect1d(split_idx['train'], torch.where(dataset.y == 0)[0])

    if cfg.dataset == "ogbg-molhiv":
        template_set = molhiv_prompt_template
    elif cfg.dataset == "ogbg-molbace":
        template_set = molbace_prompt_template
    else:
        raise ValueError("Invalid Dataset Name to find Prompt Set.")

    prompt_type = "IF"
    print('Generating {}'.format(prompt_type))
    template = template_set[prompt_type]
    list_prompt = [template.format(s) for s in smiles]
    save_prompt(dataset_name=cfg.dataset, list_prompt=list_prompt, prompt_type=prompt_type)

    prompt_type = "IFC"
    print('Generating {}'.format(prompt_type))
    template = template_set[prompt_type]
    for s, c in zip(smiles, caption):
        list_prompt.append(template.format(s, c))
    save_prompt(dataset_name=cfg.dataset, list_prompt=list_prompt, prompt_type=prompt_type)

    prompt_type = "IP"
    print('Generating {}'.format(prompt_type))
    template = template_set[prompt_type]
    list_prompt = [template.format(s) for s in smiles]
    save_prompt(dataset_name=cfg.dataset, list_prompt=list_prompt, prompt_type=prompt_type)

    prompt_type = "IPC"
    print('Generating {}'.format(prompt_type))
    template = template_set[prompt_type]
    for s, c in zip(smiles, caption):
        list_prompt.append(template.format(s, c))
    save_prompt(dataset_name=cfg.dataset, list_prompt=list_prompt, prompt_type=prompt_type)

    prompt_type = "IE"
    print('Generating {}'.format(prompt_type))
    template = template_set[prompt_type]
    list_prompt = [template.format(s) for s in smiles]
    save_prompt(dataset_name=cfg.dataset, list_prompt=list_prompt, prompt_type=prompt_type)

    prompt_type = "IEC"
    print('Generating {}'.format(prompt_type))
    template = template_set[prompt_type]
    for s, c in zip(smiles, caption):
        list_prompt.append(template.format(s, c))
    save_prompt(dataset_name=cfg.dataset, list_prompt=list_prompt, prompt_type=prompt_type)

    prompt_type = "FS-1"
    print('Generating {}'.format(prompt_type))
    list_prompt = generate_fs_prompt(
        prompt_type=prompt_type, template_set=template_set,
        smiles=smiles, index_pos=index_pos, index_neg=index_neg
    )
    save_prompt(dataset_name=cfg.dataset, list_prompt=list_prompt, prompt_type=prompt_type)

    prompt_type = "FS-2"
    print('Generating {}'.format(prompt_type))
    list_prompt = generate_fs_prompt(
        prompt_type=prompt_type, template_set=template_set,
        smiles=smiles, index_pos=index_pos, index_neg=index_neg
    )
    save_prompt(dataset_name=cfg.dataset, list_prompt=list_prompt, prompt_type=prompt_type)

    prompt_type = "FS-3"
    print('Generating {}'.format(prompt_type))
    list_prompt = generate_fs_prompt(
        prompt_type=prompt_type, template_set=template_set,
        smiles=smiles, index_pos=index_pos, index_neg=index_neg
    )
    save_prompt(dataset_name=cfg.dataset, list_prompt=list_prompt, prompt_type=prompt_type)

    prompt_type = "FSC-1"
    print('Generating {}'.format(prompt_type))
    list_prompt = generate_fsc_prompt(
        prompt_type=prompt_type, template_set=template_set,
        smiles=smiles, caption=caption, index_pos=index_pos, index_neg=index_neg
    )
    save_prompt(dataset_name=cfg.dataset, list_prompt=list_prompt, prompt_type=prompt_type)

    prompt_type = "FSC-2"
    print('Generating {}'.format(prompt_type))
    list_prompt = generate_fsc_prompt(
        prompt_type=prompt_type, template_set=template_set,
        smiles=smiles, caption=caption, index_pos=index_pos, index_neg=index_neg
    )
    save_prompt(dataset_name=cfg.dataset, list_prompt=list_prompt, prompt_type=prompt_type)

    prompt_type = "FSC-3"
    print('Generating {}'.format(prompt_type))
    list_prompt = generate_fsc_prompt(
        prompt_type=prompt_type, template_set=template_set,
        smiles=smiles, caption=caption, index_pos=index_pos, index_neg=index_neg
    )
    save_prompt(dataset_name=cfg.dataset, list_prompt=list_prompt, prompt_type=prompt_type)


if __name__ == "__main__":
    cfg = update_cfg(cfg)
    main(cfg)
