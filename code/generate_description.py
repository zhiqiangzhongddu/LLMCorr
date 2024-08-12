from rdkit import Chem

import torch

from code.data_utils.dataset import DatasetLoader
from code.config import cfg, update_cfg
from code.utils import set_seed, time_logger
from code.data_utils.utils import save_description


def generate_full_description(index, smiles_string, atom_x):
    """
    bond_x if dropped since we don't generate bond features now.
    Baseline models only use atom features? To Check.
    """

    molecule = Chem.MolFromSmiles(smiles_string)

    description = "The molecule-{} can be represented as a graph among atoms {}. In this graph:\n".format(
        index, ', '.join(['{}({})'.format(atom.GetIdx(), atom.GetSymbol()) for atom in molecule.GetAtoms()])
    )

    for atom in molecule.GetAtoms():
        atom_index = atom.GetIdx()
        neighbors = [neighbor.GetIdx() for neighbor in atom.GetNeighbors()]
        if neighbors:
            description += generate_atom_feature_description(
                atom_x=atom_x[atom_index], atom_index=atom_index, smiles_string=smiles_string
            )
            description += " Atom {} is connected to {}.\n".format(
                atom_index,
                ' and '.join(['Atom {}'.format(
                    neighbor
                ) for neighbor in neighbors])
            )
            # description += generate_bound_feature_description(x=bond_x[atom_index])
            # description += "\n"

    return description


def generate_structure_description(index, smiles_string):
    molecule = Chem.MolFromSmiles(smiles_string)

    description = "The molecule-{} can be represented as a graph among atoms {}. In this graph:\n".format(
        index, ', '.join(['{}({})'.format(atom.GetIdx(), atom.GetSymbol()) for atom in molecule.GetAtoms()])
    )

    for atom in molecule.GetAtoms():
        atom_index = atom.GetIdx()
        neighbors = [neighbor.GetIdx() for neighbor in atom.GetNeighbors()]
        if neighbors:
            description += "Atom {} is connected to {}.\n".format(
                atom_index,
                ' and '.join(['Atom {}'.format(
                    neighbor
                ) for neighbor in neighbors])
            )

    return description


def generate_atom_feature_description(atom_x, atom_index, smiles_string):

    # molecule = Chem.MolFromSmiles(smiles_string)

    if type(atom_x) is torch.Tensor:
        atom_x = atom_x.numpy()

    assert atom_x.size == 9

    chirality_list = [
        'no specified chirality',
        'a clockwise tetrahedral chirality',
        'a counter-clockwise tetrahedral chirality',
        'a non-tetrahedral chirality',
        'a miscellaneous chirality'
    ]
    hybridization_list = [
        'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'miscellaneous'
    ]
    is_aromatic_list = ['not aromatic', 'aromatic']
    is_in_ring_list = ['not part of a ring', 'part of a ring']

    description = ("Atom {} has {} atomics, "
                   "has {}, "
                   "has {} bonds with other atoms, "
                   "has a positive charge of {}, "
                   "has {} hydrogen atoms attached to it, "
                   "has {} unpaired electrons, "
                   "has a {} hybridization, "
                   "is {}, "
                   "is {}.").format(
        atom_index,
        int(atom_x[0]), chirality_list[int(atom_x[1])],
        int(atom_x[2]), int(atom_x[3]), int(atom_x[4]), int(atom_x[5]),
        hybridization_list[int(atom_x[6])],
        is_aromatic_list[int(atom_x[7])],
        is_in_ring_list[int(atom_x[8])],
    )

    return description


def generate_all_atom_feature_description(index, smiles_string, mol_x):
    molecule = Chem.MolFromSmiles(smiles_string)

    description = "The molecule-{} can be represented as a graph among atoms {}. In this graph:\n".format(
        index, ', '.join(['{}({})'.format(atom.GetIdx(), atom.GetSymbol()) for atom in molecule.GetAtoms()])
    )

    for atom in molecule.GetAtoms():
        atom_index = atom.GetIdx()
        description += generate_atom_feature_description(
            atom_x=mol_x[atom_index], atom_index=atom_index,
            smiles_string=smiles_string
        )
        description += "\n"

    return description


def generate_bond_feature_description(bond_x):

    if type(bond_x) is torch.Tensor:
        bond_x = bond_x.numpy()

    assert bond_x.size == 3

    bond_type_list = [
        'SINGLE',
        'DOUBLE',
        'TRIPLE',
        'AROMATIC',
        'misc'
    ]
    bond_stereo_list = [
        'STEREONONE',
        'STEREOZ',
        'STEREOE',
        'STEREOCIS',
        'STEREOTRANS',
        'STEREOANY',
    ]
    is_conjugated_list = [
        'not Conjugated',
        'Conjugated'
    ]

    description = ("The bound type is {}. "
                   "The bond Stereo is {}. "
                   "The bond is {}.").format(
        bond_type_list[int(bond_x[0])],
        bond_stereo_list[int(bond_x[1])],
        is_conjugated_list[int(bond_x[2])]
    )

    return description


@time_logger
def main(cfg):
    set_seed(cfg.seed)

    # Preprocess data
    dataloader = DatasetLoader(name=cfg.dataset, text='raw')
    dataset, smiles = dataloader.dataset, dataloader.text

    list_description = []
    description_type = "structure"
    print('Generating {}'.format(description_type))
    for idx, smi in enumerate(smiles):
        list_description.append(
            generate_structure_description(index=idx, smiles_string=smi)
        )
    save_description(
        dataset_name=cfg.dataset, list_description=list_description,
        description_type=description_type, demo_test=cfg.demo_test
    )

    list_description = []
    description_type = "atom"
    print('Generating {}'.format(description_type))
    for idx, smi in enumerate(smiles):
        list_description.append(
            generate_all_atom_feature_description(
                index=idx, smiles_string=smi, mol_x=dataset[idx].x)
        )
    save_description(
        dataset_name=cfg.dataset, list_description=list_description,
        description_type=description_type, demo_test=cfg.demo_test
    )

    # list_description = []
    # description_type = "bond"
    # print('Generating {}'.format(description_type))
    # for bond_x in dataset.edge_attr:
    #     list_description.append(
    #         generate_bond_feature_description(bond_x=bond_x)
    #     )
    # save_description(
    #     dataset_name=cfg.dataset, list_description=list_description,
    #     description_type=description_type, demo_test=cfg.demo_test
    # )

    list_description = []
    description_type = "full"
    print('Generating {}'.format(description_type))
    for idx, smi in enumerate(smiles):
        list_description.append(
            generate_full_description(
                index=idx, smiles_string=smi, atom_x=dataset[idx].x
            )
        )
    save_description(
        dataset_name=cfg.dataset, list_description=list_description,
        description_type=description_type, demo_test=cfg.demo_test
    )


if __name__ == "__main__":
    cfg = update_cfg(cfg)
    main(cfg)
