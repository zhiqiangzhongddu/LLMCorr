task_list_third_person = {
    "ogbg-molbace": "inhibits human β-secretase 1 (BACE-1)",
    "ogbg-molbbbp": "exhibits penetration across the blood-brain barrier",
    "ogbg-molhiv": "exhibits inhibitory effects on HIV replication",
    "ogbg-molesol": "water solubility value (in log(mol/L))",
    "ogbg-molfreesolv": "hydration free energy (in log(mol/L))",
    "ogbg-mollipo": "lipophilicity (in logP)",
}

task_list_first_person = {
    "ogbg-molbace": "inhibit human β-secretase 1 (BACE-1)",
    "ogbg-molbbbp": "exhibit penetration across the blood-brain barrier",
    "ogbg-molhiv": "exhibit inhibitory effects on HIV replication",
    "ogbg-molesol": "water solubility value (in log(mol/L))",
    "ogbg-molfreesolv": "hydration free energy (in log(mol/L))",
    "ogbg-mollipo": "lipophilicity (in logP)",
}


multi_round_instruction = {
    "role": "system",
    "content":
        "You are an expert in biomedicine and chemistry, specializing in molecules.",
}
cls_multi_round_question_template = {
    "P":
        "Predict whether molecule-{} {}. "
        "Answer the question in the format: Prediction: <True or False>.",
    "PE":
        "Predict whether molecule-{} {}. "
        "Answer the question in the format: Prediction: <True or False>; Explanation: <text>.",
    "PP":
        "Predict whether molecule-{} {}. "
        "Answer the question in the format: Prediction: <True or False>; Probability: <number>.",
    "PPE":
        "Predict whether molecule-{} {}. "
        "Answer the question in the format: Prediction: <True or False>; Probability: <number>; Explanation: <text>.",
}
cls_multi_round_question_chat_template = {
    "S": {
        "role": "user",
        "content":
            "The SMILES string of molecule-{} is {}. {} {}",
    },
    "SP": {
        "role": "user",
        "content":
            "The SMILES string of molecule-{} is {}. {} "
            "The GNN model predicts that molecule-{} {} with a probability of {:.4f}. "
            "{}",
    },
}
cls_multi_round_knowledge_template = {
    "add-train-knowledge": {
        "question": {
            "role": "user",
            "content": 
                # "The SMILES string of molecule-{} is {}. {} {}",
                "The SMILES string of molecule-{} is {}. {} "
                "Predict whether molecule-{} {}.",
        },
        "answer": {
            "role": "assistant",
            "content":
                # "Prediction: {}.",
                "Molecule-{} {}.",
        }
    },
    "add-valid-knowledge": {
        "question": {
            "role": "user",
            "content":
                "The SMILES string of molecule-{} is {}. {} "
                # "The SMILES string of molecule-{} is {}. {} {}",
                "The GNN model predicts that molecule-{} {} with a probability of {:.4f}. "
                "Predict whether molecule-{} {}.",
        },
        "answer": {
            "role": "assistant",
            "content":
                # "Prediction: {}.",
                "Molecule-{} {}.",
        },
    },
}
cls_multi_round_inquiry_template = {
    "double-check": {
        "role": "user",
        "content":
            "Your prediction differs from the GNN model's prediction, "
            "and the {} score of the GNN model on the training dataset is {:.4f}, "
            "on the validation dataset is {:.4f}. "
            "Please conduct a self-check on your thought process. "
            "Examine your reasoning carefully, and search for additional relevant information "
            "to validate or refine your current understanding. "
            "Provide your refined response after considering additional information. {}",
    },
}
reg_multi_round_question_template = {
    "P":
        "Predict the {} of molecule-{}. "
        "Answer the question in the format: Prediction: <number>.",
    "PE":
        "Predict the {} of molecule-{}. "
        "Answer the question in the format: Prediction: <number>; Explanation: <text>.",
}
reg_multi_round_question_chat_template = {
    "S": {
        "role": "user",
        "content":
            "The SMILES string of molecule-{} is {}. {} {}",
    },
    "SP": {
        "role": "user",
        "content":
            "The SMILES string of molecule-{} is {}. {} "
            "The GNN model predicts that the {} of molecule-{} is {:.4f}. {}",
    },
}
reg_multi_round_knowledge_template = {
    "add-train-knowledge": {
        "question": {
            "role": "user",
            "content":
                "The SMILES string of molecule-{} is {}. {} "
                "Predict the {} of molecule-{}.",
        },
        "answer": {
            "role": "assistant",
            "content":
                # "Prediction: {}.",
                "The {} of molecule-{} is {:.4f}.",
        }
    },
    "add-valid-knowledge": {
        "question": {
            "role": "user",
            "content":
                "The SMILES string of molecule-{} is {}. {} "
                # "The SMILES string of molecule-{} is {}. {} {}",
                "The GNN model predicts the {} of molecule-{} is {:.4f}. "
                "Predict the {} of molecule-{}.",
        },
        "answer": {
            "role": "assistant",
            "content":
                # "Prediction: {}.",
                "The {} of molecule-{} is {:.4f}.",
        },
    },
}
reg_multi_round_inquiry_template = {
    "double-check": {
        "role": "user",
        "content":
            "Your prediction significantly deviates from the GNN model's prediction, "
            "and the {} score of the GNN model on the training dataset is {:.4f}, "
            "on the validation dataset is {:.4f}. "
            "Please conduct a self-check on your thought process. "
            "Examine your reasoning carefully, and search for additional relevant information "
            "to validate or refine your current understanding. "
            "Provide your refined response after considering additional information. {}",
    },
}


compressor_instruction = {
    "role": "system",
    "content": "You are an AI molecule property analysis assistant "
               "specializing in summarising molecular atom features and graph structural features."
}
cls_compressor_template = {
    "compress_des": [
        compressor_instruction,
        {
            "role": "user",
            "content": "The SMILES string of molecule-{} is {}, "
                       "its atom features and graph structure are described as follows: {}. "
                       "Summarize its atom features and graph structure as "
                       "detailed, comprehensive, meaningful and insightful text.",
        }
    ],
}
reg_compressor_template = {
    "compress_des": [
        compressor_instruction,
        {
            "role": "user",
            "content": "The SMILES string of molecule-{} is {}, "
                       "its atom features and graph structure are described as follows: {}. "
                       "Summarize its atom features and graph structure as "
                       "detailed, comprehensive, meaningful and insightful text.",
        }
    ],
}


cls_explainer_instruction = {
    "role": "system",
    "content": "You are an AI molecule property analysis assistant "
               "specializing in explaining predictions made by Graph Neural Networks (GNNs). "
               "The prediction is about whether the molecule {}."
}
cls_explainer_template = {
    "EP": [
        cls_explainer_instruction,
        {
            "role": "user",
            "content": "The molecule-{}'s SMILES string is {}. "
                       "The molecule-{}'s prediction given by GNNs is {} "
                       "with predicted probability {:.4f}. "
                       "The true label of the molecule-{} is {}. "
                       "Indentify insightful features to explain "
                       "why GNNs make this prediction for molecule-{}.",
        }
    ],
    "EPD": [
        cls_explainer_instruction,
        {
            "role": "user",
            "content": "The molecule-{}'s SMILES string is {}. "
                       "{} "
                       "The molecule-{}'s prediction given by GNNs is {} "
                       "with predicted probability {:.4f}. "
                       "The true label of the molecule-{} is {}. "
                       "Indentify insightful features to explain "
                       "why GNNs make this prediction for molecule-{}.",
        }
    ]
}

reg_explainer_instruction = {
    "role": "system",
    "content": "You are an AI molecule property analysis assistant "
               "specializing in explaining predictions made by Graph Neural Networks (GNNs). "
               "The prediction is about the {} of a molecule."
}
reg_explainer_template = {
    "EP": [
        cls_explainer_instruction,
        {
            "role": "user",
            "content": "The molecule-{}'s SMILES string is {}. "
                       "The molecule-{}'s prediction given by GNNs is {:.4f}. "
                       "The true value of the molecule-{} is {:.4f}. "
                       "Indentify insightful features to explain "
                       "why GNNs make this prediction for molecule-{}.",
        }
    ],
    "EPD": [
        cls_explainer_instruction,
        {
            "role": "user",
            "content": "The molecule-{}'s SMILES string is {}. "
                       "{} "
                       "The molecule-{}'s prediction given by GNNs is {:.4f}. "
                       "The true value of the molecule-{} is {:.4f}. "
                       "Indentify insightful features to explain "
                       "why GNNs make this prediction for molecule-{}.",
        }
    ],
}


cls_corrector_instruction = {
    "role": "system",
    "content": "You are an expert in Molecules and Graph Machine Learning, "
               "specializing in correcting predictions made by Graph Neural Networks (GNNs)."
}
cls_corrector_message_template = {
    "CorrFS": [
        cls_corrector_instruction,
        {
            "role": "user",
            "content": "The SMILES string of molecule-{} is {}. "
                       "The GNN model predicts that molecule-{} {} "
                       "with a probability of {:.4f}. "
                       "Predict whether molecule-{} {} and provide insightful explanations. "
                       "Answer this question in the format: "
                       "Prediction: <True or False>; Explanation: <text>."
        }
    ],
    "CorrFS_knowledge": [
        {
            "role": "user",
            "content": "The SMILES string of molecule-{} is {}. "
                       "The GNN model predicts that molecule-{} {} "
                       "with a probability of {:.4f}. "
                       "Predict whether molecule-{} {}. "
                       "Answer this question in the format: "
                       "Prediction: <True or False>."
        },
        {
            "role": "assistant",
            "content": "Prediction: {}."
        },
    ],

    "CorrFSD": [
        cls_corrector_instruction,
        {
            "role": "user",
            "content": "The SMILES string of molecule-{} is {}. "
                       "The GNN model predicts that molecule-{} {} "
                       "with a probability of {:.4f}. "
                       "{} "
                       "Predict whether molecule-{} {} and provide insightful explanations. "
                       "Answer this question in the format: "
                       "Prediction: <True or False>; Explanation: <text>."
        }
    ],
    "CorrFSD_knowledge": [
        {
            "role": "user",
            "content": "The SMILES string of molecule-{} is {}. "
                       "The GNN model predicts that molecule-{} {} "
                       "with a probability of {:.4f}. "
                       "{} "
                       "Predict whether molecule-{} {}. "
                       "Answer this question in the format: "
                       "Prediction: <True or False>."
        },
        {
            "role": "assistant",
            "content": "Prediction: {}."
        },
    ],
}
