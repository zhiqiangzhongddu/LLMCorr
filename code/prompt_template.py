molhiv_prompt_template = {
    "IF":
        "SMILES string of target molecule: {}. "
        "Question: "
        "Provide helpful information to predict if this molecule inhibits HIV virus replication. "
        "Answer:",
    "IFC":
        "SMILES string of target molecule: {}. "
        "Caption of target molecule: {} "
        "Question: "
        "Provide helpful information to predict if this molecule inhibits HIV virus replication. "
        "Answer:",

    "IP":
        "SMILES string of target molecule: {}. "
        "Question: "
        "Predict if this molecule inhibits HIV virus replication. "
        "Answer this question in the form: "
        "Prediction: <number, 1: target molecule inhibits HIV virus replication, "
        "0: target molecule does not inhibit HIV virus replication>. "
        "Answer:",
    "IPC":
        "SMILES string of target molecule: {}. "
        "Caption of target molecule: {} "
        "Question: "
        "Predict if this molecule inhibits HIV virus replication. "
        "Answer this question in the form: "
        "Prediction: <number, 1: target molecule inhibits HIV virus replication, "
        "0: target molecule does not inhibit HIV virus replication>. "
        "Answer:",

    "IE":
        "SMILES string of target molecule: {}. "
        "Question: "
        "Predict if this molecule inhibits HIV virus replication. "
        "Answer this question in the form: "
        "Prediction: <number, 1: target molecule inhibits HIV virus replication, "
        "0: target molecule does not inhibit HIV virus replication>, "
        "Explanation: <text>. "
        "Answer:",
    "IEC":
        "SMILES string of target molecule: {}. "
        "Caption of target molecule: {} "
        "Question: "
        "Predict if this molecule inhibits HIV virus replication. "
        "Answer this question in the form: "
        "Prediction: <number, 1: target molecule inhibits HIV virus replication, "
        "0: target molecule does not inhibit HIV virus replication>, "
        "Explanation: <text>. "
        "Answer:",

    "FS":
        "Knowledge: {}\n"
        "SMILES string of target molecule: {}. "
        "Question: "
        "Predict if target molecule inhibits HIV virus replication. "
        "Answer this question in the form: "
        "Prediction: <number, 1: target molecule inhibits HIV virus replication, "
        "0: target molecule does not inhibit HIV virus replication>, "
        "Explanation: <text>. "
        "Answer:",
    "FS_knowledge_pos":
        "Molecule SMILES string: {}. "
        "It inhibits HIV virus replication.",
    "FS_knowledge_neg":
        "Molecule SMILES string: {}. "
        "It does not inhibit HIV virus replication.",

    "FSC":
        "Knowledge: {}\n"
        "SMILES string of target molecule: {}. "
        "Caption of target molecule: {} "
        "Question: "
        "Predict if target molecule inhibits HIV virus replication. "
        "Answer this question in the form: "
        "Prediction: <number, 1: target molecule inhibits HIV virus replication, "
        "0: target molecule does not inhibit HIV virus replication>, "
        "Explanation: <text>. "
        "Answer:",
    "FSC_knowledge_pos":
        "Molecule SMILES string: {}; "
        "Caption: {} "
        "It inhibits HIV virus replication.",
    "FSC_knowledge_neg":
        "Molecule SMILES string: {}; "
        "Caption: {} "
        "It does not inhibit HIV virus replication.",
}


molbace_prompt_template = {
    "IF":
        "SMILES string of target molecule: {}. "
        "Question: "
        "Provide helpful information to predict if this molecule inhibits human β-secretase 1(BACE-1). "
        "Answer:",
    "IFC":
        "SMILES string of target molecule: {}. "
        "Caption of target molecule: {} "
        "Question: "
        "Provide helpful information to predict if this molecule inhibits human β-secretase 1(BACE-1). "
        "Answer:",

    "IP":
        "SMILES string of target molecule: {}. "
        "Question: "
        "Predict if this molecule inhibits human β-secretase 1(BACE-1). "
        "Answer this question in the form: "
        "Prediction: <number, 1: target molecule inhibits human β-secretase 1(BACE-1)> "
        "0: target molecule does not inhibits human β-secretase 1(BACE-1)>. "
        "Answer:",
    "IPC":
        "SMILES string of target molecule: {}. "
        "Caption of target molecule: {} "
        "Question: "
        "Predict if this molecule inhibits human β-secretase 1(BACE-1). "
        "Answer this question in the form: "
        "Prediction: <number, 1: target molecule inhibits human β-secretase 1(BACE-1), "
        "0: target molecule does not inhibits human β-secretase 1(BACE-1)>. "
        "Answer:",

    "IE":
        "SMILES string of target molecule: {}. "
        "Question: "
        "Predict if this molecule inhibits human β-secretase 1(BACE-1). "
        "Answer this question in the form: "
        "Prediction: <number, 1: target molecule inhibits human β-secretase 1(BACE-1), "
        "0: target molecule does not inhibits human β-secretase 1(BACE-1)>, "
        "Explanation: <text>. "
        "Answer:",
    "IEC":
        "SMILES string of target molecule: {}. "
        "Caption of target molecule: {} "
        "Question: "
        "Predict if this molecule inhibits human β-secretase 1(BACE-1). "
        "Answer this question in the form: "
        "Prediction: <number, 1: target molecule inhibits human β-secretase 1(BACE-1), "
        "0: target molecule does not inhibits human β-secretase 1(BACE-1)>, "
        "Explanation: <text>. "
        "Answer:",

    "FS":
        "Knowledge: {}\n"
        "SMILES string of target molecule: {}. "
        "Question: "
        "Predict if target molecule inhibits human β-secretase 1(BACE-1). "
        "Answer this question in the form: "
        "Prediction: <number, 1: target molecule inhibits human β-secretase 1(BACE-1), "
        "0: target molecule does not inhibits human β-secretase 1(BACE-1)>, "
        "Explanation: <text>. "
        "Answer:",
    "FS_knowledge_pos":
        "Molecule SMILES string: {}. "
        "It inhibits human β-secretase 1(BACE-1).",
    "FS_knowledge_neg":
        "Molecule SMILES string: {}. "
        "It does not inhibits human β-secretase 1(BACE-1).",

    "FSC":
        "Knowledge: {}\n"
        "SMILES string of target molecule: {}. "
        "Caption of target molecule: {} "
        "Question: "
        "Predict if target molecule inhibits human β-secretase 1(BACE-1). "
        "Answer this question in the form: "
        "Prediction: <number, 1: target molecule inhibits human β-secretase 1(BACE-1), "
        "0: target molecule does not inhibits human β-secretase 1(BACE-1)>, "
        "Explanation: <text>. "
        "Answer:",
    "FSC_knowledge_pos":
        "Molecule SMILES string: {}; "
        "Caption: {} "
        "It inhibits human β-secretase 1(BACE-1).",
    "FSC_knowledge_neg":
        "Molecule SMILES string: {}; "
        "Caption: {} "
        "It does not inhibits human β-secretase 1(BACE-1).",
}
