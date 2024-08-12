import copy
import sys
import numpy as np
from tqdm import tqdm
from openai import OpenAI, AzureOpenAI
import os
import torch
import torch.nn.functional as F
from ogb.graphproppred import Evaluator

from code.config import cfg, update_cfg
from code.utils import time_logger
from code.data_utils.dataset import DatasetLoader
from code.LLMs.utils import num_tokens_from_messages, get_context_window_size_limit, check_response
from code.message_template import (multi_round_instruction,
                                   cls_multi_round_inquiry_template, reg_multi_round_inquiry_template,
                                   cls_multi_round_question_chat_template, reg_multi_round_question_chat_template,
                                   cls_multi_round_question_template, reg_multi_round_question_template,
                                   cls_multi_round_knowledge_template, reg_multi_round_knowledge_template,
                                   task_list_third_person, task_list_first_person)
from code.data_utils.utils import (save_llm_outputs, load_llm_outputs, check_llm_cache,
                                   load_caption, load_description,
                                   load_gnn_predictions, load_lm_predictions, get_similarity_matrix,
                                   clean_llm_cache)


def check_num_tokens(chat_history):

    # RPM limit (adjust according to your plan)
    rpm_limit = 10000
    # TPM limit (adjust according to your plan)
    tpm_limit = 1000000
    # Context window size
    cws_limit = get_context_window_size_limit(cfg.llm.model.name)

    num_tokens = num_tokens_from_messages(messages=chat_history)
    # print("Num_tokens: {}".format(num_tokens))

    if num_tokens > tpm_limit:
        sys.exit("Chat token number is large than limit {}.".format(tpm_limit))
    if num_tokens >= cws_limit:
        sys.exit("Chat context length is {}, large than limit {}.".format(num_tokens, cws_limit))


def generate_complete_template_to_save_llm_outputs(cfg):
    string = "_".join([
        cfg.llm.template,
        cfg.llm.emb.data,
        cfg.llm.emb.model,
        str(cfg.llm.model.temperature),
        str(cfg.llm.model.top_p),
    ])
    return string


def run_loading(cfg):
    # Preprocess data
    dataloader = DatasetLoader(name=cfg.dataset, text='raw')
    dataset, smiles = dataloader.dataset, dataloader.text

    split_idx = dataset.get_idx_split()
    train_indices = split_idx['train'].numpy()
    valid_indices = split_idx['valid'].numpy()
    test_indices = split_idx["test"].numpy()

    # Prepare predictions and labels
    if "classification" in dataset.task_type.lower():
        if cfg.llm.operate == "gnn":
            predictions = torch.sigmoid(load_gnn_predictions(
                dataset_name=cfg.dataset, gnn_model_name=cfg.gnn.model.name,
                feature=cfg.data.feature, lm_model_name=cfg.lm.model.name, seed=cfg.seed
            )).squeeze().numpy()
        elif cfg.llm.operate == "lm":
            predictions = F.softmax(load_lm_predictions(
                dataset_name=cfg.dataset, task=dataset.task_type, num_graphs=len(dataset),
                template="raw", lm_model_name=cfg.lm.model.name, seed=cfg.seed
            ).float(), dim=-1)[:, 1].squeeze().numpy()
        else:
            raise ValueError("{} is not a valid option.".format(cfg.llm.operate))
    else:
        if cfg.llm.operate == "gnn":
            predictions = load_gnn_predictions(
                dataset_name=cfg.dataset, gnn_model_name=cfg.gnn.model.name,
                feature=cfg.data.feature, lm_model_name=cfg.lm.model.name, seed=cfg.seed
            ).squeeze().numpy()
        elif cfg.llm.operate == "lm":
            predictions = load_lm_predictions(
                dataset_name=cfg.dataset, task=dataset.task_type, num_graphs=len(dataset),
                template="raw", lm_model_name=cfg.lm.model.name, seed=cfg.seed
            ).squeeze().numpy()
        else:
            raise ValueError("{} is not a valid option.".format(cfg.llm.operate))
    labels = dataset.y.squeeze().numpy()

    y_true = torch.Tensor(labels)[train_indices].view(-1, 1)
    y_pred = torch.Tensor(predictions)[train_indices].view(-1, 1)
    input_dict = {"y_true": y_true, "y_pred": y_pred}
    train_performance = Evaluator(cfg.dataset).eval(input_dict)[dataset.eval_metric]

    y_true = torch.Tensor(labels)[valid_indices].view(-1, 1)
    y_pred = torch.Tensor(predictions)[valid_indices].view(-1, 1)
    input_dict = {"y_true": y_true, "y_pred": y_pred}
    valid_performance = Evaluator(cfg.dataset).eval(input_dict)[dataset.eval_metric]

    caption = load_caption(dataset_name=cfg.dataset, demo_test=False)
    description = load_description(dataset_name=cfg.dataset, demo_test=cfg.demo_test)
    # remove \n in description
    description = [des.replace("\n", " ") for des in description]

    # Load Distance matrix
    similarity_matrix = get_similarity_matrix(
        dataset_name=cfg.dataset, embedding_model=cfg.llm.emb.model,
        embedding_type=cfg.llm.emb.data
    )

    return (dataset, train_indices, valid_indices, test_indices, predictions, labels,
            train_performance, valid_performance,
            similarity_matrix, smiles, caption, description)


def execute_query(cfg, client, chat_history):
    chat_completion = client.chat.completions.create(
        model=cfg.llm.model.name, messages=chat_history,
        temperature=cfg.llm.model.temperature,
        top_p=cfg.llm.model.top_p,
        # frequency_penalty=0.0,
        # presence_penalty=0,
    )

    response = chat_completion.choices[0].message.content

    return response


def add_response_into_chat_history(chat_history, response):

    add_response = {
        "role": "assistant",
        "content": response,
    }
    chat_history.append(add_response)

    return chat_history


def add_question_chat_into_chat_history(
        chat_history, index, question_template, task_first_p, task_third_p, mol_info_opt,
        smiles, description, caption, question_info, predictions, task_type,
        multi_round_question_chat_template
):
    if "classification" in task_type.lower():
        chat = copy.deepcopy(multi_round_question_chat_template[question_info])
        question = question_template.format(
            index, task_third_p
        )
    else:
        chat = copy.deepcopy(multi_round_question_chat_template[question_info])
        question = question_template.format(
            task_third_p, index
        )

    if question_info.lower() == "s":
        if mol_info_opt.lower() == "s":
            chat["content"] = chat["content"].format(
                index, smiles[index], "", question
            )
        elif mol_info_opt.lower() == "sd":
            chat["content"] = chat["content"].format(
                index, smiles[index], description[index], question
            )
        elif mol_info_opt.lower() == "sc":
            chat["content"] = chat["content"].format(
                index, smiles[index], caption[index], question
            )
        else:
            raise ValueError("Unknown option: {}".format(mol_info_opt))

    elif question_info.lower() == "sp":
        if "classification" in task_type.lower():
            text_pred = "{}".format(task_third_p) if predictions[index] > 0.5 \
                else "cannot {}".format(task_first_p)
            probability = predictions[index] if predictions[index] > 0.5 \
                else 1 - predictions[index]

            if mol_info_opt.lower() == "s":
                chat["content"] = chat["content"].format(
                    index, smiles[index], "",
                    index, text_pred, probability,
                    question
                )
            elif mol_info_opt.lower() == "sd":
                chat["content"] = chat["content"].format(
                    index, smiles[index], description[index],
                    index, text_pred, probability,
                    question
                )
            elif mol_info_opt.lower() == "sc":
                chat["content"] = chat["content"].format(
                    index, smiles[index], caption[index],
                    index, text_pred, probability,
                    question
                )
            else:
                raise ValueError("Unknown option: {}".format(mol_info_opt))
        else:
            if mol_info_opt.lower() == "s":
                chat["content"] = chat["content"].format(
                    index, smiles[index], "",
                    task_first_p, index, predictions[index],
                    question
                )
            elif mol_info_opt.lower() == "sd":
                chat["content"] = chat["content"].format(
                    index, smiles[index], description[index],
                    task_first_p, index, predictions[index],
                    question
                )
            elif mol_info_opt.lower() == "sc":
                chat["content"] = chat["content"].format(
                    index, smiles[index], caption[index],
                    task_first_p, index, predictions[index],
                    question
                )
            else:
                raise ValueError("Unknown option: {}".format(mol_info_opt))
    else:
        raise ValueError("Unknown option: {}".format(question_info))

    chat_history.append(chat)

    return chat_history


def add_train_knowledge_into_chat_histroy(
        chat_history, knowledge_template, question_template, k, select_knowledge,
        mol_info_opt, index, smiles, description, caption, task_first_p, task_third_p,
        predictions, labels, similarity_matrix, train_indices, demo_test, task_type,
):
    if select_knowledge.lower() == "top":
        similarity_train = similarity_matrix[index][train_indices]
        # Find the indices of the top k largest numbers
        knowledge_indices = train_indices[np.argsort(similarity_train)[-k:][::-1]]
    elif select_knowledge.lower() == "random":
        knowledge_indices = np.random.choice(train_indices, k)
    elif select_knowledge.lower() == "jump":
        similarity_train = similarity_matrix[index][train_indices]
        # Find the indices of the top k largest numbers
        ordered_indices = train_indices[np.argsort(similarity_train)[::-1]]
        selected_indices = np.linspace(0, len(train_indices) - 1, k, dtype=int)
        knowledge_indices = ordered_indices[selected_indices]
    else:
        raise ValueError("{} is not a valid choice".format(select_knowledge))
    if demo_test:
        print("\n Index {}, Train indices: {}".format(index, knowledge_indices))

    for knowledge_index in knowledge_indices:
        chat_question = copy.deepcopy(knowledge_template["question"])
        chat_answer = copy.deepcopy(knowledge_template["answer"])

        # question = question_template.format(knowledge_index, task_third_p)

        if mol_info_opt.lower() == "s":
            if "classification" in task_type.lower():
                # chat_question["content"] = chat_question["content"].format(
                #     knowledge_index, smiles[knowledge_index], "", question
                # )
                chat_question["content"] = chat_question["content"].format(
                    knowledge_index, smiles[knowledge_index], "",
                    knowledge_index, task_third_p
                )
            else:
                chat_question["content"] = chat_question["content"].format(
                    knowledge_index, smiles[knowledge_index], "",
                    task_third_p, knowledge_index
                )
        elif mol_info_opt.lower() == "sd":
            if "classification" in task_type.lower():
                # chat_question["content"] = chat_question["content"].format(
                #     knowledge_index, smiles[knowledge_index], description[knowledge_index], question
                # )
                chat_question["content"] = chat_question["content"].format(
                    knowledge_index, smiles[knowledge_index], description[knowledge_index],
                    knowledge_index, task_third_p
                )
            else:
                chat_question["content"] = chat_question["content"].format(
                    knowledge_index, smiles[knowledge_index], description[knowledge_index],
                    task_third_p, knowledge_index
                )
        elif mol_info_opt.lower() == "sc":
            if "classification" in task_type.lower():
                # chat_question["content"] = chat_question["content"].format(
                #     knowledge_index, smiles[knowledge_index], caption[knowledge_index], question
                # )
                chat_question["content"] = chat_question["content"].format(
                    knowledge_index, smiles[knowledge_index], caption[knowledge_index],
                    knowledge_index, task_third_p
                )
            else:
                chat_question["content"] = chat_question["content"].format(
                    knowledge_index, smiles[knowledge_index], caption[knowledge_index],
                    task_third_p, knowledge_index
                )
        else:
            raise ValueError("Unknown option: {}".format(mol_info_opt))

        if "classification" in task_type.lower():
            # text_label = "True." if labels[knowledge_index] > 0.5 else "False."
            # chat_answer["content"] = chat_answer["content"].format(text_label)
            text_label = "{}".format(task_third_p) if predictions[knowledge_index] > 0.5 \
                else "cannot {}".format(task_first_p)
            chat_answer["content"] = chat_answer["content"].format(knowledge_index, text_label)
        else:
            chat_answer["content"] = chat_answer["content"].format(
                task_third_p, knowledge_index, labels[knowledge_index]
            )

        chat_history.extend([chat_question, chat_answer])

    return chat_history


def add_valid_knowledge_into_chat_history(
        chat_history, knowledge_template, question_template, k, select_knowledge,
        mol_info_opt, task_first_p, task_third_p, index, smiles, description, caption,
        predictions, labels, similarity_matrix, valid_indices, demo_test, task_type,
):
    if select_knowledge.lower() == "top":
        candidate_indices = copy.deepcopy(valid_indices[valid_indices != index])
        similarity_valid = similarity_matrix[index][candidate_indices]
        # Find the indices of the top k largest numbers
        knowledge_indices = candidate_indices[np.argsort(similarity_valid)[-k:][::-1]]
    elif select_knowledge.lower() == "random":
        knowledge_indices = np.random.choice(valid_indices, k)
    elif select_knowledge.lower() == "jump":
        candidate_indices = copy.deepcopy(valid_indices[valid_indices != index])
        similarity_valid = similarity_matrix[index][candidate_indices]
        # Find the indices of the top k largest numbers
        ordered_indices = candidate_indices[np.argsort(similarity_valid)[::-1]]
        selected_indices = np.linspace(0, len(candidate_indices) - 1, k, dtype=int)
        knowledge_indices = ordered_indices[selected_indices]
    else:
        raise ValueError("{} is not a valid choice".format(select_knowledge))
    if demo_test:
        print("\n Index {}, Valid indices: {}".format(index, knowledge_indices))

    for knowledge_index in knowledge_indices:
        chat_question = copy.deepcopy(knowledge_template["question"])
        chat_answer = copy.deepcopy(knowledge_template["answer"])

        text_pred = "{}".format(task_third_p) if predictions[knowledge_index] > 0.5 \
            else "cannot {}".format(task_first_p)
        probability = predictions[knowledge_index] if predictions[knowledge_index] > 0.5 \
            else 1 - predictions[knowledge_index]
        question = question_template.format(knowledge_index, task_third_p)


        if mol_info_opt.lower() == "s":
            if "classification" in task_type.lower():
                # chat_question["content"] = chat_question["content"].format(
                #     knowledge_index, smiles[knowledge_index], "",
                #     knowledge_index, text_pred, probability, question
                # )
                chat_question["content"] = chat_question["content"].format(
                    knowledge_index, smiles[knowledge_index], "",
                    knowledge_index, text_pred, probability,
                    knowledge_index, task_third_p
                )
            else:
                chat_question["content"] = chat_question["content"].format(
                    knowledge_index, smiles[knowledge_index], "",
                    task_third_p, knowledge_index, predictions[knowledge_index],
                    task_third_p, knowledge_index
                )
        elif mol_info_opt.lower() == "sd":
            if "classification" in task_type.lower():
                # chat_question["content"] = chat_question["content"].format(
                #     knowledge_index, smiles[knowledge_index], description[knowledge_index],
                #     knowledge_index, text_pred, probability, question
                # )
                chat_question["content"] = chat_question["content"].format(
                    knowledge_index, smiles[knowledge_index], description[knowledge_index],
                    knowledge_index, text_pred, probability,
                    knowledge_index, task_third_p
                )
            else:
                chat_question["content"] = chat_question["content"].format(
                    knowledge_index, smiles[knowledge_index], description[knowledge_index],
                    task_third_p, knowledge_index, predictions[knowledge_index],
                    task_third_p, knowledge_index
                )
        elif mol_info_opt.lower() == "sc":
            if "classification" in task_type.lower():
                # chat_question["content"] = chat_question["content"].format(
                #     knowledge_index, smiles[knowledge_index], caption[knowledge_index],
                #     knowledge_index, text_pred, probability, question
                # )
                chat_question["content"] = chat_question["content"].format(
                    knowledge_index, smiles[knowledge_index], caption[knowledge_index],
                    knowledge_index, text_pred, probability,
                    knowledge_index, task_third_p
                )
            else:
                chat_question["content"] = chat_question["content"].format(
                    knowledge_index, smiles[knowledge_index], caption[knowledge_index],
                    task_third_p, knowledge_index, predictions[knowledge_index],
                    task_third_p, knowledge_index
                )
        else:
            raise ValueError("Unknown option: {}".format(mol_info_opt))

        if "classification" in task_type.lower():
            # text_label = "True." if labels[knowledge_index] > 0.5 else "False."
            # chat_answer["content"] = chat_answer["content"].format(text_label)
            text_label = "{}".format(task_third_p) if predictions[knowledge_index] > 0.5 \
                else "cannot {}".format(task_first_p)
            chat_answer["content"] = chat_answer["content"].format(knowledge_index, text_label)
        else:
            chat_answer["content"] = chat_answer["content"].format(
                task_third_p, knowledge_index, labels[knowledge_index]
            )

        chat_history.extend([chat_question, chat_answer])

    return chat_history


def add_knowledge_into_chat_history(
        chat_history, knowledge_template, question_template, k, select_knowledge,
        inquiry_opt, mol_info_opt, task_first_p, task_third_p, task_type,
        index, smiles, description, caption, predictions, labels,
        similarity_matrix, train_indices, valid_indices, demo_test
):
    if inquiry_opt == "add-train-knowledge":
        chat_history = add_train_knowledge_into_chat_histroy(
            chat_history=chat_history, knowledge_template=knowledge_template[inquiry_opt],
            question_template=question_template, k=k, select_knowledge=select_knowledge,
            mol_info_opt=mol_info_opt, index=index, smiles=smiles, task_first_p=task_first_p,
            task_third_p=task_third_p, description=description, caption=caption, labels=labels,
            predictions=predictions, similarity_matrix=similarity_matrix, train_indices=train_indices,
            task_type=task_type, demo_test=demo_test
        )
    elif inquiry_opt == "add-valid-knowledge":
        chat_history = add_valid_knowledge_into_chat_history(
            chat_history=chat_history, knowledge_template=knowledge_template[inquiry_opt],
            question_template=question_template, k=k, select_knowledge=select_knowledge,
            mol_info_opt=mol_info_opt, task_first_p=task_first_p, task_third_p=task_third_p,
            index=index, smiles=smiles, description=description, caption=caption,
            predictions=predictions, labels=labels, similarity_matrix=similarity_matrix,
            valid_indices=valid_indices, task_type=task_type, demo_test=demo_test
        )
    else:
        raise ValueError("Unknown option: {}".format(inquiry_opt))

    return chat_history


def add_double_check_into_chat_history(
        chat_history, inquiry_template, eval_matrix,
        train_performance, valid_performance, question
):
    inquiry = inquiry_template["double-check"]
    inquiry["content"] = inquiry["content"].format(
        eval_matrix, train_performance, valid_performance, question
    )

    chat_history.append(inquiry)

    return chat_history


def add_inquiry_into_chat_history(
        chat_history, inquiry_template, inquiry_opt, question_template,
        index, task_third_p, eval_matrix, train_performance, valid_performance,
        task_type
):
    if "classification" in task_type.lower():
        question = question_template.format(index, task_third_p)
    else:
        question = question_template.format(task_third_p, index)

    if inquiry_opt == "double-check":
        chat_history = add_double_check_into_chat_history(
            chat_history=chat_history, inquiry_template=inquiry_template,
            question=question, eval_matrix=eval_matrix,
            train_performance=train_performance, valid_performance=valid_performance
        )
    else:
        raise ValueError("Unknown option: {}".format(inquiry_opt))

    return chat_history


@time_logger
def main(cfg):

    # Set up OpenAI API
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        api_version="2023-12-01-preview",
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    ) if cfg.llm.provider == 'aoai' else OpenAI(api_key=cfg.OPENAI_API_KEY)

    (dataset, train_indices, valid_indices, test_indices, predictions,
     labels, train_performance, valid_performance,
     similarity_matrix, smiles, caption, description) = run_loading(cfg=cfg)

    # Get indices to query
    indices = np.random.choice(np.concatenate((valid_indices, test_indices)), cfg.num_sample) if cfg.demo_test \
        else np.concatenate((valid_indices, test_indices))

    task_first_p = task_list_first_person[cfg.dataset]
    task_third_p = task_list_third_person[cfg.dataset]

    # Get options S-Top-Train_Valid_10-SP-P
    mol_info_opt = cfg.llm.template.split("-")[0]
    select_knowledge = cfg.llm.template.split("-")[1]
    multi_round = cfg.llm.template.split("-")[2].split("_")[:-1]
    k = int(cfg.llm.template.split("-")[2].split("_")[-1])
    question_info = cfg.llm.template.split("-")[3]
    question_opt = cfg.llm.template.split("-")[4]

    # Get keyword for saving. It combines all tunable parameter settings.
    keyword_for_saving = generate_complete_template_to_save_llm_outputs(cfg=cfg)

    # Set up templates
    if "classification" in dataset.task_type.lower():
        question_template = copy.deepcopy(cls_multi_round_question_template[question_opt])
        multi_round_question_chat_template = copy.deepcopy(cls_multi_round_question_chat_template)
        knowledge_template = copy.deepcopy(cls_multi_round_knowledge_template)
        inquiry_template = copy.deepcopy(cls_multi_round_inquiry_template)
    else:
        question_template = copy.deepcopy(reg_multi_round_question_template[question_opt])
        multi_round_question_chat_template = copy.deepcopy(reg_multi_round_question_chat_template)
        knowledge_template = copy.deepcopy(reg_multi_round_knowledge_template)
        inquiry_template = copy.deepcopy(reg_multi_round_inquiry_template)

    # Save all queries
    chat_history_no_double_check_list = []
    chat_history_list = []
    display = "Query {} {} {}".format(cfg.dataset, cfg.llm.model.name, keyword_for_saving)
    for idx, index in enumerate(tqdm(indices, desc=display)):
        flag = check_llm_cache(
            dataset_name=cfg.dataset, template=keyword_for_saving, data_format="chat_history",
            operate_object=cfg.llm.operate, gnn_model=cfg.gnn.model.name, lm_model=cfg.lm.model.name,
            seed=cfg.seed, llm_model=cfg.llm.model.name, sample_id=index, demo_test=cfg.demo_test
        )
        if flag:
            chat_history_no_double_check = load_llm_outputs(
                dataset_name=cfg.dataset, data_format="chat_history",
                operate_object=cfg.llm.operate, gnn_model=cfg.gnn.model.name,
                lm_model=cfg.lm.model.name, seed=cfg.seed,
                template=keyword_for_saving + "_no_double_check",
                llm_model=cfg.llm.model.name,
                sample_id=index, demo_test=cfg.demo_test
            )
            chat_history = load_llm_outputs(
                dataset_name=cfg.dataset, data_format="chat_history",
                operate_object=cfg.llm.operate, gnn_model=cfg.gnn.model.name,
                lm_model=cfg.lm.model.name, seed=cfg.seed,
                template=keyword_for_saving, llm_model=cfg.llm.model.name,
                sample_id=index, demo_test=cfg.demo_test
            )
            chat_history_no_double_check_list.append(chat_history_no_double_check)
            chat_history_list.append(chat_history)
        else:
            program_state = "continue"
            while program_state != "next":
                chat_history = [multi_round_instruction]
                for step in multi_round:
                    if step.lower() == "train":
                        chat_history = add_knowledge_into_chat_history(
                            chat_history=chat_history, knowledge_template=knowledge_template,
                            question_template=question_template, k=k, select_knowledge=select_knowledge,
                            inquiry_opt="add-train-knowledge", mol_info_opt=mol_info_opt,
                            task_first_p=task_first_p, task_third_p=task_third_p, task_type=dataset.task_type,
                            index=index, smiles=smiles, description=description, caption=caption,
                            predictions=predictions, labels=labels, similarity_matrix=similarity_matrix,
                            train_indices=train_indices, valid_indices=valid_indices, demo_test=cfg.demo_test
                        )
                    elif step.lower() == "valid":
                        chat_history = add_knowledge_into_chat_history(
                            chat_history=chat_history, knowledge_template=knowledge_template,
                            question_template=question_template, k=k, select_knowledge=select_knowledge,
                            inquiry_opt="add-valid-knowledge", mol_info_opt=mol_info_opt,
                            task_first_p=task_first_p, task_third_p=task_third_p, task_type=dataset.task_type,
                            index=index, smiles=smiles, description=description, caption=caption,
                            predictions=predictions, labels=labels, similarity_matrix=similarity_matrix,
                            train_indices=train_indices, valid_indices=valid_indices, demo_test=cfg.demo_test
                        )
                    else:
                        raise ValueError("Unknown option: {}".format(step))
                chat_history = add_question_chat_into_chat_history(
                    chat_history=chat_history, index=index, question_template=question_template,
                    task_first_p=task_first_p, task_third_p=task_third_p, mol_info_opt=mol_info_opt,
                    smiles=smiles, description=description, caption=caption, question_info=question_info,
                    predictions=predictions, task_type=dataset.task_type,
                    multi_round_question_chat_template=multi_round_question_chat_template
                )
                
                # Check number of tokens
                check_num_tokens(chat_history=chat_history)
                # Execute query
                response = execute_query(
                    cfg=cfg, client=client, chat_history=chat_history,
                )
                state = check_response(
                    response=response, pred=predictions[index], task_type=dataset.task_type,
                )
                chat_history = add_response_into_chat_history(
                    chat_history=chat_history, response=response
                )
                chat_history_no_double_check = copy.deepcopy(chat_history)
                if cfg.demo_test:
                    print("\n0", "*"*20)
                    print("Chat history: {}".format(chat_history))
                    print("State: {}".format(state))

                if state == "format":
                    program_state = "repeat"
                    continue
                elif state == "next":
                    program_state = "next"
                elif state == "double-check":
                    chat_history = add_inquiry_into_chat_history(
                        chat_history=chat_history, inquiry_template=inquiry_template,
                        inquiry_opt="double-check", question_template=question_template,
                        index=index, task_third_p=task_third_p, eval_matrix=dataset.eval_metric,
                        train_performance=train_performance, valid_performance=valid_performance,
                        task_type=dataset.task_type
                    )
                    
                    # Check number of tokens
                    check_num_tokens(chat_history=chat_history)
                    # Execute query
                    _response = copy.deepcopy(response)
                    response = execute_query(
                        cfg=cfg, client=client, chat_history=chat_history,
                    )
                    state = check_response(
                        response=response, pred=predictions[index], task_type=dataset.task_type,
                        previous_response=_response
                    )
                    chat_history = add_response_into_chat_history(
                        chat_history=chat_history, response=response
                    )
                    if cfg.demo_test:
                        print("\n1", "*" * 20)
                        print("Chat history: {}".format(chat_history))
                        print("State: {}".format(state))

                    if state == "format":
                        program_state = "repeat"
                        continue
                    elif state == "double-check":
                        program_state = "next"
                    elif state == "next":
                        program_state = "next"
                    else:
                        raise ValueError("Unknown state: {}".format(state))
                else:
                    raise ValueError("Unknown state: {}".format(state))

            if cfg.demo_test:
                print("\nFinal", "*" * 20)
                print("Response: {}".format(response))
                print("{} Molecule ID: {}; Prediction: {:.4f}; Label: {:.4f}.\n\n".format(
                    idx, index, predictions[index], labels[index]
                ))

            chat_history_no_double_check_list.append(chat_history_no_double_check)
            chat_history_list.append(chat_history)
            # Save conversation of each request, in case any errors stop the execution
            save_llm_outputs(
                dataset_name=cfg.dataset, outputs=chat_history_no_double_check, data_format="chat_history",
                template=keyword_for_saving + "_no_double_check", llm_model=cfg.llm.model.name,
                operate_object=cfg.llm.operate, gnn_model=cfg.gnn.model.name, lm_model=cfg.lm.model.name,
                seed=cfg.seed, sample_id=index, demo_test=cfg.demo_test
            )
            save_llm_outputs(
                dataset_name=cfg.dataset, outputs=chat_history, data_format="chat_history",
                template=keyword_for_saving, llm_model=cfg.llm.model.name,
                operate_object=cfg.llm.operate, gnn_model=cfg.gnn.model.name, lm_model=cfg.lm.model.name,
                seed=cfg.seed, sample_id=index, demo_test=cfg.demo_test
            )

    # Save all conversations
    save_llm_outputs(
        dataset_name=cfg.dataset, outputs=chat_history_no_double_check_list, data_format="chat_history",
        template=keyword_for_saving + "_no_double_check", llm_model=cfg.llm.model.name,
        operate_object=cfg.llm.operate, gnn_model=cfg.gnn.model.name, lm_model=cfg.lm.model.name,
        seed=cfg.seed, demo_test=cfg.demo_test
    )
    save_llm_outputs(
        dataset_name=cfg.dataset, outputs=chat_history_list, data_format="chat_history",
        template=keyword_for_saving, llm_model=cfg.llm.model.name,
        operate_object=cfg.llm.operate, gnn_model=cfg.gnn.model.name, lm_model=cfg.lm.model.name,
        seed=cfg.seed, demo_test=cfg.demo_test
    )
    # Clean all cache files
    clean_llm_cache(
        dataset_name=cfg.dataset, template=keyword_for_saving + "_no_double_check", llm_model=cfg.llm.model.name,
        operate_object=cfg.llm.operate, gnn_model=cfg.gnn.model.name, lm_model=cfg.lm.model.name,
        seed=cfg.seed, clean_completion=False, clean_conversation=False,
        clean_response=False, clean_chat_history=True
    )
    clean_llm_cache(
        dataset_name=cfg.dataset, template=keyword_for_saving, llm_model=cfg.llm.model.name,
        operate_object=cfg.llm.operate, gnn_model=cfg.gnn.model.name, lm_model=cfg.lm.model.name,
        seed=cfg.seed, clean_completion=False, clean_conversation=False,
        clean_response=False, clean_chat_history=True
    )


if __name__ == "__main__":
    cfg = update_cfg(cfg)
    main(cfg)


# python -m code.query_chatgpt_multi_round dataset ogbg-molbace llm.template S-Top-Train_Valid_3-SP-PPE
# llm.emb.data smi-des llm.emb.model text-embedding-3-small llm.model.temperature 0. llm.model.top_p 1.
# num_sample 3 demo_test True
