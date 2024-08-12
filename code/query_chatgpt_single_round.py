import copy
import numpy as np
from tqdm import tqdm
from openai import OpenAI, AzureOpenAI
import os

from code.config import cfg, update_cfg
from code.utils import time_logger
from code.LLMs.utils import check_response
from code.message_template import (multi_round_instruction,
                                   cls_multi_round_inquiry_template, reg_multi_round_inquiry_template,
                                   cls_multi_round_question_chat_template, reg_multi_round_question_chat_template,
                                   cls_multi_round_question_template, reg_multi_round_question_template,
                                   cls_multi_round_knowledge_template, reg_multi_round_knowledge_template,
                                   task_list_third_person, task_list_first_person)
from code.data_utils.utils import (save_llm_outputs, load_llm_outputs, check_llm_cache,
                                   clean_llm_cache)
from code.query_chatgpt_multi_round import (run_loading, generate_complete_template_to_save_llm_outputs,
                                            add_knowledge_into_chat_history, add_question_chat_into_chat_history,
                                            check_num_tokens, execute_query, add_response_into_chat_history)


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
    chat_history_list = []
    display = "Query {} {} {}".format(cfg.dataset, cfg.llm.model.name, keyword_for_saving)
    for idx, index in enumerate(tqdm(indices, desc=display)):
        flag = check_llm_cache(
            dataset_name=cfg.dataset, template=keyword_for_saving, data_format="chat_history",
            operate_object=cfg.llm.operate, gnn_model=cfg.gnn.model.name, lm_model=cfg.lm.model.name,
            seed=cfg.seed, llm_model=cfg.llm.model.name, sample_id=index, demo_test=cfg.demo_test
        )
        if flag:
            chat_history = load_llm_outputs(
                dataset_name=cfg.dataset, data_format="chat_history",
                operate_object=cfg.llm.operate, gnn_model=cfg.gnn.model.name,
                lm_model=cfg.lm.model.name, seed=cfg.seed,
                template=keyword_for_saving, llm_model=cfg.llm.model.name,
                sample_id=index, demo_test=cfg.demo_test
            )
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
                if cfg.demo_test:
                    print("\n0", "*"*20)
                    print("Chat history: {}".format(chat_history))
                    print("State: {}".format(state))

                if state == "format":
                    program_state = "repeat"
                    continue
                else:
                    program_state = "next"

            if cfg.demo_test:
                print("\nFinal", "*" * 20)
                print("Response: {}".format(response))
                print("{} Molecule ID: {}; Prediction: {:.4f}; Label: {:.4f}.\n\n".format(
                    idx, index, predictions[index], labels[index]
                ))

            chat_history_list.append(chat_history)
            # Save conversation of each request, in case any errors stop the execution
            save_llm_outputs(
                dataset_name=cfg.dataset, outputs=chat_history, data_format="chat_history",
                template=keyword_for_saving, llm_model=cfg.llm.model.name,
                gnn_model=cfg.gnn.model.name, seed=cfg.seed,
                sample_id=index, demo_test=cfg.demo_test
            )

    # Save all conversations
    save_llm_outputs(
        dataset_name=cfg.dataset, outputs=chat_history_list, data_format="chat_history",
        template=keyword_for_saving, llm_model=cfg.llm.model.name,
        operate_object=cfg.llm.operate, gnn_model=cfg.gnn.model.name, lm_model=cfg.lm.model.name,
        seed=cfg.seed, demo_test=cfg.demo_test
    )
    # Clean all cache files
    clean_llm_cache(
        dataset_name=cfg.dataset, template=keyword_for_saving, llm_model=cfg.llm.model.name,
        operate_object=cfg.llm.operate, gnn_model=cfg.gnn.model.name, lm_model=cfg.lm.model.name,
        seed=cfg.seed, clean_completion=False, clean_conversation=False,
        clean_response=False, clean_chat_history=True
    )


if __name__ == "__main__":
    cfg = update_cfg(cfg)
    main(cfg)


# python -m code.query_chatgpt_single_round dataset ogbg-molfreesolv llm.template S-Top-Train_10-S-PE
# llm.emb.data smi-des llm.emb.model text-embedding-3-large
# llm.model.temperature 1.0 llm.model.top_p 1.0 gnn.model.name gin-v seed 42
