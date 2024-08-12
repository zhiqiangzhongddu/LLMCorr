import os
import copy
import numpy as np
from tqdm import tqdm
from time import sleep
from openai import OpenAI, AzureOpenAI
from pathlib import PurePath
import torch
from ogb.graphproppred import Evaluator

from code.config import cfg, update_cfg
from code.utils import time_logger, project_root_path
from code.data_utils.utils import (load_message, check_llm_cache,
                                   save_llm_outputs, load_llm_outputs,
                                   load_gnn_predictions, clean_llm_cache)
from code.data_utils.dataset import DatasetLoader
from code.message_template import (cls_rag_knowledge_instruction,
                                   reg_rag_knowledge_instruction,
                                   cls_rag_check_message_template,
                                   cls_rag_warm_up_message_template,
                                   task_list_first_person, task_list_third_person)
from code.LLMs.utils import check_response


def get_self_check_message(
        message_template, preds, labels, train_indices, valid_indices,
        evaluator, eval_metric
):
    y_true = torch.Tensor(labels)[train_indices].view(-1, 1)
    y_pred = torch.Tensor(preds)[train_indices].view(-1, 1)
    input_dict = {"y_true": y_true, "y_pred": y_pred}
    train_performance = evaluator.eval(input_dict)[eval_metric]

    y_true = torch.Tensor(labels)[valid_indices].view(-1, 1)
    y_pred = torch.Tensor(preds)[valid_indices].view(-1, 1)
    input_dict = {"y_true": y_true, "y_pred": y_pred}
    valid_performance = evaluator.eval(input_dict)[eval_metric]

    message_dict = copy.deepcopy(message_template)
    message_dict["double-check"] = message_template["double-check"].format(
        eval_metric, train_performance, valid_performance
    )

    return message_dict


def get_warm_up_message(message_template, task_first_p, task_third_p):
    message_dict = copy.deepcopy(message_template)

    message_dict["train"][0] = message_template["train"][0].format(
        task_third_p
    )
    message_dict["train"][1] = message_template["train"][1].format(
        task_first_p
    )
    message_dict["train"][2] = message_template["train"][2].format(
        task_first_p
    )
    message_dict["train"][3] = message_template["train"][3].format(
        task_first_p
    )
    message_dict["train"][4] = message_template["train"][4].format(
        task_first_p
    )

    message_dict["valid"][0] = message_template["valid"][0].format(
        task_first_p
    )
    message_dict["valid"][1] = message_template["valid"][1].format(
        task_first_p
    )
    message_dict["valid"][2] = message_template["valid"][2].format(
        task_first_p
    )
    message_dict["valid"][3] = message_template["valid"][3].format(
        task_first_p
    )

    return message_dict


def setup_assistant(cfg, instructions):
    # Set up OpenAI API
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        api_version="2023-12-01-preview",
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    ) if cfg.llm.provider == 'aoai' else OpenAI(api_key=cfg.OPENAI_API_KEY)

    # Upload file with an "assistants" purpose
    if cfg.llm.rag.add_knowledge is None:
        if not cfg.demo_test:
            file_path = PurePath(
                project_root_path, "input", "rag_knowledge", cfg.dataset,
                "rag_knowledge_{}_{}_{}_seed{}.json".format(
                    cfg.dataset, cfg.llm.template, cfg.gnn.model.name, cfg.seed
                )
            )
        else:
            file_path = PurePath(
                project_root_path, "input", "rag_knowledge", cfg.dataset,
                "demo_rag_knowledge_{}_{}_{}_seed{}.json".format(
                    cfg.dataset, cfg.llm.template, cfg.gnn.model.name, cfg.seed
                )
            )

        file = client.files.create(
            file=open(file_path, "rb"),
            purpose='assistants'
        )

        # Create an assistant and add the Knowledge file to the assistant
        assistant = client.beta.assistants.create(
            name="+".join([cfg.dataset, cfg.llm.model.name, cfg.gnn.model.name, str(cfg.seed)]),
            instructions=instructions,
            model=cfg.llm.model.name,
            tools=[{"type": "retrieval"}],
            file_ids=[file.id]
        )
    else:
        if not cfg.demo_test:
            file_path_main = PurePath(
                project_root_path, "input", "rag_knowledge", cfg.dataset,
                "rag_knowledge_{}_{}_{}_seed{}.json".format(
                    cfg.dataset, cfg.llm.template, cfg.gnn.model.name, cfg.seed
                )
            )
            file_path_additional = PurePath(
                project_root_path, "input", "rag_knowledge", cfg.dataset,
                "rag_knowledge_{}_{}_{}_seed{}.json".format(
                    cfg.dataset, cfg.llm.rag.add_knowledge,
                    cfg.gnn.model.name, cfg.seed
                )
            )
        else:
            file_path_main = PurePath(
                project_root_path, "input", "rag_knowledge", cfg.dataset,
                "demo_rag_knowledge_{}_{}_{}_seed{}.json".format(
                    cfg.dataset, cfg.llm.template, cfg.gnn.model.name, cfg.seed
                )
            )
            file_path_additional = PurePath(
                project_root_path, "input", "rag_knowledge", cfg.dataset,
                "demo_rag_knowledge_{}_{}_{}_seed{}.json".format(
                    cfg.dataset, cfg.llm.rag.add_knowledge,
                    cfg.gnn.model.name, cfg.seed
                )
            )

        file_main = client.files.create(
            file=open(file_path_main, "rb"),
            purpose='assistants'
        )
        file_additional = client.files.create(
            file=open(file_path_additional, "rb"),
            purpose='assistants'
        )

        # Create an assistant and add the Knowledge file to the assistant
        assistant = client.beta.assistants.create(
            name="+".join([cfg.dataset, cfg.llm.model.name, cfg.gnn.model.name, str(cfg.seed)]),
            instructions=instructions,
            model=cfg.llm.model.name,
            tools=[{"type": "retrieval"}],
            file_ids=[file_main.id, file_additional.id]
        )
    return assistant, client


def run_conversation(
        assistant, client, thread, message, pred=None, task_type=None, check_state=False
):

    if check_state:
        assert pred is not None and task_type is not None, "Please provide Pred and Task Type."

    # Add a Message to a Thread
    response_add_message = client.beta.threads.messages.create(
        thread_id=thread.id, role="user", content=message,
    )
    # Run the Assistant
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id,
    )
    while (run.status != "completed"):
        # sleep(1)
        # print("Waiting for the Assistant to respond...")
        run = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id
        )

    # Once the Run completes, you can retrieve the
    # Messages added by the Assistant to the Thread.
    conversation = client.beta.threads.messages.list(
        thread_id=thread.id
    )

    response = conversation.data[0].content[0].text.value

    state = check_response(
        response=response, pred=pred,
        task_type=task_type
    ) if check_state else None

    # Reorder conversation into sequential
    conversation = [conv.content[0].text.value for conv in conversation][::-1]

    return conversation, response, state


def run_warm_up_conversation(
        warm_up, warm_up_message_dict, assistant, client, thread
):
    if "train" in warm_up.lower():
        for message in warm_up_message_dict["train"]:
            conversation, response, state = run_conversation(
                assistant=assistant, client=client, thread=thread,
                message=message
            )
            if cfg.demo_test:
                print("Response: {}".format(response))

    if "valid" in warm_up.lower():
        for message in warm_up_message_dict["train"]:
            conversation, response, state = run_conversation(
                assistant=assistant, client=client, thread=thread,
                message=message
            )
            if cfg.demo_test:
                print("Response: {}".format(response))


def setup_template(cfg):

    # Set up file path for saving
    if cfg.llm.rag.add_knowledge is None:
        if cfg.llm.rag.warm_up:
            template_to_save = "_".join([cfg.llm.template, "warm_up", cfg.llm.rag.warm_up]) \
                if not cfg.demo_test \
                else "_".join(["demo", cfg.llm.template, "warm_up", cfg.llm.rag.warm_up])
            first_template_to_save = "_".join([cfg.llm.template, "warm_up", cfg.llm.rag.warm_up, "first"]) \
                if not cfg.demo_test \
                else "_".join(["demo", cfg.llm.template, "warm_up", cfg.llm.rag.warm_up, "first"])
            final_template_to_save = "_".join([cfg.llm.template, "warm_up", cfg.llm.rag.warm_up, "final"]) \
                if not cfg.demo_test \
                else "_".join(["demo", cfg.llm.template, "warm_up", cfg.llm.rag.warm_up, "final"])
        else:
            template_to_save = "_".join([cfg.llm.template]) if not cfg.demo_test \
                else "_".join(["demo", cfg.llm.template])
            first_template_to_save = "_".join([cfg.llm.template, "first"]) if not cfg.demo_test \
                else "_".join(["demo", cfg.llm.template, "first"])
            final_template_to_save = "_".join([cfg.llm.template, "final"]) if not cfg.demo_test \
                else "_".join(["demo", cfg.llm.template, "final"])
    else:
        if cfg.llm.rag.warm_up:
            template_to_save = "_".join([
                cfg.llm.template, cfg.llm.rag.add_knowledge, "warm_up", cfg.llm.rag.warm_up
            ]) if not cfg.demo_test else "_".join([
                "demo", cfg.llm.template, cfg.llm.rag.add_knowledge, "warm_up", cfg.llm.rag.warm_up
            ])
            first_template_to_save = "_".join([
                cfg.llm.template, cfg.llm.rag.add_knowledge, "warm_up", cfg.llm.rag.warm_up, "first"
            ]) if not cfg.demo_test else "_".join([
                "demo", cfg.llm.template, cfg.llm.rag.add_knowledge, "warm_up", cfg.llm.rag.warm_up, "first"
            ])
            final_template_to_save = "_".join([
                cfg.llm.template, cfg.llm.rag.add_knowledge, "warm_up", cfg.llm.rag.warm_up, "final"
            ]) if not cfg.demo_test else "_".join([
                "demo", cfg.llm.template, cfg.llm.rag.add_knowledge, "warm_up", cfg.llm.rag.warm_up, "final"
            ])
        else:
            template_to_save = "_".join([
                cfg.llm.template, cfg.llm.rag.add_knowledge
            ]) if not cfg.demo_test else "_".join([
                "demo", cfg.llm.template, cfg.llm.rag.add_knowledge
            ])
            first_template_to_save = "_".join([
                cfg.llm.template, cfg.llm.rag.add_knowledge, "first"
            ]) if not cfg.demo_test else "_".join([
                "demo", cfg.llm.template, cfg.llm.rag.add_knowledge, "first"
            ])
            final_template_to_save = "_".join([
                cfg.llm.template, cfg.llm.rag.add_knowledge, "final"
            ]) if not cfg.demo_test else "_".join([
                "demo", cfg.llm.template, cfg.llm.rag.add_knowledge, "final"
            ])

    return first_template_to_save, final_template_to_save, template_to_save


def load_cache(cfg, first_template_to_save, final_template_to_save, template_to_save, index):
    response_first = load_llm_outputs(
        dataset_name=cfg.dataset, data_format="response",
        template=first_template_to_save, llm_model=cfg.llm.model.name,
        gnn_model=cfg.gnn.model.name, seed=cfg.seed,
        sample_id=index, demo_test=cfg.demo_test
    )
    response_final = load_llm_outputs(
        dataset_name=cfg.dataset, data_format="response",
        template=final_template_to_save, llm_model=cfg.llm.model.name,
        gnn_model=cfg.gnn.model.name, seed=cfg.seed,
        sample_id=index, demo_test=cfg.demo_test
    )
    conversation = load_llm_outputs(
        dataset_name=cfg.dataset, data_format="conversation",
        template=template_to_save, llm_model=cfg.llm.model.name,
        gnn_model=cfg.gnn.model.name, seed=cfg.seed,
        sample_id=index, demo_test=cfg.demo_test
    )

    return response_first, response_final, conversation


@time_logger
def main(cfg):

    # Preprocess data
    dataloader = DatasetLoader(name=cfg.dataset, text='raw')
    dataset, smiles = dataloader.dataset, dataloader.text

    task_first_p = task_list_first_person[cfg.dataset]
    task_third_p = task_list_third_person[cfg.dataset]

    split_idx = dataset.get_idx_split()
    train_indices = split_idx['train'].numpy()
    valid_indices = split_idx['valid'].numpy()
    test_indices = split_idx["test"].numpy()

    predictions = torch.sigmoid(load_gnn_predictions(
        dataset_name=cfg.dataset, gnn_model_name=cfg.gnn.model.name,
        feature=cfg.data.feature, lm_model_name=cfg.lm.model.name, seed=cfg.seed
    )).squeeze().numpy()
    labels = dataset.y.squeeze().numpy()

    # Load Messages
    message_list = load_message(
        dataset_name=cfg.dataset, message_type=cfg.llm.template,
        gnn_model=cfg.gnn.model.name, seed=cfg.seed,
        demo_test=cfg.demo_test
    )

    # Set up file path for saving
    first_template_to_save, final_template_to_save, template_to_save = setup_template(cfg=cfg)
    # print(first_template_to_save, final_template_to_save)

    # Query indices
    indices = np.random.choice(test_indices, cfg.num_sample) if cfg.demo_test \
        else test_indices

    # Set instruction
    if "classification" in dataset.task_type:
        instructions = cls_rag_knowledge_instruction.format(task_third_p)
        warm_up_message_dict = cls_rag_warm_up_message_template
        self_check_message_dict = cls_rag_check_message_template
    else:
        instructions = reg_rag_knowledge_instruction.format(task_third_p)
        warm_up_message_dict = None
        self_check_message_dict = None

    # get self-check message
    self_check_message = get_self_check_message(
        message_template=self_check_message_dict,
        preds=predictions, labels=dataset.y,
        train_indices=train_indices, valid_indices=valid_indices,
        evaluator=Evaluator(cfg.dataset), eval_metric=dataset.eval_metric
    )
    # get warm-up message
    warm_up_message = get_warm_up_message(
        message_template=warm_up_message_dict,
        task_first_p=task_first_p, task_third_p=task_third_p
    )

    # Set up assistant
    assistant, client = setup_assistant(cfg=cfg, instructions=instructions)

    # Save all queries
    first_response_list = []
    final_response_list = []
    conversation_list = []

    display = "Query RaG {} {}".format(cfg.dataset, cfg.llm.model.name)
    for idx, index in enumerate(tqdm(indices, desc=display)):
        message = message_list[index]
        pred = predictions[index]
        label = labels[index]

        flag_first = check_llm_cache(
            dataset_name=cfg.dataset, data_format="response",
            template=first_template_to_save, llm_model=cfg.llm.model.name,
            gnn_model=cfg.gnn.model.name, seed=cfg.seed,
            sample_id=index, demo_test=cfg.demo_test
        )
        flag_final = check_llm_cache(
            dataset_name=cfg.dataset, data_format="response",
            template=final_template_to_save, llm_model=cfg.llm.model.name,
            gnn_model=cfg.gnn.model.name, seed=cfg.seed,
            sample_id=index, demo_test=cfg.demo_test
        )
        flag = flag_first and flag_final

        if flag:
            response_first, response_final, conversation = load_cache(
                cfg=cfg, first_template_to_save=first_template_to_save,
                final_template_to_save=final_template_to_save,
                template_to_save=template_to_save, index=index
            )
            first_response_list.append(response_first)
            final_response_list.append(response_final)
            conversation_list.append(conversation)
            if cfg.demo_test:
                print("Response: {}".format(response_final))
                print("{} Molecule ID: {}; Prediction: {:.4f}; Label: {}.\n\n".format(
                    idx, index, pred, label
                ))
        else:
            program_state = "continue"
            repeat = 0
            while program_state == "continue":
                # Create a Thread
                thread = client.beta.threads.create()

                # Run warm up
                if cfg.llm.rag.warm_up is None:
                    pass
                else:
                    run_warm_up_conversation(
                        warm_up=cfg.llm.rag.warm_up,
                        warm_up_message_dict=warm_up_message,
                        assistant=assistant, client=client, thread=thread
                    )

                conversation, response, conversation_state = run_conversation(
                    assistant=assistant, client=client, thread=thread,
                    message=message, pred=pred, task_type=dataset.task_type, check_state=True
                )
                repeat += 1

                if cfg.demo_test:
                    print("Response: {}".format(response))
                    print("State: {}".format(conversation_state))

                first_response_list.append(response)

                if conversation_state == "next":
                    program_state = "next"
                elif conversation_state == "double-check":
                    conversation, response, _state = run_conversation(
                        assistant=assistant, client=client, thread=thread,
                        message=self_check_message["double-check"],
                        pred=pred, task_type=dataset.task_type, check_state=True
                    )
                    if cfg.demo_test:
                        print("Response: {}".format(response))
                        print("State: {}".format(_state))

                    if _state == "next" or _state == "double-check":
                        program_state = "next"
                    else:
                        # Close a Thread
                        try:
                            delete_thread_response = client.beta.threads.delete(thread_id=thread.id)
                        except Exception as e:
                            print("Deleting thread with format error...", e)
                        first_response_list = first_response_list[:-1]
                    repeat += 1
                elif conversation_state == "format":
                    # Close a Thread
                    try:
                        delete_thread_response = client.beta.threads.delete(thread_id=thread.id)
                    except Exception as e:
                        print("Deleting thread with format error...", e)
                    first_response_list = first_response_list[:-1]
                else:
                    raise Exception("Unknown conversation state: {}".format(conversation_state))

                program_state = "next" if repeat > 5 else program_state

            if cfg.demo_test:
                print("Response: {}".format(response))
                print("{} Molecule ID: {}; Prediction: {:.4f}; Label: {}.\n\n".format(
                    idx, index, pred, label
                ))

            # Append true molecule information, easy to check the dialogue afterward
            conversation.append("{} Molecule ID: {}; Prediction: {:.4f}; Label: {}.".format(
                idx, index, pred, label
            ))

            final_response_list.append(response)
            conversation_list.append(conversation)

            # Save response of each request, in case any errors stop the execution
            save_llm_outputs(
                dataset_name=cfg.dataset, outputs=first_response_list[-1], sample_id=index, data_format="response",
                template=first_template_to_save, llm_model=cfg.llm.model.name,
                gnn_model=cfg.gnn.model.name, seed=cfg.seed, demo_test=cfg.demo_test
            )
            # Save response of each request, in case any errors stop the execution
            save_llm_outputs(
                dataset_name=cfg.dataset, outputs=response, sample_id=index, data_format="response",
                template=final_template_to_save, llm_model=cfg.llm.model.name,
                gnn_model=cfg.gnn.model.name, seed=cfg.seed, demo_test=cfg.demo_test
            )
            # Save conversation of each request, in case any errors stop the execution
            save_llm_outputs(
                dataset_name=cfg.dataset,
                outputs=conversation, data_format="conversation",
                template=template_to_save, llm_model=cfg.llm.model.name,
                gnn_model=cfg.gnn.model.name, seed=cfg.seed, sample_id=index, demo_test=cfg.demo_test
            )
            # Close a Thread
            try:
                delete_thread_response = client.beta.threads.delete(thread_id=thread.id)
            except Exception as e:
                print("Deleting thread after final response...", e)

    # Save all responses
    save_llm_outputs(
        dataset_name=cfg.dataset, outputs=first_response_list, data_format="response",
        template=first_template_to_save, llm_model=cfg.llm.model.name,
        demo_test=cfg.demo_test, gnn_model=cfg.gnn.model.name, seed=cfg.seed
    )
    save_llm_outputs(
        dataset_name=cfg.dataset, outputs=final_response_list, data_format="response",
        template=final_template_to_save, llm_model=cfg.llm.model.name,
        demo_test=cfg.demo_test, gnn_model=cfg.gnn.model.name, seed=cfg.seed
    )
    save_llm_outputs(
        dataset_name=cfg.dataset, outputs=conversation_list, data_format="conversation",
        template=template_to_save, llm_model=cfg.llm.model.name,
        demo_test=cfg.demo_test, gnn_model=cfg.gnn.model.name, seed=cfg.seed
    )
    # Clean all cache files
    clean_llm_cache(
        dataset_name=cfg.dataset, template=first_template_to_save,
        llm_model=cfg.llm.model.name, clean_completion=False,
        gnn_model=cfg.gnn.model.name, seed=cfg.seed,
    )
    clean_llm_cache(
        dataset_name=cfg.dataset, template=final_template_to_save,
        llm_model=cfg.llm.model.name, clean_completion=False,
        gnn_model=cfg.gnn.model.name, seed=cfg.seed,
    )
    clean_llm_cache(
        dataset_name=cfg.dataset, template=template_to_save,
        llm_model=cfg.llm.model.name, gnn_model=cfg.gnn.model.name, seed=cfg.seed,
        clean_response=False, clean_completion=False, clean_conversation=True,
    )

    # Delete Assistant
    try:
        delete_assistant_response = client.beta.assistants.delete(assistant_id=assistant.id)
    except Exception as e:
        print("Deleting assistant...", e)


if __name__ == "__main__":
    cfg = update_cfg(cfg)
    main(cfg)
