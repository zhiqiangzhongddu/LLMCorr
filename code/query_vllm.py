import os
import numpy as np
import pandas as pd
from pathlib import Path, PurePath

from vllm import LLM, SamplingParams

from code.config import cfg, update_cfg
from code.utils import set_seed


# load cfg
# cfg = update_cfg(cfg)
set_seed(cfg.seed)

# manual cfg settings
cfg.dataset = "ogbg-molhiv" # ogbg-molhiv
cfg.llm.model.name = "Mistral-7B-v0.1"
cfg.device = "0,1"
cfg.llm.prompt = "IP"
cfg.demo_test = True

os.environ["CUDA_VISIBLE_DEVICES"] = cfg.device
prompt_file_name = ("%s/input/prompt/%s_%s.csv" %
                    (PurePath(Path.home(), 'LLMaGML'), cfg.dataset, cfg.llm.prompt))
if cfg.demo_test:
    answer_file_name = ("%s/input/response/test_%s_%s_%s.csv" %
                        (PurePath(Path.home(), 'LLMaGML'), cfg.dataset, cfg.llm.prompt, cfg.llm.model.name))
else:
    answer_file_name = ("%s/input/response/%s_%s_%s.csv" %
                        (PurePath(Path.home(), 'LLMaGML'), cfg.dataset, cfg.llm.prompt, cfg.llm.model.name))

# load prompt
list_prompt = []
df_prompt = pd.read_csv(prompt_file_name)
if cfg.demo_test:
    list_prompt = df_prompt["prompt"].values.tolist()[:10]
else:
    list_prompt = df_prompt["prompt"].values.tolist()

# sampling_params = SamplingParams(
#     temperature=0.8,
#     top_p=0.95,
#     max_tokens=1000, # int = 16 Maximum number of tokens to generate per output sequence.
# )
sampling_params = SamplingParams(
    temperature=0.6,
    top_p=0.9,
    max_tokens=512,
)

if cfg.llm.model.name == "Llama-2-13b-chat-hf":
    llm = LLM(
        model="meta-llama/Llama-2-13b-chat-hf",
        tensor_parallel_size=cfg.device.count(",")+1,
    )
elif cfg.llm.model.name == "Llama-2-7b-chat-hf":
    llm = LLM(
        model="meta-llama/Llama-2-7b-chat-hf",
        tensor_parallel_size=cfg.device.count(",")+1,
    )
elif cfg.llm.model.name == "Mistral-7B-v0.1":
    llm = LLM(
        model="mistralai/Mistral-7B-v0.1",
        tensor_parallel_size=cfg.device.count(",")+1,
    )
else:
    raise ValueError("%s is not a valid LLM" % cfg.llm)

outputs = llm.generate(
    prompts=list_prompt,
    sampling_params=sampling_params,
)
list_response = [output.outputs[0].text for output in outputs]

df_res = pd.DataFrame({
    'id': np.arange(len(list_prompt)),
    'prompt': list_prompt,
    'response': list_response
})
df_res.to_csv(answer_file_name, index=False)
