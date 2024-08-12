from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration

from code.config import cfg, update_cfg
from code.utils import set_seed, time_logger
from code.data_utils.dataset import DatasetLoader
from code.data_utils.utils import save_caption


@time_logger
def main(cfg):
    set_seed(cfg.seed)

    dataloader = DatasetLoader(name=cfg.dataset, text='raw')
    text = dataloader.text
    if cfg.demo_test:
        text = text[:10]

    tokenizer = T5Tokenizer.from_pretrained(
        "laituan245/molt5-large-smiles2caption",
        model_max_length=512,
    )
    model = T5ForConditionalGeneration.from_pretrained(
        'laituan245/molt5-large-smiles2caption'
    ).to(cfg.device)

    list_caption = []
    for smiles in tqdm(text):
        input_ids = tokenizer(smiles, return_tensors="pt").input_ids.to(cfg.device)

        outputs = model.generate(input_ids, num_beams=5, max_length=512)
        # print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        list_caption.append(tokenizer.decode(outputs[0], skip_special_tokens=True))

    save_caption(dataset_name=cfg.dataset, list_caption=list_caption, demo_test=cfg.demo_test)


if __name__ == "__main__":
    cfg = update_cfg(cfg)
    main(cfg)