'''
This test.py is from the Quick_demo provided by the author.
Since the original checkpoint (pytorch_model.bin) was too large and caused OOM (out of memory) errors,
I reduced its precision to create pytorch_model_fp16_github.bin, which successfully ran the test.py in Quick_demo.
'''
import tqdm.auto as tqdm
import torch.nn.functional as F
from typing import Optional, Dict, Sequence
from typing import List, Optional, Tuple, Union
import transformers
from dataclasses import dataclass, field
from Model.RadFM.multimodality_model import MultiLLaMAForCausalLM
import torch
from transformers import LlamaTokenizer
from torchvision import transforms
from PIL import Image
import os   # Lina added


def get_tokenizer(tokenizer_path, max_img_size=100, image_num=32):
    '''
    Initialize the image special tokens
    max_img_size denotes the max image put length and image_num denotes how many patch embeddings the image will be encoded to
    '''
    if isinstance(tokenizer_path, str):
        image_padding_tokens = []
        text_tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)
        special_token = {"additional_special_tokens": ["<image>", "</image>"]}
        for i in range(max_img_size):
            image_padding_token = ""
            for j in range(image_num):
                # image_token = "<image" + str(i * image_num + j) + ">"     # Original code
                image_token = f"<image{i * image_num + j}>"     # Lina updated
                image_padding_token += image_token
                # special_token["additional_special_tokens"].append("<image" + str(i * image_num + j) + ">")     # Original code
                special_token["additional_special_tokens"].append(image_token)     # Lina updated
            image_padding_tokens.append(image_padding_token)
        text_tokenizer.add_special_tokens(special_token)
        ## make sure the bos eos pad tokens are correct for LLaMA-like models
        text_tokenizer.pad_token_id = text_tokenizer.eos_token_id     # Lina added
        # text_tokenizer.pad_token_id = 0     # Original code
        text_tokenizer.bos_token_id = 1
        text_tokenizer.eos_token_id = 2
    return text_tokenizer, image_padding_tokens


# Lina added: convert pytorch_model.bin (original) to pytorch_model_fp16_github.bin (FP16)
def convert_checkpoint_to_fp16(ckpt_path, save_path):   # Lina added
    print(f"Converting checkpoint to FP16: {ckpt_path} → {save_path}")   # Lina added
    ckpt = torch.load(ckpt_path, map_location='cpu')   # Lina added
    for k in ckpt:   # Lina added
        ckpt[k] = ckpt[k].half()   # Lina added
    torch.save(ckpt, save_path)   # Lina added
    print(f"Saved FP16 checkpoint to {save_path}")   # Lina added


def combine_and_preprocess(question, image_list, image_padding_tokens):
    transform = transforms.Compose([
        transforms.RandomResizedCrop([512, 512], scale=(0.8, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
    ])
    images = []
    new_qestions = [_ for _ in question]
    padding_index = 0
    for img in image_list:
        img_path = img['img_path']
        position = img['position']
        image = Image.open(img_path).convert('RGB')
        image = transform(image)
        image = image.unsqueeze(0).unsqueeze(-1)  # c,w,h,d

        ## pre-process the img first
        target_H = 512
        target_W = 512
        target_D = 4
        # This can be different for 3D and 2D images. For demonstration we here set this as the default sizes for 2D images.
        # images.append(torch.nn.functional.interpolate(image, size=(target_H, target_W, target_D)))  # Original code
        images.append(F.interpolate(image, size=(target_H, target_W, target_D)))     # Lina added
        ## add img placeholder to text
        new_qestions[position] = "<image>" + image_padding_tokens[padding_index] + "</image>" + new_qestions[position]
        padding_index += 1
    vision_x = torch.cat(images, dim=1).unsqueeze(0)     # cat tensors and expand the batch_size dim
    text = ''.join(new_qestions)
    return text, vision_x


def main():
    print("Setup tokenizer")
    tokenizer_path = './Language_files'
    text_tokenizer, image_padding_tokens = get_tokenizer(tokenizer_path)

    print("Setup demo case")
    question = "Please provide a diagnostic report for the given mammogram."
    image = [{'img_path': '/home/lina/RadFM_fine-tuning/original_radfm/RadFM-main/Quick_demo/converted_images/20587080_b6a4f750c6df4f90_MG_R_ML_ANON.jpg', 'position': 0}]
    text, vision_x = combine_and_preprocess(question, image, image_padding_tokens)

    print("Finish loading demo case")

    print("Setup Model")
    model = MultiLLaMAForCausalLM(lang_model_path=tokenizer_path)   ### Build up model based on LLaMa-13B config
    # ckpt = torch.load('./pytorch_model.bin',
    #                   map_location='cpu')  # riginal code
    # model.load_state_dict(ckpt) # riginal code
    print("Finish loading model")

    # model = model.to('cuda')    # original code
    # model.eval()    # original code

    # Lina added: Set path
    fp32_ckpt = os.path.join(tokenizer_path, 'pytorch_model.bin')  # Lina added
    fp16_ckpt = os.path.join(tokenizer_path, 'pytorch_model_fp16_github.bin')  # Lina added

    if not os.path.exists(fp16_ckpt):  # Lina added
        print("[INFO] FP16 checkpoint not found. Starting conversion...")  # Lina added
        convert_checkpoint_to_fp16(fp32_ckpt, fp16_ckpt)  # Lina added

    print("Loading FP16 weights...")  # Lina added
    ckpt = torch.load(fp16_ckpt, map_location='cpu')  # Lina added
    model.load_state_dict(ckpt, strict=False)  # Lina added
    model.half().to('cuda')  # Lina added

    print("Running inference...")  # Lina added
    model.eval()    # Lina added
    with torch.no_grad():
        # lang_x = text_tokenizer(
        #     text, max_length=2048, truncation=True, return_tensors="pt"
        # )['input_ids'].to('cuda') # original code
        #
        # vision_x = vision_x.to('cuda') # original code

        lang_x = text_tokenizer(text, max_length=1024, truncation=True, return_tensors="pt")['input_ids'].to('cuda')     # Lina added
        vision_x = vision_x.to('cuda').half()    # Lina updated

        generation = model.generate(lang_x, vision_x)
        generated_texts = text_tokenizer.batch_decode(generation, skip_special_tokens=True)

        print('---------------------------------------------------')
        print('Input: ', question)
        print('Output:', generated_texts[0])


if __name__ == "__main__":
    main()
