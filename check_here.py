#!/usr/bin/env python
# coding: utf-8

# In[1]:


from Model.RadFM.multimodality_model import MultiLLaMAForCausalLM
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
import torch
from transformers import LlamaTokenizer
from torchvision import transforms
from PIL import Image   
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2"


# In[2]:


def get_tokenizer(max_img_size=100, image_num=32):
    image_padding_tokens = []
    # Load the base tokenizer from the provided path
    text_tokenizer = LlamaTokenizer.from_pretrained("/home/lina/GitHub/Llama-2-13b-hf", local_files_only=True)
    special_token = {"additional_special_tokens": ["<image>", "</image>"]}

    # Generate unique tokens for each image position and patch
    for i in range(max_img_size):
        image_padding_token = ""

        for j in range(image_num):
            image_token = "<image" + str(i * image_num + j) + ">"
            image_padding_token = image_padding_token + image_token
            special_token["additional_special_tokens"].append("<image" + str(i * image_num + j) + ">")

        # Store the concatenated tokens for each image
        image_padding_tokens.append(image_padding_token)

        # Add all special tokens to the tokenizer
        text_tokenizer.add_special_tokens(
            special_token
        )

        # Configure standard special tokens for LLaMA models
        text_tokenizer.pad_token_id = 0
        text_tokenizer.bos_token_id = 1
        text_tokenizer.eos_token_id = 2

    return text_tokenizer, image_padding_tokens


# In[3]:


def combine_and_preprocess(question, image_list, image_padding_tokens):
    # Define image transformation pipeline
    transform = transforms.Compose([                        
                transforms.RandomResizedCrop([512, 512], scale=(0.8, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
            ])
    
    images = []
    new_qestions = [_ for _ in question]  # Convert question string to list of characters
    padding_index = 0
    
    # Process each image in the list
    for img in image_list:
        img_path = img['img_path']
        position = img['position']  # Where to insert the image in the text
        
        # Load and transform the image
        image = Image.open(img_path).convert('RGB')   
        image = transform(image)
        image = image.unsqueeze(0).unsqueeze(-1)  # Add batch and depth dimensions (c,w,h,d)
        
        # Resize the image to target dimensions
        target_H = 512 
        target_W = 512 
        target_D = 4 
        # This can be different for 3D and 2D images. For demonstration we here set this as the default sizes for 2D images. 
        images.append(torch.nn.functional.interpolate(image, size=(target_H, target_W, target_D)))
        
        # Insert image placeholder token at the specified position in text
        new_qestions[position] = "<image>" + image_padding_tokens[padding_index] + "</image>" + new_qestions[position]
        padding_index += 1
    
    # Stack all images into a batch and add batch dimension
    vision_x = torch.cat(images, dim=1).unsqueeze(0)  # Cat tensors and expand the batch_size dim
    
    # Join the character list back into a string
    text = ''.join(new_qestions) 
    return text, vision_x


# In[4]:


print("Setup tokenizer")
text_tokenizer, image_padding_tokens = get_tokenizer()
print("Finish loading tokenizer")


# In[12]:


print("Setup demo case")
question = "Can you identify any visible signs of Cardiomegaly in the image?"

image = [
        {
            'img_path': './view1_frontal.jpg',
            'position': 0,  # Insert at the beginning of the question
        },  # Can add arbitrary number of images
    ] 
text, vision_x = combine_and_preprocess(question, image, image_padding_tokens)    
print("Finish loading demo case")


# In[6]:


print("Setup Model")
with init_empty_weights():
    model = MultiLLaMAForCausalLM()
device_map = {
 'lang_model.model.embed_tokens': 0,
 'lang_model.model.layers.0': 0,
 'lang_model.model.layers.1': 0,
 'lang_model.model.layers.2': 0,
 'lang_model.model.layers.3': 0,
 'lang_model.model.layers.4': 0,
 'lang_model.model.layers.5': 0,
 'lang_model.model.layers.6': 0,
 'lang_model.model.layers.7': 0,
 'lang_model.model.layers.8': 0,
 'lang_model.model.layers.9': 0,
 'lang_model.model.layers.10': 0,
 'lang_model.model.layers.11': 0,
 'lang_model.model.layers.12': 0,   
 'lang_model.model.layers.13.self_attn.q_proj': 0,
 'lang_model.model.layers.13.self_attn.k_proj': 0,
 'lang_model.model.layers.13.self_attn.v_proj': 0,
 'lang_model.model.layers.13.self_attn.o_proj': 0,
 'lang_model.model.layers.13.mlp': 0,
 'lang_model.model.layers.13.input_layernorm': 0,
 'lang_model.model.layers.13.post_attention_layernorm': 0,
 'lang_model.model.layers.14': 1,
 'lang_model.model.layers.15': 1,
 'lang_model.model.layers.16': 1,
 'lang_model.model.layers.17': 1,
 'lang_model.model.layers.18': 1,
 'lang_model.model.layers.19': 1,
 'lang_model.model.layers.20': 1,
 'lang_model.model.layers.21': 1,
 'lang_model.model.layers.22': 1,
 'lang_model.model.layers.23': 1,
 'lang_model.model.layers.24': 1,
 'lang_model.model.layers.25': 1,
 'lang_model.model.layers.26': 1,  
 'lang_model.model.layers.27.self_attn.q_proj': 0,
 'lang_model.model.layers.27.self_attn.k_proj': 0,
 'lang_model.model.layers.27.self_attn.v_proj': 0,
 'lang_model.model.layers.27.self_attn.o_proj': 0,
 'lang_model.model.layers.27.mlp': 0,
 'lang_model.model.layers.27.input_layernorm': 0,
 'lang_model.model.layers.27.post_attention_layernorm': 0,
 'lang_model.model.layers.28': 2,
 'lang_model.model.layers.29': 2,
 'lang_model.model.layers.30': 2,
 'lang_model.model.layers.31': 2,
 'lang_model.model.layers.32': 2,
 'lang_model.model.layers.33': 2,
 'lang_model.model.layers.34': 2,
 'lang_model.model.layers.35': 2,
 'lang_model.model.layers.36': 2,
 'lang_model.model.layers.37': 2,
 'lang_model.model.layers.38': 2,
 'lang_model.model.layers.39': 2,
 'lang_model.model.norm': 0,
 'lang_model.model.rotary_emb': 0,
 'lang_model.lm_head': 0,
 'embedding_layer': 0}
model = load_checkpoint_and_dispatch(
    model, checkpoint='pytorch_model.bin', device_map=device_map
)
print("Finish loading model")
model.eval()


# In[7]:


model.hf_device_map


# In[13]:


with torch.no_grad():
    lang_x = text_tokenizer(
            text, max_length=2048, truncation=True, return_tensors="pt"
    )['input_ids'].to(0)

    vision_x = vision_x.to(0)
    generation = model.generate(lang_x, vision_x)


# In[14]:


generation


# In[15]:


generated_texts = text_tokenizer.batch_decode(generation, skip_special_tokens=True) 


# In[16]:


generated_texts


# In[ ]:




