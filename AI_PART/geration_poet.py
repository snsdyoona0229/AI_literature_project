import torch, os, re, pandas as pd, json
from sklearn.model_selection import train_test_split
from transformers import DataCollatorForLanguageModeling,DataCollatorWithPadding,GPT2Tokenizer,GPT2LMHeadModel,Trainer,TrainingArguments,AutoConfig
from datasets import Dataset
from transformers import BertTokenizer, GPT2LMHeadModel, TextGenerationPipeline
import jieba
import os
import sys
sys.path.append('C:\\Users\\user\\Desktop\\django\\literature_project\\AI_PART\\')
import peot_Classification_model


#Text generation models
from transformers import (
  BertTokenizerFast,
  AutoModel,
)
from transformers import AutoTokenizer,AutoModelForCausalLM

tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
model = AutoModelForCausalLM.from_pretrained('ckiplab/gpt2-base-chinese') # or other models above

# trained model loading
#model_headlines_path = 'C:\\Users\\user\\Desktop\\literature_project\\AI_MODEL\\GPT_WRITE_POET\\CChiese'

#headlines_model = AutoModelForCausalLM.from_pretrained(model_headlines_path)
#headlines_tokenizer = BertTokenizerFast.from_pretrained(model_headlines_path)

device = "cuda:0"

def gernerate_poet():
    generated_text_samples = headlines_model.generate(
    text_ids,
    max_length= 100,  
    do_sample=True,  
    top_k=0,
    temperature=0.90,
    num_return_sequences= 5
    )
    for i, beam in enumerate(generated_text_samples):
      print(f"{i}: {headlines_tokenizer.decode(beam, skip_special_tokens=True)}")
      print()

#gernerate_poet()

def Top_p_nucleus_sampling():
    # text generation example
    generated_text_samples = headlines_model.generate(
        text_ids,
        max_length= 50,  
        do_sample=True,  
        top_k=100,
        top_p=0.92,
        temperature=0.8,
        repetition_penalty= 1.5,
        num_return_sequences= 3
    )

    for i, beam in enumerate(generated_text_samples):
      print(f"{i}: {headlines_tokenizer.decode(beam, skip_special_tokens=True)}")
      print()

#Top_p_nucleus_sampling()

#reference
#https://www.modeldifferently.com/en/2021/12/generaci%C3%B3n-de-fake-news-con-gpt-2/
# https://github.com/ckiplab/ckip-transformers
# https://huggingface.co/   


def peot_generation(model_headlines_path,text):
    model_headlines_path = 'C:\\Users\\user\\Desktop\\literature_project\\AI_MODEL\\GPT_WRITE_POET\\'+ model_headlines_path
    headlines_model = AutoModelForCausalLM.from_pretrained(model_headlines_path)
    headlines_tokenizer = BertTokenizerFast.from_pretrained(model_headlines_path)
    text = text
    text_ids = headlines_tokenizer.encode(text, return_tensors = 'pt')

    generated_text_samples = headlines_model.generate(
    text_ids
    )
    #generated_text_samples
    # text generation example
    generated_text_samples = headlines_model.generate(
        text_ids,
        max_length= 150,  
        do_sample=True,  
        top_k=100,
        top_p=0.92,
        temperature=0.8,
        repetition_penalty= 1.5,
        num_return_sequences= 5
    )
    peot = []
    count_list = []
    for i, beam in enumerate(generated_text_samples):
        peot.append(f"{headlines_tokenizer.decode(beam, skip_special_tokens=True)}".replace(" ",""))
    for j in peot:
        jieba.case_sensitive = True # 可控制對於詞彙中的英文部分是否為case sensitive, 預設False
        seg_list = jieba.cut(j.replace(" ",""))
        list_jieba = list(seg_list)
        set_jieba = set(list_jieba)
        count_jieba = len(list_jieba) - len(set_jieba)
        count_list.append(count_jieba)
    p = count_list.index(min(count_list))  
    return peot[p]
    
        