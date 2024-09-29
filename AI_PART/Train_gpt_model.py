import torch, os, re, pandas as pd, json
from sklearn.model_selection import train_test_split
from transformers import DataCollatorForLanguageModeling,DataCollatorWithPadding,GPT2Tokenizer,GPT2LMHeadModel,Trainer,TrainingArguments,AutoConfig
from datasets import Dataset
from transformers import BertTokenizer, GPT2LMHeadModel, TextGenerationPipeline
from transformers import pipeline, set_seed
from transformers import (
  BertTokenizerFast,
  AutoModel,
)
from transformers import AutoTokenizer,AutoModelForCausalLM

tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
model = AutoModelForCausalLM.from_pretrained('ckiplab/gpt2-base-chinese') # or other models above


def generate_n_text_samples(model, tokenizer, input_text, device, n_samples = 5):
    text_ids = tokenizer.encode(input_text, return_tensors = 'pt')
    text_ids = text_ids.to(device)
    model = model.to(device)

    generated_text_samples = model.generate(
        text_ids, 
        max_length= 100,  
        num_return_sequences= n_samples,
        no_repeat_ngram_size= 2,
        repetition_penalty= 1.5,
        top_p= 0.92,
        temperature= .85,
        do_sample= True,
        top_k= 125,
        early_stopping= True
    )
    gen_text = []
    for t in generated_text_samples:
        text = tokenizer.decode(t, skip_special_tokens=True)
        gen_text.append(text)

        return gen_text

#########################################################################################################

# the eos and bos tokens are defined
bos = '<|endoftext|>'
eos = '<|EOS|>'
pad = '<|pad|>'

special_tokens_dict = {'eos_token': eos, 'bos_token': bos, 'pad_token': pad}

base_tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese',vocab_size=21128, model_max_len=512, is_fast=True)
# the new token is added to the tokenizer
num_added_toks = base_tokenizer.add_special_tokens(special_tokens_dict)

# the model config to which we add the special tokens
config = AutoConfig.from_pretrained('ckiplab/gpt2-base-chinese',vocab_size=21128, model_max_len=512, is_fast=True,
                                    bos_token_id=base_tokenizer.bos_token_id,
                                    eos_token_id=base_tokenizer.eos_token_id,
                                    pad_token_id=base_tokenizer.pad_token_id,
                                    output_hidden_states=False)

# the pre-trained model is loaded with the custom configuration
base_model = AutoModelForCausalLM.from_pretrained('ckiplab/gpt2-base-chinese', config=config)

# the model embedding is resized

#config.vocab_size = base_tokenizer.vocab_size
base_model.resize_token_embeddings(len(base_tokenizer))

#-------------------------------
#result: Embedding(21131, 768) |
#-------------------------------
#########################################################################################################

filepath= '/content/drive/MyDrive/poet/C.xlsx'
df = pd.read_excel(filepath, usecols=['poet_name', 'poet_content'])\
                    .rename(columns={'poet_name': 'poet_name'})

pd.set_option("display.max_colwidth", None)
#df.head(5)

#########################################################################################################

def remove_publication_headline(headline, publication):
    # publication col doesn't match exactly with newspaper in title col
    if str(publication) in str(headline):
        headline = headline.split(' - ')[0]
    return headline

def process_headlines(df, text_colname):
  
    # Remove empty and null rows
    titulo_vacio = (df['poet_content'].str.len() == 0) | df['poet_content'].isna()
    df = df[~titulo_vacio]

    # Remove publication name from title
    df['text'] = df.apply(lambda row: remove_publication_headline(row['poet_content'], row['poet_name']), axis = 1)

    # Remove headlines with less than 8 words
    titlos_len_ge8 = (df['poet_content'].str.split().apply(lambda x: len(x)) >= 8)
    df = df[titlos_len_ge8]

    # Drop duplicates
    text_df = df.drop_duplicates(subset = [text_colname])\
                [[text_colname]]

    return text_df
    
df = process_headlines(df, 'poet_content')
#########################################################################################################

df['poet_content'] = bos + ' ' + df['poet_content'] + ' ' + eos

df_train, df_val = train_test_split(df, train_size = 0.9, random_state = 77)
#print(f'There are {len(df_train)} headlines for training and {len(df_val)} for validation')
#------------------------------------------------------------------
#result: There are 85 headlines for training and 10 for validation |
#------------------------------------------------------------------
#########################################################################################################
# we load the datasets directly from a pandas df
train_dataset = Dataset.from_pandas(df_train[['poet_content']])
val_dataset = Dataset.from_pandas(df_val[['poet_content']])

#print(train_dataset)

#------------------------------------------------------------------
#result: Dataset({                                                |
#    features: ['poet_content', '__index_level_0__'],             |            
#    num_rows: 85                                                 |
#})                                                               |
#------------------------------------------------------------------
#########################################################################################################
text_generation = pipeline("text-generation", model=model, tokenizer=base_tokenizer)
data = pd.read_excel('/content/drive/MyDrive/poet/C.xlsx')
#data.head()
#########################################################################################################
#text_generation("總統大選")

def tokenize_function(examples):
  return base_tokenizer(examples['poet_content'],truncation=True,padding=True)


tokenized_train_dataset = train_dataset.map(
    tokenize_function,
    batched=True,
    num_proc=5,
    remove_columns=['poet_content'],
)
tokenized_val_dataset = val_dataset.map(
    tokenize_function,
    batched=True,
    num_proc=5,
    remove_columns=['poet_content'],
)
#########################################################################################################

# Example of the result of the tokenization process with padding
#base_tokenizer.decode(tokenized_train_dataset['input_ids'][0])

#[CLS] <|endoftext|> 嫁 耳 環 仔 叮 噹 搖 在 我 介 耳 公 邊 講 出 嫁 介 心 情 隻 隻 金 指 含 著 傳 統 介 情 愛 首 扼 仔 落 在 我 介
#  左 右 手 一 圈 一 圈 都 係 祝 福 阿 爸 送 我 三 從 四 德 阿 姑 包 分 我 一 句 話 喚 我 莫 忘 祖 宗 言 雖 然 蒙 等 一 層 濛 濛 介 
# 面 紗 我 也 讀 得 出 這 本 沉 長 介 禮 數 新 娘 車 背 響 起 嚴 肅 介 落 聲 我 會 珍 惜 潑 出 去 介 這 碗 水 紙 扇 輕 輕 跌 落 地 阿 
# 姆 撿 起 搖 清 涼 自 言 自 語 唸 四 句 公 婆 相 惜 早 供 賴 賴 <|EOS|> [SEP] <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> 
# <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> 
# <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> 
# <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|>
#  <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|>
#########################################################################################################
model_headlines_path = '/content/drive/MyDrive/poet/CChiese_1000_new8_3'

training_args = TrainingArguments(
     output_dir=model_headlines_path,          # output directory
    num_train_epochs=3,              # total # of training epochs
    per_device_train_batch_size=3,  # batch size per device during training
    per_device_eval_batch_size=2,   # batch size for evaluation
    warmup_steps=25,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir=model_headlines_path,            # directory for storing logs
    prediction_loss_only=True,
    save_steps=1000 
)
#########################################################################################################

data_collator = DataCollatorForLanguageModeling(
        tokenizer=base_tokenizer,
        mlm=False
    )
model = torch.load('/content/drive/MyDrive/poet/Cmodel_all.pt')#load model
#########################################################################################################
trainer = Trainer(
    model=base_model,                         # the instantiated  Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    data_collator=data_collator,
    train_dataset=tokenized_train_dataset,         # training dataset
    eval_dataset=tokenized_val_dataset            # evaluation dataset
)
trainer.train()
trainer.save_model()
base_tokenizer.save_pretrained(model_headlines_path)

#########################################################################################################
# save whole model
FILE = '/content/drive/MyDrive/poet/Cmodel_all.pt'
torch.save(model, FILE)
#########################################################################################################

trainer.evaluate()

#########################################################################################################
# trained model loading
model_headlines_path = '/content/drive/MyDrive/poet/CChiese_1000_new8_3'

headlines_model = AutoModelForCausalLM.from_pretrained(model_headlines_path)
headlines_tokenizer = BertTokenizerFast.from_pretrained(model_headlines_path)

device = "cuda:0"

#input_text = headlines_tokenizer.bos_token


#headlines = generate_n_text_samples(headlines_model, headlines_tokenizer, 
#                                     input_text, device, n_samples = 10)

#for h in headlines:
#    print(h)
#    print("\n")

#########################################################################################################

text = "一粒星仔"
text_ids = headlines_tokenizer.encode(text, return_tensors = 'pt')

generated_text_samples = headlines_model.generate(
    text_ids
)
generated_text_samples

#tensor([[ 101,  671, 5108, 3215,  798,  102,  671, 5108, 3215,  798,  702,  671,
#         5108, 3215,  798,  702,  671, 5108, 3215,  798]])
#########################################################################################################

# text generation example
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

###########################################################################################################
def greedy_search():
    generated_text_samples = headlines_model.generate(
    text_ids,
    max_length= 100,
    )

    for i, beam in enumerate(generated_text_samples):
      print(f"{i}: {headlines_tokenizer.decode(beam, skip_special_tokens=True)}")
      print()
###########################################################################################################

def beam_search():
    # text generation example
    generated_text_samples = headlines_model.generate(
        text_ids,
        max_length= 50,  
        num_beams=5,
        no_repeat_ngram_size=2,
        num_return_sequences= 5,
        early_stopping=True 
    )

    for i, beam in enumerate(generated_text_samples):
      print(f"{i}: {headlines_tokenizer.decode(beam, skip_special_tokens=True)}")
      print()

###########################################################################################################

def sampling():
    # text generation example
    generated_text_samples = headlines_model.generate(
        text_ids,
        max_length= 50,  
        do_sample=True,  
        top_k=0,
        temperature=0.9,
        num_return_sequences= 5
    )

    for i, beam in enumerate(generated_text_samples):
      print(f"{i}: {headlines_tokenizer.decode(beam, skip_special_tokens=True)}")
      print()
###########################################################################################################
def top_k_sampling():
    # text generation example
    generated_text_samples = headlines_model.generate(
        text_ids,
        max_length= 50,  
        do_sample=True,  
        top_k=25,
        num_return_sequences= 5
    )

    for i, beam in enumerate(generated_text_samples):
      print(f"{i}: {headlines_tokenizer.decode(beam, skip_special_tokens=True)}")
      print()
###########################################################################################################

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
        num_return_sequences= 5
    )

    for i, beam in enumerate(generated_text_samples):
      print(f"{i}: {headlines_tokenizer.decode(beam, skip_special_tokens=True)}")
      print()

#reference
#https://www.modeldifferently.com/en/2021/12/generaci%C3%B3n-de-fake-news-con-gpt-2/
# https://github.com/ckiplab/ckip-transformers
# https://huggingface.co/      