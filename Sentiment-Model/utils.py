import torch
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import accuracy_score,f1_score
from datasets import load_dataset
from tqdm import tqdm
import datasets
import torch

def scorer(prompt):

    sentiments = ['positive', 'neutral', 'negative']

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits 
        
    # Get logits for the last token (the next token to be predicted)
    last_token_logits = logits[:, -1, :]
    # Convert logits to probabilities
    probabilities = torch.softmax(last_token_logits, dim=-1)  # Shape: [batch_size, vocab_size]

    # Get probabiliies of sentiments
    sentiments_prob = [probabilities[0, tokenizer.convert_tokens_to_ids(s)].items() for s in sentiments]

    # Standarized Positive - Standarized Negative
    sentiment_score = (sentiments_prob[0] - sentiments_prob[2])/sum(sentiments_prob)
    return sentiment_score


def tokenize(args, tokenizer, feature):
    
    prompt_ids = tokenizer.encode(
        feature['prompt'].strip(), add_special_tokens=False, max_length=args.max_length, truncation=True
    )
    
    target_ids = tokenizer.encode(
        feature['output'].strip(), add_special_tokens=False
    )
    
    input_ids = prompt_ids + target_ids
    exceed_max_length = 1 if len(input_ids) > args.max_length else 0
    
     # Add EOS Token
    if input_ids[-1] != tokenizer.eos_token_id and exceed_max_length==0:
        input_ids.append(tokenizer.eos_token_id)
    
    label_ids = [tokenizer.pad_token_id] * len(prompt_ids) + input_ids[len(prompt_ids):]
    
    return {
        "input_ids": input_ids,
        "labels": label_ids,
        "exceed_max_length": exceed_max_length
    }
    
def evaluation(ref_ans, gen_ans):
    # check for accuracy
    nums = len(ref_ans)
    true_pred = [1 if ref_ans[i] in gen_ans[i] else 0 for i in range(nums)]
    accuracy = sum(true_pred) / nums

    return accuracy

def sample_by_label(dataset, label, size):
    # Filter the dataset by label
    filtered = dataset.filter(lambda x: x["output"] == label)
    # Randomly select the specified number of samples
    sampled = filtered.shuffle(seed=42).select(range(min(size, len(filtered))))
    return sampled



dic = {
    0:"negative",
    1:'positive',
    2:'neutral',
}
BINST = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>"
EINST = "<|eot_id|>"
BINPUT = "<|start_header_id|>user<|end_header_id|>"
EINPUT = "<|eot_id|>"

def format_example(example: dict) -> dict:
    context = BINST + '\n\n' + example['instruction'] + EINST + '\n'
    if example.get("input"):
        context += BINPUT + '\n\n' + example['input'] + EINPUT
    target = example["output"]
    return {"context": context, "target": target}

def change_target(x):
    if 'positive' in x or 'Positive' in x:
        return 'positive'
    elif 'negative' in x or 'Negative' in x:
        return 'negative'
    else:
        return 'neutral'

def test_tfns(model, tokenizer, batch_size = 8, prompt_fun = None):
    dataset = load_dataset('zeroshot/twitter-financial-news-sentiment')
    dataset = dataset['validation']
    dataset = dataset.to_pandas()
    dataset['label'] = dataset['label'].apply(lambda x:dic[x])
    
    if prompt_fun is None:
        dataset["instruction"] = 'What is the sentiment of this tweet? Please choose an answer from {negative/neutral/positive}.'
    else:
        dataset["instruction"] = dataset.apply(prompt_fun, axis = 1)

    dataset.columns = ['input', 'output', 'instruction']
    dataset[["context","target"]] = dataset.apply(format_example, axis = 1, result_type="expand")

    # print example
    print(f"\n\nPrompt example:\n{dataset['context'][0]}\n\n")

    context = dataset['context'].tolist()

    total_steps = dataset.shape[0]//batch_size + 1
    print(f"Total len: {len(context)}. Batchsize: {batch_size}. Total steps: {total_steps}")


    out_text_list = []
    for i in range(2):
        tmp_context = context[i* batch_size:(i+1)* batch_size]
        print(tmp_context)
        tokens = tokenizer(tmp_context, return_tensors='pt', padding=True, max_length=256, add_special_tokens=False)
        # tokens.pop('token_type_ids')
        for k in tokens.keys():
            tokens[k] = tokens[k].cuda()
        res = model.generate(**tokens, max_length=256)
        res_sentences = [tokenizer.decode(i, skip_special_tokens=False) for i in res]
        pattern = r"<\|eot_id\|>(.*?)<\|end_of_text\|>"
        out_text = [re.findall(pattern, o)[0] for o in res_sentences]
        out_text_list += out_text
        torch.cuda.empty_cache()

    dataset["out_text"] = out_text_list
    dataset["new_target"] = dataset["target"].apply(change_target)
    dataset["new_out"] = dataset["out_text"].apply(change_target)

    acc = accuracy_score(dataset["new_target"], dataset["new_out"])
    f1_macro = f1_score(dataset["new_target"], dataset["new_out"], average = "macro")
    f1_micro = f1_score(dataset["new_target"], dataset["new_out"], average = "micro")
    f1_weighted = f1_score(dataset["new_target"], dataset["new_out"], average = "weighted")

    print(f"Acc: {acc}. F1 macro: {f1_macro}. F1 micro: {f1_micro}. F1 weighted (BloombergGPT): {f1_weighted}. ")

    return dataset

