import jsonlines
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


def sentiment_score(text):
    tokens = tokenizer.encode(text, return_tensors='pt')
    result = model(tokens)
    return int(torch.argmax(result.logits))+1

file_paths = ['data/train.jsonl'] 
text=[]
label=[]
given_text = None
for file_path in file_paths:
        with jsonlines.open(file_path) as reader:
            for obj in reader:
                text.append(obj["text"])
                label.append(obj["label"])


tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

done=0
sent=[]
for i in range(0,len(text)):
    # print(text[i],label[i])
    a=sentiment_score(text[i])
    done+=1
    if done%100==0:
        print(done)
    # print(a)
    sent.append(a)

score=0
for i in range(0,len(sent)):
    if sent[i]<=2 and label[i]==1:
        score+=1
    if sent[i]>=3 and label[i]==0:
        score+=1
        
print(score,len(label))
        
                

# print(results)
# for i in range(0,len(text)):
#     print(text[i],":",label[i])