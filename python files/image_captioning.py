import torch
from deepface import DeepFace
from promptcap import PromptCap

model = PromptCap("tifa-benchmark/promptcap-coco-vqa")  # also support OFA checkpoints. e.g. "OFA-Sys/ofa-large"

if torch.cuda.is_available():
  print(123)
  model.cuda()

prompt = "asian,Ma"
image = "data/dev_images/01268.png"

print(model.caption(prompt, image))