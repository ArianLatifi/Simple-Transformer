
import torch
import torch.nn.functional as F
from train import model,tokenizer,device,block_size

ds_test_path = r"./dataset/dailyDialogCleandTest.txt"


test_text = open(ds_test_path,encoding="utf-8",mode="r").read()
test_tokens = tokenizer.encode(test_text)
test_data = torch.tensor(
    test_tokens.ids,
    dtype=torch.long,
    device=device)

checkPointPath = "./checkpoints/CLLM-I{id}-S{step}.pth"
checkpoint = torch.load(checkPointPath.format(id="0",step="4900"),map_location = torch.device("cpu"))
model.load_state_dict(checkpoint["model_state_dict"])

model.eval()
ix = torch.randint(
        len(test_data) - block_size,
        (64,)
    )
Xtest = torch.stack([
        test_data[i:i + block_size]
        for i in ix]).to(device)

Ytest = torch.stack([
        test_data[i + 1:i + block_size + 1]
        for i in ix]).to(device)

all_losses = torch.zeros(300)
for i in range(300):
    logits = model(Xtest)
    loss = F.cross_entropy(logits.transpose(1, 2),Ytest)
    all_losses[i] = loss.item()

baseLineLoss = F.cross_entropy(torch.zeros_like(logits).transpose(1, 2),Ytest).item()

print("Test Loss:", all_losses.mean())
print("Baseline Loss:", baseLineLoss)