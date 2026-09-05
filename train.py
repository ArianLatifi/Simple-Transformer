from tokenizers import Tokenizer
import torch
import torch.nn.functional as F
import time
from model import LanguageModel as LM

#device

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"device is { device}")

#constants
tokenizer_path =r"./tokenization/dialogtokens.json"
ds_train_path = r"./dataset/dailyDialogCleand.txt"
ds_validation_path = r"./dataset/dailyDialogCleandValidation.txt"
ds_test_path = r"./dataset/dailyDialogCleandTest.txt"
checkPointPath = "./checkpoints/CLLM-I{id}-S{step}.pth"

block_size = 128
batch_size = 64
n_embd = 96
n_head = 12
steps = 5000
head_size = n_embd // n_head
file_id =  str(int(time.time()))[:4]

# Data Loading and tokenization

tokenizer = Tokenizer.from_file(tokenizer_path)

train_text = open(ds_train_path,encoding="utf-8",mode="r").read()
validation_text = open(ds_validation_path,encoding="utf-8",mode="r").read()

train_tokens = tokenizer.encode(train_text)
test_tokens = tokenizer.encode(validation_text)

train_data = torch.tensor(
    train_tokens.ids,
    dtype=torch.long,
    device=device
)

test_data = torch.tensor(
    test_tokens.ids,
    dtype=torch.long,
    device=device
)

vocab_size = tokenizer.get_vocab_size()

# get batch function

def get_batch(split="train"):

    data = train_data if split == "train" else test_data

    ix = torch.randint(
        len(data) - block_size,
        (batch_size,)
    )

    x = torch.stack([
        data[i:i + block_size]
        for i in ix
    ])

    y = torch.stack([
        data[i + 1:i + block_size + 1]
        for i in ix
    ])

    return x.to(device), y.to(device)

# Loss Estimate function

@torch.no_grad()
def estimate_loss(model):
    out = {}
    losses = torch.zeros(50)
    model.eval()
    for split in ["train", "test"]:

        for i in range(50):

            Xb, Yb = get_batch(split)

            logits = model(Xb)
            loss = F.cross_entropy(logits.transpose(1, 2),Yb)

            losses[i] = loss.item()

        out[split] = losses.mean()

    model.train()

    return out

# Model Initialization

model = LM(
    vocab_size=vocab_size,
    n_embd=n_embd,
    block_size=block_size,
    device=device
).to(device)

optimizer = torch.optim.AdamW(model.parameters(),lr=1e-3)

# parameter count
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total learnable parameters: {total_params:,}")

# training loop

def main():
    times = torch.zeros(steps)

    test_losses = []
    overfitting_counter =0
    for s in range(steps):
        start = time.time()
        Xb,Yb = get_batch()
        logits = model(Xb)
        loss = F.cross_entropy(logits.transpose(1, 2),Yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        end = time.time()
        total = end - start
        times[s] = total
        if s % 10 == 0:
            mean_time = times[:s].mean()
            estimated = mean_time * (steps - s)
            e_l= estimate_loss(model)
            print(f"{s} - train loss: {e_l["train"].item()} test loss: {e_l["test"].item()} ---estimated time:({estimated})")
            test_losses.append(e_l["test"])

            if len(test_losses) >= 2 and test_losses[s//10] > test_losses[(s//10) - 1]:
                overfitting_counter += 1
            if overfitting_counter >= 3:
                print("overfitting break!")
                break
            else:
                overfitting_counter = 0
        if s % 100 == 0:
            torch.save({
                "model_state_dict":model.state_dict(),
                "optimizer_state_dict":optimizer.state_dict(),
                "loss":loss,
                "step":s
            },checkPointPath.format(id=file_id,step=s))



    torch.save(model.state_dict(),f"model_{file_id}.pth")

if __name__ == "__main__":
    main()