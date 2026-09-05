import torch
import torch.nn.functional as F
from train import block_size, tokenizer,model

checkPointPath = "./checkpoints/CLLM-I{id}-S{step}.pth"

checkpoint = torch.load(checkPointPath.format(id="0",step="4900"),map_location = torch.device("cpu"))
model.load_state_dict(checkpoint["model_state_dict"])

def generate(prompt:str):

    model.eval()
    prompt = "<user> " + prompt + "<assistant> "

    idx = torch.tensor(
        tokenizer.encode(prompt).ids,
        dtype=torch.long
    ).unsqueeze(0)


    with torch.no_grad():
        for _ in range(300):

            idx_cond = idx[:, -block_size:]
            logits = model(idx_cond)

            logits = logits[:, -1, :]

            probs = F.softmax(logits, dim=-1)

            next_token = torch.multinomial(
                probs,
                num_samples=1
            )
            idx = torch.cat(
                (idx, next_token),
                dim=1
            )
            if next_token in (1,2):
                break
    result = tokenizer.decode(idx[0].tolist(),skip_special_tokens=False)

    print(result)

generate("Hello, how are you?")