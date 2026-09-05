import torch
import torch.nn.functional as F
from train import block_size, tokenizer,model
import re

checkPointPath = "./checkpoints/CLLM-I{id}-S{step}.pth"

checkpoint = torch.load(checkPointPath.format(id="0",step="4900"),map_location = torch.device("cpu"))
model.load_state_dict(checkpoint["model_state_dict"])

def generate(prompt:str):

    model.eval()

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

    matches = re.findall(r"<assistant>(.*?)<user>", result, re.DOTALL)

    response = matches[-1].strip()
    
    return result,response

prompt ="<user> " +  input("User: ")+ "<assistant> "

while True:
    coversation , response = generate(prompt)
    print("Assistant:", response)
    prompt = coversation +  input("User: ") + "<assistant> "
    if re.findall(r"\\exit", prompt, re.IGNORECASE):
        break

print("Full Conversation:", coversation)
