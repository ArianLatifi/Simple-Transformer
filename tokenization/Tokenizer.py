from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

from tokenizers.pre_tokenizers import Whitespace

tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
tokenizer.pre_tokenizer = Whitespace()

trainer = BpeTrainer(special_tokens=["<UNK>", "<user>", "<assistant>"])

tokenizer.train([r"./dataset/dailyDialogCleand.txt"],trainer)
tokenizer.save("./dialogtokenizer.json")

tokenizer = Tokenizer.from_file("./dialogtokenizer.json")
output = tokenizer.encode("<user> hello  world this is arian",)

# print(tokenizer.get_vocab_size())
print(tokenizer.decode(output.ids))
print(output.ids)