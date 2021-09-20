from common.read_text import *

from transformers import ElectraTokenizer

from transformers import TFElectraForSequenceClassification

tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator')

from common.convert_to_tfdata import *

train_dataset,val_dataset = convert_to_tfdata(tokenizer,train_texts,val_texts,train_labels,val_labels)

#get the correct pre-trained weights
model = TFElectraForSequenceClassification.from_pretrained('google/electra-small-discriminator',num_labels=5)

from common.fine_tune_and_eval import *

fine_tune_and_eval(model,tokenizer,train_dataset,val_dataset,test_texts,test_labels)