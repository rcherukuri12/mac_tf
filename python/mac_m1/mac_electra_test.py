import tensorflow
#from tensorflow.python.framework.ops import disable_eager_execution
#disable_eager_execution()
from tensorflow.python.compiler.mlcompute import mlcompute
mlcompute.set_mlc_device(device_name='any')# 'cpu' or 'gpu' or 'any'

from common.read_text import *
path = "../../../data/bbc-text.csv"
train_texts,val_texts,test_texts,train_labels,val_labels,test_labels = data_reader(path)

from transformers import ElectraTokenizer

from transformers import TFElectraForSequenceClassification

tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator')

from common.convert_to_tfdata import *

train_dataset,val_dataset = convert_to_tfdata(tokenizer,train_texts,val_texts,train_labels,val_labels)

#get the correct pre-trained weights
model = TFElectraForSequenceClassification.from_pretrained('google/electra-small-discriminator',num_labels=5)

from common.fine_tune_and_eval import *

pred_labels = fine_tune_and_eval(model,tokenizer,train_dataset,val_dataset,test_texts)
preds       = pred_labels.tolist()
#metrics
from common.metrics import *
names_labels=["money","celeb","news","sport","tech"]
show_cm(test_labels,preds,names_labels)

