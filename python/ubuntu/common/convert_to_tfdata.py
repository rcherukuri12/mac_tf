# use this util to convert 
# datasets into tensorflow data format
# expect to use hugging face tokenizer
import tensorflow as tf
def convert_to_tfdata(tokenizer,train_texts,val_texts,train_labels,val_labels):
  train_encodings = tokenizer(train_texts,truncation=True,padding=True)
  val_encodings   = tokenizer(val_texts,truncation=True,padding=True)
  train_dataset = tf.data.Dataset.from_tensor_slices((
			dict(train_encodings),
			train_labels
		))
  val_dataset = tf.data.Dataset.from_tensor_slices((
			dict(val_encodings),
			val_labels
		))
  return train_dataset,val_dataset


