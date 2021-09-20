import pandas as pd
import tensorflow as tf


def fine_tune_and_eval(model,tokenizer,train_dataset,val_dataset,test_texts,test_labels):
  optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
  model.compile(optimizer=optimizer,loss=model.compute_loss,metrics=['accuracy'])
  
  model.fit(train_dataset.shuffle(1000).batch(16),epochs=1,batch_size=16,validation_data=val_dataset.shuffle(1000).batch(16))



