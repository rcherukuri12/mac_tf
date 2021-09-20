import pandas as pd
import tensorflow as tf


def fine_tune_and_eval(model,tokenizer,train_dataset,val_dataset,test_texts,epochs=1):
  optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
  model.compile(optimizer=optimizer,loss=model.compute_loss,metrics=['accuracy'])
  
  model.fit(train_dataset.shuffle(1000).batch(16),epochs=epochs,batch_size=16,
            validation_data=val_dataset.shuffle(1000).batch(16))
  
  # now let us encode test datasets
  test_encodings = tokenizer(test_texts,truncation=True,padding=True,return_tensors="tf")
  output = model(test_encodings)[0]
  prediction_labels = tf.argmax(output,axis=1).numpy()

  return prediction_labels


