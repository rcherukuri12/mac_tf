import pandas as pd
def data_reader(path):
    df = pd.read_csv(path)
    # basically add another column to encode category names as numbers
    df['encoded_cat']=df['category'].astype('category').cat.codes
    data_texts = df["text"].to_list() # features
    data_labels= df["encoded_cat"].to_list()

    from sklearn.model_selection import train_test_split

    train_texts,val_texts,train_labels,val_labels = train_test_split(data_texts,data_labels,test_size=0.2,random_state=0)

    val_texts,test_texts,val_labels,test_labels = train_test_split(val_texts,val_labels,test_size=0.5,random_state=0)

    return train_texts,val_texts,test_texts,train_labels,val_labels,test_labels