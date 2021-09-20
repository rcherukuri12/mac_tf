import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import confusion_matrix
def show_cm(test_labels,preds,names_labels):
    data = confusion_matrix(test_labels,preds)
    df_cm = pd.DataFrame(data,columns=np.unique(test_labels),index = np.unique(test_labels))
    df_cm.index.name = "Actual"
    df_cm.columns.name = "Predicted"
    plt.figure(figsize = (10,7))
    sn.set(font_scale=1.4)# for label size
    sn.heatmap(df_cm,cmap="Blues",annot=True,annot_kws={"size":12},
            xticklabels=names_labels,yticklabels=names_labels)
    plt.savefig('cm.png')
