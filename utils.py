import numpy as np
import random
from sklearn.metrics import confusion_matrix


def label_to_color(arr , palette):
    arr_3 = np.zeros((arr.shape[0] ,arr.shape[1],3) , dtype=np.uint8)
    for key , l in palette.items():
        m = arr == key
        arr_3[m] = l
    return arr_3

def color_to_label(arr_3d , palette):
    arr = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)
    for key , l in palette.items():
        m = np.all(arr_3d == np.array(key).reshape(1,1,3),axis=2)
        arr[m] = l
    return arr


def accuracy(input, target):
    return 100 * float(np.count_nonzero(input == target)) / target.size

def metrics(predictions, gts, label_values):
    cm = confusion_matrix(
            gts,
            predictions,
            range(len(label_values)))
    
    print("Confusion matrix :")
    print(cm)
    
    print("---")
    
    # Compute global accuracy
    total = sum(sum(cm))
    accuracy = sum([cm[x][x] for x in range(len(cm))])
    accuracy *= 100 / float(total)
    print("{} pixels processed".format(total))
    print("Total accuracy : {}%".format(accuracy))
    
    print("---")
    
    # Compute F1 score
    F1Score = np.zeros(len(label_values))
    for i in range(len(label_values)):
        try:
            F1Score[i] = 2. * cm[i,i] / (np.sum(cm[i,:]) + np.sum(cm[:,i]))
        except:
            # Ignore exception if there is no element in class i for test set
            pass
    print("F1Score :")
    for l_id, score in enumerate(F1Score):
        print("{}: {}".format(label_values[l_id], score))


    return accuracy
