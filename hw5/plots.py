import matplotlib.pyplot as plt
import itertools
import numpy as np
from sklearn.metrics import confusion_matrix

# TODO: You can use other packages if you want, e.g., Numpy, Scikit-learn, etc.


def plot_learning_curves(train_losses, valid_losses, train_accuracies, valid_accuracies):
    fig, axes = plt.subplots(1, 2)
    axes[0].plot(train_losses, label='training loss')
    axes[0].plot(valid_losses, label='validation loss')
    axes[0].legend(loc="upper center")
    axes[0].set_title('Loss Curve')
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")


    axes[1].plot(train_accuracies, label='training accuracy')
    axes[1].plot(valid_accuracies, label='validation accuracy')
    axes[1].legend(loc="upper center")
    axes[1].set_title('Accuracy Curve')
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    fig.savefig('MLP_1_LC.png')


def plot_confusion_matrix(results, class_names):
    y_true, y_pred = zip(*results)
    matrix = confusion_matrix(y_true, y_pred)
    label_matrix = matrix.astype('float')/ np.sum(matrix, axis =1)
    
    plt.figure()
    cmap = plt.get_cmap('Blues')
    plt.imshow(label_matrix , interpolation='nearest', cmap=cmap)
    plt.title('Normalized Confusion Matrix')
    plt.colorbar()
    
    numCats = len(class_names)
    ticks = np.arange(numCats)
    plt.xticks(ticks, class_names, rotation=45)
    plt.yticks(ticks, class_names)

    plt.ylabel('True')
    plt.xlabel('Predicted')
    thresh = label_matrix.max() / 2.
    for i, j in itertools.product(range(numCats), range(numCats)):
        plt.text(j, i, format(label_matrix[i, j], '.2f'),
                 horizontalalignment="center")
    plt.tight_layout()
    plt.savefig('MLP_1_CM.png')
