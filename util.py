import torch
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix


# plot loss curve for a training progress
def plot_loss_curve(loss_list) -> None:
    # plot the trend of loss
    plt.plot(loss_list)
    plt.title('Loss versus Epochs for DAGMM')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.grid(True)
    # save the figures
    plt.savefig('results/loss_figure.png', dpi=600)
    plt.show()

def plot_confusion_matrix(cm, state):
    # use confusion matrix to show the prediction result directly
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='.0f')
    plt.xlabel('predictions')
    plt.ylabel('labels')
    plt.title('Confusion Matrix for {}'.format(state))
    # automatically save the figures
    plt.savefig('results/martix.png', dpi=600)
    plt.show()

def get_classification_report(true, pred):
    report = classification_report(true, pred, output_dict=True)
    # print(report)
    tmp = pd.DataFrame(report).transpose()
    tmp.to_csv('results/report.csv', index=True)

    return report
    
def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)
