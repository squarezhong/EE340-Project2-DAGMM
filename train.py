
import time

from data import get_loader
from model import DAGMM
    
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from util import get_classification_report, plot_confusion_matrix, plot_loss_curve, init_weights


class Trainer():
    def __init__(self, hyp):
        # using cuda if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = DAGMM(hyp).to(self.device)
        self.train_loader, self.test_loader = get_loader(hyp)
        # AdamW Optimizer
        self.optimizer = optim.AdamW(self.model.parameters(), hyp['lr'], amsgrad=True)
        # change the learning rate amid learning to get better performance
        self.scheduler = optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=hyp['decay_rate'])

    def train(self, hyp):
        self.model.train()
        self.model.apply(init_weights)
        loss_list, recon_list, energy_list = [], [], []
        for epoch in range(hyp['epochs']):
            #loss_total = 0
            for data, target in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                zc, xp, z, gamma = self.model(data)
                data, xp, z, gamma = data.cpu(), xp.cpu(), z.cpu(), gamma.cpu()
                loss, reconstruct_error, sample_energy, P = self.model.criterion(data, xp, gamma, z)
                #loss_total += loss.item()
                loss.backward()
                self.optimizer.step()
                
            self.scheduler.step()
            

        #plot_loss_curve(loss_list)
        torch.save(self.model.state_dict(), 'results/dagmm.pth')
        return loss_list, recon_list, energy_list
    
    def load_model(self, hyp):
        model = DAGMM(hyp)
        try:
            model.load_state_dict(torch.load('results/dagmm.pth'))
            print('Successfully loaded model')
        except IOError as e:
            print('An IOError occurred. {}'.format(e.args[-1]))

        return model
    
    def compute_threshold(self, model):
        energy_list = []
        with torch.no_grad():
            for data, labels in self.train_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                zc,xp,z,gamma = model(data)
                _, _, _, _, E = model.sample_energy(gamma, z)
                energy_list.extend(E.detach().cpu().tolist())
        
        threshold = np.percentile(energy_list, 80.31)
        print('threshold: %.4f' %(threshold))
        return threshold

    def test(self, hyp):
        model = self.load_model(hyp)
        model.eval()

        if hyp['is_threshold'] == True:
            threshold = hyp['threshold']
        else:
            threshold = self.compute_threshold(model)

        true_list, pred_list = [], []
        index = 1


        with torch.no_grad():
            for x, y in self.test_loader:
                x = x.to(self.device)
                zc,xp,z,gamma = model(x)
                _, _, _, _, E = model.sample_energy(gamma, z)
                true_list.extend(y.cpu().tolist())
                pred_list.extend((E > threshold).cpu().tolist())

                # plot 3D scatter plot to show the difference between normal and abnormal samples
                if index <= 4:
                    fig = plt.figure(figsize = (10, 7))  
                    ax = plt.axes(projection ="3d") 
                    normal_list, abnormal_list = [], []

                    z = z.detach().cpu().tolist()
                    for i in range(hyp['batch_size']*10):
                        if y[i] == 1:
                            normal_list.append(z[i]) 
                        else:
                            abnormal_list.append(z[i])
                    print(normal_list)

                    normal_list, abnormal_list = np.array(normal_list), np.array(abnormal_list)
                    ax.scatter3D(normal_list[:,0], normal_list[:,1], normal_list[:,2], color='green', marker='o', label='normal')
                    ax.scatter3D(abnormal_list[:,0], abnormal_list[:,1], abnormal_list[:,2], color='red', marker='x', label='abnormal')

                    ax.set_xlabel('z1')
                    ax.set_ylabel('z2')
                    ax.set_zlabel('z3')
                    ax.legend()
                    
                    plt.title("3D scatter plot") 
                    plt.savefig('results/3d_scatter_plot_{}.png'.format(str(index)), dpi=600) 

                    index += 1
          
            
        accuracy = accuracy_score(true_list, pred_list) * 100
        print('Accuracy: {:.2f}%'.format(accuracy))
        # get and plot confusion matrix
        cm = confusion_matrix(true_list, pred_list)
        plot_confusion_matrix(cm, 'test')
        get_classification_report(true_list, pred_list)

        return accuracy
