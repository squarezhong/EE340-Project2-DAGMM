import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class Perceptron(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Perceptron, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.layers(x)

class DAGMM(nn.Module):
    def __init__(self, hyp) -> None:
        super(DAGMM, self).__init__()
        
        self.encoder = nn.Sequential(
            Perceptron(hyp['input_dim'], hyp['hidden_dim1']),
            Perceptron(hyp['hidden_dim1'], hyp['hidden_dim2']),
            #Perceptron(hyp['hidden_dim2'], hyp['hidden_dim3']),
            #nn.Linear(hyp['hidden_dim3'], hyp['zc_dim']),
            nn.Linear(hyp['hidden_dim2'], hyp['zc_dim'])
        )

        self.decoder = nn.Sequential(
            #Perceptron(hyp['zc_dim'], hyp['hidden_dim3']),
            #Perceptron(hyp['hidden_dim3'], hyp['hidden_dim2']),
            Perceptron(hyp['zc_dim'], hyp['hidden_dim2']),
            Perceptron(hyp['hidden_dim2'], hyp['hidden_dim1']),
            nn.Linear(hyp['hidden_dim1'], hyp['input_dim'])
        )

        self.estimation = nn.Sequential(
            Perceptron(hyp['zc_dim']+2, hyp['hidden_dim1']),
            Perceptron(hyp['hidden_dim1'], hyp['hidden_dim2']),
            nn.Dropout(p=hyp['dropout']),
            nn.Linear(hyp['hidden_dim2'], hyp['n_gmm']),
            nn.Softmax(dim=1)
        )
        
    # magic of broadcasting
    def forward(self, x):
        zc = self.encoder(x) # shape: [bs, zc_dim]
        xp = self.decoder(zc) # xp: x prime or (x') shape: [bs, input_dim]
        
        euclidean_distance = F.pairwise_distance(x, xp) # distances are computed using p-norm
        cosine_similarity = F.cosine_similarity(x, xp)
        zr = torch.cat([euclidean_distance.unsqueeze(-1), cosine_similarity.unsqueeze(-1)], dim=1) # shape: [bs, 2]
        
        z = torch.cat([zc, zr], dim=1) # shape: [bs, zc_dim + 2]

        gamma = self.estimation(z) # shape: [bs, n_gmm]
        
        return zc,xp,z,gamma

    # mixture probability, mean, covariance for component k in GMM, respectively
    def sample_energy(self, gamma, z):
        N, n_gmm = gamma.shape[0], gamma.shape[1] # bs, n_gmm
        phi = torch.sum(gamma, dim=0) / N  # shape: [n_gmm]
        
        mean = torch.sum(gamma.unsqueeze(-1) * z.unsqueeze(1), dim=0) # shape: [n_gmm, z_dim]
        mean = mean / torch.sum(gamma, dim=0).unsqueeze(-1)  # shape: [n_gmm, z_dim]
            
        tmp = z.unsqueeze(1)- mean.unsqueeze(0) # shape: [bs, n_gmm, z_dim]
        cov = torch.sum(gamma.unsqueeze(-1).unsqueeze(-1) * tmp.unsqueeze(-1) * tmp.unsqueeze(-2), dim = 0) # shape: [n_gmm, z_dim, z_dim]
        cov = cov / torch.sum(gamma, dim=0).unsqueeze(-1).unsqueeze(-1) # shape: [n_gmm, z_dim, z_dim]
            
        # [bs, n_gmm, 1, z_dim] * [n_gmm, z_dim, z_dim] * [bs, n_gmm, z_dim, 1]
        # result: [bs, n_gmm, 1, 1] -> [bs, n_gmm]
        # add 1e-12 to avoid singular matrix
        E = torch.exp(-0.5 * tmp.unsqueeze(-2) @ torch.matmul(torch.inverse(cov + 1e-12), tmp.unsqueeze(-1))).squeeze() 
        # print(E.shape)
        E = E * phi / torch.sqrt(torch.abs(torch.det(2 * math.pi * cov))) # shape: [bs, n_gmm]
        
        E = -torch.log(torch.sum(E, dim=1)) # shape: [bs]

        sample_energy = 0.1 * torch.mean(E) # lambda1 = 0.1
        return sample_energy, phi, mean, cov, E


    def criterion(self, x, xp, gamma, z):
        # reconstruction error
        reconstruct_error = torch.mean(torch.sum((x - xp) ** 2, dim=1))
        
        # the probabilities that we could observe the input samples
        sample_energy, _, _, cov, _ = self.sample_energy(gamma, z)
        
        # penalty term
        P = 0.0001 * torch.sum(1 / torch.diagonal(cov, dim1=-2, dim2=-1)) # lambda2 = 0.005

        loss = reconstruct_error + sample_energy  + P
        
        return loss, reconstruct_error, sample_energy, P
    