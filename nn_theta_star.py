import torch.nn as nn 
import torch
from utils import *
import torch.optim as optim
import time

class ThetaStarNN(nn.Module):
    def __init__(self, d, k, M,lambda_regression= 0.01 ):
        super(ThetaStarNN, self).__init__()
        self.d = d 
        self.k = k
        self.M = M
        self.lambda_regression = lambda_regression
        self.W = nn.ParameterList([nn.Parameter(torch.randn(k)) for _ in range(M)])
        self.w_target = nn.Parameter(torch.randn(k))
        self.lambda_regression = lambda_regression
        self.linear = nn.Linear(d, k)
        self.activation = nn.ReLU()
    
    def forward(self, x, weight):
        x = self.linear(x)
        out = torch.mv(x, weight)
        return out

    def source_loss(self, X, Y):
        loss = 0
        
        for m in range(self.M):
            y_predict = self.forward(X[m], self.W[m])
            mse_loss = torch.mean((Y[m] - y_predict) ** 2)
            regression_loss = self.lambda_regression * 0.5 * (self.linear.weight.norm()**2+ self.W[m].norm() ** 2)
            loss += mse_loss + regression_loss

        return loss / self.M
    
    def target_loss(self, X_target, Y_target):
        
        y_predict = self.forward(X_target, self.w_target)
        mse_loss = torch.mean((Y_target - y_predict) ** 2)
        
        return mse_loss
        
    def target_task(self, X_target, Y_target, epochs = 100, lr = 0.1):
        
        optimizer = optim.SGD([self.w_target], lr = lr)
        
        for epoch in range(epochs):
            
            optimizer.zero_grad()
            loss = self.target_loss(X_target, Y_target)
            loss.backward()
            optimizer.step()
            
            if epoch % 10 == 0:
                
                print(f"Epoch {epoch}, Loss: {loss.item()}")


