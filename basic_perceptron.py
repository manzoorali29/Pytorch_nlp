import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class Perceptron(nn.Module):
    def __init__(self, input_features):
        super(Perceptron, self).__init__()
        self.fc1 = nn.Linear(input_features, 1)
        self.prelu = nn.PReLU()
        self.softmax = nn.Softmax()

    def forward(self, inp):
        r = self.fc1(inp)
        
        return [torch.sigmoid(r).squeeze(), torch.tanh(r).squeeze(), torch.relu(r).squeeze(),self.prelu(r).squeeze(),self.softmax(r).squeeze()]

model = Perceptron(1)  # Adjust input_features to match the dimensions

x = torch.arange(-5., 5., 0.1)

y = model.forward(x.unsqueeze(1))

plt.plot(x.numpy(), y[0].detach().numpy())
plt.plot(x.numpy(),y[1].detach().numpy())
plt.plot(x.numpy(),y[2].detach().numpy())
plt.plot(x.numpy(),y[3].detach().numpy())
plt.plot(x.numpy(),y[4].detach().numpy())
#plt.plot(x.numpy(),y[4].detach().numpy())
plt.xlabel('x')
plt.ylabel('Range')
plt.legend(['Sigmoid','Tanh','Relu','prelu','Softmax'])
plt.title('Activation Functions')
plt.grid(True)
plt.show()





