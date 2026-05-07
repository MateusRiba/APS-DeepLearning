import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    # num_canais=1 pro mnist (pb) e 3 pro cifar (colorido)
    def __init__(self, num_canais=1, num_classes=10):
        super(CNN, self).__init__()
        
        # primeira camada pegando a imagem
        self.conv1 = nn.Conv2d(in_channels=num_canais, out_channels=16, kernel_size=3, padding=1)
        # batchnorm ajuda a rede a não se perder nos pesos no começo
        self.bn1 = nn.BatchNorm2d(16)
        
        # segunda camada engrossando os filtros
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        # pooling basico pra reduzir o tamanho da imagem pela metade
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # dps de dois poolings, uma imagem 28x28 vira 7x7. 
        # (Vamos ter q dar resize na imagem lá no notebook pra 28x28 pra não quebrar aqui)
        self.fc1 = nn.Linear(32 * 7 * 7, 128) 
        
        # regularização exigida na atividade (desliga 50% dos neuronios pra evitar overfit)
        self.dropout = nn.Dropout(0.5) 
        
        # saida final com as 10 classes (serve pros 2 datasets)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # passa na conv1, ativa com relu, normaliza e diminui o tamanho
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # mesma coisa na conv2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # achata o tensor pra virar um vetorzão pra camada linear
        x = torch.flatten(x, 1)
        
        # passa nas camadas finais
        x = F.relu(self.fc1(x))
        x = self.dropout(x) 
        x = self.fc2(x)
        
        return x