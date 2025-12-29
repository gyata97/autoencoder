import torch
import torch.nn as nn
import torch.optim as optim
import settings

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(settings.IMAGE_FLATTENED_SIZE, settings.AE_ENCODER_LAYERS[0]),
            nn.ReLU(),
            nn.Linear(settings.AE_ENCODER_LAYERS[0], settings.AE_ENCODER_LAYERS[1]),
            nn.ReLU(),
            nn.Linear(settings.AE_ENCODER_LAYERS[1], settings.AE_ENCODER_LAYERS[2]),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(settings.AE_ENCODER_LAYERS[2], settings.AE_DECODER_LAYERS[0]),
            nn.ReLU(),
            nn.Linear(settings.AE_DECODER_LAYERS[0], settings.AE_DECODER_LAYERS[1]),
            nn.ReLU(),
            nn.Linear(settings.AE_DECODER_LAYERS[1], settings.AE_DECODER_LAYERS[2]),
            nn.Sigmoid(),
        )
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, x):
        return self.decoder(x)

                

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

model = Autoencoder()

optimizer = optim.Adam(model.parameters(), lr=settings.LEARNING_RATE)

for epoch in range(settings.NUM_EPOCHS):
    for i, (x, _) in enumerate(train_loader):
        x = x.view(-1, settings.IMAGE_FLATTENED_SIZE)
        optimizer.zero_grad()
        recon_batch = model(x)
        loss = F.binary_cross_entropy(recon_batch, x)
        loss.backward()
        optimizer.step()
        if i % settings.PRINT_INTERVAL == 0:
            print(f'Epoch [{epoch+1}/{settings.NUM_EPOCHS}], Step [{i+1}/600], Loss: {loss.item():.4f}')






