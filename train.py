import torch
import torch.nn as nn
import torch.optim as optim

def train(net,
          trainloader,
          num_epochs=10,
          learn_rate=0.001,
          report_freq=5):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'using device: {device}')
    net.to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learn_rate)

    for epoch in range(num_epochs):
        print(f'epoch ... {epoch+1}/{num_epochs}')
        running_loss = 0.0
        for i, batch in enumerate(trainloader):
            inputs, labels = batch

            inputs.to(device)
            labels.to(device)

            optimizer.zero_grad()

            outputs = net(inputs)

            loss = loss_function(outputs, labels)
            running_loss += loss
            loss.backward()

            optimizer.step()

            if (i % report_freq == 0 and i != 0) or report_freq == 1:
                print(f'batch: {i+1} / {len(trainloader)}')
                print(f'loss: {running_loss / report_freq:.3f}')
                running_loss=0.0