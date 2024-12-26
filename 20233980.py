import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision.models as models  # Import SqueezeNet
from tqdm import tqdm
import numpy as np
import os
import random

def set_seed(seed=0):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False

# Set the seed at the very beginning
set_seed(0)


# Device configuration (use GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
batch_size = 64
learning_rate = 0.001
num_epochs = 20
num_class = 10

# Transformation (normalize Fashion-MNIST images and resize to 224x224)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224 for DenseNet
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize the grayscale image
])

# Download and load training data
train_dataset = datasets.FashionMNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.FashionMNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Define CNN model
class Model(nn.Module):
    def __init__(self, num_classes):
        super(Model, self).__init__()
        
        # Load pre-trained DenseNet
        self.densenet = models.densenet121(pretrained=True)
        
        # Modify first conv layer to accept grayscale images
        self.densenet.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2,
                                                padding=3, bias=False)
        
        # Freeze feature extraction layers if specified
        # if freeze_features:
        #     for param in self.densenet.features.parameters():
        #         param.requires_grad = False
        
        # Get the number of features from the last layer
        num_features = self.densenet.classifier.in_features
        
        # Replace classifier with custom 2-layer MLP
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        # Replace DenseNet classifier
        self.densenet.classifier = nn.Identity()
    
    def forward(self, x):
        #print('x shape:',x.shape) #torch.Size([64, 1, 224, 224])
        features = self.densenet(x)
        #print('features shape:',features.shape) # torch.Size([64, 1024])
        output = self.classifier(features)
        #print('output shape:',output.shape) #torch.Size([64, 10])
        return output 

# Initialize the model, loss function, and optimizer
model = Model(num_classes=num_class).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# print("Params to learn:")
# for name,param in model.named_parameters():
#     if param.requires_grad == True:
#         print("\t",name)
# exit(0)

def train(model, train_loader, criterion, optimizer, num_epochs, device): #save_path for model state save
    
    
    #best_val_loss = float('inf')
    best_val_accuracy = 0.0
    best_model_state = None
    

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0
        total_batches = len(train_loader)

        # Initialize the tqdm progress bar
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', total=total_batches)
        for img, labels in progress_bar: 
            # print('img shape:',img.shape) 
            # print('labels shape:',labels.shape) 
            img =img.to(device)           
                
            out = model(img) 
            #print ('out from model shape:',out.shape) #torch.Size([64, 10])
            
            
            labels=labels.to(device)
            #print('LABELS:',labels)
            
            # Compute loss
            loss =  criterion(out, labels) 

            _, preds = torch.max(out, 1)
            #print('preds::',preds)
            correct_preds += torch.sum(preds == labels)
            total_preds += labels.size(0)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

                       
            running_loss += loss.item()* img.size(0)  
        epoch_loss = running_loss / len(train_loader.dataset) 
        train_accuracy = correct_preds / total_preds   
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, accuracy: {train_accuracy:.2f}")                     
            
      
        
        if train_accuracy > best_val_accuracy:
            best_val_accuracy = train_accuracy
            
            #print('save logits len',len(save_logits))
            best_model_state = model.state_dict()
            torch.save(best_model_state, './best_model_val_accuracy.pth')
            print(f'Saved model with highest validation accuracy at epoch {epoch + 1}')

    return best_val_accuracy

model_path_existing = 'best_model_val_accuracy.pth'
if os.path.exists(model_path_existing):    
    model.load_state_dict(torch.load(model_path_existing)) # Load the model state_dict
    print("Loaded pre-trained model from", model_path_existing)
else:
    print("No pre-trained model found. Starting training from scratch.")


train_accuracy = train(model, train_loader, criterion, optimizer, num_epochs, device) 



#Function to evaluate the model
def evaluate(model, data_loader, device):
    model.eval()
    
    total_loss = 0.0
    correct = 0
    total = 0
    all_outputs = []
    criterion = nn.CrossEntropyLoss()
    
    
    with torch.no_grad():
        #with open('output_probabilities.txt', 'w') as f:  # Open a text file to write the probabilities
        for img, labels in data_loader:
            
            img =img.to(device)
            labels=labels.to(device)
                        
            out = model(img)
            # Store outputs
            all_outputs.append(out.cpu())
            # Compute loss
            loss = criterion(out, labels)
            total_loss += loss.item() * img.size(0)

            # Compute accuracy
            _, predicted = torch.max(out, 1)
            #print('predicted:',predicted)
            total += labels.size(0)
            #print('total:',total)
            correct += (predicted == labels).sum().item()
            #print('correct:',correct)
       
    
    accuracy = correct / total 
    save_logits = all_outputs    

        
    return save_logits, accuracy

#load the best model
model_weights_path_test = 'best_model_val_accuracy.pth'
model.load_state_dict(torch.load(model_weights_path_test))

save_logits, test_accuracy = evaluate(model, test_loader, device)
print(f'Accuracy on test dataset: {test_accuracy:.2f}')

# Concatenate all outputs
all_best_logits = torch.cat(save_logits, dim=0)
#print('all_outputs appended shape:',all_best_logits.shape)
all_best_logits=all_best_logits.cpu().numpy()

# Save logits if accuracy threshold is met
accuracy = 100*test_accuracy
#print('test accuracy: ',accuracy)
if accuracy > 92.15:
    np.save('20233980.npy', all_best_logits)
    print(f'\nAccuracy {accuracy:.2f}% exceeds threshold 92.15%')
    #print(f'Logits shape: {all_best_logits.shape}')
    print('20233980.npy file saved')
else:
    print('\nAccuracy threshold of 92.15% not met. Logits not saved.')
