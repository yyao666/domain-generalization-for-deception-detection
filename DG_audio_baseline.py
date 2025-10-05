import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
torch.manual_seed(1111)
import torchvision


#############################################################################################################################
#############################################################################################################################

class audio_model(nn.Module):
    def __init__(self, ):
        super(audio_model, self).__init__()

        # audio tower
        r50 = torchvision.models.resnet50(weights="ResNet50_Weights.IMAGENET1K_V1")    # introduce the ResNet architecture from PyTorch
        r50.conv1 = nn.Conv2d(1, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)
        r50_num_in_feat = r50.fc.in_features
        r50.fc = nn.Identity()
        self.audio_backbone = r50

        # linear classifier
        self.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Dropout(0.1),
        nn.Linear(r50_num_in_feat, 2)
        )

    def forward(self, x):
        x = x.float()
        audio_preds = self.audio_backbone(x.unsqueeze(1))
        return self.classifier(audio_preds)

#############################################################################################################################
############################################################################################################################# 
 
class Spec_Dataset(Dataset):
    def __init__(self, annotations_file, spec_dir, domain):

        # domian is a python list; domain = ["CHINESE", "MALAY", "HINDI"]; choose accordingly!
        annotations = pd.read_csv(annotations_file) # load protocols/all_samples.csv
        self.spec_dir = spec_dir

        self.annos = []
        for i in range(annotations.shape[0]):

            mono_or_interro = annotations.iloc[i,4] # mono, monologue, interrogation
            ethnicity = annotations.iloc[i,1].split("_")[0] # EA, SEA, SA
            language = annotations.iloc[i,-1] # chinese, english, English etc. CAUTION !! - language names are case sensitive

            # if language in ["English","english"]:
            #     if ethnicity == "EA" and "CHINESE" in domain: self.annos.append(annotations.iloc[i])
            #     if ethnicity == "SEA" and "MALAY" in domain: self.annos.append(annotations.iloc[i])
            #     if ethnicity == "SA" and "HINDI" in domain: self.annos.append(annotations.iloc[i])
            # else:
            #     continue # ignore native languages

            if language in ["Chinese","chinese","Malay","malay","Hindi","hindi"]:
                if ethnicity == "EA" and "CHINESE" in domain: self.annos.append(annotations.iloc[i])
                if ethnicity == "SEA" and "MALAY" in domain: self.annos.append(annotations.iloc[i])
                if ethnicity == "SA" and "HINDI" in domain: self.annos.append(annotations.iloc[i])
            else:
                continue # ignore english language

    def __len__(self):
        return len(self.annos)

    def __getitem__(self, idx):
        
        # spectrograms
        spectrogram = torch.load(self.spec_dir + self.annos[idx][0] + ".pth")

        # labels
        gt = self.annos[idx][5]
        if gt == 'T':
           label = 0
        elif gt in ['F','L']:
           label = 1
        label = torch.tensor(label)

        return spectrogram, label

def af_pad_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    return batch.permute(0, 2, 1)

def af_collate_fn(batch):
    spec_tensors, targets = [], []

    # Gather in lists, and encode labels as indices
    for spec, label in batch:
        spec_tensors += [spec]
        targets += [label]

    # Group the list of tensors into a batched tensor
    spec_tensors = af_pad_sequence(spec_tensors)
    targets = torch.stack(targets)

    return spec_tensors, targets

#############################################################################################################################
#############################################################################################################################

def train_one_epoch(train_data_loader,model,optimizer,loss_fn):
    epoch_loss = []
    sum_correct_pred = 0
    total_samples = 0

    model.train()
    for i,(spec,labels) in enumerate(train_data_loader):

        spec = spec.to(device)
        labels = labels.to(device) 

        #Reseting Gradients
        optimizer.zero_grad()
        #Forward
        preds = model(spec)
        _loss = loss_fn(preds, labels)
        loss = _loss.item()
        epoch_loss.append(loss)      
        #Backward
        _loss.backward()
        optimizer.step()

        # calculate acc per minibatch
        sum_correct_pred += (torch.argmax(preds, dim=-1) == labels).sum().item()
        total_samples += len(labels)

        # if i%50 == 0: print("\t loss = ", np.mean(epoch_loss))

    epoch_loss = np.mean(epoch_loss)
    acc = round(sum_correct_pred/total_samples,4)*100
    return epoch_loss, acc

def val_one_epoch(val_data_loader, model,loss_fn):

    epoch_loss = []
    sum_correct_pred = 0
    total_samples = 0

    model.eval()
    with torch.no_grad():
      for spec,labels in val_data_loader:       
        spec = spec.to(device)
        labels = labels.to(device)

        preds = model(spec)
        _loss = loss_fn(preds, labels)
        epoch_loss.append(_loss.item())
        
        # calculate acc per minibatch
        sum_correct_pred += (torch.argmax(preds, dim=-1) == labels).sum().item()
        total_samples += len(labels)
    
    epoch_loss = np.mean(epoch_loss)
    acc = round(sum_correct_pred/total_samples,4)*100
    return epoch_loss, acc

#############################################################################################################################
#############################################################################################################################

device = torch.device("cuda:0")
batch_size = 16
num_epochs = 10
learning_rate = 3e-4

protocols = [
    [["CHINESE", "MALAY"],["HINDI"]],
    [["CHINESE", "HINDI"],["MALAY"]],
    [["MALAY", "HINDI"],["CHINESE"]]
    ]

for tr,te in protocols: 

    print("\n Train domain = ",tr)
    print("\n Test domain = ",te,"\n")

    train_dataset = Spec_Dataset("/home/DSO_SSD/ROSE_V2/scripts/protocols/all_samples.csv", "/home/DSO_SSD/ROSE_V2/hr_spectrograms/",domain=tr)
    test_dataset = Spec_Dataset("/home/DSO_SSD/ROSE_V2/scripts/protocols/all_samples.csv", "/home/DSO_SSD/ROSE_V2/hr_spectrograms/",domain=te)
    # print("\t",len(train_dataset),len(test_dataset))

    train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True, num_workers=16, collate_fn=af_collate_fn)
    test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False, num_workers=16, collate_fn=af_collate_fn)


    model = audio_model()
    model = nn.DataParallel(model, device_ids=[0,1,2,3,4,5,6,7])
    model.to(device)
    print("\t Model Loaded")
    print("\t Trainable Params = ",sum(p.numel() for p in model.parameters() if p.requires_grad))

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    #Loss Function
    loss_fn = nn.CrossEntropyLoss()

    best_val_acc = 0.0

    for epoch in range(num_epochs):
             
        ###Training
        train_loss, train_acc = train_one_epoch(train_loader,model,optimizer,loss_fn)
        ###Validation
        val_loss, val_acc = val_one_epoch(test_loader,model,loss_fn)

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        print('\n\tEpoch...........',epoch+1) 
        print("\tTrain loss = {} \t Train accuracy = {}".format(train_loss, train_acc ))
        print("\tVal loss = {} \t Val accuracy = {}".format(val_loss, val_acc ))

    print("\nBest Accuracy........", best_val_acc)