# audio_DG_GRL method

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
torch.manual_seed(1111)
import torchvision


#############################################################################################################################
#############################################################################################################################

class gradient_reversal_layer(nn.Module):
    def __init__(self, ):
        super(gradient_reversal_layer, self).__init__()

    # donot perform any operation during the forward pass
    # just allow the tensor as it is
    def forward(self, x): 
        return x
    # reverse the gradients during backward pass
    def backward(self, grad_output):
        return -grad_output

class audio_model(nn.Module):
    def __init__(self, ):
        super(audio_model, self).__init__()

        # audio tower
        r50 = torchvision.models.resnet50(weights="ResNet50_Weights.IMAGENET1K_V1")
        r50.conv1 = nn.Conv2d(1, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)
        r50_num_in_feat = r50.fc.in_features
        r50.fc = nn.Identity()
        self.audio_backbone = r50

        # linear classifier  
        self.deception_classifier = nn.Sequential(
        nn.Flatten(),
        nn.Dropout(0.5),
        nn.Linear(r50_num_in_feat, 2)
        )

        self.domain_classifier = nn.Sequential(
        nn.Flatten(),
        gradient_reversal_layer(),
        nn.Linear(r50_num_in_feat, 3)
        )

    def forward(self, x):
        x = x.float()
        audio_preds = self.audio_backbone(x.unsqueeze(1))
        return self.deception_classifier(audio_preds), self.domain_classifier(audio_preds)

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

            if language in ["English","english"]:
                if ethnicity == "EA" and "CHINESE" in domain: self.annos.append(annotations.iloc[i])
                if ethnicity == "SEA" and "MALAY" in domain: self.annos.append(annotations.iloc[i])
                if ethnicity == "SA" and "HINDI" in domain: self.annos.append(annotations.iloc[i])
            else:
                continue # ignore native languages

            # if language in ["Chinese","chinese","Malay","malay","Hindi","hindi"]:
            #     if ethnicity == "EA" and "CHINESE" in domain: self.annos.append(annotations.iloc[i])
            #     if ethnicity == "SEA" and "MALAY" in domain: self.annos.append(annotations.iloc[i])
            #     if ethnicity == "SA" and "HINDI" in domain: self.annos.append(annotations.iloc[i])
            # else:
            #     continue # ignore english language

    def __len__(self):
        return len(self.annos)

    def __getitem__(self, idx):
        
        # spectrograms
        spectrogram = torch.load(self.spec_dir + self.annos[idx][0] + ".pth")

        # deception labels
        gt = self.annos[idx][5]
        if gt == 'T':
           deception_label = 0
        elif gt in ['F','L']:
           deception_label = 1
        deception_label = torch.tensor(deception_label)

        # domian labels
        domains = ["EA", "SEA", "SA"]
        domain_label = torch.tensor(domains.index(self.annos[idx][1].split("_")[0]))

        return spectrogram, deception_label, domain_label

def af_pad_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    return batch.permute(0, 2, 1)

def af_collate_fn(batch):
    spec_tensors, targets1, targets2 = [], [], []

    # Gather in lists, and encode labels as indices
    for spec, deception_label, domain_label in batch:
        spec_tensors += [spec]
        targets1 += [deception_label]
        targets2 += [domain_label]

    # Group the list of tensors into a batched tensor
    spec_tensors = af_pad_sequence(spec_tensors)
    targets1 = torch.stack(targets1)
    targets2 = torch.stack(targets2)

    return spec_tensors, targets1, targets2


#############################################################################################################################
#############################################################################################################################

def train_one_epoch(train_data_loader, model, optimizer, loss1, loss2, alpha):
    deception_loss = []
    domain_loss = []
    sum_correct_pred = 0
    total_samples = 0

    model.train()
    for i,(spec,deception_labels,domain_labels) in enumerate(train_data_loader):

        spec = spec.to(device)
        deception_labels = deception_labels.to(device)
        domain_labels = domain_labels.to(device) 

        #Reseting Gradients
        optimizer.zero_grad()
        #Forward
        deception_preds, domain_preds = model(spec)

        _deception_loss = loss1(deception_preds,deception_labels)
        _domain_loss = loss2(domain_preds,domain_labels)
        _loss = alpha * _deception_loss + (1.0 - alpha) * _domain_loss #####                 <<<<<<<<<<<<<<<<<<<======================
        deception_loss.append(_deception_loss.item())
        domain_loss.append(_domain_loss.item())     

        #Backward
        _loss.backward()
        optimizer.step()

        # calculate acc per minibatch
        sum_correct_pred += (torch.argmax(deception_preds, dim=-1) == deception_labels).sum().item()
        total_samples += len(deception_labels)


    deception_loss = np.mean(deception_loss)
    domain_loss = np.mean(domain_loss)
    acc = round(sum_correct_pred/total_samples,4)*100
    return deception_loss, domain_loss, acc

def val_one_epoch(val_data_loader,model,loss1):

    epoch_loss = []
    sum_correct_pred = 0
    total_samples = 0

    model.eval()
    with torch.no_grad():
      for spec, deception_labels, _ in val_data_loader:       
        spec = spec.to(device)
        deception_labels = deception_labels.to(device)

        deception_preds, _ = model(spec)
        _loss = loss1(deception_preds,deception_labels)
        epoch_loss.append(_loss.item())
        
        # calculate acc per minibatch
        sum_correct_pred += (torch.argmax(deception_preds, dim=-1) == deception_labels).sum().item()
        total_samples += len(deception_labels)
    
    epoch_loss = np.mean(epoch_loss)
    acc = round(sum_correct_pred/total_samples,4)*100
    return epoch_loss, acc

#############################################################################################################################
#############################################################################################################################

device = torch.device("cuda:0")
batch_size = 16
num_epochs = 15
learning_rate = 3e-4
alpha = 0.75

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
    # print("\n\t Dataset Loaded")

    model = audio_model()
    model = nn.DataParallel(model, device_ids=[0,1,2,3,4,5,6,7])
    model.to(device)
    print("\n\t Model Loaded")
    print("\t Trainable Params = ",sum(p.numel() for p in model.parameters() if p.requires_grad))


    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    #Loss Function
    deception_loss_fn = nn.CrossEntropyLoss()
    domain_loss_fn = nn.CrossEntropyLoss()

    best_val_acc = 0.0

    for epoch in range(num_epochs):
             
        ###Training
        train_deception_loss, train_domain_loss, train_acc = train_one_epoch(train_loader,model,optimizer,deception_loss_fn, domain_loss_fn, alpha)
        ###Validation
        val_loss, val_acc = val_one_epoch(test_loader,model,deception_loss_fn)

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        print('\n\tEpoch...........',epoch+1) 
        print("\tTrain deception loss = {}; \tTrain domain loss = {}; \tTrain accuracy = {}".format(train_deception_loss, train_domain_loss, train_acc))
        print("\tVal loss = {} \t Val accuracy = {}".format(val_loss, val_acc ))

    print("\nBest Accuracy........", best_val_acc)
