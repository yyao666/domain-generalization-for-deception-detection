import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
torch.manual_seed(1111)


########################################################################  
########################################################################

# Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss


# NT-Xent Loss
class NT_XentLoss(nn.Module):
    def __init__(self, temperature=0.7):
        super(NT_XentLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        # normalized dot
        dot_product = F.normalize(features, dim=1) @ F.normalize(features, dim=1).T

        batch_size = features.size(0)
        labels = labels.view(-1, 1)
        labels = labels.expand(batch_size, batch_size).eq(labels.expand(batch_size, batch_size).t())
        labels = labels.float()

        # tempreature method 
        logits = torch.log(torch.exp(dot_product / self.temperature) + 1e-7) - torch.logsumexp(dot_product / self.temperature, dim=1)

        # NT-Xent Loss
        loss = -torch.mean(torch.sum(labels * logits, dim=1))



class audio_model(nn.Module):
    def __init__(self, ):
        super(audio_model, self).__init__()

        # audio tower
        r50 = torchvision.models.resnet50(weights="ResNet50_Weights.IMAGENET1K_V1")
        r50.conv1 = nn.Conv2d(1, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)
        r50_num_in_feat = r50.fc.in_features
        r50.fc = nn.Identity()
        self.audio_backbone = r50

        self.deception_classifier = nn.Sequential(
        nn.Flatten(),
        nn.Dropout(0.5),
        nn.Linear(r50_num_in_feat, 2)
        )

        self.contrastive_projection = nn.Sequential(
        nn.Flatten(),
        nn.Linear(r50_num_in_feat, 128),
        nn.GELU(),
        nn.Linear(128,128)
        )

    def forward(self, x):
        x = x.float()
        audio_preds = self.audio_backbone(x.unsqueeze(1))
        return self.deception_classifier(audio_preds), self.contrastive_projection(audio_preds)


########################################################################
########################################################################


class Spec_Dataset(Dataset):
    def __init__(self, annotations_file, spec_dir, domain):

        annotations = pd.read_csv(annotations_file) # load protocols/all_samples.csv
        self.spec_dir = spec_dir

        self.annos = []
        for i in range(annotations.shape[0]):

            mono_or_interro = annotations.iloc[i,4] 
            ethnicity = annotations.iloc[i,1].split("_")[0] 
            language = annotations.iloc[i,-1] 
            
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

        return spectrogram, deception_label

def af_pad_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    return batch.permute(0, 2, 1)

def af_collate_fn(batch):
    spec_tensors, targets1 = [], []

    for spec, deception_label in batch:
        spec_tensors += [spec]
        targets1 += [deception_label]

    spec_tensors = af_pad_sequence(spec_tensors)
    targets1 = torch.stack(targets1)

    return spec_tensors, targets1


########################################################################
########################################################################


def train_one_epoch(train_data_loader, model, optimizer, loss1, focal_loss, nt_xent_loss, alpha, beta):

    deception_loss = []
    contrastive_loss = []
    focal_loss = []
    sum_correct_pred = 0
    total_samples = 0

    model.train()

    for i, (spec, deception_labels) in enumerate(train_data_loader):
        spec = spec.to(device)
        deception_labels = deception_labels.to(device)

        # reset gradients
        optimizer.zero_grad()
        # Forward
        deception_preds, contrastive_projections = model(spec)
        
        # three loss functions
        _deception_loss = loss1(deception_preds, deception_labels)
        _focal_loss_value = focal_loss(deception_preds, deception_labels)
        _nt_xent_loss_value = nt_xent_loss(contrastive_projections, deception_labels)
        
        # COMBINE 
        total_loss = _deception_loss + alpha * _focal_loss_value + beta * _nt_xent_loss_value

        deception_loss.append(_deception_loss.item())
        contrastive_loss.append(_nt_xent_loss_value.item())   
        focal_loss.qppend(_focal_loss_value.item())

        # Backward 
        total_loss.backward()
        optimizer.step()

        # calculate acc per minibatch
        sum_correct_pred += (torch.argmax(deception_preds, dim=-1) == deception_labels).sum().item()
        total_samples += len(deception_labels)

    deception_loss = np.mean(deception_loss)
    contrastive_loss = np.mean(contrastive_loss)
    focal_loss = np.mean(focal_loss)
    acc = round(sum_correct_pred/total_samples,4)*100
    return deception_loss, contrastive_loss, focal_loss, acc


def val_one_epoch(val_data_loader,model,loss1):

    epoch_loss = []
    sum_correct_pred = 0
    total_samples = 0

    model.eval()
    with torch.no_grad():
      for spec, deception_labels in val_data_loader:       
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


# hyper parameters setting
alpha = 0.5  # weight of Focal Loss 
beta = 0.5   # weight of NT-Xent Loss 
num_epochs = 20
batch_size = 16
learning_rate = 3e-4


protocols = [
    [["CHINESE", "MALAY"], ["HINDI"]],
    [["CHINESE", "HINDI"], ["MALAY"]],
    [["MALAY", "HINDI"], ["CHINESE"]]
]


beta_rates = [0.1, 0.5, 1.0]
for betas in beta_rates:
    print("\n\n\n BETA VALUE = ", betas)

    for tr, te in protocols:
        print("\n Train domain = ", tr)
        print("\n Test domain = ", te, "\n")

        train_dataset = Spec_Dataset("/home/DSO_SSD/ROSE_V2/scripts/protocols/all_samples.csv", "/home/DSO_SSD/ROSE_V2/hr_spectrograms/", domain=tr)
        test_dataset = Spec_Dataset("/home/DSO_SSD/ROSE_V2/scripts/protocols/all_samples.csv", "/home/DSO_SSD/ROSE_V2/hr_spectrograms/", domain=te)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16, collate_fn=af_collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=16, collate_fn=af_collate_fn)

        model = audio_model()
        model = nn.DataParallel(model, device_ids=[0, 1, 2, 3, 4, 5, 6, 7])
        model.to(device)

        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        # Loss Functions
        deception_loss_fn = nn.CrossEntropyLoss()
        focal_loss = FocalLoss(alpha=0.25, gamma=2, reduction='mean')
        nt_xent_loss = NT_XentLoss(temperature=0.7)

        best_val_acc = 0.0

        for epoch in range(num_epochs):
            # Training
            train_deception_loss, train_contrastive_loss, train_acc = train_one_epoch(
                train_loader, model, optimizer, deception_loss_fn, focal_loss, nt_xent_loss, beta=betas)

            # Validation
            val_loss, val_acc = val_one_epoch(test_loader, model, deception_loss_fn)

            # Check for improvement
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), 'best_model.pth')  # Save best model checkpoint

            print('\n\tEpoch...........', epoch + 1)
            print("\tTrain Deception Loss = {:.4f}, \tTrain Contrastive Loss = {:.4f}, \tTrain Accuracy = {:.2f}%".format(
                train_deception_loss, train_contrastive_loss, train_acc))
            print("\tVal Loss = {:.4f}, \tVal Accuracy = {:.2f}%".format(val_loss, val_acc))

        print("\nBest Accuracy........", best_val_acc)
        print(("* " * 2 + "\n") * 50, end="")
