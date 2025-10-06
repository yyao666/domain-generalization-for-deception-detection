import os
import pandas as pd
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
torch.manual_seed(1111)
import torchvision.transforms as T


class face_model(nn.Module):
    def __init__(self, ):
        super(face_model, self).__init__()

        r3d = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)
        r3d.blocks[5].proj = torch.nn.Identity()
        self.visual_backbone = nn.Sequential(
            r3d, 
            nn.Flatten(),
            )

        # linear classifier
        self.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(2048, 2) # 2048 = output dim of slow_r50 model
        )

    def forward(self, y):
        visual_preds =  self.visual_backbone(y)
        return self.classifier(visual_preds)


###################################################################################################################


def parse_train_csv_file(annotations_file, mode, img_dir, time_window):
    annos = pd.read_csv(annotations_file)
    with open("/home/DSO_SSD/yuyao/parsed_protocols/parsed_train.txt","w") as f:
        for i in range(annos.shape[0]):

            # mono vs interro. If mode = None, then all samples are used
            if mode == "monologue":
                if annos.iloc[i, 4] == "interrogation": continue
            if mode == "interrogation":
                if annos.iloc[i, 4] in ["mono", "monologue"]: continue

            # calculate the duration in seconds for each clip
            num_frames = len(os.listdir(img_dir + annos.iloc[i, 0]))
            time = round(num_frames/5.0,2)

            # if the original video duration is less than the time_window use it as is
            if time <= time_window:
                f.write(annos.iloc[i, 0]+','+annos.iloc[i,5]+','+str(0)+','+str(num_frames)+'\n')

            # else split the larger video into smaller chunks
            else:
                # calculate the number of small segments
                num_splits = round(time/time_window)
                # use NumPy to split the indices
                split_indices = np.array_split(np.arange(num_frames), num_splits)
                # update txt file
                for split in split_indices: f.write(annos.iloc[i, 0]+','+annos.iloc[i,5]+','+str(split[0])+','+str(split[-1])+'\n')
    f.close()
    csv = pd.read_csv('/home/DSO_SSD/yuyao/parsed_protocols/parsed_train.txt',header=None)
    csv.to_csv('/home/DSO_SSD/yuyao/parsed_protocols/parsed_train.csv',index=None)

def parse_test_csv_file(annotations_file, mode, img_dir, time_window):
    annos = pd.read_csv(annotations_file)
    with open("/home/DSO_SSD/yuyao/parsed_protocols/parsed_test.txt","w") as f:
        for i in range(annos.shape[0]):

            # mono vs interro. If mode = None, then all samples are used
            if mode == "monologue":
                if annos.iloc[i, 4] == "interrogation": continue
            if mode == "interrogation":
                if annos.iloc[i, 4] in ["mono", "monologue"]: continue
            
            # calculate the duration in seconds for each clip
            num_frames = len(os.listdir(img_dir + annos.iloc[i, 0]))
            time = round(num_frames/5.0,2)

            # if the original video duration is less than the time_window use it as is
            if time <= time_window:
                f.write(annos.iloc[i, 0]+','+annos.iloc[i,5]+','+str(0)+','+str(num_frames)+'\n')
                
            # else split the larger video into smaller chunks
            else:
                # calculate the number of small segments
                num_splits = round(time/time_window)
                # use NumPy to split the indices
                split_indices = np.array_split(np.arange(num_frames), num_splits)
                # update txt file
                for split in split_indices: f.write(annos.iloc[i, 0]+','+annos.iloc[i,5]+','+str(split[0])+','+str(split[-1])+'\n')
    f.close()
    csv = pd.read_csv('/home/DSO_SSD/yuyao/parsed_protocols/parsed_test.txt',header=None)
    csv.to_csv('/home/DSO_SSD/yuyao/parsed_protocols/parsed_test.csv',index=None)


#######################################################################################################################################

class Parsed_Face_Dataset(Dataset):
    def __init__(self, annotations_file, img_dir, time_window, fps, domain):

        self.annos = pd.read_csv(annotations_file)  # load all samples .csv
        self.img_dir = img_dir
        self.visual_transforms = T.Compose([
                                    T.ToTensor(),
                                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                ])
        self.time_window = time_window
        self.fps = fps
        self.number_of_target_frames = int(time_window * fps)  

        # self.annos = []   ###???

        for i in range(self.annos.shape[0]):

           # mono_or_interro = self.annos.iloc[i,4] # mono, monologue, interrogation
            ethnicity = self.annos.iloc[i,1].split("_")[0] # EA, SEA, SA
            language = self.annos.iloc[i,-1] # chinese, english, English etc. CAUTION !! - language names are case sensitive

            if language in ["Chinese","chinese","Malay","malay","Hindi","hindi"]:
                if ethnicity == "EA" and "CHINESE" in domain: self.annos.append(self.annos.iloc[i])
                if ethnicity == "SEA" and "MALAY" in domain: self.annos.append(self.annos.iloc[i])
                if ethnicity == "SA" and "HINDI" in domain: self.annos.append(self.annos.iloc[i])
            else:
                continue # ignore english language


###############################################################################
        
 

    def __len__(self):
        return len(self.annos)

    def __getitem__(self, idx):

        file_path = self.img_dir + self.annos.iloc[idx, 0] + "/"
        start_index=self.annos.iloc[idx, 2]
        end_index=self.annos.iloc[idx, 3]

        # face frames
        frame_names = os.listdir(file_path) # all face frames saved at 5 fps
        window_frames = frame_names[start_index : end_index]
        target_frames = np.linspace(start=0,stop=len(window_frames)-1,num=self.number_of_target_frames,dtype=np.int32)

        face_frames = []
        for i in target_frames:
            img = np.asarray(Image.open(file_path + window_frames[i]), dtype=np.float32) / 255.0
            face_frames.append(self.visual_transforms(img))
        face_frames = torch.stack(face_frames, 0).permute(1,0,2,3) # for slow_r50

        # labels
        gt = self.annos.iloc[idx, 1]
        if gt == 'T':
           label = 0
        elif gt in ['F','L']:
           label = 1
        label = torch.tensor(label)

        # the sample name is required to calculate accuracy over entire video clips
        return self.annos.iloc[idx, 0].split('/')[1], face_frames, label


###############################################################################################################
###############################################################################################################

def train_one_epoch(train_data_loader,model,optimizer,loss_fn):
    epoch_loss = []
    model.train()
    sum_correct_pred = 0
    total_samples = 0
    for i,(_,face,labels) in enumerate(train_data_loader):

        face = face.to(device)
        labels = labels.to(device) 

        #Reseting Gradients
        optimizer.zero_grad()
        #Forward
        preds = model(face)
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

def estimate_clip_level_labels():
    
    clip_level_names = []
    # this adds all unique file names as a list
    for line in open("/home/DSO_SSD/yuyao/parsed_protocols/parsed_test.txt").readlines():
        name = line.split(",")[0].split('/')[1]
        if name not in clip_level_names: clip_level_names.append(name)

    clip_level_labels = np.zeros(len(clip_level_names),dtype=np.int8)
    splits_per_clip = np.zeros(len(clip_level_names),dtype=np.int8)
    # append clip level labels and number of splits per clip
    f = open("/home/DSO_SSD/yuyao/parsed_protocols/parsed_test.txt")
    for line in f.readlines():

        gt = line.split(',')[1]
        if gt == 'T':
           label = 0
        elif gt in ['F','L']:
           label = 1
        idx = clip_level_names.index(line.split(",")[0].split('/')[1])

        clip_level_labels[idx] = label
        splits_per_clip[idx] += 1
    
    return clip_level_names, clip_level_labels, splits_per_clip

def val_one_epoch(val_data_loader, model,loss_fn):
    epoch_loss = []

    clip_level_names, clip_level_labels, splits_per_clip = estimate_clip_level_labels()
    clip_level_preds = np.zeros(len(clip_level_names),dtype=np.int8)

    model.eval()
    with torch.no_grad():
      for sample_names, face,labels in val_data_loader:       
        face = face.to(device)
        labels = labels.to(device)
        preds = model(face)

        _loss = loss_fn(preds, labels)
        epoch_loss.append(_loss.item())
        
        for sample_name,pred in zip(sample_names,preds):
            idx = clip_level_names.index(sample_name)
            clip_level_preds[idx] += torch.argmax(pred).item()
    
    epoch_loss = np.mean(epoch_loss)

    avg_preds = np.round(clip_level_preds/splits_per_clip)
    avg_preds = np.asarray(avg_preds, dtype=np.int8)
    sum_correct = np.sum(clip_level_labels==avg_preds)
    acc = round(sum_correct/len(clip_level_names),2)*100
    return epoch_loss, acc


###############################################################
###############################################################

time_window = 30.0
fps = 2
device = torch.device("cuda:0")
batch_size = 32
num_epochs = 10
learning_rate = 1e-4


protocols = [
    [["CHINESE", "MALAY"],["HINDI"]],
    [["CHINESE", "HINDI"],["MALAY"]],
    [["MALAY", "HINDI"],["CHINESE"]]
    ]


for tr,te in protocols: 

    print("\n Train domain = ",tr)
    print("\n Test domain = ",te,"\n")

###
    # print("\n\tFold {}".format(str(fold)))

    # train_csv = "protocols/train"+str(fold)+".csv"
    # test_csv = "protocols/test"+str(fold)+".csv"
###
    
    # train_csv = "protocols/train.csv"
    # test_csv = "protocols/test.csv"

    ###########

    ###  train_dataset = Spec_Dataset("/home/DSO_SSD/ROSE_V2/scripts/protocols/all_samples.csv", "/home/DSO_SSD/ROSE_V2/hr_spectrograms/",domain=tr)
    ###  test_dataset = Spec_Dataset("/home/DSO_SSD/ROSE_V2/scripts/protocols/all_samples.csv", "/home/DSO_SSD/ROSE_V2/hr_spectrograms/",domain=te)

    ### CHECK 

    # Monologue experiments
    # parse_train_csv_file(train_csv, "monologue", "/home/DSO_SSD/ROSE_V2/interro_masked_face_frames/", time_window)
    # parse_test_csv_file(test_csv, "monologue", "/home/DSO_SSD/ROSE_V2/interro_masked_face_frames/", time_window)
    
    # Interrogation experiments
    parse_train_csv_file("/home/DSO_SSD/ROSE_V2/scripts/protocols/all_samples.csv", "interrogation", "/home/DSO_SSD/ROSE_V2/interro_masked_face_frames/", time_window)
    parse_test_csv_file("/home/DSO_SSD/ROSE_V2/scripts/protocols/all_samples.csv", "interrogation", "/home/DSO_SSD/ROSE_V2/interro_masked_face_frames/", time_window)

    # Full experiments
    # parse_train_csv_file(train_csv, None, "/home/DSO_SSD/ROSE_V2/interro_masked_face_frames/", time_window)
    # parse_test_csv_file(test_csv, None, "/home/DSO_SSD/ROSE_V2/interro_masked_face_frames/", time_window)

    train_dataset = Parsed_Face_Dataset(
        "/home/DSO_SSD/yuyao/parsed_protocols/parsed_train.csv",
        "/home/DSO_SSD/ROSE_V2/interro_masked_face_frames/", 
        time_window, fps, domain=tr)
    test_dataset = Parsed_Face_Dataset(
        "/home/DSO_SSD/yuyao/parsed_protocols/parsed_test.csv",
        "/home/DSO_SSD/ROSE_V2/interro_masked_face_frames/", 
        time_window, fps, domain=te)

    train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True, num_workers=16)
    test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False, num_workers=16)
    print("\n\t Dataset Loaded")


    #####################

    model = face_model()
    model = nn.DataParallel(model, device_ids=[0,1,2,3,4,5,6,7])
    model.to(device)
    print("\n\t Model Loaded")
    print("\t Trainable Params = ",sum(p.numel() for p in model.parameters() if p.requires_grad))

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    #Loss Function
    loss_fn = nn.CrossEntropyLoss()

    best_val_acc = []

    for epoch in range(num_epochs):
        print('\n\tEpoch...........',epoch+1,"\n")      
        ###Training
        train_loss, train_acc = train_one_epoch(train_loader,model,optimizer,loss_fn)
        ###Validation
        val_loss, val_acc = val_one_epoch(test_loader,model,loss_fn)

        print("\n\tTrain loss = {} \t Train accuracy = {}".format(train_loss, train_acc ))
        print("\tVal loss = {} \t Val accuracy = {}".format(val_loss, val_acc ))
        best_val_acc.append(val_acc)

    print("\n\tBest Accuracy........", round(np.max(np.asarray(best_val_acc)),2))










