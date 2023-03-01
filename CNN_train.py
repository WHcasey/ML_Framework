from model import NeuralNetwork
from dataset import POEMMS_Dataset
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import os

config = {
    'Dataset_path':'/home/*/*.pkl',
    'device':torch.device("cuda:0" if torch.cuda.is_available else "cpu"),
    'save_path':'./CNN_MkI.pth',
    'lr':1e-3,
    'epochs':1,
    'batch_size':16,
    'train_split':0.7,
    }
    
    
def train_loop():
    print(f"Using: {config['device']}\n")

    try:
        os.makedirs(config['save_path'].split('/')[1])
    except FileExistsError:
        pass

    dataset = POEMMS_Dataset(config['Dataset_path'])
    train_set, test_set = random_split(dataset, [int(dataset.__len__()*config['train_split']), dataset.__len__() - int(dataset.__len__()*config['train_split'])])
    train_loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False)

    model = NeuralNetwork()
    model.to(config['device'])
    print(model)

    loss_fn = nn.BCELoss()
    opt = optim.Adam(model.parameters(), lr=config['lr'])

    old_loss = 1000
    for epoch in range(config['epochs']):
        with tqdm(train_loader, unit='batch') as tepoch:
            correct=0
            total=0
            for data, labels in tepoch:
                tepoch.set_description(f"Epoch {epoch+1}/{config['epochs']}")
                data, labels = data.to(config['device']), labels.to(config['device'])
                opt.zero_grad()
                outputs = model(data.float())
                loss = loss_fn(outputs, labels.float())
                loss.backward()
                opt.step()
                
                preds = []
                acts = []
                for i in range(len(outputs[0])):
                    if outputs[0][i] >= 0.5:
                        preds.append(dataset.solutions[i])
                    if labels[0][i] == 1:
                        acts.append(dataset.solutions[i])
                        
                total+=len(acts)
                for p in preds:
                    if p in acts:
                        correct+=1
                
                tepoch.set_postfix(loss=round(loss.item(),8), accuracy=100*correct//total)
                
            if loss.item() < old_loss:
                torch.save(model.state_dict(), config['save_path'])
                old_loss = loss.item()
                
    print('Finished Training')
    
    ### need to finish setting up stat functions
    
def eval_one():
    test_loader = DataLoader(test_set, batch_size=1, shuffle=True)
    dataiter = iter(test_loader)
    data, labels = dataiter.next()
    #plt.figure(figsize=(10,8))
    #plt.imshow(data[0][0])
    #print(labels[0])

    model = NeuralNetwork()
    model.load_state_dict(torch.load(path))
    model.eval()

    with torch.no_grad():
        outputs = model(data.float())
        
        preds = []
        acts = []
        for i in range(len(outputs[0])):
            if outputs[0][i] >= 0.5:
                preds.append(dataset.solutions[i])
            if labels[0][i] == 1:
                acts.append(dataset.solutions[i])

        print(f'Predicted Solutions: {preds}')
        print(f'Actual Solutions: {acts}')
        
def eval_set():
    # eval entire dataset
    test_loader = DataLoader(test_set, batch_size=1, shuffle=True)
    dataiter = iter(test_loader)
    model = NeuralNetwork()
    model.load_state_dict(torch.load(path))
    model.eval()
    tc = 0
    t = 0
    with torch.no_grad():
        for j in range(len(dataiter)):
            data, labels = dataiter.next()
            outputs = model(data.float())
            
            preds = []
            acts = []
            for i in range(len(outputs[0])):
                if outputs[0][i] >= 0.5:
                    preds.append(dataset.solutions[i])
                if labels[0][i] == 1:
                    acts.append(dataset.solutions[i]) 

            t += len(acts)
            if acts == preds:
                tc += len(acts)
    print(f'Prediction Accuracy: {tc*100//t} %')

def confusion_matrix():
    # create confusion matrix
    solution_data = {}
    for file in dataset.files:
        s = file.split('/')[-2]
        if s not in solution_data:
            solution_data[s] = np.zeros(len(dataset.solutions))

    test_loader = DataLoader(test_set, batch_size=1, shuffle=True)
    dataiter = iter(test_loader)
    model = NeuralNetwork()
    model.load_state_dict(torch.load(path))
    model.eval()

    with torch.no_grad():
        for j in range(len(dataiter)):
            data, labels = dataiter.next()
            outputs = model(data.float())
            
            pred_idx = []
            acts = ''
            
            for i in range(len(outputs[0])):
                if outputs[0][i] >= 0.5:
                    pred_idx.append(i)
            for i in range(len(labels[0])):
                if labels[0][i] == 1:
                    acts+=dataset.solutions[i]+'_'

            for idx in pred_idx:
                solution_data[acts][idx] += 1
    print(solution_data)
    
if __name__ == "__main__":
    train_loop()
    #eval_one()
    #eval_set()
    #confusion_matrix()
