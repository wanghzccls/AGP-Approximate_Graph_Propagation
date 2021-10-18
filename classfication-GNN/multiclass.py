import time
import random
import argparse
import uuid
import resource
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from model import GnnAGP
from utils import load_inductive,muticlass_f1

# Training settings
parser = argparse.ArgumentParser()
# Dataset and Algorithom
parser.add_argument('--seed', type=int, default=20159, help='random seed..')
parser.add_argument('--dataset', default='Amazon2M', help='dateset.')
parser.add_argument('--agp_alg',default='appnp_agp',help='APG algorithm.')
# Algorithm parameters
parser.add_argument('--alpha', type=float, default=0.2, help='alpha for APPNP_AGP.')
parser.add_argument('--ti',type=float,default=4,help='t for GDC_AGP.')
parser.add_argument('--rmax', type=float, default=1e-8, help='threshold.')
parser.add_argument('--L', type=int, default=3,help='propagation levels.')
# Learining parameters
parser.add_argument('--lr', type=float, default=0.01, help='learning rate.')
parser.add_argument('--weight_decay', type=float, default=0, help='weight decay.')
parser.add_argument('--layer', type=int, default=4, help='number of layers.')
parser.add_argument('--hidden', type=int, default=1024, help='hidden dimensions.')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate.')
parser.add_argument('--bias', default='bn', help='bias.')
parser.add_argument('--epochs', type=int, default=1000, help='number of epochs.')
parser.add_argument('--batch', type=int, default=100000, help='batch size.')
parser.add_argument('--patience', type=int, default=100, help='patience.')
parser.add_argument('--dev', type=int, default=1, help='device id.')
args = parser.parse_args()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

print("--------------------------")
print(args)
features_train,features,labels,idx_train,idx_val,idx_test,memory_dataset = load_inductive(args.dataset,args.agp_alg, args.alpha,args.ti,args.rmax,args.L)

checkpt_file = 'pretrained/'+uuid.uuid4().hex+'.pt'

model = GnnAGP(nfeat=features_train.shape[1],nlayers=args.layer,nhidden=args.hidden,nclass=int(labels.max()) + 1,dropout=args.dropout,bias = args.bias).cuda(args.dev)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
loss_fn = nn.CrossEntropyLoss()

torch_dataset = Data.TensorDataset(features_train, labels[idx_train])
loader = Data.DataLoader(dataset=torch_dataset,batch_size=args.batch,shuffle=True,num_workers=40)

def train():
    model.train()
    loss_list = []
    time_epoch = 0
    
    for step, (batch_x, batch_y) in enumerate(loader):
        batch_x = batch_x.cuda(args.dev)
        batch_y = batch_y.cuda(args.dev)
        t1 = time.time()
        optimizer.zero_grad()
        output = model(batch_x)
        loss_train = loss_fn(output, batch_y)
        loss_train.backward()
        optimizer.step()
        time_epoch+=(time.time()-t1)
        loss_list.append(loss_train.item())
    return np.mean(loss_list),time_epoch


def validate():
    model.eval()
    with torch.no_grad():
        output = model(features[idx_val].cuda(args.dev))
        micro_val = muticlass_f1(output, labels[idx_val])
        return micro_val.item()

def test():
    model.load_state_dict(torch.load(checkpt_file))
    model.eval()
    with torch.no_grad():
        output = model(features[idx_test].cuda(args.dev))
        micro_test = muticlass_f1(output, labels[idx_test])
        return micro_test.item()
    
train_time = 0
bad_counter = 0
best = 0
best_epoch = 0
print("--------------------------")
print("Training...")
for epoch in range(args.epochs):
    loss_tra,train_ep = train()
    train_time+=train_ep

    val_acc=validate()
    if(epoch+1)%50== 0:
        print(f'Epoch:{epoch+1:02d},'
            f'Train_loss:{loss_tra:.3f}',
            f'Valid_acc:{100*val_acc:.2f}%',
            f'Time_cost{train_time:.3f}')
    if val_acc > best:
        best = val_acc
        best_epoch = epoch+1
        torch.save(model.state_dict(), checkpt_file)
        bad_counter = 0
    else:
        bad_counter += 1

    if bad_counter == args.patience:
        break

test_acc = test()
print(f"Train cost: {train_time:.2f}s")
print('Load {}th epoch'.format(best_epoch))
print(f"Test accuracy:{100*test_acc:.2f}%")

memory_main = 1024 * resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**30
memory=memory_main-memory_dataset
print("Memory overhead:{:.2f}GB".format(memory))
print("--------------------------")



