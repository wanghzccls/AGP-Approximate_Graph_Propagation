import time
import uuid
import random
import argparse
import gc
import torch
import resource
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from ogb.nodeproppred import Evaluator
from utils import SimpleDataset
from model import MLP
from utils import load_transductive

parser = argparse.ArgumentParser()
# Dataset and Algorithom
parser.add_argument('--seed', type=int, default=20159, help='random seed..')
parser.add_argument('--dataset', default='papers100M', help='dateset.')
parser.add_argument('--agp_alg',default='sgc_agp',help='APG algorithm.')
# Algorithm parameters
parser.add_argument('--alpha', type=float, default=0.2, help='alpha for APPNP_AGP.')
parser.add_argument('--ti',type=float,default=4,help='t for GDC_AGP.')
parser.add_argument('--rmax', type=float, default=1e-7, help='threshold.')
parser.add_argument('--L', type=int, default=10,help='propagation levels.')
# Learining parameters
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate.')
parser.add_argument('--weight_decay', type=float, default=0, help='weight decay.')
parser.add_argument('--layer', type=int, default=3, help='number of layers.')
parser.add_argument('--hidden', type=int, default=2048, help='hidden dimensions.')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate.')
parser.add_argument('--bias', default='none', help='bias.')
parser.add_argument('--epochs', type=int, default=1000, help='number of epochs.')
parser.add_argument('--batch', type=int, default=10000, help='batch size.')
parser.add_argument('--patience', type=int, default=50, help='patience.')
parser.add_argument('--dev', type=int, default=1, help='device id.')
args = parser.parse_args()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
print("--------------------------")
print(args)
checkpt_file = 'pretrained/'+uuid.uuid4().hex+'.pt'

features,train_labels,val_labels,test_labels,train_idx,val_idx,test_idx,memory_dataset = load_transductive(args.dataset,args.agp_alg, args.alpha,args.ti,args.rmax,args.L)

features_train = features[train_idx]
features_val = features[val_idx]
features_test = features[test_idx]
del features
gc.collect()

label_dim = int(max(train_labels.max(),val_labels.max(),test_labels.max()))+1
train_dataset = SimpleDataset(features_train,train_labels)
valid_dataset = SimpleDataset(features_val,val_labels)
test_dataset = SimpleDataset(features_test, test_labels)

train_loader = DataLoader(train_dataset, batch_size=args.batch,shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=128, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

model = MLP(features_train.size(-1),args.hidden,label_dim,args.layer,args.dropout).cuda(args.dev)
evaluator = Evaluator(name='ogbn-papers100M')
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def train(model, device, train_loader, optimizer):
	model.train()

	time_epoch=0
	loss_list=[]
	for step, (x, y) in enumerate(train_loader):
		t_st=time.time()
		x, y = x.cuda(device), y.cuda(device)
		optimizer.zero_grad()
		out = model(x)
		loss = F.nll_loss(out, y.squeeze(1))
		loss.backward()
		optimizer.step()
		time_epoch+=(time.time()-t_st)
		loss_list.append(loss.item())
	return np.mean(loss_list),time_epoch


@torch.no_grad()
def validate(model, device, loader, evaluator):
	model.eval()
	y_pred, y_true = [], []
	for step,(x,y) in enumerate(loader):
		x = x.cuda(device)
		out = model(x)
		y_pred.append(torch.argmax(out, dim=1, keepdim=True).cpu())
		y_true.append(y)
	return evaluator.eval({
		"y_true": torch.cat(y_true, dim=0),
		"y_pred": torch.cat(y_pred, dim=0),
	})['acc']


@torch.no_grad()
def test(model, device, loader, evaluator,checkpt_file):
	model.load_state_dict(torch.load(checkpt_file))
	model.eval()
	y_pred, y_true = [], []
	for step,(x,y) in enumerate(loader):
		x = x.cuda(device)
		out = model(x)
		y_pred.append(torch.argmax(out, dim=1, keepdim=True).cpu())
		y_true.append(y)
	return evaluator.eval({
		"y_true": torch.cat(y_true, dim=0),
		"y_pred": torch.cat(y_pred, dim=0),
	})['acc']

bad_counter = 0
best = 0
best_epoch = 0
train_time = 0
model.reset_parameters()
print("--------------------------")
print("Training...")
for epoch in range(args.epochs):
	loss_tra,train_ep = train(model,args.dev,train_loader,optimizer)
	f1_val = validate(model, args.dev, valid_loader, evaluator)
	train_time+=train_ep
	if(epoch+1)%20 == 0: 
		print(f'Epoch:{epoch+1:02d},'
			f'Train_loss:{loss_tra:.3f}',
			f'Valid_acc:{100*f1_val:.2f}%',
			f'Time_cost{train_time:.3f}')
	if f1_val > best:
		best = f1_val
		best_epoch = epoch+1
		torch.save(model.state_dict(), checkpt_file)
		bad_counter = 0
	else:
		bad_counter += 1

	if bad_counter == args.patience:
		break

test_acc = test(model, args.dev, test_loader, evaluator,checkpt_file)
print(f"Train cost: {train_time:.2f}s")
print('Load {}th epoch'.format(best_epoch))
print(f"Test accuracy:{100*test_acc:.2f}%")

memory_main = 1024 * resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**30
memory=memory_main-memory_dataset
print("Memory overhead:{:.2f}GB".format(memory))
print("--------------------------")

