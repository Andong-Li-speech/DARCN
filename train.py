import sys
import os
import argparse
import torch
import numpy as np
from data import TrainDataset, CvDataset, TrainDataLoader, CvDataLoader
from solver import Solver
from Backup import numParams
# fix random seed
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)


def main(args, model):
    tr_dataset = TrainDataset(json_dir= args.json_path,
                              batch_size = args.batch_size)
    cv_dataset = CvDataset(json_dir= args.json_path,
                           batch_size= args.cv_batch_size)
    tr_loader = TrainDataLoader(data_set= tr_dataset,
                                batch_size = 1,
                                num_workers= args.num_workers)
    cv_loader = CvDataLoader(data_set= cv_dataset,
                             batch_size = 1,
                             num_workers=args.num_workers)
    data= {'tr_loader': tr_loader, 'cv_loader': cv_loader}

    print(model)
    # count the parameter number of the network
    print('The number of trainable parameters of the net is:%d' %(numParams(model)))
    model.cuda()

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.l2)
    solver = Solver(data , model, optimizer, args)
    solver.train()

# if __name__ == '__main__':
#     args = parser.parse_args()
#     model = train_model
#     print(args)
#     main(args, model)



