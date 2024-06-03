import logging
import argparse
import os
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

import random
import numpy as np

from transformers import BartModel
from collections import OrderedDict
from torch.utils.data import DataLoader

from src.utils import DatasetConfig, MyDataset, collate_fn, MyModelDataset
from src.bart import MyConfig
from src.bart import BartModel as BM
from src.bart import BartForConditionalGeneration
from train_eval import train, test, get_loss


def set_seed(seed=42):
    # seed init.
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
 
    # torch seed init.
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False # train speed is slower after enabling this opts.
 
    # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
 
    # avoiding nondeterministic algorithms (see https://pytorch.org/docs/stable/notes/randomness.html)
    torch.use_deterministic_algorithms(True)


def get_parser():
    parser = argparse.ArgumentParser(description='Logging Demo')
    parser.add_argument('--not-save', default=False, action='store_true', help='If yes, only output log to terminal.')
    parser.add_argument('--test-only', default=False, action='store_true', help='If yes, only run test.')
    parser.add_argument('--valid-only', default=False, action='store_true', help='If yes, only test valid.')
    parser.add_argument('--init-embed', default=False, action='store_true', help='If yes, initialize kg embeddings with give data.')
    parser.add_argument('--archive-file', default='./pretrained-bart/bart-base-w-lm.bin', help='Model weights archive.')
    parser.add_argument('--save-dir', default='./saved_model/work_dir', help='The folder for storing results')
    parser.add_argument('--kg-dir', default='/path2kg', help='The folder for storing kg data')
    parser.add_argument('--data-set', required=True, help='Which dataset to use.')
    parser.add_argument('--config-file', default=None, help='Config file path.')
    parser.add_argument('--device', default='cpu', help='Which device to use during running.')
    parser.add_argument('--modal-match', default='sft', help='`mm` for modal match, `sft` for supervised finetune')
    parser.add_argument('--loss-only', required=True, help='If yes, only calculate the loss, and do not decode.')
    parser.add_argument('--cur-epoch', default='0', help='Current epoch number for model save.')
    parser.add_argument('--seed', default='42', help='Random seed setting.')
    return parser


def load_logger(args):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        fmt='[ %(asctime)s ] %(message)s', datefmt='%a %b %d %H:%M:%S %Y')

    sHandler = logging.StreamHandler()
    sHandler.setFormatter(formatter)
    logger.addHandler(sHandler)

    if not args.not_save:
        save_dir = os.path.join(args.save_dir, time.strftime(
            '%Y-%m-%d--%H-%M-%S', time.localtime()))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        fHandler = logging.FileHandler(save_dir + '/log.txt', mode='w')
        fHandler.setLevel(logging.DEBUG)
        fHandler.setFormatter(formatter)
        logger.addHandler(fHandler)

    return logger


def print_args(args):
    logger_config = {}
    logger_config['args'] = args.__dict__
    return json.dumps(logger_config, indent=4)


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    logger = load_logger(args)

    set_seed(int(args.seed))
    
    logger.info(print_args(args))
    # KG
    new_embedding_file = os.path.join(args.kg_dir, 'transe_200dim_1000iter.pth')
    new_embedding = torch.load(new_embedding_file)
    num_entity, d_entity = new_embedding.shape

    # load config
    my_config = MyConfig(config_file=args.config_file, save_dir=args.save_dir, device=args.device, num_entity=num_entity, d_entity=d_entity)
    bart_model = my_config.bart_model
    bart_config = BartModel.config_class.from_pretrained(bart_model)
    my_config.d_model = bart_config.d_model

    # load dataset
    dataset_config = DatasetConfig(args.data_set, args.kg_dir)
    dataset = MyDataset(dataset_config, bart_config, my_config, is_test=args.test_only, is_valid=args.valid_only)

    # load model
    state_dict = torch.load(args.archive_file, map_location='cpu')

    model = BartForConditionalGeneration(bart_config, my_config)

    # initialize kg embeddings
    if args.init_embed:
        new_state_dict = OrderedDict()
        for key in state_dict.keys():
            if not key.startswith('model.sent_ent_gate'):
                new_state_dict[key] = state_dict[key]
        model.load_state_dict(new_state_dict, strict=False)
        model.set_kg_embeddings(new_embedding)
    else:
        model.load_state_dict(state_dict, strict=True)
    
    # load data
    is_train = not args.test_only and not args.valid_only
    data_iter = DataLoader(
        dataset=dataset,
        batch_size=my_config.batch_size,
        shuffle=is_train,
        collate_fn=collate_fn
    )
    
    if args.loss_only == 'not_loss_only':
        if is_train:
            if args.modal_match == 'mm':
                print('Modal match training.')
                for k, v in model.named_parameters():
                    v.requires_grad_(False)
                for k, v in model.model.encoder.embed_images.named_parameters():
                    v.requires_grad_(True)
            elif args.modal_match == 'sft':
                pass
            else:
                raise ValueError('`modal_match` has to be `mm` or `sft`')
            train(logger, my_config, dataset_config, model, data_iter, cur_epoch=args.cur_epoch)
        else:
            archive_file = args.archive_file.strip().split('/')[-1]
            archive_name = archive_file.split('.')[0]
            hyp_file = f'{my_config.save_dir}/{archive_name}-hyps.txt'
            ref_file = f'{my_config.save_dir}/{archive_name}-refs.txt'
            test(logger, my_config, model, data_iter, hyp_file, ref_file)
    elif args.loss_only == 'loss_only':
        loss = get_loss(logger, my_config, dataset_config, model, data_iter)
        loss_file = f'{my_config.save_dir}/loss.txt'
        with open(loss_file, 'a') as fw:
            fw.write(f'{loss}\t{args.data_set}\t{args.archive_file}\n')
