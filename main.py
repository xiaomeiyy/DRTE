### after training with bce resnet101, try to use lstm to further improve the performance
import logging
from datetime import datetime
import argparse
import os
from munkres import Munkres
from scipy.stats import logistic
from future.utils import iteritems
import numpy as np
from collections import OrderedDict
from sklearn.metrics import precision_recall_fscore_support

from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.utils.data.dataloader import default_collate

from model import CNN_Encoder
from model import TransformerModel
from dataset import COCOMultiLabel
from dataset import categories


"""Initialize loss and m function"""
criterion = nn.CrossEntropyLoss()
m = Munkres()


def get_logger(filename, verbosity=1, name=None):
    """logger function for print logs."""
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter("[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s")
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


"""Create log file"""
now = datetime.now()
timestr = now.strftime("%Y%m%d%H%M")
tim_str_file= 'save_model/exp' + "_" + timestr + ".log"
logger = get_logger(tim_str_file)


def my_collate(batch):
    batch = [b for b in batch if b is not None]
    return default_collate(batch)

def adjust_learning_rate(optimizer, shrink_factor):
    logger.info( "DECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    logger.info ("The new learning rate is {}".format(optimizer.param_groups[0]['lr']))

def convert_weights(state_dict):
    """convert wights when load model."""
    tmp_weights = OrderedDict()
    for name, params in iteritems(state_dict):
        tmp_weights[name.replace('module.', '')] = params
    return tmp_weights


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-batch_size', type=int, default=32, metavar='N',help='input batch size for training (default: 32)')
    parser.add_argument('-epochs', type=int, default=30, metavar='N', help='number of epochs to train (default: 30)')
    parser.add_argument('-log_interval', type=int, default=50, metavar='N', help='how many batches to wait before logging training status')
    parser.add_argument('-image_path', type=str, default = '/home/notebook/code/personal/S9031003/DATASET/MSCOCO2014/', help='path for the training and validation folders')
    parser.add_argument('-save_path', type=str, default='save_model/', help='save training models')
    parser.add_argument('-encoder_weights', type=str, default='save_model/resnet101_model_ft.pt',  help='pretrained resnet model')
    parser.add_argument('-decoder_lr', default=1e-4, type=float, help='learning rate for transformer encoder')
    parser.add_argument('-encoder_lr', default=1e-4, type=float, help='learning rate for cnn')
    parser.add_argument('-max_length', default=15, type=int, help='set maximum number of labels for each image')
    parser.add_argument('-dropout', type=float, default=0.1, help='dropout coefficient')
    parser.add_argument('-num_workers', default=6, type=int, help='number of workers')
    parser.add_argument('-coeff', default=0.5, type=float, help='learning rate decrease coefficient')
    
    ## for transfomer model
    parser.add_argument('-num_layers', default=6, type=int, help="number of transformer encoder")
    parser.add_argument('-input_fea_size', default=2048, type=int, help='initial input feature size of transformer')
    parser.add_argument('-d_ff', default=1024, type=int, help='dimension of feedforward')
    parser.add_argument('-embed_size', default=512, type=int, help='dimension of embedded size')
    parser.add_argument('-use_bn', default=0, type=int, help='used when embedding cnn features')
    parser.add_argument('-vocab_size', default=80, type=int, help='label category size')
    parser.add_argument('-threshold', type=float, default=0.5, help='threshold for the evaluation (default: 0.5)')
    parser.add_argument('-ef', type=float, default=0.9, help='the trade-off of classification and distance loss')
    parser.add_argument('-C', type=float, default=7.0, help = 'margin in distrance loss function')
    
    ## evaluation or not
    parser.add_argument('-use_eval', default=False, type=bool, help='open when only evaluation')
    parser.add_argument('-use_model', type=str, default='save_model/trained_model.pt.tar',  help='trained model only for evaluation')
    
    args = parser.parse_args()
    return args


def get_dis_loss(embs, label_mask, label_glove, loss_hinge):
    """Caculate distance loss"""
    ## get size of each dimension
    batch_size, label_size, dim = label_mask.size(0), label_mask.size(1), label_glove.size(2)
    mask_n = 1 - label_mask
    
    ## preprare unify dimensions
    embs = embs.unsqueeze(1).expand(batch_size, label_size, dim)
    mask = label_mask * 2 - 1
    dis = torch.sqrt(torch.sum((embs - label_glove) ** 2, dim=2))
    dis_p = torch.sum(dis * label_mask) / torch.sum(label_mask)
    loss = dis_p + loss_hinge(dis, mask)
    
    return loss


def train(args, encoder, decoder, train_loader, encoder_optimizer, decoder_optimizer, epoch, loss_hinge):
    encoder.train()
    decoder.train()
    for batch_idx, (data, label_cls, label_steps, label_length, label_glove,img_name) in enumerate(train_loader):
        ### set input data format, send the data to cnn and then to transformer encoder
        data, label_cls, label_steps, label_length, label_glove = data.cuda(), label_cls.cuda(), label_steps.cuda(), label_length.cuda(), label_glove.cuda()
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        att_feats, fc_feats = encoder(data)
        att_feats = att_feats.view(fc_feats.size(0), -1, fc_feats.size(1))
        pre_label, pre_emb = decoder(fc_feats, att_feats, label_steps)
        
        ### calculate loss and update parameters
        emb_loss = get_dis_loss(pre_emb, label_cls, label_glove, loss_hinge)
        cls_loss = F.binary_cross_entropy_with_logits(pre_label, label_cls)
        loss = cls_loss +  args.ef * emb_loss
        encoder_optimizer.step()
        loss.backward()
        decoder_optimizer.step()
        ### log for check performance
        if batch_idx % args.log_interval == 0 and batch_idx != 0:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item()))

            
def test(args, encoder, decoder, test_loader, threshold, epoch):
    encoder.eval()
    decoder.eval()
    
    ### whether save test model
    if args.use_eval:
        res_fh = open('test_result_raw_coco_vte_drl.txt', 'w')
        
    with torch.no_grad():
        for batch_idx, (data,label_cls,label_steps, label_length, label_glove,img_name) in enumerate(test_loader):
            ### set input data format, send the data to cnn and then to transformer encoder
            data, label_cls, label_steps, label_glove = data.cuda(), label_cls.cuda(), label_steps.cuda(), label_glove.cuda()
            att_feats, fc_feats = encoder(data)
            att_feats = att_feats.view(fc_feats.size(0), -1, fc_feats.size(1))
            output, pre_emb = decoder(fc_feats, att_feats, label_steps)
            
            ### get final predicted label
            output_arr = output.data.cpu().numpy()
            output_arr = logistic.cdf(output_arr)
            junk = output_arr.copy()
            output_arr[output_arr >= threshold] = 1
            output_arr[output_arr < threshold] = 0
            if batch_idx == 0:
                labels = label_cls.data.cpu().numpy()
                outputs = output_arr
            else:
                labels = np.concatenate((labels, label_cls.data.cpu().numpy()),axis=0)
                outputs = np.concatenate((outputs, output_arr),axis=0)
                
            ### write output predicted  and ground truth labels to .txt file
            if args.use_eval:
                for i in range(len(img_name)):
                    pred_labels = list([categories[j] for j in range(args.vocab_size) if output_arr[i][j] > 0])
                    gt_labels = list([categories[j] for j in range(args.vocab_size) if label_cls[i][j] > 0])
                    res_fh.write('{}\t{}\t{}\n'.format(img_name[i], ','.join(pred_labels), ','.join(gt_labels))) 

            ### log for check performance   
            if batch_idx % args.log_interval == 0 and batch_idx != 0:
                logger.info('Val Epoch: {}[{}/{} ({:.0f}%)]'.format( epoch, batch_idx * len(data), len(test_loader.dataset),100. * batch_idx / len(test_loader)))
                
    ### Caculate precision and print log
    prec, recall, _, _ = precision_recall_fscore_support(outputs,labels,average='macro')
    f1 = 2 * prec * recall / (prec + recall)
    logger.info('\nMACRO prec: {:.2f}, recall: {:.2f}, f1: {:.2f}\n'.format(100*recall, 100*prec, 100*f1))
    prec, recall, f1, _ = precision_recall_fscore_support(outputs,labels,average='micro')
    logger.info('\nMICRO prec: {:.2f}, recall: {:.2f}, f1: {:.2f}\n'.format(100*recall, 100*prec, 100*f1))
    
    ### close write file"
    if args.use_eval:
        res_fh.close()

    return f1
            
    

### main function 
def main():
    ### settings
    args = set_args()
    save_path = args.save_path
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    logger.info(args)
    
    ### prepare for data
    train_dataset = COCOMultiLabel(args, train=True, image_path=args.image_path)
    test_dataset = COCOMultiLabel(args, train=False, image_path=args.image_path)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              num_workers=args.num_workers, pin_memory=True, 
                              shuffle=True, drop_last=True, collate_fn=my_collate)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             num_workers=args.num_workers, pin_memory=True,
                             shuffle=False, drop_last=False, collate_fn=my_collate)
    
    ## prepare for models
    encoder = CNN_Encoder().cuda()
    decoder = TransformerModel(args).cuda()
    ## set different parameter for training or only evaluation'
    if args.use_eval:
        weights_dic = torch.load(args.use_model) 
        encoder.load_state_dict(convert_weights(weights_dic['encoder_state_dict']))
        decoder.load_state_dict(convert_weights(weights_dic['decoder_state_dict']))
    else:
        encoder.load_state_dict(convert_weights(torch.load(args.encoder_weights)))
        encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=args.encoder_lr)
        decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=args.decoder_lr)
    
    ## whether using dataparallel'
    if torch.cuda.device_count() > 1:
        encoder = nn.DataParallel(encoder)
        decoder = nn.DataParallel(decoder)
        
    ## set hinge loss function'
    loss_hinge = torch.nn.HingeEmbeddingLoss(margin=args.C, size_average=None, reduce=None, reduction='mean')
    
    ## if only evaluation, return"
    if args.use_eval:
        f1 = test(args, encoder, decoder, test_loader, args.threshold, 1)
        return
    
    ##  training stage
    highest_f1 = 0
    epochs_without_improve = 0
    for epoch in range(args.epochs):
        ## train and test
        train(args, encoder, decoder, train_loader,encoder_optimizer, decoder_optimizer, epoch, loss_hinge)
        f1 = test(args, encoder, decoder, test_loader, args.threshold, epoch)

        ### save parameter
        save_dict = {'encoder_state_dict': encoder.state_dict(), 
                     'decoder_state_dict': decoder.state_dict(),
                     'epoch': epoch, 'f1': f1, 
                     'decoder_optimizer_state_dict': decoder_optimizer.state_dict(), 
                     'encoder_optimizer_state_dict': encoder_optimizer.state_dict(), 
                     'epochs_without_improve': epochs_without_improve} 

        ### save models'
        torch.save(save_dict, args.save_path + "/checkpoint_" + timestr + '.pt.tar')
        if f1 > highest_f1:
            torch.save(save_dict, args.save_path + "/BEST_checkpoint_" + timestr + '.pt.tar')   
            logger.info("Now the highest f1 is {}, it was {}".format(100*f1, 100*highest_f1))
            highest_f1 = f1
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            if epochs_without_improve == 3:
                adjust_learning_rate(decoder_optimizer, args.coeff)
                adjust_learning_rate(encoder_optimizer, args.coeff)
                epochs_without_imp = 0
        
    
if __name__ == '__main__':
    main()
