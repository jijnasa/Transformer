'''
This script handling the training and testing process.
'''

import sys
import time
import argparse
import math
from sklearn.utils import shuffle
import numpy as np
from tensorboardX import SummaryWriter
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from model_sa import Transformer
from data_preprocess import DataManager
import tensorflow as tf
#from Optim import ScheduledOptim
from torch.autograd import Variable

def batchify(data, batch_size):
    ''' Input: training_data_list [[(question, pos_relas, pos_words, neg_relas, neg_words) * neg_size] * q_size]
        Return: [[(question, pos_relas, pos_words, neg_relas, neg_words)*neg_size] * batch_size] * nb_batch]
    '''
    nb_batch = math.ceil(len(data) / batch_size)
    batch_data = [data[idx*batch_size:(idx+1)*batch_size] for idx in range(nb_batch)]
    print('nb_batch', len(batch_data), 'batch_size', len(batch_data[0]))
    return batch_data

def cal_acc(sorted_score_label):
    if sorted_score_label[0][1] == 1:
        return 1
    else:
        return 0

def save_best_model(model):
    import datetime
    now = datetime.datetime.now()
    if args.save_model_path == '':
        args.save_model_path = f'save_model/{now.month}{now.day}_{now.hour}h{now.minute}m.pt'
        with open('log.txt','a') as outfile:
            outfile.write(str(args)+'\n')
    print('save model at {}'.format(args.save_model_path))
    with open(args.save_model_path, 'wb') as outfile:
        torch.save(model, outfile)
def train(args):
    #Build Model
    print('Building model')
    
    q_len = corpus.maxlen_q
    r_len = corpus.maxlen_w + corpus.maxlen_r
    transformer = Transformer(corpus.word_embedding, corpus.rela_embedding, q_len, r_len, n_layers=6, n_head=8, d_word_vec=512, d_model=512, d_inner_hid=1024, d_k=64,dropout=0.1)

    if args.optimizer == 'Adadelta':
        print('optimizer: Adadelta')
        optimizer = torch.optim.Adadelta(filter(lambda p: p.requires_grad, transformer.parameters()), lr=args.learning_rate)
    #optimizer = ScheduledOptim(optim.Adam(transformer.get_trainable_parameters(), betas=(0.9, 0.98), eps=1e-09),opt.d_model, opt.n_warmup_steps)

    best_model = None
    best_val_loss = None
    train_start_time = time.time()


    earlystop_counter = 0
    global_step = 0

    for epoch_count in range(0, args.epoch_num):
        transformer.train()
        total_loss, total_acc = 0.0, 0.0
        nb_question = 0
        epoch_start_time = time.time()
        for batch_count, batch_data in enumerate(train_data, 1):
            variable_start_time = time.time()
            if args.batch_type == 'batch_question':
                training_objs = [obj for q_obj in batch_data for obj in q_obj]
                question, pos_relas, pos_words, neg_relas, neg_words = zip(*training_objs)
                nb_question += len(batch_data)
            elif args.batch_type == 'batch_obj':
                question, pos_relas, pos_words, neg_relas, neg_words = zip(*batch_data)
                q_len, pos_r_len, pos_w_len, neg_r_len, neg_w_len = zip(*train_data_len[batch_count-1])
            q = Variable(torch.LongTensor(question))
            p_relas = Variable(torch.LongTensor(pos_relas))
            p_words = Variable(torch.LongTensor(pos_words))
            n_relas = Variable(torch.LongTensor(neg_relas))
            n_words = Variable(torch.LongTensor(neg_words))
            ones = Variable(torch.ones(len(question)))
            variable_end_time = time.time()
            print('question is :')
            #tf.Print(q, [q])
            optimizer.zero_grad()
            all_pos_score = transformer(q, p_relas, p_words)
            all_neg_score = transformer(q, n_relas, n_words)

            model_end_time = time.time()

            loss = loss_function(all_pos_score, all_neg_score, ones)
            loss.backward()
            optimizer.step()
            loss_backward_time = time.time()
            writer.add_scalar('data/pre_gen_loss', loss.item(), global_step)
            global_step += 1
            if torch.__version__ == '0.3.0.post4':
                total_loss += loss.data.cpu().numpy()[0]
            else:
                total_loss += loss.data.cpu().numpy()
            average_loss = total_loss / batch_count

            #calculate accuracy and f1
            if args.batch_type == 'batch_question':
                all_pos = all_pos_score.data.cpu().numpy()
                all_neg = all_pos_score.data.cpu().numpy()
                start, end = 0, 0
                for idx, q_obj in enumerate(batch_data):
                    end +=len(q_obj)
                    score_list = [all_pos[start]]
                    batch_neg_score = all_neg[start:end]
                    start = end
                    label_list = [1]
                    for ns in batch_neg_score:
                        score_list.append(ns)
                    label_list += [0] * len(batch_neg_score)
                    score_label = [(x, y) for x, y in zip(score_list, label_list)]
                    sorted_score_label = sorted(score_label, key=lambda x:x[0], reverse=True)
                    total_acc += cal_acc(sorted_score_label)
                average_acc = total_acc / nb_question
                elapsed = time.time() - epoch_start_time
                print_str = f'Epoch {eoch_count} batch {batch_count} Spend Time:{elapsed:.2f}s Loss:{average_loss*1000:.4f} Acc:{average_acc:.4f} #_question:{nb_question}'
            else:
                elapsed = time.time() - epoch_start_time
                print_str = f'Epoch {epoch_count} batch {batch_count} Spend Time:{elapsed:.2f}s Loss:{average_loss*1000:.4f}'
            if batch_count % 10 == 0:
                 print('\r', print_str, end='')
        print('\r', print_str, end='')
        print()
        val_print_str, val_loss, _ = evaluation(transformer, 'dev', global_step)
        print('Val', val_print_str)

        #Handle earlystopping
        if not best_val_loss or val_loss < best_val_loss:
            earlystop_counter = 0
            best_model = transformer
            save_best_model(best_model)
            best_val_loss = val_loss

        else:
            earlystop_counter += 1
        if earlystop_counter >= args.earlystop_tolerance:
            print('EarlyStopping!')
            print(f'Total training time {time.time()-train_start_time:.2f}')
            break
    return best_model


def evaluation(model, mode='dev', global_step=None):
   # print('hello')
    model_test = model.eval()
    start_time = time.time()
    total_loss, total_acc = 0.0, 0.0
    if model == 'test':
        input_data = test_data
        print(model_test)
    else:
        input_data = val_data
    nb_question = sum(len(batch_data) for batch_data in input_data)
    count = 0
    print('nb_question', nb_question)
    for batch_count, batch_data in enumerate(input_data, 1):
        count+=1
        training_objs = [obj for q_obj in batch_data for obj in q_obj]
        question, pos_relas, pos_words, neg_relas, neg_words = zip(*training_objs)
        q = Variable(torch.LongTensor(question)).cuda()
        p_relas = Variable(torch.LongTensor(pos_relas)).cuda()
        p_words = Variable(torch.LongTensor(pos_words)).cuda()
        n_relas = Variable(torch.LongTensor(neg_relas)).cuda()
        n_words = Variable(torch.LongTensor(neg_words)).cuda()
        ones = Variable(torch.ones(len(question))).cuda()
        pos_score = model_test(q, p_relas, p_words)
        neg_score = model_test(q, n_relas, n_words)
        loss = loss_function(pos_score, neg_scores, ones)
        if torch.__version__ == '0.3.0.post4':
            total_loss += loss.data.cpu().numpy()[0]
        else:
            total_loss += loss.data.cpu().numpy()
        averahe_loss = total_loss / batch_count

        #calculate accuracy and f1
        all_pos = pos_score.data.cpu().numpy()
        all_neg = neg_score.data.cpu().numpy()
        start, end = 0, 0
        for idx, q_obj in enumerate(batch_data):
            end += len(q_obj)
            score_list = [all_pos[start]]
            label_list = [1]
            batch_neg_score = all_neg[start:end]
            for ns in batch_neg_score:
                score_list.append(ns)
            label_list += [0] * len(batch_neg_score)
            start = end
            score_label = [(x, y) for x, u in zip(score_list, label_list)]
            sorted_score_label = sorted(score_label, key=lambda x:x[0], reverse=True)
            total_acc += cal_acc(sorted_score_label)
    if mode == 'dev':
        writer.add_scalar('val_loss', average_loss.item(), global_step)

    time_elapsed = time.time()-start_time
    average_acc = total_acc / nb_question
    print_str = f'Batch {batch_count} Spend Time:{time_elapsed:.2f}s Loss:{average_loss*1000:.4f} Acc:{average_acc:.4f} # question:{nb_question}'
    return print_str, average_loss, average_acc
        
if __name__ == '__main__':
    ''' Main function '''
   # print('hi')
    #set random seed
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    np.random.seed(1234)
    #print('hello')
    parser = argparse.ArgumentParser()
    #setting
    parser.add_argument('-train', default=False, action='store_true')
    parser.add_argument('-test', default=False, action='store_true')
    parser.add_argument('-epoch_num', type=int, default=1000)
    #parser.add_argument('-batch_size', type=int, default=64)
    parser.add_argument('-batch_question_size', type=int, default=32)
    parser.add_argument('-batch_obj_size', type=int, default=128)
    parser.add_argument('-batch_type', type=str, default='batch_question')#[batch_question/batch_obj]

    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-d_inner_hid', type=int, default=1024)
    parser.add_argument('-d_k', type=int, default=64)
    parser.add_argument('-d_v', type=int, default=64)
    
    parser.add_argument('-n_head', type=int, default=8)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-n_warmup_steps', type=int, default=4000)

    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-log', default=None)
    parser.add_argument('-save_model', default=None)
    parser.add_argument('-optimizer', type=str, default='Adadelta')
    
    #parser.add_argument('-save_model', type=str, choices=['all', 'best'], default='best')
    parser.add_argument('-margin', type=float, default=0.1)
    parser.add_argument('-learning_rate', type=float, default=2.0)
    #parse.add_argument('-optimizer', type=str, default='Adadelta')

    parser.add_argument('-earlystop_tolerance', type=int, default=5)
    parser.add_argument('-save_model_path', type=str, default='')
    parser.add_argument('-pretrain_model', type=str, default=None)
    parser.add_argument('-data_type', type=str, default='SQ') #[SQ/WQ]
    args = parser.parse_args()
    loss_function = torch.nn.MarginRankingLoss(margin=args.margin)

    #Load Data
    corpus = DataManager(args.data_type)
    if args.train:
        
        if args.data_type == 'SQ':
            train_data = corpus.token_train_data
            train_data_len = corpus.train_data_len
            val_data = corpus.token_val_data
            print('training data length:', len(train_data))
            print('validation data length:', len(val_data))

        else:
            #shuffle training data
            shuffle(corpus.token_train_data, corpus.train_data_len, random_state=1234)
            #splitting the training data to train an validation
            split_num = int(0.9*len(corpus.token_train_data))
            print('split_num', split_num)
            train_data = corpus.token_train_data[:split_num]
            train_data_len = corpus.token_train_data_len[:split_num]
            val_data = corpus.token_train_data[split_num:]
            print('training data length:', len(train_data))
            print('validation data length:', len(val_data))

        if args.batch_type == 'batch_question':
            train_data = batchify(train_data, args.batch_question_size)
        elif args.batch_type == 'batch_obj':
            flat_train_data = [obj for q_obj in train_data for obj in q_obj]
            flat_train_data_len = [obj for q_obj in train_data_len for obj in q_obj]
            print('len(flat_train_data)', len(flat_train_data))
            print('len(flat_train_data_len)', len(flat_train_data_len))
            shuffle(flat_train_data, flat_train_data_len, random_state=1234)
            train_data = batchify(flat_train_data_len, args.batch_obj_size)
            train_data_len = batchify(flat_train_data_len, args.batch_obj_size)
        val_data = batchify(val_data, args.batch_question_size)

        writer = SummaryWriter(log_dir='save_model_sa/tensorboard_log')
        train(args)
    if args.test:
        print('test data length:', len(corpus.token_test_data))
        test_data = batchify(corpus.token_test_data, args.batch_question_size)
        if args.pretrained_model == None:
            print('Load best model', args.save_model_path)
            with open(args.save_model_path, 'rb') as infile:
                model = torch.load(infile)

        else:
            print('Load Pretrained Model', args.pretrain_model)
            with open(args.pretrain_model, 'rb') as infile:
                model = torch.load(infile)
        log_str, _, test_acc = evaluation(model, 'test')
        print(log_str)
        print(test_acc)
        with open('log.txt', 'a') as outfile:
            if args.pretrain_model == None:
                outfile.write(str(test_acc)+'\t'+args.save_model_path+'\n')
            else:
                outfile.write(str(test_acc)+'\t'+args.pretrain_model+'\n')

    


   



