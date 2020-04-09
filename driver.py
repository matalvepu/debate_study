from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random
import pickle
import sys
from global_config import *

import wandb
import numpy as np

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from torch.nn import CrossEntropyLoss, L1Loss
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef

from transformers import (
    AlbertConfig,
    AlbertTokenizer,
    AlbertForSequenceClassification,
    BertForNextSentencePrediction,
    BertTokenizer,
    get_linear_schedule_with_warmup,
)
from modeling import AlbertForDebateSequenceClassification
from transformers.optimization import AdamW


def return_unk():
    return 0


parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    type=str,
    choices=["Albert", "XLNet"],
    default="Albert",
)

parser.add_argument(
    "--task", type=str, choices=["train", "extract_NSP"], default="train"
)
parser.add_argument("--dataset", type=str, choices=["debate"], default="debate")
parser.add_argument("--punchline", action="store_true", default=False)
parser.add_argument("--batch_size", type=int, default=5)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--warmup_ratio", type=float, default=0.1)
parser.add_argument("--max_seq_length", type=int, default=400)
parser.add_argument("--learning_rate", type=float, default=1e-5)
parser.add_argument("--fc1_out", type=int, default=512)
parser.add_argument("--fc1_drop", type=float, default=0.2)

args = parser.parse_args()


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

    def __str__(self):
        print(
            "guid:{0},text_a:{1},text_b:{2},label:{3}".format(
                self.guid, self.text_a, self.text_b, self.label
            )
        )


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids,label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        

def convert_examples_to_features(examples, tokenizer):

    
    features = []

    for (ex_index, example) in enumerate(examples):
        
        (
            text,
            label,
            sid,
        ) = example
                

             
        example = InputExample(
            guid=sid, text_a=text, text_b=None, label=label.item()
        )

        tokens = tokenizer.tokenize(example.text_a)
        tokens = tokens[: args.max_seq_length - 2]


        tokens = ["[CLS]"] + tokens + ["[SEP]"]


        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        segment_ids = [0] * len(input_ids)
        input_mask = [1] * len(input_ids)


        padding = [0] * (args.max_seq_length - len(input_ids))

        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == args.max_seq_length
        assert len(input_mask) == args.max_seq_length
        assert len(segment_ids) == args.max_seq_length

        label_id = example.label>3

        features.append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label_id=label_id,
            )
        )
    return features


def get_appropriate_dataset(data, tokenizer, parition):

    features = convert_examples_to_features(data, tokenizer)

        
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)



    dataset = TensorDataset(
        all_input_ids,
        all_input_mask,
        all_segment_ids,
        all_label_ids,
    )
    return dataset


def set_up_data_loader():
    
    with open(
        os.path.join(DATASET_LOCATION, args.dataset, "dataset.pkl"), "rb"
    ) as handle:
        all_data = pickle.load(handle)
        
    train_data = all_data["train"]
    dev_data = all_data["dev"]
    test_data = all_data["test"]

    if args.model == "Albert" :
        tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")
    elif args.model == "XLNet":
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    else:
        raise ValueError(args.model + " is not supported model")

    train_dataset = get_appropriate_dataset(train_data, tokenizer, "train")
    dev_dataset = get_appropriate_dataset(dev_data, tokenizer, "dev")
    test_dataset = get_appropriate_dataset(test_data, tokenizer, "test")

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1
    )

    dev_dataloader = DataLoader(
        dev_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1
    )

    test_dataloader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1
    )

    num_training_steps = len(train_dataloader) * args.epochs

    return (train_dataloader, dev_dataloader, test_dataloader, num_training_steps)


def set_random_seed(seed):
    """
    This function controls the randomness by setting seed in all the libraries we will use.
    Parameter:
        seed: It is set in @ex.config and will be passed through variable injection.
    """
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prep_for_training(num_training_steps):

    if args.model == "Albert":
        model = AlbertForDebateSequenceClassification.from_pretrained(
            "albert-base-v2", newly_added_config=args
        )
        tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")
    elif args.model == "BertNSP":
        model = BertForNextSentencePrediction.from_pretrained("bert-base-uncased")
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    else:
        raise ValueError("request model is not available")

    model.to(DEVICE)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(num_training_steps * args.warmup_ratio),
        num_training_steps=num_training_steps,
    )
    return model, optimizer, scheduler, tokenizer


def train_epoch(model, train_dataloader, optimizer, scheduler):
    model.train()
    tr_loss = 0
    tr_cls_loss = 0
    tr_incog_loss = 0
    
    nb_tr_examples, nb_tr_steps = 0, 0
    logit_list = []

    for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
        batch = tuple(t.to(DEVICE) for t in batch)
        input_ids, input_mask, segment_ids,  label_ids = batch

        if args.model == "Albert":
            outputs = model(
                input_ids,
                token_type_ids=segment_ids,
                attention_mask=input_mask,
                labels=None,
            )

        logits = outputs[0]
        #loss_fct = L1Loss()
        loss_fct = nn.BCEWithLogitsLoss()
        total_loss = loss_fct(logits.view(-1), label_ids.view(-1))
        

                
        tr_loss += total_loss.item()

        nb_tr_examples += input_ids.size(0)
        nb_tr_steps += 1

        total_loss.backward()
        optimizer.step()
        scheduler.step()
        model.zero_grad()

    return tr_loss


def eval_epoch(model, dev_dataloader, optimizer):
    model.eval()
    dev_loss = 0
    dev_cls_loss=0
    dev_incog_loss=0
    nb_dev_examples, nb_dev_steps = 0, 0

    logit_list = []
    with torch.no_grad():
        for step, batch in enumerate(tqdm(dev_dataloader, desc="Iteration")):
            batch = tuple(t.to(DEVICE) for t in batch)

            input_ids, input_mask, segment_ids,label_ids = batch


            if args.model == "Albert":
                outputs = model(
                    input_ids,
                    token_type_ids=segment_ids,
                    attention_mask=input_mask,
                    labels=None,
                )

            
            logits = outputs[0]
            
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1), label_ids.view(-1))
            
            total_loss = loss
            
            
            dev_loss += total_loss.item()

            nb_dev_examples += input_ids.size(0)
            nb_dev_steps += 1

    return dev_loss


def test_epoch(model, data_loader):
    """ Epoch operation in evaluation phase """
    model.eval()

    test_loss = 0.0
    test_cls_loss = 0.0
    test_incog_loss = 0.0
    nb_eval_steps = 0
    preds = []
    all_labels = []
    logit_list = []

    with torch.no_grad():
        for batch in tqdm(
            data_loader, mininterval=2, desc="  - (Validation)   ", leave=False
        ):
            batch = tuple(t.to(DEVICE) for t in batch)

            input_ids, input_mask, segment_ids, label_ids = batch

            if args.model == "Albert":
                outputs = model(
                    input_ids,
                    token_type_ids=segment_ids,
                    attention_mask=input_mask,
                    labels=None,
                )
            
            logits = outputs[0]
            
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1), label_ids.view(-1))
            

            test_loss += loss.item()
            
            nb_eval_steps += 1
            #because bcewithlogitloss is used
            logits=torch.round(torch.sigmoid(logits))
            
            
            if len(preds) == 0:
                preds.append(logits.detach().cpu().numpy())
                all_labels.append(label_ids.detach().cpu().numpy())
            else:
                preds[0] = np.append(preds[0], logits.detach().cpu().numpy(), axis=0)
                all_labels[0] = np.append(
                    all_labels[0], label_ids.detach().cpu().numpy(), axis=0
                )
        

        preds = preds[0]
        all_labels = all_labels[0]
        preds = np.squeeze(preds)
        all_labels = np.squeeze(all_labels)

    return preds, all_labels, test_loss


def multiclass_acc(preds, truths):
    """
    Compute the multiclass accuracy w.r.t. groundtruth
    :param preds: Float array representing the predictions, dimension (N,)
    :param truths: Float/int array representing the groundtruth classes, dimension (N,)
    :return: Classification accuracy
    """
    return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))


def test_score_model(model, test_data_loader, exclude_zero=False):

    predictions, y_test, test_loss = test_epoch(model, test_data_loader)
    print(predictions)
    print(y_test)
    #predictions = torch.round(torch.sigmoid(predictions))
    #print("predictions",predictions[:10])
    #print("ytest",y_test[:10])
    f_score = f1_score(y_test, predictions)
    confusion_matrix_result = confusion_matrix(y_test, predictions)
    classification_report_score = classification_report(y_test, predictions, digits=5)
    accuracy = accuracy_score(y_test, predictions)

    print("Accuracy ", accuracy)

    r = {
        "accuracy": accuracy,
        "mult_f_score": f_score,
        "confusion Matrix": confusion_matrix_result,
        "classification Report": classification_report_score,
    }

    return accuracy, f_score, test_loss


def train(
    model,
    train_dataloader,
    validation_dataloader,
    test_data_loader,
    optimizer,
    scheduler,
):
    best_valid_loss = 0.0

    valid_losses = []
    for epoch_i in range(args.epochs):
        train_loss  = train_epoch(model, train_dataloader, optimizer, scheduler)
        valid_loss  = eval_epoch(model, validation_dataloader, optimizer)

        valid_losses.append(valid_loss)
        print(
            "\nepoch:{},train_loss:{}, valid_loss:{}".format(
                epoch_i, train_loss, valid_loss
            )
        )

        model_state_dict = model.state_dict()

        test_accuracy, test_f_score, test_loss = test_score_model(
            model, test_data_loader
        )
        
        if epoch_i==0:
            best_valid_loss=valid_loss
            
        
        if valid_loss <= best_valid_loss:
            best_valid_loss=valid_loss
            wandb.log(
                {
                    "best_test_acc": test_accuracy       
                }
            )
            
            
        wandb.log(
            {
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                "test_loss": test_loss,
                "test_acc": test_accuracy,
            }
        )
        
        if len(valid_losses)>6:
            if valid_losses[-1]>valid_losses[-2] and valid_losses[-2]>valid_losses[-3] and valid_losses[-3]>valid_losses[-4] and valid_losses[-4]>valid_losses[-5] :
                break


def get_NSP(model, train_data_loader, dev_data_loader, test_data_loader):
    model.eval()
    dataloaders = [train_data_loader, dev_data_loader, test_data_loader]
    for loader in dataloaders:
        positive_count = 0
        negative_count = 0
        humor_instance_count = 0
        for batch in tqdm(loader):
            batch = tuple(t.to(DEVICE) for t in batch)
            input_ids, visual, acoustic, input_mask, segment_ids, label_ids = batch
            outputs = model(
                input_ids, token_type_ids=segment_ids, attention_mask=input_mask,
            )
            NSP_score = F.softmax(outputs[0])[0, 0].item()
            if NSP_score >= 0.5:
                positive_count += 1
            else:
                if label_ids >= 0.5:
                    humor_instance_count += 1
                negative_count += 1
        print(positive_count)
        print(negative_count)
        print(humor_instance_count)


def main():
    wandb.init(project="debate")
    wandb.config.update(args)

    set_random_seed(9999)
    (
        train_data_loader,
        dev_data_loader,
        test_data_loader,
        num_training_steps,
    ) = set_up_data_loader()

    model, optimizer, scheduler, tokenizer = prep_for_training(num_training_steps)
    #wandb.watch(model)

    if args.task == "train":
        train(
            model,
            train_data_loader,
            dev_data_loader,
            test_data_loader,
            optimizer,
            scheduler,
        )
    elif args.task == "extract_NSP":
        get_NSP(model, train_data_loader, dev_data_loader, test_data_loader)
    else:
        raise ValueError(args.task + " is invalid task")


if __name__ == "__main__":
    main()
