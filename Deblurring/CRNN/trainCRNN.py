import torch
import numpy as np
import argparse
from torch.nn import CTCLoss
import torch.optim as optim
import torchvision.transforms as transforms
from model_crnn import CRNN
from dataset import PatchDataset
from util import get_char_maps,  PadWhite, get_text_stack
from util import extract_patches_with_labels
import properties as properties
from tqdm import tqdm
import wandb
import os

class TrainCRNN():

    def __init__(self, args):
        self.batch_size = args.batch_size
        self.random_seed = args.random_seed
        self.lr = args.lr
        self.max_epochs = args.epoch

        #WANDB LOGIN AND SET UP
        self.wandb = args.wandb
        if self.wandb == True:
            self.wandb = True
            wandb.login()
            run = wandb.init(
                # Set the project where this run will be logged
                project="TrainCRNN",
                # Track hyperparameters and run metadata
                config={
                    "learning_rate": self.lr,
                    "epochs": 10,
                    "batch_size": self.batch_size
            })
            wandb.define_metric("CTC_Loss_Train", summary="min")
            wandb.define_metric("CTC_Loss_Val", summary="min")
            wandb.define_metric("Training_Loss", summary="min")
            
        else:
            self.wandb = False 

        self.decay = 0.8
        self.decay_step = 10
        torch.manual_seed(self.random_seed)
        np.random.seed(torch.initial_seed())


        self.train_image_folder = properties.train_image_folder
        self.train_text_folder  = properties.train_text_folder
        self.val_image_folder = properties.val_image_folder
        self.val_text_folder  = properties.val_text_folder

        self.input_size = properties.input_size
        self.char_to_index, self.index_to_char, self.vocab_size = get_char_maps(
            properties.char_set)
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

        self.model = CRNN(self.vocab_size, False).to(self.device)

        if properties.continue_training:
            #self.model = torch.load(properties.model_path).to(self.device)
            self.model.load_state_dict(torch.load(properties.model_path))


        # TRAIN DATASET AND DATALOADERS
        dataset = PatchDataset(
                self.train_image_folder,self.train_text_folder,pad=True, include_name=False)
        self.loader_train = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, collate_fn=PatchDataset.collate)

        # VAL DATASET AND DATALOADERS
        self.validation_set = PatchDataset(
            self.val_image_folder,self.val_text_folder, pad=True)
        
        self.train_set_size = len(dataset)
        self.val_set_size = len(self.validation_set)

        self.loss_function = CTCLoss().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=self.decay_step, gamma=self.decay)

    def _call_model(self, images, labels):
        X_var = images.to(self.device)
        scores = self.model(X_var)
        out_size = torch.tensor(
            [scores.shape[0]] * images.shape[0], dtype=torch.int)
        y_size = torch.tensor([len(l) for l in labels], dtype=torch.int)
        conc_label = ''.join(labels)
        y = [self.char_to_index[c] for c in conc_label]
        y_var = torch.tensor(y, dtype=torch.int)
        return scores, y_var, out_size, y_size

    def train(self):

        step = 0
        epoch_start=0
        if properties.continue_training:
            epoch_start = properties.epoch_restart
            step+= (self.train_set_size//self.batch_size)*epoch_start
        for epoch in range(epoch_start,self.max_epochs):
            self.model.train()
            training_loss = 0
            tq = tqdm(self.loader_train)
            for images, labels in tq:
                self.model.zero_grad()


                text_crops, labels_stacks = extract_patches_with_labels(
                    images, labels, self.input_size
                )

                scores, y, pred_size, y_size = self._call_model(
                        text_crops, labels_stacks)
                    
                loss = self.loss_function(
                            scores, y, pred_size, y_size)
                        
                loss.backward()

                training_loss += loss.item()
                self.optimizer.step()

                if step % 100 == 0:
                    #print("Iteration: %d => %f" % (step, loss.item()))
                    tq.set_postfix(CTC_loss=loss.item())
                    if self.wandb:
                        wandb.log({"CTC_Loss_Train":loss.item()},step=step)
                step += 1


            if self.wandb:
                wandb.log({"Training_Loss": training_loss / (self.train_set_size//self.batch_size) }, step = step)
            
            #EVALUATION
            self.model.eval()
            validation_loss = 0
            validation_step = 0
            with torch.no_grad():
                tq = tqdm(self.validation_set)
                for image, labels_dict in tq:
                    n_text_crops, labels = get_text_stack(
                        image, labels_dict, self.input_size)
                    scores, y, pred_size, y_size = self._call_model(
                        n_text_crops, labels)
                    
                    loss = self.loss_function(scores, y, pred_size, y_size)
                    validation_loss += loss.item()
                    validation_step += 1

                    if validation_step==500:
                        break

            if self.wandb:
                wandb.log({"CTC_Loss_Val":validation_loss/validation_step},step=step)
            print("Epoch: %d/%d => Training loss: %f | Validation loss: %f" % ((epoch + 1),
                                                                               self.max_epochs,
                                                                               training_loss / (self.train_set_size//self.batch_size)  ,
                                                                               validation_loss/validation_step))

            self.scheduler.step()
            
            torch.save(self.model.state_dict(), os.path.join(properties.crnn_model_path,f"model_{epoch}.pth"))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Trains the CRNN model')
    parser.add_argument('--batch_size', type=int,
                        default=32, help='input batch size')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate, not used by adadealta')
    parser.add_argument('--epoch', type=int,
                        default=10, help='number of epochs')
    parser.add_argument('--random_seed', type=int,
                        default=42, help='random seed for shuffles')
    parser.add_argument('--wandb', type= bool, default=False, help='True if you want to log results in wandb')

    args = parser.parse_args()
    print(args)
    trainer = TrainCRNN(args)
    trainer.train()