import json
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
import os

from .archive import Archive

class TinyTrainer:
    def __init__(self, 
                 lr, 
                 num_epochs, 
                 weight_decay=5e-4, 
                 criterion=None, 
                 lr_milestone="default",
                 device="cuda",
                 archive: Archive=None,
                 save_best=True) -> None:
        self.lr = lr
        self.num_epochs = num_epochs 
        self.weight_decay = weight_decay
        self.criterion = criterion
        self.lr_milestone = lr_milestone
        self.device = device
        self.archive = archive
        self.save_best = save_best
        
        if not criterion:
            self.criterion = nn.CrossEntropyLoss()
        
        if isinstance(lr_milestone, str) and lr_milestone == "default":
            # milestones: [20, 50, 80, 100] 四个挡
            self.lr_milestone = [int(num_epochs * 0.2), int(num_epochs * 0.5), int(num_epochs * 0.8)]

        # 解析保存
        if archive:
            if not isinstance(archive, Archive):
                raise TypeError("archive should be instance of Archive")
        
    def before_loss(self, loss, inputs, labels):
        pass

    def after_inference(self):
        pass

    def train(self, 
              model: nn.Module,
              train_loader, 
              val_loader, 
              optimizer="Adam"):
        
        device = self.device
        criterion = self.criterion
        model = model.to(device)
        
        
        if optimizer == "Adam":
            optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif optimizer == "SGD":
            optimizer = optim.SGD(model.parameters(), lr=self.lr, weight_decay=self.weight_decay, momentum=0.9)

        
        if self.lr_milestone:
            train_scheduler = optim.lr_scheduler.MultiStepLR( optimizer, milestones=self.lr_milestone, gamma=0.2)

        
        logs = []
        
        best_acc = 0
        best_epoch = 0
        
        for epoch in range(self.num_epochs):
            train_loss = 0.0
            val_loss = 0.0
            train_correct = 0
            train_total = 0
            
            with tqdm(total=(len(train_loader))) as _tqdm:
                _tqdm.set_description(
                    '[Training] Epoch [{:03}/{:03}] Lr [{:.1e}]'.format(
                        epoch + 1, self.num_epochs, optimizer.param_groups[0]['lr'],
                    )
                )
                model.train()
                for i, (inputs, labels) in enumerate(train_loader):
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    outputs = model(inputs)
                    self.before_loss(criterion, inputs, labels)

                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    train_total += labels.size(0)
                    train_correct += (predicted == labels).sum().item()

                    _tqdm.set_postfix(
                        Loss="{:.3f}".format(train_loss / train_total),
                        Acc="{:.2f}%".format(100 * train_correct / train_total)
                    )
                    _tqdm.update(1)
                    
                self.after_inference()

            model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                with tqdm(total=(len(val_loader))) as _tqdm:
                    _tqdm.set_description(
                        '[Testing ] Epoch [{:03}/{:03}] Lr [{:.1e}]'.format(epoch + 1, self.num_epochs, optimizer.param_groups[0]['lr'],)
                    )
                    for i, (inputs, labels) in enumerate(val_loader):
                        inputs = inputs.to(device)
                        labels = labels.to(device)

                        outputs = model(inputs)

                        self.before_loss(criterion, inputs, labels)
                        loss = criterion(outputs, labels)

                        val_loss += loss.item() * inputs.size(0)
                        _, predicted = torch.max(outputs.data, 1)
                        val_total += labels.size(0)
                        val_correct += (predicted == labels).sum().item()

                        _tqdm.set_postfix(
                            Loss="{:.3f}".format(val_loss / val_total),
                            Acc="{:.2f}%".format(100 * val_correct / val_total)
                        )
                        _tqdm.update(1)

            if self.lr_milestone:
                train_scheduler.step()
                
            
            if self.archive:
                if self.archive.save_interval == -1:
                    save_interval = self.num_epochs
                else:
                    save_interval = self.archive.save_interval
                
                if best_acc < val_correct / val_total:

                    if best_epoch > 0:
                        pre_best_path = self.archive.get_weight_path(Type="BEST", E=best_epoch)
                        os.remove(pre_best_path)
                    
                    best_epoch = epoch + 1
                    best_acc = val_correct / val_total
                    best_path = self.archive.get_weight_path(Type="BEST", E=best_epoch)
                    torch.save(model, best_path)

                if (epoch + 1) % save_interval == 0 and epoch + 1 != self.num_epochs:
                    weight_path = self.archive.get_weight_path(Type="SAVE", E=epoch+1)
                    torch.save(model, weight_path)
                
                if epoch + 1 == self.num_epochs:
                    weight_path = self.archive.get_weight_path(Type="LAST", E=epoch+1)
                    torch.save(model, weight_path)
                
                
                log = {
                    "Epoch": epoch + 1,
                    "lr": optimizer.param_groups[0]['lr'],
                    "train_loss": "{:.6f}".format(train_loss / train_total),
                    "test_loss": "{:.6f}".format(val_loss / val_total),
                    "train_acc": "{:.2f}%".format(100 * train_correct / train_total),
                    "test_acc": "{:.2f}%".format(100 * val_correct / val_total)
                }
                logs.append(log)
                with open(self.archive.get_log_path(), "w") as f:
                    json.dump(logs, f, indent=4)
            
            print()
        return model

    def eval(self, test_loader):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device)

        correct = 0
        total = 0

        self.model.eval()
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(test_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        print("Test Accuracy: {:.2f}%".format(accuracy * 100))