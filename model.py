import torch
from torch import nn
from torch_geometric import nn as gnn
import pytorch_lightning as pl


class GCN(pl.LightningModule):
  
  
  def __init__(self, in_channels, num_hidden_layers, hidden_dim, out_channels, dropout=0.5):
    
    super().__init__()
    self.conv = gnn.GCNConv(in_channels, hidden_dim)
    hidden_layers = []
    for _ in range(num_hidden_layers):
        hidden_layers.extend([(gnn.GCNConv(hidden_dim, hidden_dim), 'x, edge_index -> x'),
                             gnn.BatchNorm(hidden_dim),
                             nn.ReLU(),
                             nn.Dropout(dropout)])
    self.hidden_layers = gnn.Sequential('x, edge_index', hidden_layers)
    self.out_mlp = nn.Sequential([nn.Linear(hidden_dim, hidden_dim),
                                  nn.ReLU(),
                                  nn.Linear(hidden_dim, hidden_dim),
                                  nn.ReLU(),
                                  nn.Linear(hidden_dim, out_channels),
                                  nn.ReLU()
                                 ])
    self.loss = nn.CrossEntropyLoss()
    
    
    def forward(self, graph):
      
      x, edge_index = graph.x, graph.edge_index
      x = self.conv(x)
      x = self.hidden_layers(x)
      x = gnn.global_max_pool(x, graph.batch)
      return self.out_mlp(x)
    
    
    def model_step(self, batch, batch_idx, mode):
        """
        Function to handle training and validation steps.
        """
        y = batch.y
        y_hat = self(batch)
        loss = self.loss(y_hat, y)
        batch_metrics = self.metrics(y_hat, y)
        

        total = len(y)
        logs = {f'{mode}_loss' : loss}
        

        batch_dict = {
            "loss" : loss,
            "log" : logs,            
            "total" : total,            
            "y" : y,
            **batch_metrics
        }
       
        return batch_dict


    def metrics(self, y_hat, y):
      
        correct = y_hat.argmax(dim=1).eq(y).sum().item()
        total = len(y)
        total_positives = y.sum() 
        false_negatives = ((((y-y_hat.argmax(dim=1))  == 1)) * (y == 1)).sum()
        false_positives = total - (false_negatives+correct)
        total_negatives = total - total_positives
        y_probs = F.softmax(y_hat, dim=1)
        f1_score = f1(y_probs, y)

        return {"F1" : f1_score,
                "correct" : correct,
                "y_hat" : y_probs.argmax(dim=1),
                "false_negative" : false_negatives,
                "total_positive" : total_positives,
                "false_positive" : false_positives,
                "total_negative" : total_negatives,
                "y_probs" : y_probs
               }


    def training_step(self, train_batch, batch_idx):
        """
        A single training step for the model.
        """
        return self.model_step(train_batch, batch_idx, 'train')


    def validation_step(self, val_batch, batch_idx):
        """
        A single validation step for the model.
        """        
        return self.model_step(val_batch, batch_idx, 'val')


    def test_step(self, test_batch, batch_idx):
        """
        A single test step for the model.
        """
        return self.model_step(test_batch, batch_idx, 'test')


    def loss_accuracy_logging(self, outputs, mode):
        """
        Used for logging to Tensorboard after each epoch.
        """
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        

        self.logger.experiment.add_scalar(f"{mode} Loss",
                                            avg_loss,
                                            self.current_epoch)
      
        correct = sum([x["correct"] for x in outputs])
        total = sum([x["total"] for x in outputs])
        self.logger.experiment.add_scalar(f"{mode} Accuracy",
                                          correct/total,
                                          self.current_epoch)


    def training_epoch_end(self, outputs):

        self.loss_accuracy_logging(outputs, 'Training')


    def validation_epoch_end(self, outputs):
      
        self.loss_accuracy_logging(outputs, 'Validation')
        avg_f1 = torch.stack([x['F1'] for x in outputs]).mean()
        false_negative_count = torch.stack([x['false_negative'] for x in outputs]).sum()
        total_positve_count = torch.stack([x['total_positive'] for x in outputs]).sum()
        false_positive_count = torch.stack([x['false_positive'] for x in outputs]).sum()
        total_negative_count = torch.stack([x['total_negative'] for x in outputs]).sum()
        y_probs = torch.cat([x['y_probs'] for x in outputs]).view(-1, 2).cuda()
        y = torch.cat([x['y'] for x in outputs])
        true_pos = total_positve_count - false_positive_count
        self.y_probs = torch.cat([y_probs, y], dim=1)

        self.logger.experiment.add_scalar("Micro-F1 Score", avg_f1, self.current_epoch)
        self.logger.experiment.add_scalar("Macro-F1 Score", true_pos/(true_pos+.5*(false_positive_count+false_negative_count)), self.current_epoch)
        self.logger.experiment.add_scalar("False Negative Rate", false_negative_count/total_positve_count, self.current_epoch)
        self.logger.experiment.add_scalar("False Positive Rate", false_positive_count/total_negative_count, self.current_epoch)                    
