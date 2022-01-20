from pathlib import Path
from pytorch_lightning import LightningModule, Trainer, loggers
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.functional import cross_entropy
import torch
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1
from argparse import ArgumentParser

from data_processors import process_esc50_to_openl3, ESC50OpenL3Reader


class OpenL3Classifier(LightningModule):

    def __init__(self, in_channels, out_channels, learning_rate, batch_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(256, out_channels),
        )
        self.learning_rate = learning_rate

        metrics = MetricCollection([
            Accuracy(average="micro", num_classes=out_channels),
            Precision(average="micro", num_classes=out_channels),
            Recall(average="micro", num_classes=out_channels),
            F1(average="micro", num_classes=out_channels)
        ])
        self.training_metrics = metrics.clone(postfix="/Train")
        self.validation_metrics = metrics.clone(postfix="/Test")

        self.batch_size = batch_size
        self.save_hyperparameters()

    def forward(self, x):
        x = (x - x.mean()) / x.std()
        x = self.net(x)
        x = nn.Softmax(dim=-1)(x)
        return x

    def _step(self, batch):
        x, y = batch
        y_hat = self(x)
        loss = cross_entropy(y_hat, y.to(torch.float32))
        return loss, y_hat.argmax(axis=1), y.argmax(axis=1)

    def training_step(self, batch, batch_idx):
        train_loss, pred, gt = self._step(batch)
        self.training_metrics(pred, gt)
        return train_loss

    def validation_step(self, batch, batch_idx):
        val_loss, pred, gt = self._step(batch)
        self.validation_metrics(pred, gt)
        return val_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def validation_epoch_end(self, outputs):
        mean_scores = self.validation_metrics.compute()
        mean_scores = {f"{key}/Epoch": value for key, value in mean_scores.items()}
        mean_scores['step'] = self.current_epoch
        self.log_dict(mean_scores, on_step=False, on_epoch=True, batch_size=self.batch_size)

    def training_epoch_end(self, outputs):
        mean_scores = self.training_metrics.compute()
        mean_scores = {f"{key}/Epoch": value for key, value in mean_scores.items()}
        mean_scores['step'] = self.current_epoch
        self.log_dict(mean_scores, on_step=False, on_epoch=True, batch_size=self.batch_size)


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("--esc50-path", required=True, type=Path)
    parser.add_argument("--embedding-size", default=512, type=int)
    parser.add_argument("--experiment-name", type=str, default="experiment")
    parser.add_argument("--test-fold", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=400)
    args = parser.parse_args()

    embedding_path = Path(f"esc50_openl3_{args.embedding_size}")
    embedding_path.mkdir(exist_ok=True)

    if not any(embedding_path.iterdir()):
        process_esc50_to_openl3(args.esc50_path, args.embedding_size)

    train_reader = ESC50OpenL3Reader(args.esc50_path, embedding_path, train=True, test_fold=args.test_fold)
    test_reader = ESC50OpenL3Reader(args.esc50_path, embedding_path, train=False, test_fold=args.test_fold)

    train_loader = DataLoader(dataset=train_reader, batch_size=args.batch_size, num_workers=8, persistent_workers=True,
                              shuffle=True)
    eval_loader = DataLoader(dataset=test_reader, num_workers=8, batch_size=args.batch_size, persistent_workers=True)

    model = OpenL3Classifier(args.embedding_size, 50, 1e-3, args.batch_size)
    trainer = Trainer(max_epochs=150, gpus=1, check_val_every_n_epoch=1,
                      logger=loggers.TensorBoardLogger(save_dir='lightning_logs', name=args.experiment_name,
                                                       version=f"fold{args.test_fold}"))
    trainer.fit(model, train_loader, eval_loader)
