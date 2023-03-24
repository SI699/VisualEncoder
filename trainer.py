import wandb
import torch

import utils

from tqdm import tqdm


class Trainer:
    # AMP may hurt the precision for reconstructing the image, not implementing here for code simplicity
    def __init__(self, model, optimizer, scheduler, train_loader, val_loader,
                 test_loader, loss_fn, metrics, logger, configs):
        self.model = model
        if configs['cuda']:
            self.model.cuda()
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.loss_fn = loss_fn
        self.metrics = metrics
        self.configs = configs
        self.logger = logger

    def train_one_epoch(self):
        self.model.train()

        avg_metrics_dict = {
            metric: utils.AverageMeter()
            for metric in self.metrics
        }
        avg_loss = utils.AverageMeter()

        with tqdm(total=len(self.train_loader)) as t:
            for i, img in enumerate(self.train_loader):
                if self.configs['cuda']:
                    img = img.cuda()

                self.optimizer.zero_grad()
                output = self.model(img)
                loss = self.loss_fn(output, img)
                loss.backward()
                self.optimizer.step()

                if i % self.configs['save_summary_steps'] == 0:
                    summary_batch = {}
                    summary_batch['targer_hanzi'] = utils.make_grid(
                        img, 'target hanzi')
                    summary_batch['generated_hanzi'] = utils.make_grid(
                        output, 'generated hanzi')
                    output = output.data.cpu().numpy()
                    img = img.cpu().data.numpy()

                    for metric in self.metrics:
                        val = self.metrics[metric](output, img)
                        summary_batch[metric] = val
                        avg_metrics_dict[metric].update(val, img.shape[0])
                    summary_batch['loss'] = loss.item()

                    if len(self.train_loader
                           ) - i < self.configs['save_summary_steps']:
                        wandb.log({'train': summary_batch}, commit=False)
                    else:
                        wandb.log({'train': summary_batch})

                avg_loss.update(loss.item(), img.shape[0])
                t.set_postfix(loss=avg_loss, **avg_metrics_dict)
                t.update()

        avg_metrics_dict['loss'] = avg_loss()
        self.log_metrics(avg_metrics_dict, 'train')

    def eval(self, mode='validate'):
        # TODO: enable upload image to wandb
        self.model.eval()

        avg_loss = utils.AverageMeter()
        avg_metrics_dict = {
            metric: utils.AverageMeter()
            for metric in self.metrics
        }

        data_loader = self.val_loader if mode == 'validate' else self.test_loader

        summary = {}
        with torch.no_grad():
            for i, img in tqdm(enumerate(data_loader)):
                if self.configs['cuda']:
                    img = img.cuda()

                output = self.model(img)
                loss = self.loss_fn(output, img)
                avg_loss.update(loss.item(), img.shape[0])

                if i == 0:
                    summary['targer_hanzi'] = utils.make_grid(
                        img, 'target hanzi')
                    summary['generated_hanzi'] = utils.make_grid(
                        output, 'generated hanzi')

                output = output.cpu().numpy()
                img = img.cpu().numpy()

                for metric in self.metrics:
                    avg_metrics_dict[metric].update(
                        self.metrics[metric](output, img), img.shape[0])

            for metric in self.metrics:
                summary[metric] = avg_metrics_dict[metric]()
            summary['loss'] = avg_loss()
            wandb.log({mode: summary})
            summary.pop('targer_hanzi')
            summary.pop('generated_hanzi')
            self.log_metrics(summary, mode)
            return summary

    def log_metrics(self, metrics_dict, mode='train'):
        metrics_string = " ; ".join("{}: {}".format(k, str(v))
                                    for k, v in metrics_dict.items())
        self.log(f"***** {mode.capitalize()} metrics: " + metrics_string)

    def log(self, msg):
        if self.logger is not None:
            self.logger.info(msg)
        else:
            print(msg)

    def train(self, restore_path=None):
        start_epoch = 0
        if restore_path is not None:
            start_epoch = utils.load_checkpoint(restore_path, self.model,
                                                self.optimizer, self.scheduler)
        best_val_metric = 0

        for epoch in range(start_epoch, self.configs['epochs']):
            print('epoch: ', epoch)
            self.train_one_epoch()
            val_metrics = self.eval('validate')
            if self.scheduler is not None:
                self.scheduler.step()
            is_best = val_metrics[
                self.configs['eval_metric_name']] > best_val_metric
            utils.save_checkpoint(
                {
                    'epoch': epoch + 1,
                    'state_dict': self.model.state_dict(),
                    'optim_dict': self.optimizer.state_dict(),
                    'scheduler_dict': self.scheduler.state_dict(),
                },
                is_best=is_best,
                checkpoint=self.configs['save_dir'])
            if is_best:
                self.log(
                    f"***** New best {self.configs['eval_metric_name']}: {val_metrics[self.configs['eval_metric_name']]} *****"
                )
                best_val_metric = val_metrics[self.configs['eval_metric_name']]
                best_json_path = self.configs[
                    'save_dir'] / "best_val_metrics.json"
                utils.save_dict_to_json(val_metrics, best_json_path)
