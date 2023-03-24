import configargparse
import torch
import wandb
import utils
import json

from pathlib2 import Path
from torch.utils.data import DataLoader
from torch import optim

from dataset import HanziDataset
from trainer import Trainer
from utils import set_all_random_seed, set_logger
from model import create_AutoEncoder
from metrics import ssim_score, psnr_score


# TODO enable define model by args
def set_parse():
    config_file_path = './config/config.yml'
    parser = configargparse.ArgParser(
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        default_config_files=[config_file_path])
    parser.add('-c',
               '--config',
               is_config_file=True,
               default=config_file_path,
               help='config file path')
    parser.add_argument('--batch_size',
                        type=int,
                        required=True,
                        help='The batch size of the training')
    parser.add_argument('--epochs',
                        type=int,
                        required=True,
                        help='The epochs of the training')
    parser.add_argument('--save_dir',
                        type=str,
                        required=True,
                        help='The directory to save this experiment')
    parser.add_argument('--wrap_size',
                        type=int,
                        required=True,
                        help='The size of the image to be wrapped')
    parser.add_argument('--wandb_project',
                        type=str,
                        required=True,
                        help='The project name of wandb')
    parser.add_argument('--seed',
                        type=int,
                        required=True,
                        help='The seed of the random number generator')
    parser.add_argument('--learning_rate',
                        type=float,
                        required=True,
                        help='The learning rate')
    parser.add_argument('--weight_path',
                        type=str,
                        help='The path of saved weights')
    parser.add_argument('--weight_decay',
                        type=float,
                        required=True,
                        help='The weight decay factor of the regressor')
    parser.add_argument(
        '--save_summary_steps',
        type=int,
        required=True,
        help='The number of gap steps to save summary during training')
    parser.add_argument(
        '--eval_metric_name',
        type=str,
        required=True,
        help='The name of the metric to be evaluated for validation and test')
    parser.add_argument('--data_dir',
                        type=str,
                        required=True,
                        help='The directory of the dataset')
    parser.add_argument(
        '--restore_path',
        type=str,
        help='the path to the saved checkpoint file for restore training')
    parser.add_argument('--scheduler_config',
                        type=eval,
                        help='the config of the scheduler')
    parser.add('--model', required=True, help='The model to be used', type=eval)
    parser.add('--font', required=True, help='The font to be used', type=str)

    return parser


def create_configs(args):
    wandb_config = {
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'wrap_size': args.wrap_size,
        'seed': args.seed,
        'eval_metric_name': args.eval_metric_name,
        'scheduler_config': args.scheduler_config,
        'model': args.model,
        'font': args.font,
    }

    trainer_config = {
        'cuda': torch.cuda.is_available(),
        'epochs': args.epochs,
        'save_summary_steps': args.save_summary_steps,
        'eval_metric_name': args.eval_metric_name,
        'save_dir': Path(args.save_dir) / args.font,
    }

    return wandb_config, trainer_config


if __name__ == '__main__':
    parser = set_parse()
    args = parser.parse_args()
    wandb_config, trainer_config = create_configs(args)
    #TODO: use tensorboard to save model visualization
    #tb_writer = SummaryWriter(log_dir=args.save_dir)

    set_all_random_seed(args.seed)
    logger = set_logger(Path(args.save_dir) / 'experiment.log')

    wandb_resume = True if args.restore_path is not None else False

    # init wandb for logging
    wandb.init(project=args.wandb_project, resume=wandb_resume)
    wandb.config.update(wandb_config)

    model = create_AutoEncoder(args.model)
    optimizer = optim.Adam(model.parameters(),
                           weight_decay=args.weight_decay,
                           lr=args.learning_rate)

    data_dir = Path(args.data_dir)

    train_data = HanziDataset(data_dir, 'train.csv', args.wrap_size, args.font)
    train_dataloader = DataLoader(train_data,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=4,
                                  pin_memory=True)

    val_data = HanziDataset(data_dir, 'val.csv', args.wrap_size)
    val_dataloader = DataLoader(val_data,
                                batch_size=args.batch_size,
                                shuffle=False,
                                num_workers=8,
                                pin_memory=True)

    test_data = HanziDataset(data_dir, 'test.csv', args.wrap_size)
    test_dataloader = DataLoader(test_data,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 num_workers=8,
                                 pin_memory=True)

    metrics = {'ssim': ssim_score, 'psnr': psnr_score}
    mse_loss = torch.nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.__dict__[
        args.scheduler_config['type']](optimizer,
                                       **args.scheduler_config['args'])
    trainer = Trainer(
        model,
        optimizer,
        scheduler,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        mse_loss,
        metrics,
        logger,
        trainer_config,
    )

    trainer.train(restore_path=args.restore_path)
    utils.load_best_checkpoint(trainer_config['save_dir'], trainer.model)
    trainer.eval(mode='test')

    wandb.finish()
