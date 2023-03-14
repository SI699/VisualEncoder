import configargparse
import torch
import wandb

from pathlib2 import Path
from torch.utils.data import DataLoader
from torch import optim

from dataset import AVADatasetEmp
from trainer import Trainer
from utils import set_all_random_seed, set_logger
from model import create_ASAIAANet

from torch.utils.tensorboard import SummaryWriter


def set_parse():
    config_file_path = './config/config.yml'
    parser = configargparse.ArgParser(
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        default_config_files=[config_file_path])
    parser.add('-c', '--config', is_config_file=True, help='config file path')
    parser.add_argument('--batch_size',
                        type=int,
                        required=True,
                        help='The batch size of the training')
    parser.add_argument('--epochs',
                        type=int,
                        required=True,
                        help='The epochs of the training')
    parser.add_argument('--feature_channels_num',
                        type=int,
                        required=True,
                        help='The number of feature channels')
    parser.add_argument('--feature_h',
                        type=int,
                        required=True,
                        help='The height of feature map')
    parser.add_argument('--feature_w',
                        type=int,
                        required=True,
                        help='The width of feature map')
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
    }

    trainer_config = {
        'cuda': torch.cuda.is_available(),
        'epochs': args.epochs,
        'save_summary_steps': args.save_summary_steps,
        'eval_metric_name': args.eval_metric_name,
        'momentum_D_backbone': args.momentum_D_backbone,
    }

    return wandb_config, trainer_config


if __name__ == '__main__':
    parser = set_parse()
    args = parser.parse_args()
    wandb_config, trainer_config = create_configs(args)
    #tb_writer = SummaryWriter(log_dir=args.save_dir)

    set_all_random_seed(args.seed)
    logger = set_logger(Path(args.save_dir) / 'experiment.log')

    wandb_resume = True if args.restore_path is not None else False

    # init wandb for logging
    wandb.init(project=args.wandb_project, resume=wandb_resume)
    wandb.config.update(wandb_config)

    model = create_ASAIAANet(args)
    optimizer = optim.Adam(model.regressor.parameters(),
                           weight_decay=args.weight_decay,
                           lr=args.learning_rate)

    data_dir = Path(args.data_dir)

    train_data = AVADatasetEmp('train.pickle', data_dir, args.wrap_size)
    train_dataloader = DataLoader(train_data,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=4,
                                  pin_memory=True)

    val_data = AVADatasetEmp('val.pickle', data_dir, args.wrap_size)
    val_dataloader = DataLoader(val_data,
                                batch_size=args.batch_size,
                                shuffle=False,
                                num_workers=8,
                                pin_memory=True)

    test_data = AVADatasetEmp('test.pickle', data_dir, args.wrap_size)
    test_dataloader = DataLoader(test_data,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=8,
                                 pin_memory=True)

    metrics = None
    mse_loss = torch.nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=5,
                                                gamma=0.1)
    trainer = Trainer(
        model,
        optimizer,
        scheduler,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        mse_loss,
        metrics,
        Path(args.save_dir),
        logger,
        trainer_config,
    )

    trainer.train(restore_path=args.restore_path)
    trainer.test()

    wandb.finish()