import sys
import yaml
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import argparse
from feeders import build_dataset
from networks import build_model
from networks import losses
from utils import yaml_config_hook, save_model, knns2ordered_nbrs
from utils.clustering import Clustering

from tensorboardX import SummaryWriter

os.environ['OMP_NUM_THREADS'] = '4'
os.environ['CUDA_VISIBLE_DEVICES'] = '7'


class Run():
    def __init__(self, parent_parser):

        self.parent_parser = parent_parser
        self.load_args()
        self.init_environment()
        self.device()
        self.load_data()
        self.load_model()
        self.load_criterion()

    def load_args(self):
        parser = argparse.ArgumentParser(add_help=True,
                                         parents=[self.parent_parser],
                                         description='Run Parser')
        self.args = parser.parse_args()
        if not os.path.exists(self.args.config_file):
            print(
                "Error: config file do not exist, please provide config file path via -c."
            )
            raise ValueError()
        config = yaml_config_hook(self.args.config_file)
        for k, v in config.items():
            parser.add_argument(f"--{k}", default=v, type=type(v))
        self.args = parser.parse_args()

    def init_environment(self):
        if not os.path.exists(self.args.work_dir):
            os.makedirs(self.args.work_dir)

        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed_all(self.args.seed)
        torch.cuda.manual_seed(self.args.seed)
        np.random.seed(self.args.seed)
        random.seed(self.args.seed)

        # save arg
        self.session_file = '{}/config.yaml'.format(self.args.work_dir)
        arg_dict = vars(self.args)
        with open(self.session_file, 'w') as f:
            f.write('# command line: {}\n\n'.format(' '.join(sys.argv)))
            yaml.dump(arg_dict, f, default_flow_style=False, indent=4)

        self.train_logger = SummaryWriter(log_dir=os.path.join(
            self.args.work_dir, 'train'),
                                          comment='train')
        self.validation_logger = SummaryWriter(log_dir=os.path.join(
            self.args.work_dir, 'validation'),
                                               comment='validation')

        self.best_epoch = 0
        self.best_bcubed_fscore = 0
        self.best_pairwise_fscore = 0

    def device(self):
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_data(self):
        # prepare training data
        train_dataset = build_dataset(self.args.dataset_name,
                                      self.args.dataset_args)

        self.data_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.args.workers,
        )

        self.data_loader_train = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.args.workers,
        )

    def load_model(self):
        # initialize model
        self.model = build_model(self.args.model_name, self.args.model_args)
        self.model = self.model.to(self.dev)
        # optimizer / loss
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.args.learning_rate,
                                          weight_decay=self.args.weight_decay)
        if self.args.reload:
            model_fp = os.path.join(
                self.args.work_dir,
                "checkpoint_{}.tar".format(self.args.start_epoch))
            checkpoint = torch.load(model_fp)
            self.model.load_state_dict(checkpoint['net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.args.start_epoch = checkpoint['epoch'] + 1

    def load_criterion(self):
        # criterion
        self.criterions = dict()
        if 'pair' in self.args.loss_list:
            self.criterions['pair'] = losses.PairLoss(
                **self.args.loss_args).to(self.dev)

    def train(self):
        self.model.train()
        loss_epoch = 0
        labels = []
        dists = []
        for step, (_, label, feature) in enumerate(self.data_loader):

            if self.args.debug:
                if step >= 10:
                    break

            feature = feature.float().to('cuda')
            label = label.float().to('cuda')

            output = self.model(feature)

            feats = output.cpu().detach().numpy()
            for idx, feat in enumerate(feats):
                dists.append(2 - 2 * np.matmul(feat[0], feat.T))

            labels.extend(label.cpu().detach().numpy())

            losses = dict()
            for k, v in self.criterions.items():
                if k == 'pair':
                    losses[k] = self.args.lambdas[k] * \
                        self.criterions[k](output, label)

            self.loss = 0
            for k, v in losses.items():
                self.loss += losses[k]

            self.optimizer.zero_grad()
            self.loss.backward()
            self.optimizer.step()

            if self.args.clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               self.args.clip_max_norm)

            self.optimizer.step()
            if step % self.args.print_interval == 0:
                print_info = f""
                print_info += f"Step [{step}/{len(self.data_loader)}]\t"
                for k, v in losses.items():
                    print_info += f"{k}_loss: {losses[k].item()}, "
                print_info += f"total_loss: {self.loss.item()}"
                print(print_info)
            loss_epoch += self.loss.item()

        self.train_logger.add_scalar('loss', loss_epoch, self.epoch)
        for k, v in losses.items():
            self.train_logger.add_scalar(k, losses[k], self.epoch)

        labels = np.array(labels)
        dists = np.array(dists)

        return loss_epoch

    def inference(self):
        self.model.eval()
        features = []
        labels = []
        dists = []

        data_loader = self.data_loader_train

        for step, (_, label, feature) in enumerate(data_loader):

            if self.args.debug:
                if step >= 10:
                    break

            feature = feature.float().to('cuda')
            label = label.float().to('cuda')

            with torch.no_grad():
                output = self.model(feature)

            batch_size, topk, dim = output.shape
            output = F.sigmoid(output)
            values = output.squeeze()
            dists.extend(1 - values.cpu().detach().numpy())
            labels.extend(label.cpu().detach().numpy())

            if step % self.args.print_interval == 0:
                print(
                    f"Step [{step}/{len(self.data_loader)}]\t Processing features for training data..."
                )
        labels = np.array(labels)
        self.dists = np.array(dists)

        if not os.path.exists(os.path.join(self.args.work_dir, 'inference')):
            os.makedirs(os.path.join(self.args.work_dir, 'inference'))

        self.labels = labels

        return features, labels

    def clustering(self):

        if len(self.labels.shape) >= 2:
            gt = self.labels[:, 0]
        else:
            gt = self.labels

        knns = np.load(self.args.train_knn_path)['data']
        _, knn = knns2ordered_nbrs(knns)

        # class sim
        if 'class_sim_path' in self.args:
            sim = np.load(self.args.class_sim_path, allow_pickle=True).item()
        else:
            sim = None

        print("Distance range: [{}, {}]".format(self.dists.min(),
                                                self.dists.max()))

        new_dist = np.sort(self.dists, axis=1)
        order = np.argsort(self.dists, axis=1)
        first_order = np.arange(order.shape[0])[:, None]
        new_knn = knn[first_order, order]

        if not os.path.exists(os.path.join(self.args.work_dir, 'inference')):
            os.makedirs(os.path.join(self.args.work_dir, 'inference'))

        np.save(
            os.path.join(
                self.args.work_dir, 'inference', self.args.dataset_name +
                '_knn_' + '_epoch{}'.format(self.epoch)), new_knn)
        np.save(
            os.path.join(
                self.args.work_dir, 'inference', self.args.dataset_name +
                '_dist_' + '_epoch{}'.format(self.epoch)), new_dist)

        print("knn shape: {}, dist shape: {}".format(new_knn.shape,
                                                     new_dist.shape))

        pairwise_fscore, bcubed_fscore = Clustering(new_knn,
                                                    new_dist,
                                                    gt,
                                                    self.args.density_args,
                                                    self.args.cluster_args,
                                                    self.args.work_dir,
                                                    str(self.epoch),
                                                    verbose=False,
                                                    sim=sim).run()

        if self.best_pairwise_fscore < pairwise_fscore:
            self.best_bcubed_fscore = bcubed_fscore
            self.best_pairwise_fscore = pairwise_fscore
            self.best_epoch = self.epoch

            np.savetxt(
                os.path.join(self.args.work_dir, 'results',
                             'best_pairwise_f1.txt'),
                np.array([self.best_pairwise_fscore]))
            np.savetxt(
                os.path.join(self.args.work_dir, 'results',
                             'best_bcubed_f1.txt'),
                np.array([self.best_bcubed_fscore]))
            np.savetxt(
                os.path.join(self.args.work_dir, 'results', 'best_epoch.txt'),
                np.array([self.best_epoch]))

        return

    def start(self):
        for epoch in range(self.args.start_epoch, self.args.epochs):

            print(f"Starting Epoch [{epoch}/{self.args.epochs}]...")
            self.epoch = epoch
            lr = self.optimizer.param_groups[0]["lr"]
            loss_epoch = self.train()

            if epoch % self.args.save_interval == 0:
                save_model(self.args.work_dir, self.model, self.optimizer,
                           epoch)

            if epoch == 0 or epoch % self.args.inference_interval == 0 or epoch == self.args.epochs - 1:
                    self.inference()
                    self.clustering()

            print(f"=======> {self.args.work_dir}")
            print(
                f"Ending Epoch [{epoch}/{self.args.epochs}]\t Loss: {loss_epoch / len(self.data_loader)}"
            )
        save_model(self.args.work_dir, self.model, self.optimizer,
                   self.args.epochs)

        return


if __name__ == "__main__":

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-c',
                        '--config_file',
                        type=str,
                        default='./config/config.yaml',
                        help='config file')

    processor = Run(parent_parser=parser)

    processor.start()
