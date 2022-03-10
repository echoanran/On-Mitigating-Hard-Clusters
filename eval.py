import sys
import yaml
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import argparse
from feeders import build_dataset
from networks import build_model
from networks import losses
from utils import yaml_config_hook, knns2ordered_nbrs
from utils.clustering import Clustering

os.environ['OMP_NUM_THREADS'] = '4'
os.environ['CUDA_VISIBLE_DEVICES'] = '2'


class Eval():
    def __init__(self, parent_parser):

        self.parent_parser = parent_parser
        self.load_args()
        self.init_environment()
        self.device()
        self.load_data()
        self.load_model()

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
        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed_all(self.args.seed)
        torch.cuda.manual_seed(self.args.seed)
        np.random.seed(self.args.seed)
        random.seed(self.args.seed)

    def device(self):
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_data(self):
        # prepare training data

        if self.args.mode == 'train':
            train_dataset = build_dataset(self.args.dataset_name,
                                          self.args.dataset_args)

            self.data_loader_train = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=self.args.batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=self.args.workers,
            )

        if self.args.mode == 'test':
            test_dataset = build_dataset(self.args.test_dataset_name,
                                         self.args.test_dataset_args)

            self.data_loader_test = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=self.args.test_batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=self.args.workers,
            )

    def load_model(self):
        # initialize model
        self.model = build_model(self.args.model_name, self.args.model_args)
        self.model = self.model.to(self.dev)

        model_fp = os.path.join(self.args.work_dir, "checkpoint.tar")
        checkpoint = torch.load(model_fp)
        self.model.load_state_dict(checkpoint['net'])

    def inference(self, mode='test'):
        self.model.eval()
        features = []
        labels = []
        dists = []
        if mode == 'train':
            data_loader = self.data_loader_train
        else:
            data_loader = self.data_loader_test

        for step, (_, label, feature) in enumerate(data_loader):

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
                    f"Step [{step}/{len(self.data_loader_test)}]\t Processing features for {mode} data..."
                )
        labels = np.array(labels)
        self.dists = np.array(dists)

        if not os.path.exists(os.path.join(self.args.work_dir, 'inference')):
            os.makedirs(os.path.join(self.args.work_dir, 'inference'))

        self.labels = labels

        return features, labels

    def clustering(self, mode='test'):

        if self.args.recompute_dist:

            self.inference(mode='test')

            gt = self.labels[:, 0]

            if mode == 'train':
                knns = np.load(self.args.train_knn_path)['data']
                _, knn = knns2ordered_nbrs(knns)
            else:
                knns = np.load(self.args.test_knn_path)['data']
                _, knn = knns2ordered_nbrs(knns)

            print("Distance range: [{}, {}]".format(self.dists.min(),
                                                    self.dists.max()))

            new_dist = np.sort(self.dists, axis=1)
            order = np.argsort(self.dists, axis=1)
            first_order = np.arange(order.shape[0])[:, None]
            new_knn = knn[first_order, order]

            np.save(
                os.path.join(
                    self.args.work_dir, 'inference', self.args.prefix + ' ' +
                    self.args.dataset_name + '_knn_' + mode), new_knn)
            np.save(
                os.path.join(
                    self.args.work_dir, 'inference', self.args.prefix + ' ' +
                    self.args.dataset_name + '_dist_' + mode), new_dist)

            np.save(
                os.path.join(
                    self.args.work_dir, 'inference', self.args.prefix + ' ' +
                    self.args.dataset_name + '_labels_' + mode), self.labels)
        else:
            gt = np.load(
                os.path.join(
                    self.args.work_dir, 'inference', self.args.prefix + ' ' +
                    self.args.dataset_name + '_labels_' + mode))[:, 0]

            new_dist = np.load(
                os.path.join(
                    self.args.work_dir, 'inference', self.args.prefix + ' ' +
                    self.args.dataset_name + '_dist_' + mode))
            new_knn = np.load(
                os.path.join(
                    self.args.work_dir, 'inference', self.args.prefix + ' ' +
                    self.args.dataset_name + '_knn_' + mode))

        print("knn shape: {}, dist shape: {}".format(new_knn.shape,
                                                     new_dist.shape))

        print("Distance range: [{}, {}]".format(new_dist.min(),
                                                new_dist.max()))

        # class sim
        if 'class_sim_path' in self.args:
            sim = np.load(self.args.class_sim_path, allow_pickle=True).item()
        else:
            sim = None

        Clustering(new_knn,
                   new_dist,
                   gt,
                   self.args.density_args,
                   self.args.cluster_args,
                   self.args.work_dir,
                   self.args.prefix,
                   verbose=False,
                   ori_knn=None,
                   ori_dist=None,
                   sim=sim).run()

        print("End of {}, {}".format(self.args.work_dir, self.args.prefix))

        return


if __name__ == "__main__":

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-c',
                        '--config_file',
                        type=str,
                        default='./config/config_eval.yaml',
                        help='config file')

    processor = Eval(parent_parser=parser)

    processor.clustering(processor.args.mode)
