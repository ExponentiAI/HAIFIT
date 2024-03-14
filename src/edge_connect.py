import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from .dataset import LoadMyDataset
from .models import EdgeModel, Transfer_Model
from .utils import Progbar, create_dir, stitch_images, resize, to_tensor
from .metrics import PSNR, EdgeAccuracy
import pytorch_ssim
import time
import torchvision


class EdgeConnect():
    def __init__(self, config):
        if config.MODEL == 1:
            model_name = 'edge'
        elif config.MODEL == 2:
            model_name = 'transfer'
        elif config.MODEL == 3:
            model_name = 'edge_inpaint'
        elif config.MODEL == 4:
            model_name = 'joint'
        else:
            raise ValueError('MODEL must be in [1, 2, 3, 4]')

        self.debug = False
        self.config = config
        self.model_name = model_name

        self.num_scale = int(np.round(np.log(self.config.img_size_max / self.config.img_size_min)
                                      / np.log(self.config.scale_factor)))
        self.size_list = [int(self.config.img_size_min * self.config.scale_factor ** i) for i in
                          range(self.num_scale + 1)]

        self.edge_model = EdgeModel(config).to(config.DEVICE)
        self.transfer_model = Transfer_Model(config, self.num_scale,
                                             self.config.scale_factor, self.size_list).to(config.DEVICE)

        self.psnr = PSNR(255.0).to(config.DEVICE)
        self.edgeacc = EdgeAccuracy(config.EDGE_THRESHOLD).to(config.DEVICE)

        # test mode
        if self.config.MODE == 2:
            self.test_dataset = LoadMyDataset(edge_path=config.TEST_EDGE, img_path=config.TEST_IMG,
                                              im_size=config.INPUT_SIZE, train=False)
        else:
            self.train_dataset = LoadMyDataset(edge_path=config.TRAIN_EDGE, img_path=config.TRAIN_IMG,
                                               im_size=config.INPUT_SIZE, train=True)
            self.val_dataset = LoadMyDataset(edge_path=config.TRAIN_EDGE, img_path=config.TRAIN_IMG,
                                             im_size=config.INPUT_SIZE, train=True)
            self.val_dataloader = iter(DataLoader(
                dataset=self.val_dataset,
                batch_size=self.config.BATCH_SIZE,
                drop_last=True))

        self.samples_path = os.path.join(config.PATH, 'samples')
        self.results_path = os.path.join(config.PATH, 'results')

        if not os.path.exists(self.samples_path):
            os.makedirs(self.samples_path)
        if not os.path.exists(self.results_path):
            os.makedirs(self.results_path)

        if config.DEBUG is not None and config.DEBUG != 0:
            self.debug = True

        self.log_file = os.path.join(config.PATH, 'log_' + model_name + '.dat')

    def load(self):
        if self.config.MODEL == 1:
            self.edge_model.load()

        elif self.config.MODEL == 2:
            self.transfer_model.load()

        else:
            self.edge_model.load()
            self.transfer_model.load()

    def save(self, stage):
        if self.config.MODEL == 1:
            self.edge_model.save(stage)

        elif self.config.MODEL == 2 or self.config.MODEL == 3:
            self.transfer_model.save(stage)

        else:
            self.edge_model.save(stage)
            self.transfer_model.save(stage)

    def train(self):
        train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.BATCH_SIZE,
            num_workers=0,
            drop_last=True,
            shuffle=True
        )

        model = self.config.MODEL
        max_iteration = int(float((self.config.MAX_ITERS)))
        total = len(self.train_dataset)

        if total == 0:
            print('No training data was provided! Check \'TRAIN_FLIST\' value in the configuration file.')
            return

        dis_optimizer = torch.optim.Adam(self.transfer_model.discriminator.sub_discriminators[0].parameters(),
                                         lr=float(self.config.LR) * float(self.config.D2G_LR),
                                         betas=(self.config.BETA1, self.config.BETA2))
        gen_optimizer = torch.optim.Adam(self.transfer_model.generator.sub_generators[0].parameters(),
                                         lr=float(self.config.LR), betas=(self.config.BETA1, self.config.BETA2))

        self.transfer_model.gen_optimizer = gen_optimizer
        self.transfer_model.dis_optimizer = dis_optimizer

        epoch_list = [self.config.max_epoch * 10 ** i for i in range(self.num_scale + 1)]
        print(epoch_list)

        with open(self.config.matrics_path, 'a+') as file:
            for stage in range(0, self.num_scale + 1):
                print('\n########################################')
                print('########   Training stage [{}]   ########'.format(stage))
                print('########################################\n')

                time_stage_start = time.time()
                file.write('stage: ' + str(stage) + '\n')

                if stage < self.transfer_model.stage:
                    self.transfer_model = self.transfer_model.cuda()
                    dis_optimizer = torch.optim.Adam(self.transfer_model.discriminator.sub_discriminators[
                                                         self.transfer_model.discriminator.current_scale].parameters(),
                                                     5e-4, (0.5, 0.999))
                    gen_optimizer = torch.optim.Adam(self.transfer_model.generator.sub_generators[
                                                         self.transfer_model.generator.current_scale].parameters(),
                                                     5e-4, (0.5, 0.999))
                    self.transfer_model.gen_optimizer = gen_optimizer
                    self.transfer_model.dis_optimizer = dis_optimizer
                    continue

                progbar = Progbar(total, width=20, stateful_metrics=['epoch', 'iter'])
                ssim_record = {'epoch': 0, 'ssim': 0}
                epoch = 0
                keep_training = True
                while keep_training:
                    epoch += 1
                    time_epcoch_start = time.time()
                    ssim_list = []
                    for items in train_loader:
                        self.edge_model.train()
                        self.transfer_model.train()

                        input, target = self.cuda(*items)

                        # inpaint model
                        if model == 2:
                            # train
                            outputs, target, gen_loss, dis_loss, logs = \
                                self.transfer_model.process(input, target, stage)

                            # metrics
                            psnr = self.psnr(self.postprocess(target), self.postprocess(outputs))
                            mae = (torch.sum(torch.abs(target - outputs)) / torch.sum(target)).float()
                            ssim = pytorch_ssim.ssim(target.cpu(), outputs.cpu())
                            ssim_list.append(ssim.item())
                            file.write('epoch: ' + str(epoch) + "  psnr: " + str(psnr.item()) +
                                       "  mae: " + str(mae.item()) + "  ssim: " + str(ssim.item()) + '\n')

                            if len(ssim_list) == len(train_loader):
                                ssim_avg = 0
                                for i in ssim_list:
                                    ssim_avg = ssim_avg + i
                                ssim_avg = ssim_avg / len(ssim_list)
                                file.write('ssim_avg: ' + str(ssim_avg) + '\n')
                                if ssim_avg > ssim_record['ssim']:
                                    ssim_record['epoch'] = epoch
                                    ssim_record['ssim'] = ssim_avg

                            else:
                                if epoch - ssim_record['epoch'] > self.config.early_stopping:
                                    keep_training = False

                            logs.append(('psnr', psnr.item()))
                            logs.append(('mae', mae.item()))
                            logs.append(('ssim', ssim.item()))

                            # backward
                            self.transfer_model.backward(gen_loss, dis_loss)
                            iteration = self.transfer_model.iteration

                        if iteration >= max_iteration:
                            keep_training = False
                            break

                        logs = [("epoch", epoch),
                                ("iter", iteration)] + logs

                        progbar.add(len(target), values=logs if self.config.VERBOSE else [x for x in logs if
                                                                                          not x[0].startswith('l_')])

                        # log model at checkpoints
                        if self.config.LOG_INTERVAL and iteration % self.config.LOG_INTERVAL == 0:
                            self.log(logs)

                        # sample model at checkpoints
                        if self.config.SAMPLE_INTERVAL and iteration % self.config.SAMPLE_INTERVAL == 0:
                            self.sample()

                        # evaluate model at checkpoints
                        if self.config.EVAL_INTERVAL and iteration % self.config.EVAL_INTERVAL == 0:
                            print('\nstart eval...\n')
                            self.eval()

                        # save model at checkpoints
                        if self.config.SAVE_INTERVAL and iteration % self.config.SAVE_INTERVAL == 0:
                            self.save(stage)

                    print('\n========  ssim avg : {}  ========'.format(ssim_avg))

                    time_epoch_end = time.time()
                    file.write('epoch: ' + str(epoch) + 'time' + str(time_epoch_end - time_epcoch_start) + '\n')
                    if epoch >= epoch_list[stage]:
                        break

                if stage == self.num_scale:
                    break

                self.transfer_model.generator.progress()
                self.transfer_model.discriminator.progress()

                for net_idx in range(self.transfer_model.generator.current_scale):
                    for param in self.transfer_model.generator.sub_generators[net_idx].parameters():
                        param.requires_grad = False
                    for param in self.transfer_model.discriminator.sub_discriminators[net_idx].parameters():
                        param.requires_grad = False

                self.transfer_model = self.transfer_model.cuda()
                dis_optimizer = torch.optim.Adam(self.transfer_model.discriminator.sub_discriminators[
                                                     self.transfer_model.discriminator.current_scale].parameters(),
                                                 5e-4, (0.5, 0.999))
                gen_optimizer = torch.optim.Adam(self.transfer_model.generator.sub_generators[
                                                     self.transfer_model.generator.current_scale].parameters(),
                                                 5e-4, (0.5, 0.999))
                self.transfer_model.gen_optimizer = gen_optimizer
                self.transfer_model.dis_optimizer = dis_optimizer

                time_stage_end = time.time()
                file.write('stage: ' + str(stage) + 'time' + str(time_stage_end - time_stage_start) + '\n')
            time_all_end = time.time()
            file.write('time' + str(time_all_end - time_all_end) + '\n')
            print('\nEnd training....')

    def eval(self):
        val_loader = DataLoader(
            dataset=self.val_dataset,
            batch_size=self.config.BATCH_SIZE,
            drop_last=True,
            shuffle=True
        )

        model = self.config.MODEL
        total = len(self.val_dataset)

        self.edge_model.eval()
        self.transfer_model.eval()

        progbar = Progbar(total, width=20, stateful_metrics=['it'])
        iteration = 0

        for items in val_loader:
            iteration += 1
            images, outline, color_domain = self.cuda(*items)

            # inpaint model
            if model == 2:
                # eval
                outputs, gen_loss, dis_loss, logs = self.transfer_model.process(images, outline, color_domain)

                # metrics
                psnr = self.psnr(self.postprocess(images), self.postprocess(outputs))
                mae = (torch.sum(torch.abs(images - outputs)) / torch.sum(images)).float()
                logs.append(('psnr', psnr.item()))
                logs.append(('mae', mae.item()))

            logs = [("it", iteration), ] + logs
            progbar.add(len(images), values=logs)

    def test(self):
        with torch.no_grad():
            self.transfer_model.eval()
            self.transfer_model.to(self.config.DEVICE)

            create_dir(self.results_path)

            test_loader = DataLoader(
                dataset=self.test_dataset,
                batch_size=1,
                shuffle=False
            )

            create_dir(self.results_path)
            for idx, items in enumerate(test_loader):
                name = self.test_dataset.load_name(idx)
                input, target = self.cuda(*items)
                model_name = 'lstm'

                outputs = self.transfer_model(input)[-1]
                print(name)
                psnr = self.psnr(self.postprocess(target), self.postprocess(outputs))
                mae = (torch.sum(torch.abs(target - outputs)) / torch.sum(target)).float()
                ssim = pytorch_ssim.ssim(target.cpu(), outputs.cpu())
                print('psnr: ', psnr.item(), 'mae: ', mae.item(), 'ssim: ', ssim.item())
                torchvision.utils.save_image(outputs, './generate/{}/gen/{}.png'.format(model_name,
                                                                                        str(name.split('.')[0]).zfill(5)))
                torchvision.utils.save_image(target, './generate/{}/org/{}.png'.format(model_name,
                                                                                       str(name.split('.')[0]).zfill(5)))

                print('\nsaving sample ' + name)

            print('\nEnd test....')

    def draw(self, color_domain, edge):
        self.transfer_model.eval()
        size = self.config.INPUT_SIZE
        color_domain = resize(color_domain, size, size, interp='lanczos')
        edge = resize(edge, size, size, interp='lanczos')
        edge[edge <= 69] = 0
        edge[edge > 69] = 255

        color_domain = to_tensor(color_domain)
        edge = to_tensor(edge)

        color_domain, edge = self.cuda(color_domain, edge)

        if self.config.DEBUG:
            print('In model.draw():---> \n color domain size is {}, edges size is {}'.format(color_domain.size(),
                                                                                             edge.size()))
        outputs = self.transfer_model(edge, color_domain)
        outputs = self.postprocess(outputs)[0]
        output = outputs.cpu().numpy().astype(np.uint8).squeeze()
        edge = self.postprocess(edge)[0]
        edge = edge.cpu().numpy().astype(np.uint8).squeeze()

        return output

    def sample(self, it=None):
        # do not sample when validation set is empty
        with torch.no_grad():
            if len(self.val_dataset) == 0:
                return

            self.edge_model.eval()
            self.transfer_model.eval()

            model = self.config.MODEL
            items = next(self.val_dataloader)
            input, target = self.cuda(*items)

            # model
            if model == 2:
                iteration = self.transfer_model.iteration
                outputs = self.transfer_model(input)[-1]

            if it is not None:
                iteration = it

            image_per_row = 2
            if self.config.SAMPLE_SIZE <= 6:
                image_per_row = 1

            images = stitch_images(
                self.postprocess(input),
                self.postprocess(target),
                self.postprocess(outputs),
                img_per_row=image_per_row
            )

            path = os.path.join(self.samples_path, str(iteration).zfill(5) + ".png")
            print('\nsaving sample ' + path)
            images.save(path)

    def log(self, logs):
        with open(self.log_file, 'a') as f:
            f.write('%s\n' % ' '.join([str(item[1]) for item in logs]))

    def cuda(self, *args):
        return (item.to(self.config.DEVICE) for item in args)

    def postprocess(self, img):
        # [0, 1] => [0, 255]
        img = img * 255.0
        img = img.permute(0, 2, 3, 1)
        return img.int()
