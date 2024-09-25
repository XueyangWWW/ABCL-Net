import numpy as np
import os
import sys
import ntpath
import time
from . import util, html
from subprocess import Popen, PIPE

if sys.version_info[0] == 2:
    VisdomExceptionBase = Exception
else:
    VisdomExceptionBase = ConnectionError


def save_images(webpage, visuals, image_path, aspect_ratio=1.0, width=256):

    image_dir = webpage.get_image_dir()
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]

    webpage.add_header(name)
    ims, txts, links = [], [], []

    for label, im_data in visuals.items():
        im = util.tensor2im(im_data)
        image_name = '%s/%s.nii.gz' % (label, name)
        os.makedirs(os.path.join(image_dir, label), exist_ok=True)
        save_path = os.path.join(image_dir, image_name)
        util.save_image(im, save_path, aspect_ratio=aspect_ratio)
        ims.append(image_name)
        txts.append(label)
        links.append(image_name)
    webpage.add_images(ims, txts, links, width=width)


class Visualizer():
    """This class includes several functions that can display/save images and print/save logging information.

    It uses a Python library 'visdom' for display, and a Python library 'dominate' (wrapped in 'HTML') for creating HTML files with images.
    """

    def __init__(self, opt):

        self.opt = opt  # cache the option
        if opt.display_id is None:
            self.display_id = np.random.randint(100000) * 10  # just a random display id
        else:
            self.display_id = opt.display_id
        self.use_html = opt.isTrain and not opt.no_html
        self.win_size = opt.display_winsize
        self.name = opt.name
        self.port = opt.display_port
        self.saved = False
        if self.display_id > 0:  # connect to a visdom server given <display_port> and <display_server>
            import visdom
            self.plot_data = {}
            self.ncols = opt.display_ncols
            if "tensorboard_base_url" not in os.environ:
                self.vis = visdom.Visdom(server=opt.display_server, port=opt.display_port, env=opt.display_env)
            else:
                self.vis = visdom.Visdom(port=2004,
                                         base_url=os.environ['tensorboard_base_url'] + '/visdom')
            if not self.vis.check_connection():
                self.create_visdom_connections()

        if self.use_html:  # create an HTML object at <checkpoints_dir>/web/; images will be saved under <checkpoints_dir>/web/images/
            self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('create web directory %s...' % self.web_dir)
            util.mkdirs([self.web_dir, self.img_dir])
        # create a logging file to store training losses
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        self.plot_loss = os.path.join(opt.checkpoints_dir, opt.name, 'web')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def reset(self):
        """Reset the self.saved status"""
        self.saved = False

    def create_visdom_connections(self):
        """If the program could not connect to Visdom server, this function will start a new server at port < self.port > """
        cmd = sys.executable + ' -m visdom.server -p %d &>/dev/null &' % self.port
        print('\n\nCould not connect to Visdom server. \n Trying to start a server....')
        print('Command: %s' % cmd)
        Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)

    def display_current_results(self, visuals, epoch, save_result):


        if self.use_html and (save_result or not self.saved):  # save images to an HTML file if they haven't been saved.
            self.saved = True
            # save images to the disk
            for label, image in visuals.items():


                if str(label) == 'seg_fake_B' or str(label) == 'seg_real_A':
                    image_numpy = util.tensor2seg_output_test(image)
                else:
                    image_numpy = util.tensor2im(image)

                img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.nii.gz' % (epoch, label))
                util.save_image(image_numpy, img_path)

            # update website
            webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, refresh=0)
            for n in range(epoch, 0, -1):
                webpage.add_header('epoch [%d]' % n)
                ims, txts, links = [], [], []

                for label, image_numpy in visuals.items():
                    image_numpy = util.tensor2im(image)
                    img_path = 'epoch%.3d_%s.png' % (n, label)
                    ims.append(img_path)
                    txts.append(label)
                    links.append(img_path)
                webpage.add_images(ims, txts, links, width=self.win_size)
            webpage.save()

    def plot_current_losses(self, epoch, loss):
        from torch.utils.tensorboard import SummaryWriter
        summaryWriter = SummaryWriter(self.plot_loss)
        summaryWriter.add_scalar("losses", loss, epoch)

    # losses: same format as |losses| of plot_current_losses
    def print_current_losses(self, epoch, iters, losses, metrics, t_comp):

        message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, iters, t_comp)
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)
        message += '\n'
        for m, n in metrics.items():
            message += '%s: %.3f ' % (m, n)


        print(message)  # print the message
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message

    def mkdir(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    def save_seg_images_to_dir(self, save_dir_seg, save_dir_syn_t2, visuals, image_path):
        short_path = []
        name = []

        for i in range(len(image_path)):
            short_path.append(ntpath.basename(image_path[i]))

            name.append(short_path[i].replace('_T1_', '_'))

        for label, image in visuals.items():

            image_numpy = image
            for j in range(image_numpy.shape[0]):
                if str(label) == 'fake_B':
                    image_name = name[j].replace('ths_', 'ths_T2_')
                    save_path_syn_t2 = os.path.join(save_dir_syn_t2, image_name)
                    util.save_image(image_numpy[j], save_path_syn_t2)
                if str(label) == 'seg_fake_B':
                    image_name = name[j].replace('ths_', 'ths_Seg_')
                    save_path_seg = os.path.join(save_dir_seg, image_name)
                    util.save_image(image_numpy[j], save_path_seg)