from .base_options import BaseOptions


class TestOptions(BaseOptions):

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options

        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        parser.add_argument('--which_epoch_S', type=str, default='Best_dice',
                            help='which epoch to load for S set to latest to use latest cached model')
        parser.add_argument('--which_epoch_G', type=str, default='Best_dice',
                            help='which epoch to load for G set to latest to use latest cached model')

        parser.add_argument('--test_A_dir', type=str, default='./', help='INPUT PATH for test T1')
        parser.add_argument('--test_B_dir', type=str, default='./', help='INPUT PATH for test T2')
        parser.add_argument('--test_0_dir', type=str, default='./', help='INPUT PATH for test T2')
        parser.add_argument('--test_seg_dir', type=str, default='./', help='INPUT PATH for test seg')

        parser.add_argument('--test_seg_output_dir', type=str,
                            default='./',help='OUTPUT PATH for test seg, save test segmentation output results')
        parser.add_argument('--model', type=str, default='gen2seg_model_test')
        parser.add_argument('--dataset_mode', type=str, default='gen2seg_test')

        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.add_argument('--num_test', type=int, default=float("inf"), help='how many test images to run')

        parser.set_defaults(load_size=parser.get_default('crop_size'))

        self.isTrain = False
        return parser
