import argparse


class Hparams:
    parser = argparse.ArgumentParser()

    parser.add_argument('--gfdl_dataset_dir',
                        default='/media/dl/Skye_Cui/vit/data/GFDL/meta_data')
    parser.add_argument('--soda_dataset_dir',
                        default='/home/dl/Desktop/ENSO-MC/data/soda_data/')
    parser.add_argument('--godas_dataset_dir',
                        default='/home/dl/Desktop/ENSO-MC/data/godas_data/')
    parser.add_argument('--era5_dataset_dir',
                        default='/home/dl/Desktop/ENSO-MC/data/era5/')

    parser.add_argument('--input_dataset_dir',
                        default='/home/dl/Desktop/ENSO_MC/data_process/')

    # data
    parser.add_argument('--in_seqlen', default=12, type=int)
    parser.add_argument('--out_seqlen', default=18)
    parser.add_argument('--lead_time', default=0)
    parser.add_argument('--rolling_len', default=0)
    parser.add_argument('--width', default=160)
    parser.add_argument('--height', default=80)
    parser.add_argument('--num_predictor', default=4)
    parser.add_argument('--input_variables', default=["sst", "t300", "u10", "v10"])
    parser.add_argument('--num_output', default=4)
    parser.add_argument('--output_variables', default=["sst", "t300", "u10", "v10"])

    # training scheme
    parser.add_argument('--strategy', default='DMS')
    # parser.add_argument('--train_eval_split', default=0.2)
    parser.add_argument('--random_seed', default=42)
    parser.add_argument('--batch_size', default=1, type=int)
    # parser.add_argument('--eval_batch_size', default=128, type=int)
    parser.add_argument('--num_epochs', default=1000, type=int)
    parser.add_argument('--num_epoch_record', default=1, help="Number of step to record checkpoint.")

    parser.add_argument('--ckpt',
                        default='uconvlstm-ckp_2',
                        help="checkpoint file path")
    parser.add_argument('--single_gpu_model_dir',
                        default="/home/dl/Desktop/ENSO_MC/ckpt")
    # parser.add_argument('--multi_gpu_model_dir', default="ckpt/checkpoints_multi")
    parser.add_argument('--lr', default=0.0001, type=float, help="learning rate")
    parser.add_argument('--warmup_steps', default=4000, type=int)
    parser.add_argument('--logdir', default="logs", help="log directory")

    # model
    parser.add_argument('--model', default='uconvlstm')
    parser.add_argument('--kunit', default=16, type=int)

    # test scheme
    parser.add_argument('--test_month_start', default=0)
    parser.add_argument('--delivery_model_dir',
                        default='/home/dl/Desktop/ENSO_MC/ckpt/ten/')
    parser.add_argument('--delivery_model_file', default='uconvlstm-ckp_2')
