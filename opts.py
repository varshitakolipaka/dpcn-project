import argparse

def parse_args():
    # Main Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task', type=str, default='nc', choices=['gc', 'nc'], help='Task to run')
    parser.add_argument('-d', '--dataset', type=str, default='MOLT-4', help='Dataset to load')
    parser.add_argument('-n', '--num_samples', type=int, default=10, help='number of samples to be manipulated')
    parser.add_argument('-nt', '--num_test_samples', type=int, default=10, help='number of samples to be manipulated')
    parser.add_argument('-m', '--manip', action='store_true', help='Use the manipulated dataset')
    parser.add_argument('-p', '--poison', action='store_true', help='Use the poisoned dataset')
    parser.add_argument('-r', '--retrain', action='store_true', help='Run retrain from scratch')
    parser.add_argument('-ssd','--ssd', action='store_true', help='Run SSD')
    parser.add_argument('-scrub','--scrub', action='store_true', help='Run SCRUB')
    parser.add_argument('--save_model', action='store_true', help='For Saving the current Model')
    parser.add_argument('--load_model', action='store_true', help='For Loading the Model')
    parser.add_argument('-cp','--corrective_percentage_size', type=float, default=100, help='Percentage of samples to be known to the unlearning method')
    parser.add_argument('--replace', action='store_true', help='Replace the manipulated samples with new samples in retraining', default=False)
    parser.add_argument('--model', type=str, default='GCN', choices=['GCN', 'GAT', 'GIN'], help='Model to use')
    parser.add_argument('-pa', '--poison_all', action='store_true', help='Poison the entire dataset', default=False)
    parser.add_argument('--hidden_dim', type=int, default=16, help='Hidden dimension of the model')
    
    # parser.add_argument('--dataset', type=str, default='CIFAR10', choices=['CIFAR10','CIFAR100','PCAM', 'LFWPeople', 'CelebA', 'DermNet', 'Pneumonia'])
    # parser.add_argument('--model', type=str, default='resnet9', choices=['resnet9', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnetwide28x10', 'vitb16'])
    parser.add_argument('--dataset_method', type=str, default='labelrandom', choices=['randomlabelswap', 'interclasslabelswap', 'poisoning'], help='Number of Classes')
    parser.add_argument('--unlearn_method', type=str, default='SSD', choices=['Naive', 'EU', 'CF', 'Scrub', 'BadT', 'SSD'], help='Method for unlearning')
    parser.add_argument('--num_classes', type=int, default=7, choices=[2, 10, 100], help='Number of Classes')
    parser.add_argument('--forget_set_size', type=int, default=500, help='Number of samples to be manipulated')
    parser.add_argument('--patch_size', type=int, default=3, help='Creates a patch of size patch_size x patch_size for poisoning at bottom right corner of image')
    parser.add_argument('--deletion_size', type=int, default=None, help='Number of samples to be deleted')

    # Method Specific Params
    parser.add_argument('--factor', type=float, default=0.1, help='Magnitude to decrease weights')
    parser.add_argument('--kd_T', type=float, default=4, help='Knowledge distilation temperature for SCRUB')
    parser.add_argument('--alpha', type=float, default=1, help='KL from og_model constant for SCRUB, higher incentivizes closeness to ogmodel')
    parser.add_argument('--msteps', type=int, default=15, help='Maximization steps on forget set for SCRUB')
    parser.add_argument('--SSDdampening', type=float, default=1.0, help='SSD: lambda aka dampening constant, lower leads to more forgetting')
    parser.add_argument('--SSDselectwt', type=float, default=10.0, help='SSD: alpha aka selection weight, lower leads to more forgetting')
    parser.add_argument('--rsteps', type=int, default=800, help='InfRe when to stop retain set gradient descent')
    parser.add_argument('--ascLRscale', type=float, default=1.0, help='AL/InfRe: scaling of lr to use for gradient ascent')
    
    # Optimizer Params
    parser.add_argument('--batch_size', type=int, default=512, help='input batch size for training (default: 128)')
    parser.add_argument('--pretrain_iters', type=int, default=300, help='number of epochs to train (default: 31)')
    parser.add_argument('--unlearn_iters', type=int, default=50, help='number of epochs to train (default: 31)')
    parser.add_argument('--unlearn_lr', type=float, default=0.015, help='learning rate (default: 0.025)')
    parser.add_argument('--pretrain_lr', type=float, default=0.0001, help='learning rate (default: 0.025)')
    parser.add_argument('--wd', type=float, default=0.0005, help='learning rate (default: 0.01)')
    
    # Defaults
    parser.add_argument('--data_dir', type=str, default='../data/')
    parser.add_argument('--save_dir', type=str, default='../logs/')
    parser.add_argument('--exp_name', type=str, default='unlearn')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    return args