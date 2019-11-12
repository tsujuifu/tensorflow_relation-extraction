from Src.Lib import *

def proc_args():
    parser = ArgumentParser()
    
    parser.add_argument('--emb', choices=['glove', 'bert'], required=True)
    
    parser.add_argument('--epoch', default=100, type=int)
    parser.add_argument('--batch_size', default=50, type=int)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--decay', default=0.00005, type=float)
    
    args = parser.parse_args()
    args.str = '%s' % (args.emb)
    
    return args

if __name__=='__main__':
    args = proc_args()
    print(args)
    