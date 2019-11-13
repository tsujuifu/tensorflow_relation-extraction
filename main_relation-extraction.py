from Src.Lib import *
from Src.Model import *
from Src.DataLoader import *

def proc_args():
    parser = ArgumentParser()
    
    parser.add_argument('--emb', choices=['glove', 'bert'], required=True)
    
    parser.add_argument('--size_hid', default=256, type=int)
    parser.add_argument('--dropout', default=0.5, type=float)
    
    parser.add_argument('--epoch', default=100, type=int)
    parser.add_argument('--size_batch', default=50, type=int)
    parser.add_argument('--lr', default=0.0001, type=float)
    
    parser.add_argument('--es', default=5, type=int)
    
    args = parser.parse_args()
    args.str = '%s' % (args.emb)
    
    return args

class EarlyStop(object):
    def __init__(self, es):
        super().__init__()
        
        self.acc = 0
        self.cnt = 0
        self.es = es
        
    def step(self, acc):
        if acc>self.acc:
            self.acc = acc
            self.cnt = 0
        else:
            self.cnt += 1
        
        if self.cnt>=self.es:
            return False
        else:
            return True

if __name__=='__main__':
    args = proc_args()
    print(args)
    
    rels = pickle.load(open('Dataset/RELs.pkl', 'rb'))
    
    tr = pickle.load(open('Dataset/tr_%s.pkl' % (args.emb), 'rb'))
    vl = pickle.load(open('Dataset/vl_%s.pkl' % (args.emb), 'rb'))
    ts = pickle.load(open('Dataset/ts_%s.pkl' % (args.emb), 'rb'))
    
    ld_tr = DataLoader(args=args, dat=tr, rels=rels)
    ld_vl = DataLoader(args=args, dat=vl, rels=rels)
    ld_ts = DataLoader(args=args, dat=ts, rels=rels)
    
    model = Model(args=args, size_emb=tr[0][0][1].shape[0], size_out=len(rels))
    ES = EarlyStop(es=args.es)
    
    best = args.epoch
    with tqdm(range(args.epoch), ascii=True) as TQ:
        for e in TQ:
            
            ls_ep = 0
            for _ in range(ld_tr.num_batch):
                inp_sent, inp_len, gdt = ld_tr.next_batch()
        
                ls_bh = model.train(inp_sent, inp_len, gdt)
                TQ.set_postfix(ls_bh='%.3f' % (ls_bh))
                ls_ep += ls_bh
            ls_ep /= ld_tr.num_batch
            
            ac_ep = 0
            for _ in range(ld_vl.num_batch):
                inp_sent, inp_len, gdt = ld_vl.next_batch()
                
                out = model.test(inp_sent, inp_len)
                
                out = np.argmax(out, axis=1)
                gdt = np.argmax(gdt, axis=1)
                
                ac_ep += np.average(out==gdt)
            ac_ep /= ld_vl.num_batch
            
            model.save(filename='Model/model_%s_%d.ckpt' % (args.emb, e+1))
            print('Ep %d: loss=%.4f, acc=%.2f%%' % (e+1, ls_ep, 100*ac_ep))
            
            if ES.step(ac_ep)==False:
                best = (e+1)-args.es
                print('Best: %d' % (best))
                
                break
    
    model.load(filename='Model/model_%s_%d.ckpt' % (args.emb, best))
    ac_ep = 0
    for _ in range(ld_ts.num_batch):
        inp_sent, inp_len, gdt = ld_ts.next_batch()
        
        out = model.test(inp_sent, inp_len)
        
        out = np.argmax(out, axis=1)
        gdt = np.argmax(gdt, axis=1)
        
        ac_ep += np.average(out==gdt)
    ac_ep /= ld_ts.num_batch
    print('Test: %.3f%%' % (100*ac_ep))
    
