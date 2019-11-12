from Src.Lib import *

class DataLoader(object):
    def __init__(self, args, dat, rels):
        super().__init__()
        
        self.args = args
        self.dat = dat
        self.rels = rels
        
        RD.shuffle(self.dat)
        self.idx = 0
        self.num_batch = len(self.dat)//self.args.size_batch
    
    def next_batch(self):
        if self.idx+self.args.size_batch>len(self.dat):
            RD.shuffle(self.dat)
            self.idx = 0
        
        mxl = 0
        for i in range(self.idx, self.idx+self.args.size_batch):
            mxl = max(mxl, len(self.dat[i]))
        
        inp_sent = np.zeros((self.args.size_batch, mxl, self.dat[0][0][1].shape[0]), np.float32)
        inp_len = np.zeros((self.args.size_batch, ), np.int32)
        gdt = np.zeros((self.args.size_batch, len(self.rels)), np.float32)
        
        for i in range(self.idx, self.idx+self.args.size_batch):
            l = len(self.dat[i])-1
            
            for j in range(l):
                inp_sent[i-self.idx, j] = self.dat[i][j][1]
            inp_len[i-self.idx] = l
            gdt[i-self.idx][self.dat[i][-1]] = 1
        
        self.idx += self.args.size_batch
        
        return inp_sent, inp_len, gdt
        