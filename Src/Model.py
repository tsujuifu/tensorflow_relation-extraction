from Src.Lib import *

class Model(object):
    def __init__(self, args, size_emb, size_out):
        super().__init__()
        
        self.args = args
        
        tf.compat.v1.reset_default_graph()
        self.sess = tf.compat.v1.Session()
        
        self.inp_sent = tf.compat.v1.placeholder(tf.float32, shape=[None, None, size_emb])
        self.inp_len = tf.compat.v1.placeholder(tf.int32, shape=[None])
        self.gdt = tf.compat.v1.placeholder(tf.float32, shape=[None, size_out])
        
        self.dp = tf.placeholder_with_default(self.args.dropout, shape=())
        
        inp = tf.compat.v1.nn.dropout(self.inp_sent, rate=self.dp)
        cell = tf.compat.v1.nn.rnn_cell.LSTMCell(args.size_hid)
        out_rnn, _ = tf.compat.v1.nn.dynamic_rnn(cell=cell, inputs=inp, sequence_length=self.inp_len, dtype=tf.float32)
        
        out_feat = tf.compat.v1.math.reduce_max(out_rnn, axis=1)
        
        out_fc = tf.contrib.layers.fully_connected(out_feat, size_out**2)
        out_fc = tf.compat.v1.nn.dropout(out_fc, rate=self.dp)
        self.out_fc = tf.contrib.layers.fully_connected(out_feat, size_out, activation_fn=None)
        
        loss = tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(labels=self.gdt, logits=self.out_fc)
        self.loss = tf.compat.v1.math.reduce_mean(loss)
        self.optim = tf.compat.v1.train.AdamOptimizer(args.lr).minimize(self.loss)
        
        self.sess.run(tf.compat.v1.global_variables_initializer())
        self.saver = tf.compat.v1.train.Saver(max_to_keep=10)
    
    def train(self, inp_sent, inp_len, gdt):
        _, loss = self.sess.run([self.optim, self.loss], 
                                {self.inp_sent: inp_sent, self.inp_len: inp_len, self.gdt: gdt})
        
        return loss
    
    def test(self, inp_sent, inp_len):
        out = self.sess.run(self.out_fc, 
                            {self.inp_sent: inp_sent, self.inp_len: inp_len, self.dp: 0.0})
        
        return out
    
    def save(self, filename):
        self.saver.save(self.sess, filename)
    
    def load(self, filename):
        self.saver.restore(self.sess, filename)
        