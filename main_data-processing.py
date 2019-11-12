from Src.Lib import *

def proc_args():
    parser = ArgumentParser()
    
    parser.add_argument('--emb', choices=['glove', 'bert'], required=True)
    
    args = parser.parse_args()
    args.str = '%s' % (args.emb)
    
    return args

def process_rel(filename):
    dat = pickle.load(open(filename, 'rb'))
    
    ret = dict()
    for d in dat:
        _, rel = d
        
        if not rel in ret:
            ret[rel] = len(ret)
    
    return ret

def process_sent(filename, rels, EMB):
    dat = pickle.load(open(filename, 'rb'))
    
    ret = []
    for d in tqdm(dat, ascii=True):
        sent, rel = d
        
        sent = Sentence(sent, use_tokenizer=True)
        EMB.embed(sent)
        
        ret.append([[w.text, w.embedding.data.cpu().numpy()] for w in sent]+[rels[rel]])
    
    return ret

if __name__=='__main__':
    args = proc_args()
    print(args)
    
    import flair
    from flair.data import Sentence
    if args.emb=='glove':
        from flair.embeddings import WordEmbeddings
        EMB = WordEmbeddings('glove')
        
    elif args.emb=='bert':
        from flair.embeddings import BertEmbeddings
        EMB = embedding = BertEmbeddings()
    
    
    rels = process_rel('Dataset/tr.pkl')
    pickle.dump(rels, open('Dataset/RELs.pkl', 'wb'))
    
    tr = process_sent('Dataset/tr.pkl', rels, EMB)
    pickle.dump(tr, open('Dataset/tr_%s.pkl' % (args.emb), 'wb'))
    vl = process_sent('Dataset/vl.pkl', rels, EMB)
    pickle.dump(vl, open('Dataset/vl_%s.pkl' % (args.emb), 'wb'))
    ts = process_sent('Dataset/ts.pkl', rels, EMB)
    pickle.dump(ts, open('Dataset/ts_%s.pkl' % (args.emb), 'wb'))
    