from model.GCN import GCN
from model.SAGE import SAGE

def get_model(args, num_feats, num_cls):
    if args.model == 'GCN':
        model = GCN(in_dim = num_feats,
                    hid_dim = args.hidden_dim,
                    n_cls = num_cls,
                    num_layers = args.num_layers,
                    dropout = args.dropout,
                    jk = args.jk,
                    norm = args.normalize,
                    conv_norm = args.conv_norm).cuda()
    elif args.model == 'SAGE':
        model = SAGE(in_dim=num_feats,
                     hid_dim=args.hidden_dim,
                     n_cls=num_cls,
                     num_layers=args.num_layers,
                     dropout=args.dropout,
                     norm = args.normalize,
                     proj = args.proj).cuda()
    return model