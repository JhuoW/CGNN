from model.GCN import GCN
from model.SAGE import SAGE
from model.MixHop import MixHop
from model.GPRGNN import GPRGNN
from model.GCNII import GCNII
from model.CGNN import CGNN

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
    elif args.model == 'MixHop':
        model = MixHop(in_channels=num_feats, hidden_channels=args.hidden_dim, out_channels=num_cls, num_layers=args.num_layers, dropout=args.dropout, hops = args.hops).cuda()
    elif args.model == 'GPRGNN':
        model = GPRGNN(num_features=num_feats,
                       hidden_dim=args.hidden_dim,
                       num_classes=num_cls,
                       K = args.K,
                       beta = args.beta,
                       init = args.init,
                       Gamma = args.Gamma,
                       dprate = args.dprate,
                       dropout = args.dropout).cuda()
    elif args.model == 'GCNII':
        model = GCNII(in_channels=num_feats,
                      hid_channels=args.hidden_dim,
                      out_channels=num_cls,
                      num_layers=args.num_layers,
                      alpha=args.alpha,
                      theta=args.theta,
                      shared_weights=args.shared_weights,
                      dropout=args.dropout).cuda()
    elif args.model == 'CGNN':
        model = CGNN(in_dim = num_feats,
                    hid_dim = args.hidden_dim,
                    n_cls = num_cls,
                    num_layers = args.num_layers,
                    dropout = args.dropout,
                    jk = args.jk,
                    norm = args.normalize,
                    conv_norm = args.conv_norm).cuda()
    return model