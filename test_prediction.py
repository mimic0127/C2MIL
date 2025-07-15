import os
import argparse
import setproctitle
import pandas as pd
setproctitle.setproctitle("ESCA")
def get_params():
    parser = argparse.ArgumentParser(description="All the parameters of this network.")
    parser.add_argument("--cancer_type", type=str, default="ESCA_UNI", help="cancer_type")
    parser.add_argument("--model_name", type=str, default="causal_double", help="choose model to train")
    parser.add_argument("--clusterK", type=int, default=0, help="choose model to train")
    parser.add_argument("--lossver", type=str, default="", help="choose model to train")
    parser.add_argument("--details", type=str, default="UNI_0205_GTprob_adaptivecluster", help="details") #causal10ratio01
    parser.add_argument("--slide_in_feats", type=int, default=1024, help="slide_in_feats")
    parser.add_argument("--fusion_dim", type=int, default=128, help="fusion_dim")
    parser.add_argument("--output_dim", type=int, default=1, help="output_dim")
    parser.add_argument("--slide_size_arg", type=str, choices=["small","big"], default="small", help="slide_size_arg")
    # parser.add_argument("--omic_size_arg", type=str, choices=["small","big"], default="small", help="omic_size_arg")
    parser.add_argument("--lr", type=float, default=1e-4, help="lr")
    parser.add_argument("--seed", type=int, default=8, help="seed")
    parser.add_argument("--epochs", type=int, default=100, help="epochs")
    parser.add_argument("--folds_num", type=int, default=5, help="folds_num")
    parser.add_argument("--batch_size", type=int, default=16, help="batch_size")
    parser.add_argument("--if_adjust_lr", action="store_true", default=True, help="if_adjust_lr")
    parser.add_argument("--dropout", type=float, default=0.25, help="dropout")
    parser.add_argument("--dc_loss_ratio", type=float, default=0.05, help="dc_loss_ratio")
    parser.add_argument("--fusion_loss_ratio", type=float, default=0.005, help="fusion_loss_ratio")
    parser.add_argument("--result_dir", type=str, default="./results_cluster", help="result_dir")
    parser.add_argument("--output_dir", type=str,default = "")
    parser.add_argument("--gpu", type=str, default="0", help="gpu")
    parser.add_argument("--fea_type", type=str, default="UNI")
    args, _ = parser.parse_known_args()
    return args
args = get_params()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
#  nohup python -u train_ESCA.py --gpu 6 --batch_size 16 --seed 8 > log_uni_esca_external.txt &
# 1: GT
# 2: GTprob
# 3: GTprobquanbu
# 4: prob
import copy
import torch
import joblib
import random
import numpy as np
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm 
from utils import *
import sys
sys.path.append('/cache/cenmin/causual_graph/survival')
sys.path.append('/cache/cenmin/causual_graph/survival/RRT_MIL')
from modules import attmil,clam,dsmil,transmil,mean_max,rrt,attmil_ibmil
from model import AMIL_Surv, PatchGCN_Surv, DeepGraphConv_Surv
from model_causal_double import GraphCausal2_Surv
device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')

# %%

def setup_seed(seed):
	import random
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if device.type == 'cuda':
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

def prediction(model,patients,slide_feats_dict,cli_dict,args):
    val_all_hazards = []
    val_all_hazards_c = []
    val_all_surv_time = []
    val_all_censorship = []


    val_patients_hazards = {}
    val_patients_hazards_c = {}
    val_patients_surv_time = {}
    val_patients_censorship = {}
    
    model.eval()
    with torch.no_grad():
        for n_iter, patient_id in tqdm(enumerate(patients)):
            surv_time = cli_dict[patient_id][1]
            censorship = cli_dict[patient_id][0]
            val_all_surv_time.append(surv_time)
            val_all_censorship.append(censorship)
            val_patients_surv_time[patient_id] = surv_time
            val_patients_censorship[patient_id] = censorship
            # if args.model_name in ['PatchGCN','DeepGraphConv']:
                # slide_feats_dict[patient_id].x = slide_feats_dict[patient_id].x[:,:1024]
            # else:
                # slide_feats_dict[patient_id] = slide_feats_dict[patient_id][:,:1024]
            slide_data = slide_feats_dict[patient_id].to(device)
            if args.cancer_type == 'KIRC_UNI':
                thumb = torch.load(f'/cache/cenmin/causual_graph/data/thumbnail_fea/TCGA_KIRC_{args.fea_type}/{patient_id}.pt',map_location='cpu').detach().to(device)
            if args.cancer_type == 'ESCA_UNI':
                thumb = torch.load(f'/cache/cenmin/causual_graph/data/thumbnail_fea/TCGA_ESCA_{args.fea_type}/{patient_id}.pt',map_location='cpu').detach().to(device)
            if args.cancer_type == 'BRCA_UNI':
                thumb = torch.load(f'/cache/cenmin/causual_graph/data/thumbnail_fea/TCGA_BLCA_{args.fea_type}/{patient_id}.pt',map_location='cpu').detach().to(device)
            
            # thumb = torch.load(f'/cache/cenmin/causual_graph/data/thumbnail_fea/TCGA_ESCA_{args.fea_type}/{patient_id}.pt',map_location='cpu').detach().to(device)
            # hazards,h,hazards_c, h_causal,h_remaining,causal_ratio,z_old,z_new = model(slide_data,thumb)
            hazards,_,hazards_c,_,_,_,_,_ = model(slide_data,thumb)

            hazards = hazards.squeeze(0)
            val_all_hazards.append(hazards)
            val_patients_hazards[patient_id] = hazards.detach().cpu().numpy()

            hazards_c = hazards_c.squeeze(0)
            val_all_hazards_c.append(hazards_c)
            val_patients_hazards_c[patient_id] = hazards_c.detach().cpu().numpy()
    
    val_all_hazards = torch.cat(val_all_hazards, dim=0)
    val_all_hazards_c = torch.cat(val_all_hazards_c, dim=0)

            
    val_surv_loss = CoxSurvLoss(hazards=val_all_hazards,
                                time=val_all_surv_time,
                                censorship=val_all_censorship)
    val_surv_loss_c = CoxSurvLoss(hazards=val_all_hazards_c,
                                time=val_all_surv_time,
                                censorship=val_all_censorship)
    
    val_all_loss = val_surv_loss
    val_all_loss = val_all_loss.item()
    val_ci = calculate_ci(val_patients_surv_time, val_patients_hazards_c, val_patients_censorship)
    
    return val_all_loss, val_ci, val_patients_hazards_c

def prediction_norm(model,patients,slide_feats_dict,cli_dict,args):
    val_all_hazards = []
    val_all_surv_time = []
    val_all_censorship = []


    val_patients_hazards = {}
    val_patients_surv_time = {}
    val_patients_censorship = {}
    
    model.eval()
    with torch.no_grad():
        for n_iter, patient_id in tqdm(enumerate(patients)):
            surv_time = cli_dict[patient_id][1]
            censorship = cli_dict[patient_id][0]
            val_all_surv_time.append(surv_time)
            val_all_censorship.append(censorship)
            val_patients_surv_time[patient_id] = surv_time
            val_patients_censorship[patient_id] = censorship
            # if args.model_name in ['PatchGCN','DeepGraphConv']:
                # slide_feats_dict[patient_id].x = slide_feats_dict[patient_id].x[:,:1024]
            # else:
                # slide_feats_dict[patient_id] = slide_feats_dict[patient_id][:,:1024]
            slide_data = slide_feats_dict[patient_id].to(device)
            
            hazards = model(slide_data)
            if len(hazards) == 2:
                hazards = hazards[1]
            hazards = hazards.squeeze(0)
            val_all_hazards.append(hazards)
            val_patients_hazards[patient_id] = hazards.detach().cpu().numpy()
    
    val_all_hazards = torch.cat(val_all_hazards, dim=0)

            
    val_surv_loss = CoxSurvLoss(hazards=val_all_hazards,
                                time=val_all_surv_time,
                                censorship=val_all_censorship)
    
    val_all_loss = val_surv_loss
    val_all_loss = val_all_loss.item()
    val_ci = calculate_ci(val_patients_surv_time, val_patients_hazards, val_patients_censorship)
    
    return val_all_loss, val_ci, val_patients_hazards


def main(args):
    save_dir = args.output_dir
    log_name = os.path.join(save_dir, "test_ci.log")
    
    f_ = open(log_name,'w')
    f_.truncate()

    if args.cancer_type == 'KIRC_UNI':
        cli_dict = joblib.load('/cache/cenmin/causual_graph/data/Feat_result/TCGA_KIRC_input/kirc_sur_and_time.pkl')
        train_val_dict = joblib.load('/cache/cenmin/causual_graph/data/Feat_result/TCGA_KIRC_input/kirc_five_folds.pkl')
    elif args.cancer_type == 'BLCA_UNI':
        cli_dict = joblib.load('/cache/cenmin/causual_graph/data/Feat_result/TCGA_BLCA_input/blca_sur_and_time.pkl')
        train_val_dict = joblib.load('/cache/cenmin/causual_graph/data/Feat_result/TCGA_BLCA_input/blca_five_folds.pkl')
    elif args.cancer_type == 'ESCA_UNI':
        cli_dict = joblib.load('/cache/cenmin/causual_graph/data/Feat_result/TCGA_ESCA_input/esca_sur_and_time.pkl')
        train_val_dict = joblib.load('/cache/cenmin/causual_graph/data/Feat_result/TCGA_ESCA_input/esca_five_folds.pkl')     
    print(f'load {args.cancer_type}...')
    if args.model_name in ['PatchGCN','DeepGraphConv','PatchGCN_Surv_causal','causal_double']:
        if args.cancer_type == "KIRC_UNI":
            slide_feats_dict = joblib.load('/cache/cenmin/causual_graph/data/Feat_result/TCGA_KIRC_input/KIRC_UNI_graphs.pkl')
        if args.cancer_type == "ESCA_UNI":
            slide_feats_dict = joblib.load('/cache/cenmin/causual_graph/data/Feat_result/TCGA_ESCA_input/ESCA_UNI_graphs.pkl')
        if args.cancer_type == "BLCA_UNI":
            slide_feats_dict = joblib.load('/cache/cenmin/causual_graph/data/Feat_result/TCGA_BLCA_input/BLCA_UNI_graphs.pkl')    
    else:
        if args.cancer_type == "KIRC_UNI":
            slide_feats_dict = joblib.load('/cache/cenmin/causual_graph/data/Feat_result/TCGA_KIRC_input/KIRC_UNI_features.pkl')
        if args.cancer_type == "ESCA_UNI":
            slide_feats_dict = joblib.load('/cache/cenmin/causual_graph/data/Feat_result/TCGA_ESCA_input/ESCA_UNI_features.pkl')
        if args.cancer_type == "BLCA_UNI":
            slide_feats_dict = joblib.load('/cache/cenmin/causual_graph/data/Feat_result/TCGA_BLCA_input/BLCA_UNI_features.pkl')
    print('Load ok')
    for fi in range(args.folds_num):
        setup_seed(args.seed)
        if args.model_name=='ABMIL':
            model = AMIL_Surv(in_feats=args.slide_in_feats, size_arg=args.slide_size_arg, dropout = args.dropout, fusion_dim = args.fusion_dim,output_dim = args.output_dim).to(device)
        elif args.model_name=='DeepGraphConv':
            model = DeepGraphConv_Surv(edge_agg='spatial', num_features=args.slide_in_feats, hidden_dim=256, linear_dim=args.slide_size_arg, dropout=args.dropout, n_classes=args.output_dim,device=device).to(device)
        elif args.model_name=='PatchGCN':
            model = PatchGCN_Surv(num_features=args.slide_in_feats, dropout = args.dropout, hidden_dim = args.fusion_dim, n_classes = args.output_dim).to(device)
        elif args.model_name=='TransMIL':
            model = transmil.TransMIL(input_dim=args.slide_in_feats,n_classes=args.output_dim,dropout=args.dropout,act='relu').to(device)
        elif args.model_name == 'IBMIL':
            _confounder_path = os.path.join(save_dir,str(fi),'train_bag_cls_agnostic_feats_proto_'+str(5)+'.npy')
            model = attmil_ibmil.Dattention_ori(out_dim=args.output_dim,dropout=args.dropout,in_size=args.slide_in_feats,confounder_path=_confounder_path).to(device)
        elif args.model_name=='causal_double':
            if args.cancer_type == 'KIRC_UNI':
                thumbnail_path = f'/cache/cenmin/causual_graph/data/Feat_result/TCGA_KIRC_input/thumbnail_all_{args.fea_type}.pkl'
                patch_sample_path = '/cache/cenmin/causual_graph/data/Feat_result/TCGA_KIRC_input/patch_1500_all.pkl'
            if args.cancer_type == 'ESCA_UNI':
                thumbnail_path = f'/cache/cenmin/causual_graph/data/Feat_result/TCGA_ESCA_input/thumbnail_all_{args.fea_type}.pkl'
                patch_sample_path = '/cache/cenmin/causual_graph/data/Feat_result/TCGA_ESCA_input/patch_1500_all.pkl'
            if args.cancer_type == 'BLCA_UNI':
                thumbnail_path = f'/cache/cenmin/causual_graph/data/Feat_result/TCGA_BLCA_input/thumbnail_all_{args.fea_type}.pkl'
                patch_sample_path = '/cache/cenmin/causual_graph/data/Feat_result/TCGA_BLCA_input/patch_1500_all.pkl'
            thumb_all = joblib.load(thumbnail_path)[f'fold{str(fi)}'].to(device)
            patch_all = joblib.load(patch_sample_path)[f'fold{str(fi)}'].to(device)
            model = GraphCausal2_Surv(
                thumb_all=thumb_all, patch_all=patch_all, K = args.clusterK, num_features=args.slide_in_feats,
                dropout=args.dropout, hidden_dim=args.fusion_dim, n_classes=args.output_dim
            ).to(device)
        elif args.model_name=='RRTMIL':
            model_params = {
                'input_dim': args.slide_in_feats,
                'n_classes': args.output_dim,
                'dropout': args.dropout,
                'act': 'relu',
                'region_num': 8,
                'pos': 'none',
                'pos_pos': 0,
                'pool': 'attn',
                'peg_k': 7,
                'drop_path': 0.0,
                'n_layers': 2,
                'n_heads': 8,
                'attn': 'rmsa',
                'da_act': 'relu',
                'trans_dropout': 0.1,
                'ffn': False,
                'mlp_ratio': 4.0,
                'trans_dim': 64,
                'epeg': False,
                'min_region_num': 0,
                'qkv_bias': False,
                'epeg_k': 15,
                'epeg_2d': False,
                'epeg_bias': False,
                'epeg_type': 'attn',
                'region_attn': 'native',
                'peg_1d': False,
                'cr_msa': False,
                'crmsa_k': 3,
                'all_shortcut': False,
                'crmsa_mlp': False,
                'crmsa_heads': 8,
            }
            model = rrt.RRTMIL(**model_params).to(device)        

        train_patients = train_val_dict["train_{}".format(fi)]
        val_patients = train_val_dict["val_{}".format(fi)]
        test_patients = train_val_dict["test_{}".format(fi)]


        best_model_path = os.path.join(save_dir, "fold{}_best_model.pth".format(fi))
        if args.model_name == 'causal_double':
            model_state = torch.load(best_model_path)
            model.load_state_dict(model_state['model_state_dict'])
            if model_state['saved_cluster_centers'] is not None:
                model.biasremoval.cluster.saved_cluster_centers = model_state['saved_cluster_centers']
            if model_state['biases'] is not None:
                model.biasremoval.biases = model_state['biases']
        else:
            model.load_state_dict(torch.load(best_model_path))
        model.eval()

        if args.model_name == 'causal_double':
            train_loss, train_ci, t_train_hazards = prediction(model,train_patients,slide_feats_dict,cli_dict,args)
            val_loss, val_ci, t_val_hazards = prediction(model,val_patients,slide_feats_dict,cli_dict,args)
            test_loss, test_ci, t_test_hazards = prediction(model,test_patients,slide_feats_dict,cli_dict,args)
        else:
            train_loss, train_ci, t_train_hazards = prediction_norm(model,train_patients,slide_feats_dict,cli_dict,args)
            val_loss, val_ci, t_val_hazards = prediction_norm(model,val_patients,slide_feats_dict,cli_dict,args)
            test_loss, test_ci, t_test_hazards = prediction_norm(model,test_patients,slide_feats_dict,cli_dict,args)

        print_info(log_name,f"Fold{fi}:\n")
        print_info(log_name,f"train_ci: {train_ci}:\n")
        print_info(log_name,f"val_ci: {val_ci}\n")
        print_info(log_name,f"test_ci: {test_ci}\n")
        print(t_test_hazards)
        train_data = [(patient_id, hazard[0]) for patient_id, hazard in t_train_hazards.items()]
        val_data = [(patient_id, hazard[0]) for patient_id, hazard in t_val_hazards.items()]
        test_data = [(patient_id, hazard[0]) for patient_id, hazard in t_test_hazards.items()]

        # 创建 DataFrame
        train_df = pd.DataFrame(train_data, columns=['patient_id', 'hazards'])
        val_df = pd.DataFrame(val_data, columns=['patient_id', 'hazards'])
        test_df = pd.DataFrame(test_data, columns=['patient_id', 'hazards'])
        train_df.to_csv(os.path.join(save_dir, f"train_folds{fi}_results.csv"), index=False)
        val_df.to_csv(os.path.join(save_dir, f"val_folds{fi}_results.csv"), index=False)
        test_df.to_csv(os.path.join(save_dir, f"test_folds{fi}_results.csv"), index=False)

    f_.close()

if __name__ == "__main__":
    setup_seed(args.seed)
    main(args)