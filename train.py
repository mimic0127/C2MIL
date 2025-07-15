 # %%
import os
import argparse
import setproctitle
setproctitle.setproctitle("BLCA")
def get_params():
    parser = argparse.ArgumentParser(description="All the parameters of this network.")
    parser.add_argument("--cancer_type", type=str, default="BLCA_UNI", help="cancer_type")
    parser.add_argument("--model_name", type=str, default="causal_double", help="choose model to train")
    parser.add_argument("--clusterK", type=int, default=0, help="choose model to train")
    parser.add_argument("--lossver", type=str, default="", help="choose model to train")  
    parser.add_argument("--details", type=str, default="UNI_0113_GTprob_adaptivecluster", help="details") #causal10ratio01
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
    parser.add_argument("--result_dir", type=str, default="./results_loss", help="result_dir")
    parser.add_argument("--gpu", type=str, default="0", help="gpu")
    parser.add_argument("--fea_type", type=str, default="UNI")
    args, _ = parser.parse_known_args()
    return args
args = get_params()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
#  nohup python -u train_BLCA.py --gpu 1 --batch_size 16 --seed 8 > log_file/log_uni_blca.txt &
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
# from model import AMIL_Surv, PatchGCN_Surv, DeepGraphConv_Surv
# from model_causal3 import PatchGCN_Surv_causal
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

def train_one_epoch(model,patients,slide_feats_dict,cli_dict,optimizer,args):
    train_batch_hazards = []
    train_batch_hazards_c = []
    train_batch_surv_time = []
    train_batch_censorship = []
    train_batch_h = []
    train_batch_h_c = []
    train_batch_h_r = []
    batch_z_old = []
    batch_z_new = []
    ratio_loss = 0.
    train_all_loss = 0.
    train_patients_hazards = {}
    train_patients_hazards_c = {}
    train_patients_surv_time = {}
    train_patients_censorship = {}
    
    model.train()
    random.shuffle(patients)
    print(patients)
    # print(model.fc.state_dict())
    # exit()
    for n_iter, patient_id in tqdm(enumerate(patients)):
        print('patient_id:',patient_id)
        surv_time = cli_dict[patient_id][1]
        censorship = cli_dict[patient_id][0]
        train_patients_surv_time[patient_id] = surv_time
        train_patients_censorship[patient_id] = censorship
        train_batch_censorship.append(censorship)
        train_batch_surv_time.append(surv_time)

        slide_data = slide_feats_dict[patient_id]
        slide_data = slide_data.to(device)

        thumb = torch.load(f'./data/thumbnail_fea/TCGA_BLCA_{args.fea_type}/{patient_id}.pt',map_location='cpu').detach().to(device)
        hazards,h,hazards_c, h_causal,h_remaining,causal_ratio,z_old,z_new = model(slide_data,thumb)

        ratio_loss = ratio_loss + causal_ratio**2
        hazards = hazards.squeeze(0)
        hazards_c = hazards_c.squeeze(0)
        train_batch_hazards.append(hazards)
        train_batch_hazards_c.append(hazards_c)
        train_batch_h.append(h)
        train_batch_h_c.append(h_causal)
        train_batch_h_r.append(h_remaining)
        batch_z_old.append(z_old)
        batch_z_new.append(z_new)
        train_patients_hazards[patient_id] = hazards.detach().cpu().numpy()
        train_patients_hazards_c[patient_id] = hazards_c.detach().cpu().numpy()
        print(hazards)
        print(hazards_c)
        if ((n_iter+1)%args.batch_size == 0 and (n_iter+1)//args.batch_size != len(patients)//args.batch_size) or n_iter == len(patients) - 1:
        # if ((n_iter+1)%args.batch_size == 0 and (n_iter+1)//args.batch_size != len(patients)//args.batch_size):
            # print('batch_size:',args.batch_size)
            train_batch_hazards = torch.cat(train_batch_hazards, dim=0)
            train_batch_hazards_c = torch.cat(train_batch_hazards_c, dim=0)
            train_batch_h = torch.stack(train_batch_h, dim=0)
            train_batch_h_c = torch.stack(train_batch_h_c, dim=0)
            train_batch_h_r = torch.stack(train_batch_h_r, dim=0)
            batch_z_old = torch.cat(batch_z_old, dim=0)
            batch_z_new = torch.cat(batch_z_new, dim=0)
            # print('train_batch_h:',train_batch_h.size())
            surv_loss = CoxSurvLoss(hazards = train_batch_hazards,
                                    time = train_batch_surv_time,
                                    censorship = train_batch_censorship)
            surv_loss_c = CoxSurvLoss(hazards = train_batch_hazards_c,
                                    time = train_batch_surv_time,
                                    censorship = train_batch_censorship)
            # 不变性
            # print(torch.randperm(train_batch_h_r.size(0)))
            # train_batch_h_r_shuffle = train_batch_h_r[torch.randperm(train_batch_h_r.size(0))]
            # train_batch_h_mix = []
            # for t in range(len(train_batch_h_r)):
            #     # perturbation = torch.randn_like(train_batch_h_r_shuffle[t]) * 0.01  # 设置扰动强度为 0.01
            #     perturbation = 0
            #     train_batch_h_mix.append(train_batch_h_c[t] + train_batch_h_r_shuffle[t] + perturbation)
            # train_batch_h_mix_tensor = torch.stack(train_batch_h_mix)
            # train_batch_hazards_mix = model.classifier(train_batch_h_mix_tensor)
            # causal_loss = CoxSurvLoss(hazards = train_batch_hazards_mix,
            #                         time = train_batch_surv_time,
            #                         censorship = train_batch_censorship)
            causal_loss2 = causal_contrastive_loss(train_batch_h,train_batch_h_c,train_batch_h_r)
            mse_loss_batch = F.mse_loss(batch_z_new, batch_z_old)
            print('surv_loss:',surv_loss)
            print('surv_loss:',surv_loss_c)
            # print('causal_loss:',causal_loss)
            print('causal_loss2:',causal_loss2)
            print('mse_loss:',mse_loss_batch)
            print('ratio_loss:',ratio_loss/args.batch_size)
            # train1
            if args.lossver == "":
                loss = (surv_loss + surv_loss_c + causal_loss2*0.1 + ratio_loss/args.batch_size*0.1 + mse_loss_batch * 0.1)
            if args.lossver == "woprob":
                loss = (surv_loss + surv_loss_c + causal_loss2*0.1 + ratio_loss/args.batch_size*0.1)
            if args.lossver == "woratio":
                loss = (surv_loss + surv_loss_c + causal_loss2*0.1 + mse_loss_batch * 0.1)
            if args.lossver == "wocausal":
                loss = (surv_loss + surv_loss_c + ratio_loss/args.batch_size*0.1 + mse_loss_batch * 0.1)           
            # loss = (surv_loss + surv_loss_c + ratio_loss/args.batch_size*0.1 + mse_loss_batch * 0.1)
            # loss = (surv_loss + surv_loss_c + causal_loss2*0.1 + ratio_loss/args.batch_size*0.2)
            # train0
            # loss = surv_loss
            # train5
            # loss = surv_loss + surv_loss_c + 0.2 * causal_loss + 0.1 * causal_loss2 + 0.1 * ratio_loss/args.batch_size
            # train3
            # loss = surv_loss + surv_loss_c + 0.1 * causal_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_all_loss += loss.item()
            del train_batch_hazards, train_batch_hazards_c, train_batch_h, train_batch_h_c, train_batch_h_r
            # torch.cuda.empty_cache()
            train_batch_hazards = []
            train_batch_hazards_c = []
            train_batch_surv_time = []
            train_batch_censorship = []
            batch_z_old = []
            batch_z_new = []
            train_batch_h = []
            train_batch_h_c = []
            train_batch_h_r = []
            ratio_loss = 0.
    print("Final batch sizes:")
    print("train_patients_surv_time:", len(train_patients_surv_time))
    print("train_patients_hazards_c:", len(train_patients_hazards_c))
    print("train_patients_censorship:", len(train_patients_censorship))
    train_all_loss = train_all_loss / (len(patients)//args.batch_size)
    train_ci = calculate_ci(train_patients_surv_time, train_patients_hazards_c, train_patients_censorship)
        
    return train_all_loss, train_ci
    


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
            thumb = torch.load(f'./data/thumbnail_fea/TCGA_BLCA_{args.fea_type}/{patient_id}.pt',map_location='cpu').detach().to(device)
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

# %%

def main(args):
    os.makedirs(args.result_dir, exist_ok=True)
    save_dir_name = "{} {} cluster{} {} {}_lr_{}_b_{}_fd_{}_epoch_{}_seed_{}_dc_{}_fusion_{}".format(args.cancer_type,args.lossver,args.clusterK,args.details,args.model_name,args.lr,args.batch_size,args.fusion_dim,args.epochs,args.seed,args.dc_loss_ratio,args.fusion_loss_ratio)
    save_dir = os.path.join(args.result_dir,save_dir_name)
    os.makedirs(save_dir, exist_ok=True)
    log_name = os.path.join(save_dir, "train_info.log")
    
    f_ = open(log_name,'w')
    f_.truncate()
    
    cli_dict = joblib.load('./data/Feat_result/TCGA_BLCA_input/blca_sur_and_time.pkl')
    train_val_dict = joblib.load('./data/Feat_result/TCGA_BLCA_input/blca_five_folds.pkl')  
    print('load BLCA...')
    if args.model_name in ['PatchGCN','DeepGraphConv','PatchGCN_Surv_causal','causal_double']:
        if args.cancer_type == "BLCA_UNI":
            slide_feats_dict = joblib.load('./data/Feat_result/TCGA_BLCA_input/BLCA_UNI_graphs.pkl')
    else:
        if args.cancer_type == "BLCA_UNI":
            slide_feats_dict = joblib.load('./data/Feat_result/TCGA_BLCA_input/BLCA_UNI_features.pkl')
    print('Load ok')
    all_folds_test_hazards = {}
    all_folds_external_hazards = {}
    for fi in range(args.folds_num):
    # for fi in [1]:
        
        if args.model_name=='ABMIL':
            model = AMIL_Surv(in_feats=args.slide_in_feats, size_arg=args.slide_size_arg, dropout = args.dropout, fusion_dim = args.fusion_dim,output_dim = args.output_dim).to(device)
        elif args.model_name=='DeepGraphConv':
            model = DeepGraphConv_Surv(edge_agg='spatial', num_features=args.slide_in_feats, hidden_dim=256, linear_dim=args.slide_size_arg, dropout=args.dropout, n_classes=args.output_dim,device=device).to(device)
        elif args.model_name=='PatchGCN':
            model = PatchGCN_Surv(num_features=args.slide_in_feats, dropout = args.dropout, hidden_dim = args.fusion_dim, n_classes = args.output_dim).to(device)
        elif args.model_name=='PatchGCN_Surv_causal':
            model = PatchGCN_Surv_causal(num_features=args.slide_in_feats, dropout = args.dropout, hidden_dim = args.fusion_dim, n_classes = args.output_dim).to(device)
        elif args.model_name=='causal_double':
            thumbnail_path = f'./data/Feat_result/TCGA_BLCA_input/thumbnail_all_{args.fea_type}.pkl'
            patch_sample_path = './data/Feat_result/TCGA_BLCA_input/patch_1500_all.pkl'
            thumb_all = joblib.load(thumbnail_path)[f'fold{str(fi)}'].to(device)
            patch_all = joblib.load(patch_sample_path)[f'fold{str(fi)}'].to(device)
            model = GraphCausal2_Surv(
                thumb_all=thumb_all, patch_all=patch_all, K = args.clusterK, num_features=args.slide_in_feats,
                dropout=args.dropout, hidden_dim=args.fusion_dim, n_classes=args.output_dim
            ).to(device)
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
        
        train_patients = train_val_dict["train_{}".format(fi)]
        val_patients = train_val_dict["val_{}".format(fi)]
        test_patients = train_val_dict["test_{}".format(fi)]
        # external_patients = joblib.load('./data/Feat_result/TCGA_BLCA_input_external/blca_external_id.pkl')
        # external_patients = list(external_patients)
        
        print_info(log_name, "---------Fold {}, {} train patients, {} val patients---------\n".format(fi,len(train_patients),len(val_patients)))
        
        best_val_ci = 0.
        best_val_loss = 99.
        best_model_path = os.path.join(save_dir, "fold{}_best_model.pth".format(fi))
        for epoch in range(args.epochs):
            # if epoch == 3:
            #     exit()
            if args.if_adjust_lr:
                adjust_learning_rate(optimizer, args.lr, epoch, lr_step=40, lr_gamma=0.5)
            
            train_loss, train_ci = train_one_epoch(model=model,patients=train_patients,slide_feats_dict=slide_feats_dict,cli_dict=cli_dict,optimizer=optimizer,args=args)
            val_loss, val_ci, _ = prediction(model,val_patients,slide_feats_dict,cli_dict,args)
            test_loss, test_ci, _ = prediction(model,test_patients,slide_feats_dict,cli_dict,args)
            # external_loss, external_ci, _ = prediction(model,external_patients,slide_feats_dict,cli_dict,args)

            # if epoch>=round(args.epochs*0.2) and epoch <= args.epochs - 1 and val_loss<best_val_loss:
            #     best_model = copy.deepcopy(model)
            #     best_val_loss = val_loss
            if epoch>=round(2) and epoch <= args.epochs - 1 and val_ci>=best_val_ci:
                model_state = {
                    'model_state_dict': model.state_dict(),  # 模型参数
                    'saved_cluster_centers': model.biasremoval.cluster.saved_cluster_centers,  # SoftCluster 的聚类中心
                    'biases': model.biasremoval.biases  # BiasRemoval 的偏置
                }
                torch.save(model_state, best_model_path)
                best_val_ci = val_ci
                
            print_info(log_name, "Epoch {:03d}----train loss: {:4f}, train ci: {:4f}, val loss: {:4f}, val ci: {:4f}, test loss: {:4f}, test ci: {:4f}\n".format(epoch,train_loss,train_ci,val_loss,val_ci,test_loss,test_ci))
            
        model_state = torch.load(best_model_path)
        best_model = GraphCausal2_Surv(
            thumb_all=thumb_all, patch_all=patch_all, num_features=args.slide_in_feats,
            dropout=args.dropout, hidden_dim=args.fusion_dim, n_classes=args.output_dim
        ).to(device)
        best_model.load_state_dict(model_state['model_state_dict'])
        if model_state['saved_cluster_centers'] is not None:
            best_model.biasremoval.cluster.saved_cluster_centers = model_state['saved_cluster_centers']
        if model_state['biases'] is not None:
            best_model.biasremoval.biases = model_state['biases']
        torch.save(best_model.state_dict(), os.path.join(save_dir,"fold{}_best_model.pth".format(fi)))
        best_model.eval()

        t_test_loss, t_test_ci, t_test_hazards = prediction(best_model,test_patients,slide_feats_dict,cli_dict,args)
        # t_external_loss, t_external_ci, t_external_hazards = prediction(best_model,external_patients,slide_feats_dict,cli_dict,args)
        print_info(log_name, "---------Fold {}----test loss: {:4f}, test ci: {:4f}---------\n\n".format(fi,t_test_loss,t_test_ci))
        
        for patient_id in t_test_hazards:
            all_folds_test_hazards[patient_id] = t_test_hazards[patient_id]
        # for patient_id in t_external_hazards:
        #     all_folds_external_hazards[patient_id] = t_external_hazards[patient_id]
        
        del model, best_model, train_patients, val_patients
    
    all_folds_val_ci = calculate_all_ci(cli_dict, all_folds_test_hazards)
    print_info(log_name, "All folds val ci: {:4f}\n".format(all_folds_val_ci))
    
    all_folds_test_info = {}
    all_folds_test_info["hazards"] = all_folds_test_hazards
    joblib.dump(all_folds_test_info, os.path.join(save_dir,"test_all_folds_results.pkl"))
    # all_folds_external_info = {}
    # all_folds_external_info["hazards"] = all_folds_external_hazards
    # joblib.dump(all_folds_external_info, os.path.join(save_dir,"external_all_folds_results.pkl"))
    
    f_.close()    

if __name__ == "__main__":
    setup_seed(args.seed)
    main(args)  