import os
import csv
import copy
import numpy as np

from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn.functional as F

from torch_geometric.data import DataLoader

from utils import metrics_pharma
from utils.args import ArgsInit

from data.dataset import load_dataset
from data.dataset import demo_molecule ##For demo
from model.model import DeeperGCN
from model.model_bonds import TransformerPAU
from model.model_transformer import Transformer

from model.model_concatenation import PLANet


@torch.no_grad()
def eval(model, device, loader, num_classes, args):
    model.eval()
    y_true, y_pred = [], []
    y_true_fold1, y_pred_fold1 = [], []
    y_true_fold2, y_pred_fold2 = [], []
    y_true_fold3, y_pred_fold3 = [], []
    y_true_fold4, y_pred_fold4 = [], []
    correct = 0

    print("------Copying model 1---------")
    prop_predictor1 = copy.deepcopy(model)
    print("------Copying model 2---------")
    prop_predictor2 = copy.deepcopy(model)
    print("------Copying model 3---------")
    prop_predictor3 = copy.deepcopy(model)
    print("------Copying model 4---------")
    prop_predictor4 = copy.deepcopy(model)

    test_model_path = os.path.join(args.save)

    test_model_path1 = "/media/SSD3/jpuentes/ENTREGAFINAL/BaselineAML/log/results/pruebaBONDS/Fold1/model_ckpt/Checkpoint_valid_best.pth"
    test_model_path2 = "/media/SSD3/jpuentes/ENTREGAFINAL/BaselineAML/log/results/pruebaBONDS/Fold2/model_ckpt/Checkpoint_valid_best.pth"
    test_model_path3 = "/media/SSD3/jpuentes/ENTREGAFINAL/BaselineAML/log/results/pruebaBONDS/Fold3/model_ckpt/Checkpoint_valid_best.pth"
    test_model_path4 = "/media/SSD3/jpuentes/ENTREGAFINAL/BaselineAML/log/results/pruebaBONDS/Fold4/model_ckpt/Checkpoint_valid_best.pth"

    # LOAD MODELS
    print("------- Loading weights----------")
    prop_predictor1.load_state_dict(torch.load(test_model_path1)["model_state_dict"])
    prop_predictor1.to(device)

    prop_predictor2.load_state_dict(torch.load(test_model_path2)["model_state_dict"])
    prop_predictor2.to(device)

    prop_predictor3.load_state_dict(torch.load(test_model_path3)["model_state_dict"])
    prop_predictor3.to(device)

    prop_predictor4.load_state_dict(torch.load(test_model_path4)["model_state_dict"])
    prop_predictor4.to(device)

    # METHOD.EVAL
    prop_predictor1.eval()
    prop_predictor2.eval()
    prop_predictor3.eval()
    prop_predictor4.eval()

    if args.mode == "demo":
        batch_mol = loader.to(device)
        if args.feature == "full":
            pass
        elif args.feature == "simple":
            # only retain the top two node/edge features
            num_features = args.num_features
            batch_mol.x = batch_mol.x[:, :num_features]
            batch_mol.edge_attr = batch_mol.edge_attr[:, :num_features]
        if batch_mol.x.shape[0] == 1:
            pass
        else:
            with torch.set_grad_enabled(False):
                pred1 = F.softmax(prop_predictor1(batch_mol), dim=1)
                pred2 = F.softmax(prop_predictor2(batch_mol), dim=1)
                pred3 = F.softmax(prop_predictor3(batch_mol), dim=1)
                pred4 = F.softmax(prop_predictor4(batch_mol), dim=1)

                pred = (pred1 + pred2 + pred3 + pred4) / 4
                y_true.append(batch_mol.y.view(batch_mol.y.shape).detach().cpu())
                y_pred.append(pred.detach().cpu())

                _, prediction_class = torch.max(pred, 1)
                correct += torch.sum(prediction_class == batch_mol.y)

                y_pred_fold1.append(pred1.detach().cpu())

                y_pred_fold2.append(pred2.detach().cpu())

                y_pred_fold3.append(pred3.detach().cpu())

                y_pred_fold4.append(pred4.detach().cpu())

        acc = correct / 1

    else:
        for step, batch in enumerate(tqdm(loader, desc="Iteration")):
            batch_mol = batch.to(device)
            if args.feature == "full":
                pass
            elif args.feature == "simple":
                # only retain the top two node/edge features
                num_features = args.num_features
                batch_mol.x = batch_mol.x[:, :num_features]
                batch_mol.edge_attr = batch_mol.edge_attr[:, :num_features]
            if batch_mol.x.shape[0] == 1:
                pass
            else:
                with torch.set_grad_enabled(False):
                    pred1 = F.softmax(prop_predictor1(batch_mol), dim=1)
                    pred2 = F.softmax(prop_predictor2(batch_mol), dim=1)
                    pred3 = F.softmax(prop_predictor3(batch_mol), dim=1)
                    pred4 = F.softmax(prop_predictor4(batch_mol), dim=1)

                    pred = (pred1 + pred2 + pred3 + pred4) / 4
                    y_true.append(batch_mol.y.view(batch_mol.y.shape).detach().cpu())
                    y_pred.append(pred.detach().cpu())

                    _, prediction_class = torch.max(pred, 1)
                    correct += torch.sum(prediction_class == batch_mol.y)

                    y_pred_fold1.append(pred1.detach().cpu())

                    y_pred_fold2.append(pred2.detach().cpu())

                    y_pred_fold3.append(pred3.detach().cpu())

                    y_pred_fold4.append(pred4.detach().cpu())

        acc = correct / len(loader.dataset) #

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()

    if args.binary:
        auc = metrics_pharma.plotbinauc(y_pred, y_true)
        nap, f = metrics_pharma.pltmap_bin(y_pred, y_true)
    else:
        nap, f = metrics_pharma.norm_ap(y_pred, y_true, num_classes)
        #auc = metrics_pharma.pltauc(y_pred, y_true, num_classes)

    #acc = correct / len(loader.dataset)
    if args.mode == "demo":
        return acc, f, nap, y_true, prediction_class

    else:
        if not args.binary:
            auc = metrics_pharma.pltauc(y_pred, y_true, num_classes)
        return acc, auc, f, nap, y_true, prediction_class

def main(target):

    args = ArgsInit().args
    if args.target is None:
        args.target = target

    if args.use_gpu:
        device = (
            torch.device("cuda:" + str(args.device))
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
    else:
        device = torch.device("cpu")

    if args.binary:
        args.nclasses = 2

    # Numpy and torch seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(args.seed)

    print(args)

    ( _,_,test_dataset,_,_,_,) = load_dataset(
        cross_val=args.cross_val,
        binary_task=args.binary,
        target=args.target,
        use_prot=args.use_prot,
        args=args,
        test=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    if args.use_bonds:
        model = TransformerPAU(args).to(device)
    else:
        model = Transformer(args).to(device)

    acc, auc, f, nap, xx, xxx = eval(model, device, test_loader, args.nclasses, args)

    save_items = {"Target": [], "NAP": [], "AUC": [], "ACC": [], "F_Med": []}

    save_items["Target"] = args.target
    save_items["NAP"] = nap
    save_items["AUC"] = auc
    save_items["ACC"] = acc.item()
    save_items["F_Med"] = f

    fieldnames = list(save_items.keys())

    csv_file = os.path.join(
        args.save,'Performance.csv'
    )
    if not os.path.exists(csv_file):
        create_header = True
    else:
        create_header = False

    with open(csv_file, "a+") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if create_header:
            writer.writeheader()
        writer.writerow(save_items)

    print({"ACC": acc, "AUC": auc, "F-medida": f, "NAP": nap})
    return nap

def PLIT_single(args):
    if args.use_gpu:
        device = (
            torch.device("cuda:" + str(args.device))
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
    else:
        device = torch.device("cpu")

    # Numpy and torch seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(args.seed)

    print(args)
    print("Loading molecule...")
    data_molec = demo_molecule(args.molecule, args)

    if args.use_bonds:
        model = TransformerPAU(args).to(device)
    else:
        model = Transformer(args).to(device)

    acc, f, nap, y_true, predic_class = eval(model, device, data_molec, args.nclasses, args)
    save_items = {"NAP": [], "ACC": [], "F_Med": []}

    save_items["NAP"] = nap
    save_items["ACC"] = acc.item()
    save_items["F_Med"] = f

    fieldnames = list(save_items.keys())

    csv_file = os.path.join(
        args.save, 'Performance.csv'
    )
    if not os.path.exists(csv_file):
        create_header = True
    else:
        create_header = False

    with open(csv_file, "a+") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if create_header:
            writer.writeheader()
        writer.writerow(save_items)

    print("PLIT prediction: ", predic_class[0])
    print("Molecule's real label: ", y_true)
    print({"ACC": acc, "F-medida": f, "NAP": nap})

def PLIT_test(args):
    targets = ['aa2ar']

    results = {'Target': [], 'Mean_Test': []}

    for target in targets:
        nap_result = main(target)
        results['Target'].append(target)
        results['Mean_Test'].append(nap_result)

    torch.save(results, os.path.join(args.save, 'Overall_test_results.pth'))
    print('Mean Test: {}'.format(np.mean(results['Mean_Test'])))

if __name__ == "__main__":

    args = ArgsInit().args

    if args.mode == "demo" and args.molecule is not None:
        PLIT_single(args)

    elif args.target is None:

        PLIT_test(args)

        '''targets = ['aa2ar']
        
        results = {'Target': [], 'Mean_Test': []}
        
        for target in targets:
            nap_result = main(target)
            results['Target'].append(target)
            results['Mean_Test'].append(nap_result)
        
        torch.save(results,os.path.join(args.save,'Overall_test_results.pth'))
        print('Mean Test: {}'.format(np.mean(results['Mean_Test'])))'''

    else:
        main()
        print("se usó else:")