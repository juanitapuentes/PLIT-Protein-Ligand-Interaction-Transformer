import os
import sys
import copy
import time
import pickle
import logging
import numpy as np #
import pandas as pd
from tqdm import tqdm

from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import DataLoader


from utils import metrics_pharma
from utils.args import ArgsInit
from utils.ckpt_util import save_ckpt

from model.model2 import DeeperGCN
from model.model_transformer import Transformer
from model.model_bonds import TransformerPAU
from model.model_concatenation import PLANet

from data.dataset import load_dataset, reload_dataset, get_perturbed_dataset
from ensamble import PLIT_single, PLIT_test


def train(
    model, device, loader, optimizer, num_classes, args, threshold, trainset=None
):
    """
    Perform training for one epoch.
    Args:
        model:
        device:
        loader (loader): Training loader
        optimizer:
        num_classes (int): Number of classes
        args (parser): Model's configuration
        threshold (dict):
        trainset:
    Return:
        loss:
    """
    loss_list = []
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):

        batch_mol = batch.to(device)

        model.train()
        if args.feature == "full":
            pass
        elif args.feature == "simple":
            num_features = args.num_features
            batch_mol.x = batch_mol.x[:, :num_features]
            batch_mol.edge_attr = batch_mol.edge_attr[:, :num_features]
        if batch_mol.x.shape[0] == 1:
            pass
        else:
            optimizer.zero_grad()
            loss = 0
            pred = model(batch_mol)
            if not args.binary:
                for i in range(0,num_classes):
                    class_mask = batch.y.clone()
                    class_mask[batch.y == i] = 1
                    class_mask[batch.y != i] = 0
                    class_loss = cls_criterion(F.sigmoid(pred[:,i]).to(torch.float32), class_mask.to(torch.float32))
                    loss += class_loss
            else:
                class_loss = cls_criterion(F.sigmoid(pred[:,1]).to(torch.float32), batch.y.to(torch.float32))
                loss += class_loss
                
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
        

    return np.mean(loss_list)


@torch.no_grad()
def eval_gcn(model, device, loader, num_classes, args):
    """
    Evaluate the model on the validation set.
    Args:
    Return:
    """
    model.eval()
    loss_list = []
    y_true = []
    y_pred = []
    correct = 0

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch_mol = batch.to(device)

        if args.feature == "full":
            pass
        elif args.feature == "simple":
            num_features = args.num_features
            batch_mol.x = batch_mol.x[:, :num_features]
            batch_mol.edge_attr = batch_mol.edge_attr[:, :num_features]
        if batch_mol.x.shape[0] == 1:
            pass
        else:
            with torch.set_grad_enabled(False):
                pred = model(batch_mol)
                loss = 0
                if not args.binary:
                    for i in range(0,num_classes):
                        class_mask = batch.y.clone()
                        class_mask[batch.y == i] = 1
                        class_mask[batch.y != i] = 0
                        class_loss = cls_criterion(F.sigmoid(pred[:,i]).to(torch.float32), class_mask.to(torch.float32))
                        loss += class_loss
                else:
                    class_loss = cls_criterion(F.sigmoid(pred[:,1]).to(torch.float32), batch.y.to(torch.float32))
                    loss += class_loss
                
                loss_list.append(loss.item())
                pred = F.softmax(pred,dim=1)
                y_true.append(batch.y.view(batch.y.shape).detach().cpu())
                y_pred.append(pred.detach().cpu())
                _, prediction_class = torch.max(pred,1)
                correct+=torch.sum(prediction_class == batch.y)


    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()
    if args.binary:
        nap, f = metrics_pharma.pltmap_bin(y_pred, y_true, num_classes)
        auc = metrics_pharma.plotbinauc(y_pred, y_true)

    else:
        nap, f = metrics_pharma.pltmap(y_pred, y_true, num_classes)
        auc = metrics_pharma.pltauc(y_pred, y_true, num_classes)

    acc = correct / len(loader.dataset)

    return acc, auc, f, nap, np.mean(loss_list)


def make_weights_for_balanced_classes(data, nclasses):
    """
    Generate weights for a balance training loader.
    Args:
        data (list): Labels of each molecule
        nclasses (int): number of classes
    Return:
        weight (list): Weights for each class
    """
    count = [0] * nclasses
    for item in data:
        count[item] += 1
    weight_per_class = [0.0] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N / float(count[i])
    weight = [0] * len(data)
    for idx, val in enumerate(data):
        weight[idx] = weight_per_class[val]

    return weight


def main_t():
    """
    Train a model on the train set and evaluate it on validation set.
    """
    # Init args
    args = ArgsInit().save_exp()

    # Read threshold-files use to select augmented molecules.
    if args.advs:
        args.edge_dict = {}

        df = pd.read_csv("./threshold/Umbral_Molecules_Maximium.csv")
        class_label = np.asarray(df["Class"])
        thresh = np.asarray(df["Umbral"])
        threshold = {k: v for k, v in zip(class_label, thresh)}
    else:
        threshold = {}

    # Set device
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

    # Set random seed for numpy, torch and cuda
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(args.seed)

    # Print model configuration
    logging.info("%s" % args)

    # Load all data splits
    train_dataset, valid_dataset, test_dataset, data_train, _, _ = load_dataset(
        cross_val=args.cross_val,
        binary_task=args.binary,
        target=args.target,
        args=args,
        use_prot=args.use_prot,
        advs=args.advs,
    )

    # Create a balance traning loader
    if args.balanced_loader:

        weights_train = make_weights_for_balanced_classes(
            list(data_train.Label), args.nclasses
        )
        weights_train = torch.DoubleTensor(weights_train)
        sampler_train = torch.utils.data.sampler.WeightedRandomSampler(
            weights_train, len(weights_train)
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=sampler_train,
            num_workers=args.num_workers,
        )
    else:
        # Create an unbalance traning loader

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )
    # Create validation and test loaders.
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    # Define the model based on the configuration (With or without PM Module)
    if args.use_bonds:
        model = TransformerPAU(args).to(device)
    else:
        model = Transformer(args).to(device)

    # Save model's configuration
    logging.info(model)

    # Set the optimizer and it's parameters
    optimizer = optim.Adamax(model.parameters(), lr=args.lr)

    # Set dictionary that is used to save the best results on every epoch.
    results = {
        "lowest_valid_loss": 100,
        "highest_valid": 0,
        "highest_train": 0,
        "epoch": 0,
    }

    start_time = time.time()

    # Set lists to save overall metrics and loss
    train_epoch_loss = []
    val_epoch_loss = []
    train_epoch_nap = []
    val_epoch_nap = []

    # Load model to resume training
    if args.resume:
        model_name = os.path.join(args.save, "model_ckpt", args.model_load_path)
        checkpoint = torch.load(model_name, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        init_epoch = checkpoint["epoch"] + 1
        train_epoch_loss = checkpoint["loss_train"]
        val_epoch_loss = checkpoint["loss_val"]
        train_epoch_nap = checkpoint["nap_train"]
        val_epoch_nap = checkpoint["nap_val"]
        results["highest_valid"] = max(val_epoch_nap)
        results["lowest_valid_loss"] = min(val_epoch_loss)
        results["highest_train"] = max(train_epoch_nap)
        results["epoch"] = init_epoch
        logging.info("Model loaded")
    else:
        init_epoch = 1

    if args.init_adv_training:
        model_name = os.path.join(
            args.model_load_init_path, "model_ckpt", args.model_load_path
        )
        checkpoint = torch.load(model_name, map_location=lambda storage, loc: storage)
        for i in range(args.num_layers):
            checkpoint["model_state_dict"][
                "gcns." + str(i) + ".edge_encoder.bond_embedding_list.0.weight"
            ] = checkpoint["model_state_dict"][
                "gcns." + str(i) + ".edge_encoder.bond_embedding_list.0.weight"
            ].t()
            checkpoint["model_state_dict"][
                "gcns." + str(i) + ".edge_encoder.bond_embedding_list.1.weight"
            ] = checkpoint["model_state_dict"][
                "gcns." + str(i) + ".edge_encoder.bond_embedding_list.1.weight"
            ].t()
            checkpoint["model_state_dict"][
                "gcns." + str(i) + ".edge_encoder.bond_embedding_list.2.weight"
            ] = checkpoint["model_state_dict"][
                "gcns." + str(i) + ".edge_encoder.bond_embedding_list.2.weight"
            ].t()
        model.load_state_dict(checkpoint["model_state_dict"])
        init_epoch = 1
        print("Model loaded")

    elif args.LMPM:

        full_model_path = "/media/SSD3/jpuentes/BaselineAML/log/results/LM/" \
                          "Original/Fold{}/model_ckpt/Checkpoint_valid_best.pth".format(args.cross_val)

        pre_model = torch.load(full_model_path)
        pre_model_dict = pre_model["model_state_dict"]
        model_weights = {}

        if args.use_bonds:
            del pre_model_dict["graph_pred_linear.weight"]
            del pre_model_dict["graph_pred_linear.bias"]

            for k, v in pre_model_dict.items():
                if args.use_prot:
                    if v.shape == model.molecule_gcn.state_dict()[k].shape:
                        model_weights[k] = v
                    else:
                        model_weights[k] = torch.transpose(v, 0, 1)
                else:
                    if v.shape == model.state_dict()[k].shape:
                        model_weights[k] = v
                    else:
                        model_weights[k] = torch.transpose(v, 0, 1)

            model.molecule_gcn.load_state_dict(model_weights)

        else:
            for k, v in pre_model_dict.items():
                if args.use_prot:
                    if v.shape == model.molecule_gcn.state_dict()[k].shape:
                        model_weights[k] = v
                    else:
                        model_weights[k] = torch.transpose(v, 0, 1)
                else:
                    if v.shape == model.state_dict()[k].shape:
                        model_weights[k] = v
                    else:
                        model_weights[k] = torch.transpose(v, 0, 1)

            model.molecule_gcn.load_state_dict(model_weights)

            dict_clasificacion = {
                "weight": pre_model["model_state_dict"]["graph_pred_linear.weight"],
                "bias": pre_model["model_state_dict"]["graph_pred_linear.bias"],
            }
            model.classification.load_state_dict(dict_clasificacion, strict=False)


        all_params = []
        variable_params = []
        for name, param in model.named_parameters():
            all_params.append(name)
            if param.requires_grad:
                variable_params.append(name)

        if len(variable_params) < len(all_params):
            logging.info(
                "Molecule model loaded and freezed."
            )

    loss_track = 0
    past_loss = 0
    # Training
    for epoch in range(init_epoch, args.epochs + 1):

        logging.info("=====Epoch {}".format(epoch))
        logging.info("Training...")

        if epoch == 1:
            # Evaluate loaded models
            logging.info("Evaluating First Epoch...")
            val_acc, val_auc, val_f, val_nap, val_loss = eval_gcn(
                model, device, valid_loader, args.nclasses, args
            )
            logging.info(
                "Valid:Loss {}, ACC {}, AUC {}, F-Measure {}, AP {}".format(
                    val_loss, val_acc, val_auc, val_f, val_nap
                )
            )
        if args.advs:
            tr_loss = train(
                model,
                device,
                train_loader,
                optimizer,
                args.nclasses,
                args,
                threshold,
                trainset=train_dataset,
            )
        else:
            tr_loss = train(
                model, device, train_loader, optimizer, args.nclasses, args, threshold
            )

        logging.info("Evaluating...")
        tr_acc, tr_auc, tr_f, tr_nap, tr_loss = eval_gcn(
            model, device, train_loader, args.nclasses, args
        )
        val_acc, val_auc, val_f, val_nap, val_loss = eval_gcn(
            model, device, valid_loader, args.nclasses, args
        )

        train_epoch_loss.append(tr_loss)
        val_epoch_loss.append(val_loss)
        train_epoch_nap.append(tr_nap)
        val_epoch_nap.append(val_nap)

        metrics_pharma.plot_loss(
            train_epoch_loss, val_epoch_loss, save_dir=args.save, num_epoch=args.epochs
        )
        metrics_pharma.plot_nap(
            train_epoch_nap, val_epoch_nap, save_dir=args.save, num_epoch=args.epochs
        )

        logging.info(
            "Train:Loss {}, ACC {}, AUC {}, F-Measure {}, AP {}".format(
                tr_loss, tr_acc, tr_auc, tr_f, tr_nap
            )
        )
        logging.info(
            "Valid:Loss {}, ACC {}, AUC {}, F-Measure {}, AP {}".format(
                val_loss, val_acc, val_auc, val_f, val_nap
            )
        )

        logging.info("Learning Rate: {}".format(optimizer.param_groups[0]["lr"]))

        sub_dir = "Checkpoint"
        save_ckpt(
            model,
            optimizer,
            train_epoch_loss,
            val_epoch_loss,
            train_epoch_nap,
            val_epoch_nap,
            epoch,
            args.model_save_path,
            sub_dir,
            name_post="Last_model",
        )

        if tr_nap > results["highest_train"]:

            results["highest_train"] = tr_nap

        if val_loss < results["lowest_valid_loss"]:
            results["lowest_valid_loss"] = val_loss
            results["epoch"] = epoch

            save_ckpt(
                model,
                optimizer,
                train_epoch_loss,
                val_epoch_loss,
                train_epoch_nap,
                val_epoch_nap,
                epoch,
                args.model_save_path,
                sub_dir,
                name_post="valid_best",
            )
        if args.advs or args.PLANET:
            if val_nap > results["highest_valid"]:
                results["highest_valid"] = val_nap
                results["epoch"] = epoch

                save_ckpt(
                    model,
                    optimizer,
                    train_epoch_loss,
                    val_epoch_loss,
                    train_epoch_nap,
                    val_epoch_nap,
                    epoch,
                    args.model_save_path,
                    sub_dir,
                    name_post="valid_best_nap",
                )

        if args.PLANET or args.advs:
            if val_loss >= past_loss:
                loss_track += 1
            else:
                loss_track = 0
            past_loss = val_loss

            if args.PLANET and loss_track >= 5:
                logging.info("Early exit due to overfitting")
                end_time = time.time()
                total_time = end_time - start_time
                logging.info("Best model in epoch: {}".format(results["epoch"]))
                logging.info(
                    "Total time: {}".format(
                        time.strftime("%H:%M:%S", time.gmtime(total_time))
                    )
                )
                sys.exit()
            if args.advs and loss_track >= 15:
                logging.info("Early exit due to overfitting")
                end_time = time.time()
                total_time = end_time - start_time
                logging.info("Best model in epoch: {}".format(results["epoch"]))
                logging.info(
                    "Total time: {}".format(
                        time.strftime("%H:%M:%S", time.gmtime(total_time))
                    )
                )
                sys.exit()
        if args.advs:
            train_dataset, data_train = reload_dataset(
                cross_val=args.cross_val,
                binary_task=args.binary,
                target=args.target,
                args=args,
                advs=args.advs,
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
            )
    end_time = time.time()
    total_time = end_time - start_time
    logging.info("Best model in epoch: {}".format(results["epoch"]))
    logging.info(
        "Total time: {}".format(time.strftime("%H:%M:%S", time.gmtime(total_time)))
    )


if __name__ == "__main__":
    args = ArgsInit().args

    if args.mode == "test":
        args.save = "TestResults/test"
        args.batch_size = 30
        PLIT_test(args)

    elif args.mode == "demo":
        args.save = "TestResults/demo"
        args.batch_size = 30
        PLIT_single(args)

    else:
        cls_criterion = torch.nn.BCELoss()
        reg_criterion = torch.nn.MSELoss()
        main_t()
