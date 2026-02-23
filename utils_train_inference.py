import torch
from torch import nn, optim
import models as my_models
from utils import *
import os
import time
import numpy as np
from torch.nn import functional as F
from contrastive_loss import SupConLoss
from coral import CorrelationAlignmentLoss
from torchvision import transforms
from masker import Masker
from masker_Opt import get_optim_and_scheduler

def inference(model, device, loader):
    model.eval()
    set_size = len(loader.dataset)
    acc_list = None
    with torch.no_grad():
        for input_image, y, path in loader:
            y = y.to(device)
            input_image = [x.to(device) for x in input_image]
            outputs = model(x=input_image, tg=[], train=False, flag=False)
            if acc_list is None:
                acc_list = [0] * len(outputs)
            for idx, output in enumerate(outputs):
                preds = output.argmax(dim=1)
                acc_list[idx] += (preds == y).sum().item()
    acc_list = [round(acc / set_size, 4) for acc in acc_list]
    return acc_list


def training_function(
    args, loader_info, lr, method_loss, save_path, experiment_dir, device
):
    model_name_path = os.path.join(
        ".", save_path, f"Method_loss_{method_loss}_lr_{lr}.pt"
    )

    model = my_models.PseudoCombiner(
        no_classes=len(loader_info["classes"]),
        pretrained=args.pretrained,
        backbone_name=args.backbone,
    )
    model.to(device)
    model.train()
    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    gamma = pow(
        args.bt_exp_scheduler_gamma, (1.0 / float(args.epochs))
    )
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma, last_epoch=-1)

    convertor = my_models.AugNet(1).to(device)
    convertor_opt = torch.optim.SGD(convertor.parameters(), lr=10)

    masker = Masker(in_dim=512, num_classes = 512, middle = 4*512,k=308).to(device)
    masker_optim, masker_sched = get_optim_and_scheduler(masker)
    masker.train()

    model2 = my_models.PseudoCombiner(
        no_classes=len(loader_info["classes"]),
        pretrained=args.pretrained,
        backbone_name=args.backbone,
    )
    model2.to(device)
    model2.train()
    optimizer2 = optim.SGD(
        model2.parameters(),
        lr=lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    gamma = pow(
        args.bt_exp_scheduler_gamma, (1.0 / float(args.epochs))
    )
    scheduler2 = optim.lr_scheduler.ExponentialLR(optimizer2, gamma, last_epoch=-1)

    save_architecture(network=model, direct=experiment_dir)
    if args.search_mode.lower() == "new_test" and os.path.isfile(
        model_name_path
    ):
        load_model(network=model, model_location=model_name_path)
        return model
    elif os.path.isfile(model_name_path):
        print(f"{model_name_path} exists.")
        return

    criterion = nn.CrossEntropyLoss(reduction="sum").to(device)
    correlation_alignment_loss = CorrelationAlignmentLoss().to(device)
    con = SupConLoss()
    Cyc = torch.nn.MSELoss().to(device)
    tran = transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
    class_num = 7
    epochs = args.epochs
    for epoch in range(epochs):
        start = time.time()
        train_acc, train_count = 0, 0

        for idx, (input_image, y, path) in enumerate(loader_info["train_loader"]):
            y = y.to(device)
            input_image = [x.to(device) for x in input_image]
            inputs1_max = tran(torch.sigmoid(convertor(input_image[1])))
            inputs1_max = inputs1_max * 0.6 + input_image[0] * 0.4
            data1_aug = [input_image[0], inputs1_max]
            outputs1, tuples1 = model(x=data1_aug, tg=[], train=True, flag=False)
            emb_aug = F.normalize(tuples1['Embedding'][y.size(0):]).unsqueeze(1)
            emb_src = F.normalize(tuples1['Embedding'][:y.size(0)]).unsqueeze(1)
            combine = torch.cat([emb_src, emb_aug], dim=1)
            con1 = con(combine , y)
            mu = tuples1['mu'][:y.size(0)]
            logvar = tuples1['logvar'][:y.size(0)]
            y_samples = tuples1['Embedding'][y.size(0):]
            likeli = -(-(mu - y_samples)**2 /logvar.exp()-logvar).mean()
            loss = torch.zeros(1)[0].to(device)
            batches = [x.shape[0] for x in outputs1]
            for idx2, output in enumerate(outputs1):
                current_loss = criterion(output, y)
                if idx2 != 0:
                    loss = current_loss * method_loss + loss
                else:
                    loss = current_loss + loss

            loss = loss / sum(batches)
            loss += likeli + con1
            _, preds = torch.max(nn.functional.softmax(outputs1[0], dim=1).data, 1)
            train_acc += torch.sum(preds == y.data).item()
            train_count += len(y)

            masker_optim.zero_grad()
            feature_a = tuples1['Embedding'][y.size(0):]
            masks_sup = masker(feature_a.detach())
            masks_inf = torch.ones_like(masks_sup) - masks_sup
            if epoch <= 5:
                masks_sup = torch.ones_like(feature_a.detach())
                masks_inf = torch.ones_like(feature_a.detach())
            features_sup = feature_a * masks_sup
            features_inf = feature_a * masks_inf
            scores_sup = model.classifier(features_sup)
            scores_inf = model2.classifier(features_inf)
            loss_cls_sup = criterion(scores_sup, y)
            loss_cls_inf = criterion(scores_inf, y)
            loss_cls_sup /= y.size(0)
            loss_cls_inf /= y.size(0)
            loss += 0.5 * loss_cls_sup + 0.5 * loss_cls_inf

            optimizer.zero_grad()
            optimizer2.zero_grad()
            loss.backward()
            if nan_in_grad(model=model):
                save_model(
                    network=model,
                    optimizer=optimizer,
                    epoch=args.epochs,
                    direct=model_name_path,
                    is_nan=True,
                )
                return model
            optimizer.step()
            optimizer2.step()

            inputs3_max = tran(torch.sigmoid(convertor(input_image[1])))
            inputs3_max = inputs3_max * 0.6 + input_image[0] * 0.4
            data3_aug = [input_image[0], inputs3_max]
            outputs3, tuples3 = model(x=data3_aug, tg=[], train=True, flag=False)
            feature_a = tuples3['Embedding'][y.size(0):]
            masks_sup = masker(feature_a.detach())
            masks_inf = torch.ones_like(masks_sup) - masks_sup
            features_sup = feature_a * masks_sup
            features_inf = feature_a * masks_inf
            scores_sup = model.classifier(features_sup)
            scores_inf = model2.classifier(features_inf)
            loss_cls_sup = criterion(scores_sup, y)
            loss_cls_inf = criterion(scores_inf, y)
            softmax_desc = nn.Softmax(dim=1)(scores_inf)
            epsilon = 1e-5
            entropy = -softmax_desc * torch.log(softmax_desc + epsilon)
            entropy = torch.sum(entropy, dim=1)
            entropy_desc = torch.mean(entropy)
            demix_loss = entropy_desc            
            loss_cls_sup /= y.size(0)
            loss_cls_inf /= y.size(0)
            total_loss = 0.5 * loss_cls_sup - 0.1 * demix_loss - 0.5 * loss_cls_inf
            masker_optim.zero_grad()
            total_loss.backward()
            masker_optim.step()

            inputs2_max = tran(torch.sigmoid(convertor(input_image[1], estimation=True)))
            inputs2_max = inputs2_max * 0.6 + input_image[0] * 0.4
            loss_l2 = Cyc(inputs2_max, input_image[1])
            data2_aug = [input_image[0], inputs2_max]
            outputs2, tuples2 = model(x=data2_aug, tg=[], train=True, flag=False)
            feature_a = tuples2['Embedding'][y.size(0):]
            masks_sup = masker(feature_a.detach())
            masks_inf = torch.ones_like(masks_sup) - masks_sup
            features_inf = feature_a * masks_inf
            tuples2, Ldml = model(x=data2_aug, tg=features_inf, train=False, flag=True)
            e = tuples2['Embedding']
            e1 = e[:y.size(0)]
            e2 = e[y.size(0):]
            chunks1 = torch.chunk(e1, 7, dim=0)
            chunks2 = torch.chunk(e2, 7, dim=0)
            loss_coral = 0
            f_m = []
            f_mp = []
            for i in range(class_num):
                for j in range(i+1, class_num):
                    f_m = chunks1[i]
                    f_mp = chunks2[j]
                    loss_coral += correlation_alignment_loss(f_m, f_mp)
            loss_coral /= (len(torch.unique(y)) * (len(torch.unique(y)) - 1) / 2)
            loss1 = loss_coral + 0.1*loss_l2 + 0.1*Ldml
            convertor_opt.zero_grad()
            loss1.backward()
            convertor_opt.step()
        scheduler.step()
        scheduler2.step()
        masker_sched.step()
        print(
            f"Epoch: {epoch} "
            f"LR: {round(scheduler.get_lr()[0], 8)} "
            f"Acc: {round(train_acc / train_count, 4)} "
            f"Time: {report_time(time.time() - start)}"
        )
    save_model(
        network=model,
        optimizer=optimizer,
        epoch=args.epochs,
        direct=model_name_path,
    )
    return model


def testing_function(
    model, loader_info, test_idx_list, lr, method_loss, csv_file_name, domains, device
):
    if model is not None:
        val_only_acc, test_acc, val_only_acc, imgaug_average = ([] for _ in range(4))
        with torch.no_grad():
            print("Validating Normal")
            val_list = inference(
                model=model, device=device, loader=loader_info["val_loader"]
            )
            if loader_info["val_only_loader"] is not None:
                for idx, vo_loader in enumerate(loader_info["val_only_loader"]):
                    print(
                        f"Validating {loader_info['pd_names_val_only'][idx].replace('_', ' ')}"
                    )
                    val_only_acc.append(
                        inference(model=model, device=device, loader=vo_loader)
                    )
                imgaug_average = [
                    round(sum(col) / len(col), 4) for col in zip(*val_only_acc)
                ]
                val_only_acc = [item for sublist in val_only_acc for item in sublist]
            for idx2, test_idx in enumerate(test_idx_list):
                test_acc.append(
                    inference(
                        model=model,
                        device=device,
                        loader=loader_info["test_loader_list"][idx2],
                    )
                )
                for idx, output in enumerate(loader_info["output_names_val"]):
                    print(
                        f"Test {domains[test_idx]} domain, {output} Acc: {test_acc[-1][idx]}"
                    )

            total_test = [round(sum(col) / len(col), 4) for col in zip(*test_acc)]
            test_acc_list_out = [item for sublist in test_acc for item in sublist]
        add_free_log(
            data=[[str(lr)]]
            + [[str(method_loss)]]
            + [[str(x)] for x in val_list]
            + [[str(x)] for x in imgaug_average]
            + [[str(x)] for x in total_test]
            + [[str(x)] for x in val_only_acc]
            + [[str(x)] for x in test_acc_list_out],
            save_dir=csv_file_name,
        )
        for idx, output in enumerate(loader_info["output_names_val"]):
            print(f"Total Test {output} Acc: {total_test[idx]}")
    return


def search_hyperparameters(
    args,
    loader_info,
    test_idx_list,
    csv_file_name,
    domains,
    save_path,
    experiment_dir,
    device,
):
    lr_min, lr_max = args.lr_search_range
    ml_min, ml_max = args.ml_search_range
    if lr_max < lr_min:
        lr_max, lr_min = lr_min, lr_max
    if ml_max < ml_min:
        ml_max, ml_min = ml_min, ml_max
    if args.lr is None and args.method_loss is None:
        lr_search_range = generate_points(
            range_tuple=(lr_min, lr_max),
            points=args.lr_search_no,
            log_scale=True,
        )
        ml_search_range = generate_points(
            range_tuple=(ml_min, ml_max), points=args.ml_search_no
        )
        print("Searching for the best Learning Rate and the best Method Loss...")
        print(
            f"We will train a total of {len(lr_search_range)*len(ml_search_range)} models"
        )
        for lr in lr_search_range:
            for method_loss in ml_search_range:
                print(f"Trying Learning Rate: {lr} and Method Loss: {method_loss}")
                model = training_function(
                    args,
                    loader_info,
                    lr,
                    method_loss,
                    save_path,
                    experiment_dir,
                    device,
                )
                testing_function(
                    model,
                    loader_info,
                    test_idx_list,
                    lr,
                    method_loss,
                    csv_file_name,
                    domains,
                    device,
                )
    elif args.lr is None:
        lr_search_range = generate_points(
            range_tuple=(lr_min, lr_max),
            points=args.lr_search_no,
            log_scale=True,
        )
        print("Searching for the best Learning Rate...")
        print(f"We will train a total of {len(lr_search_range)} models")
        for lr in lr_search_range:
            print(f"Trying Learning Rate: {lr}")
            model = training_function(
                args,
                loader_info,
                lr,
                args.method_loss,
                save_path,
                experiment_dir,
                device,
            )
            testing_function(
                model,
                loader_info,
                test_idx_list,
                lr,
                args.method_loss,
                csv_file_name,
                domains,
                device,
            )
    elif args.method_loss is None:
        ml_search_range = generate_points(
            range_tuple=(ml_min, ml_max), points=args.ml_search_no
        )
        print("Searching for the best Method Loss...")
        print(f"We will train a total of {len(ml_search_range)} models")
        for method_loss in ml_search_range:
            print(f"Trying Method Loss: {method_loss}")
            model = training_function(
                args,
                loader_info,
                args.lr,
                method_loss,
                save_path,
                experiment_dir,
                device,
            )
            testing_function(
                model,
                loader_info,
                test_idx_list,
                args.lr,
                method_loss,
                csv_file_name,
                domains,
                device,
            )