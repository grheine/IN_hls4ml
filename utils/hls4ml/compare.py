import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_absolute_error, mean_squared_error, roc_curve

def compare_hls4ml(graphs, output_dir, torch_model, hls_model, torch_wrapper, all_metrics=False):
    all_torch_error = {
    "MAE": [],
    "MSE": [],
    "RMSE": [],
    'Accuracy': [],
    "f1": [],
    "AUC": []
    }
    all_hls_error = {
        "MAE": [],
        "MSE": [],
        "RMSE": [],
        'Accuracy': [],
        "f1": [],
        "AUC": []
    }
    all_torch_hls_diff = {
        "MAE": [],
        "MSE": [],
        "RMSE": [],
        "Accuracy": [],
        "f1": [],
        "AUC": []
    }
    targets, torch_preds, hls_preds = [], [], []
    wrapper_MAE = []

    for i, data in enumerate(graphs):

        target = data.np_target

        # torch prediction
        torch_pred = torch_model(data).detach().cpu().numpy()
        torch_pred = np.reshape(torch_pred[:target.shape[0]], newshape=(target.shape[0],)) #drop dummy edges
        if i==0: np.savetxt(f'{output_dir}/tb_data/output_predictions.dat', torch_pred.reshape(1, -1), fmt='%f', delimiter=' ')
        
        # hls prediction
        hls_pred = hls_model.predict(data.hls_data)
        hls_pred = np.reshape(hls_pred[:target.shape[0]], newshape=(target.shape[0],)) #drop dummy edges

        if all_metrics:
            # get errors
            all_torch_error["MAE"].append(mean_absolute_error(target, torch_pred))
            all_torch_error["MSE"].append(mean_squared_error(target, torch_pred))
            all_torch_error["RMSE"].append(mean_squared_error(target, torch_pred, squared=False))
            all_torch_error["Accuracy"].append(accuracy_score(target, np.round(torch_pred)))
            all_torch_error["f1"].append(f1_score(target, np.round(torch_pred)))
            try:
                all_torch_error["AUC"].append(roc_auc_score(target, torch_pred))
            except ValueError:
                all_torch_error["AUC"].append(0.5) #0.5=random number generator

            all_hls_error["MAE"].append(mean_absolute_error(target, hls_pred))
            all_hls_error["MSE"].append(mean_squared_error(target, hls_pred))
            all_hls_error["RMSE"].append(mean_squared_error(target, hls_pred, squared=False))
            all_hls_error["Accuracy"].append(accuracy_score(target, np.round(hls_pred)))
            all_hls_error["f1"].append(f1_score(target, np.round(hls_pred)))
            try:
                all_hls_error["AUC"].append(roc_auc_score(target, hls_pred))
            except:
                all_hls_error["AUC"].append(0.5)

            all_torch_hls_diff["MAE"].append(mean_absolute_error(torch_pred, hls_pred))
            all_torch_hls_diff["MSE"].append(mean_squared_error(torch_pred, hls_pred))
            all_torch_hls_diff["RMSE"].append(mean_squared_error(torch_pred, hls_pred, squared=False))
            all_torch_hls_diff["Accuracy"].append(accuracy_score(np.round(torch_pred), np.round(hls_pred)))
            all_torch_hls_diff["f1"].append(f1_score(np.round(torch_pred), np.round(hls_pred)))
            try:
                all_torch_hls_diff["AUC"].append(roc_auc_score(np.round(torch_pred), hls_pred))
            except ValueError:
                all_torch_hls_diff["AUC"].append(0.5)

            # test torch_wrapper
            if i==len(graphs)-1:
                wrapper_pred = torch_wrapper.forward(data) #saves intermediates
                wrapper_pred = wrapper_pred.detach().cpu().numpy()
                wrapper_pred = np.reshape(wrapper_pred[:target.shape[0]], newshape=(target.shape[0],)) #drop dummy edges
                wrapper_MAE = mean_absolute_error(torch_pred, wrapper_pred)
            
        #ROC
        targets.append(target)
        torch_preds.append(torch_pred)
        hls_preds.append(hls_pred)
        
    targets = np.concatenate(targets)
    torch_preds = np.concatenate(torch_preds)
    hls_preds = np.concatenate(hls_preds)
    all_torch_error["ROC"] = roc_curve(targets, torch_preds)
    all_hls_error["ROC"] = roc_curve(targets, hls_preds)
    if all_metrics:
        print(f"     single-graph wrapper-->torch MAE: {wrapper_MAE}")
        print("")
        for err_type in ["MAE", "MSE", "RMSE"]:#, "Accuracy", "f1"]:#, "MCE"]:
            print(f"     with error criteria = {err_type}:")
            print(f"          mean torch error: %s" %np.mean(all_torch_error["%s" %err_type]))
            print(f"          mean hls error: %s" %np.mean(all_hls_error["%s" %err_type]))
            print(f"          mean hls-->torch error: %s" %np.mean(all_torch_hls_diff["%s" %err_type]))
            print("")
        for score_type in ["Accuracy", "f1", "AUC"]:
            print(f"     with score criteria = {score_type}:")
            print(f"          mean torch score: %s" %np.mean(all_torch_error["%s"%score_type]))
            print(f"          mean hls score: %s" %np.mean(all_hls_error["%s"%score_type]))
            print(f"          mean hls-->torch score: %s" % np.mean(all_torch_hls_diff["%s" % score_type]))
            print("")
        
    return all_torch_error, all_hls_error, all_torch_hls_diff, wrapper_MAE, hls_preds