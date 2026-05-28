import os
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import shap
from torch.utils.data import DataLoader
from opt import parse_opts
from data.dataset import FBGDataset
from models.lmf import HealthPredictor
from collections import Counter
import numpy as np

def compute_weighted_accuracy(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    class_counts = Counter(y_true)  # samples per class
    total = len(y_true)

    weighted_acc = 0.0
    for cls in class_counts:
        cls_mask = (y_true == cls)
        correct = (y_pred[cls_mask] == y_true[cls_mask]).sum()
        acc_c = correct / cls_mask.sum()  # per-class accuracy
        weight = cls_mask.sum() / total   # per-class weight
        weighted_acc += acc_c * weight

    return weighted_acc

def model_wrapper(input_list, model, device, label=None):
    id, ls, pi, pe, ni = input_list
    with torch.no_grad():
        output, _ = model(id, ls, pi, pe, ni, label)
        return output

def test_model(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    args.device = device

    print("Loading test dataset...")
    test_data = FBGDataset(args, subset='test')
    test_loader = DataLoader(test_data, batch_size=args.batch_size, num_workers=args.n_threads, shuffle=False)
    
    print("Loading model...")
    model = HealthPredictor(args).to(device)
    # checkpoint_path = os.path.join(args.result_path, 'checkpoint/best_model.pth')
    checkpoint_path = os.path.join('./results/best', 'checkpoint/best_model.pth')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for id, ls, pi, pe, ni, label in test_loader:
            ls = ls.to(device)
            pi = pi.to(device)
            pe = pe.to(device)
            ni = ni.to(device)
            label = label.to(device)

            output, _ = model(id, ls, pi, pe, ni, label)
            preds = (output >= 0.5).float()
            
            # DeepExplainer only supports single input tensor, so adapt the model
            # Method 1: analyze one input modality (e.g., ls)
            # explainer = shap.DeepExplainer(lambda x: model_wrapper([id, x, pi, pe, ni])[0], ls)
            # shap_values = explainer.shap_values(ls)

            # # Visualize (visualize SHAP values of the first sample)
            # shap.summary_plot(shap_values, ls.cpu().numpy())
            
            all_preds.append(preds.cpu())
            all_labels.append(label.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    print("\nEvaluation Metrics:")
    acc  = accuracy_score(all_labels, all_preds)
    
    cm   = confusion_matrix(all_labels, all_preds)
    w_acc = compute_weighted_accuracy(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    w_f1 = f1_score(all_labels, all_preds, average='weighted')
    print(f"Accuracy : {acc:.4f}")
    print(f"W-ACC: {w_acc:.4f}")
    print(f"F1   : {f1:.4f}")
    print(f"W-F1 : {w_f1:.4f}")
    print("Confusion Matrix:")
    print(cm)
    # print("\nClassification Report:")
    # print(classification_report(all_labels, all_preds, digits=4))

if __name__ == "__main__":
    args = parse_opts()
    test_model(args)
