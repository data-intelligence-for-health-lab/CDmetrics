from sklearn import metrics
from keras.utils import to_categorical
import numpy as np


def binary_metrics(TP, FP, TN, FN):
    accuracy = (TP + TN) / (TP + FP + TN + FN)
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    ppv = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    npv = TN / (TN + FN) if (TN + FN) > 0 else 0.0
    return accuracy, sensitivity, specificity, ppv, npv


def binary_confusion_matrix(true_labels, y_pred, class_weights):
    TP = sum((a == 1) and (p == 1) for a, p in zip(true_labels, y_pred))
    FP = sum((a != 1) and (p == 1) for a, p in zip(true_labels, y_pred))
    TN = sum((a != 1) and (p != 1) for a, p in zip(true_labels, y_pred))
    FN = sum((a == 1) and (p != 1) for a, p in zip(true_labels, y_pred))

    d_TP = sum(
        w * ((a == 1) and (p == 1))
        for a, p, w in zip(true_labels, y_pred, class_weights)
    )
    d_FP = sum(
        (1 - w) * ((a != 1) and (p == 1))
        for a, p, w in zip(true_labels, y_pred, class_weights)
    )
    d_TN = sum(
        w * ((a != 1) and (p != 1))
        for a, p, w in zip(true_labels, y_pred, class_weights)
    )
    d_FN = sum(
        (1 - w) * ((a == 1) and (p != 1))
        for a, p, w in zip(true_labels, y_pred, class_weights)
    )

    if sum(class_weights) == 0:  # If all weights are 0, accuracy is 1
        print("Sum of class weights is 0")
        return TP, FP, TN, FN, TP, FP, TN, FN
    else:
        return TP, FP, TN, FN, d_TP, d_FP, d_TN, d_FN


def binary_evaluation(true_labels, predicted_labels, class_weights):
    predicted_labels = np.array(predicted_labels)[:, 1]
    y_pred_binary = np.where(predicted_labels >= 0.5, 1, 0)
    TP, FP, TN, FN, d_TP, d_FP, d_TN, d_FN = binary_confusion_matrix(
        true_labels, y_pred_binary, class_weights
    )
    accuracy, sensitivity, specificity, ppv, npv = binary_metrics(TP, FP, TN, FN)
    d_accuracy, d_sensitivity, d_specificity, d_ppv, d_npv = binary_metrics(
        d_TP, d_FP, d_TN, d_FN
    )

    # AUC
    thresholds = np.linspace(0, 1, 100)
    tpr_list = []
    fpr_list = []
    d_tpr_list = []
    d_fpr_list = []
    for threshold in thresholds:  # Calculate TPR and FPR for each threshold
        y_pred_binary = np.where(predicted_labels >= threshold, 1, 0)
        TP, FP, TN, FN, d_TP, d_FP, d_TN, d_FN = binary_confusion_matrix(
            true_labels, y_pred_binary, class_weights
        )

        tpr = TP / (TP + FN)
        fpr = FP / (FP + TN)
        d_tpr = d_TP / (d_TP + d_FN)
        d_fpr = d_FP / (d_FP + d_TN)

        tpr_list.append(tpr)
        fpr_list.append(fpr)
        d_tpr_list.append(d_tpr)
        d_fpr_list.append(d_fpr)

    roc_auc = metrics.auc(fpr_list, tpr_list)
    d_roc_auc = metrics.auc(d_fpr_list, d_tpr_list)

    return [
        accuracy,
        sensitivity,
        specificity,
        ppv,
        npv,
        roc_auc,
        d_accuracy,
        d_sensitivity,
        d_specificity,
        d_ppv,
        d_npv,
        d_roc_auc,
    ]


# Multiclas
def multiclass_metrics(confusion_matrix):
    num_classes = len(confusion_matrix)
    macro_accuracy = 0.0
    macro_sensitivity = 0.0
    macro_specificity = 0.0
    macro_ppv = 0.0
    macro_npv = 0.0

    for cls in confusion_matrix:
        TP = confusion_matrix[cls]["TP"]
        FP = confusion_matrix[cls]["FP"]
        TN = confusion_matrix[cls]["TN"]
        FN = confusion_matrix[cls]["FN"]

        # Calculate metrics for the current class
        accuracy = (TP + TN) / (TP + FP + TN + FN)
        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
        ppv = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        npv = TN / (TN + FN) if (TN + FN) > 0 else 0.0

        macro_accuracy += accuracy
        macro_sensitivity += sensitivity
        macro_specificity += specificity
        macro_ppv += ppv
        macro_npv += npv

    # Average the metrics across all classes
    macro_accuracy /= num_classes
    macro_sensitivity /= num_classes
    macro_specificity /= num_classes
    macro_ppv /= num_classes
    macro_npv /= num_classes

    # calculate_micro_averaged_metrics
    total_TP = sum(confusion_matrix[cls]["TP"] for cls in confusion_matrix)
    total_FP = sum(confusion_matrix[cls]["FP"] for cls in confusion_matrix)
    total_TN = sum(confusion_matrix[cls]["TN"] for cls in confusion_matrix)
    total_FN = sum(confusion_matrix[cls]["FN"] for cls in confusion_matrix)

    # Calculate metrics using aggregated values
    micro_accuracy = (total_TP + total_TN) / (total_TP + total_FP + total_TN + total_FN)
    micro_sensitivity = (
        total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0.0
    )
    micro_specificity = (
        total_TN / (total_TN + total_FP) if (total_TN + total_FP) > 0 else 0.0
    )
    micro_ppv = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0.0
    micro_npv = total_TN / (total_TN + total_FN) if (total_TN + total_FN) > 0 else 0.0

    metrics = [
        micro_accuracy,
        micro_sensitivity,
        micro_specificity,
        micro_ppv,
        micro_npv,
        macro_accuracy,
        macro_sensitivity,
        macro_specificity,
        macro_ppv,
        macro_npv,
    ]
    return metrics


def multiclass_confusion_matrix(true_labels, predicted_labels, classes, case_weights):
    confusion_matrix = {cls: {"TP": 0, "FP": 0, "TN": 0, "FN": 0} for cls in classes}
    for a, p in zip(true_labels, predicted_labels):
        for cls in classes:
            TP = (a == cls) and (p == cls)
            FP = (a != cls) and (p == cls)
            TN = (a != cls) and (p != cls)
            FN = (a == cls) and (p != cls)
            confusion_matrix[cls]["TP"] += TP
            confusion_matrix[cls]["FP"] += FP
            confusion_matrix[cls]["TN"] += TN
            confusion_matrix[cls]["FN"] += FN

    d_confusion_matrix = {cls: {"TP": 0, "FP": 0, "TN": 0, "FN": 0} for cls in classes}
    for a, p, w in zip(true_labels, predicted_labels, case_weights):
        for cls in classes:
            TP = w * ((a == cls) and (p == cls))
            FP = (1 - w) * ((a != cls) and (p == cls))
            TN = w * ((a != cls) and (p != cls))
            FN = (1 - w) * ((a == cls) and (p != cls))
            d_confusion_matrix[cls]["TP"] += TP
            d_confusion_matrix[cls]["FP"] += FP
            d_confusion_matrix[cls]["TN"] += TN
            d_confusion_matrix[cls]["FN"] += FN

    if sum(case_weights) == 0:  # If all weights are 0, accuracy is 1
        print("Sum of class weights is 0")
        return confusion_matrix, confusion_matrix
    else:
        return confusion_matrix, d_confusion_matrix


def multi_evaluation(true_labels, prediction_probabilities, classes, case_weights):
    confusion_matrix, d_confusion_matrix = multiclass_confusion_matrix(
        true_labels, np.argmax(prediction_probabilities, axis=1), classes, case_weights
    )
    conventional_result = multiclass_metrics(confusion_matrix)
    new_result = multiclass_metrics(d_confusion_matrix)

    # AUC
    y_true = true_labels
    y_true = to_categorical(y_true)
    y_pred = prediction_probabilities
    expanded_weight = np.repeat(case_weights, len(classes))

    # Flatten the true labels and predicted probabilities
    y_true_flat = y_true.ravel()
    y_pred_flat = y_pred.ravel()

    # Compute TPR and FPR for binary classification
    def calculate_tpr_fpr_binary(y_true, y_pred, threshold, expanded_weight):
        y_pred_binary = (y_pred >= threshold).astype(int)
        tp = np.sum((y_true == 1) & (y_pred_binary == 1))
        fp = np.sum((y_true == 0) & (y_pred_binary == 1))
        tn = np.sum((y_true == 0) & (y_pred_binary == 0))
        fn = np.sum((y_true == 1) & (y_pred_binary == 0))
        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)

        d_tp = np.sum(expanded_weight * ((y_true == 1) & (y_pred_binary == 1)))
        d_fp = np.sum(
            (1.0 - np.array(expanded_weight)) * ((y_true == 0) & (y_pred_binary == 1))
        )
        d_tn = np.sum(expanded_weight * ((y_true == 0) & (y_pred_binary == 0)))
        d_fn = np.sum(
            (1.0 - np.array(expanded_weight)) * ((y_true == 1) & (y_pred_binary == 0))
        )
        d_tpr = d_tp / (d_tp + d_fn)
        d_fpr = d_fp / (d_fp + d_tn)

        if sum(expanded_weight) == 0:  # If all weights are 0, accuracy is 1
            print("Sum of class weights is 0")
            return tpr, fpr, tpr, fpr
        else:
            return tpr, fpr, d_tpr, d_fpr

    # Calculate TPR and FPR for different thresholds for binary classification
    thresholds = np.linspace(0, 1, 100)
    tprs = []
    fprs = []
    d_tprs = []
    d_fprs = []
    for threshold in thresholds:
        tpr, fpr, d_tpr, d_fpr = calculate_tpr_fpr_binary(
            y_true_flat, y_pred_flat, threshold, expanded_weight
        )
        tprs.append(tpr)
        fprs.append(fpr)
        d_tprs.append(d_tpr)
        d_fprs.append(d_fpr)

    # Calculate the micro AUC by integrating the area under the micro-average ROC curve
    micro_auc = np.trapz(tprs, fprs)
    d_micro_auc = np.trapz(d_tprs, d_fprs)

    # Calculate the macro AUC by averaging the AUC for each class
    macro_auc = 0.0
    d_macro_auc = 0.0
    for i in range(len(classes)):
        class_tprs = []
        class_fprs = []
        d_class_tprs = []
        d_class_fprs = []
        for threshold in thresholds:
            tpr, fpr, d_tpr, d_fpr = calculate_tpr_fpr_binary(
                y_true[:, i], y_pred[:, i], threshold, case_weights
            )
            class_tprs.append(tpr)
            class_fprs.append(fpr)
            d_class_tprs.append(d_tpr)
            d_class_fprs.append(d_fpr)

        auc_i = np.trapz(class_tprs, class_fprs)
        d_auc_i = np.trapz(d_class_tprs, d_class_fprs)
        macro_auc += auc_i
        d_macro_auc += d_auc_i

    macro_auc /= len(classes)
    d_macro_auc /= len(classes)

    conventional_result.insert(5, abs(micro_auc))
    conventional_result.insert(11, abs(macro_auc))
    new_result.insert(5, abs(d_micro_auc))
    new_result.insert(11, abs(d_macro_auc))

    conventional_result = [np.round(x, 3) for x in conventional_result]
    new_result = [np.round(x, 3) for x in new_result]
    return conventional_result + new_result
