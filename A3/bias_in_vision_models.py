"""Implementation points for Assignment 3 Part 2.

The notebook uses these helpers so that group metrics and fairness
interventions are computed consistently. The autograder imports this file
directly.
"""

import numpy as np
import torch


COCO_CATEGORY_OBJECTS = {
    "sports": {
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
    },
}


def scores_to_predictions(scores, threshold=0.5):
    """Convert positive-class scores or probabilities into 0/1 predictions."""
    # TODO (Part 2): support 1D scores and 2D class-score arrays.
    scores = np.asarray(scores)
    if scores.ndim == 2:
        if scores.shape[1] == 1:
            scores = scores[:, 0]
        elif scores.shape[1] == 2:
            # Use the positive class. Convert logits to probabilities if needed.
            if np.all((scores >= 0) & (scores <= 1)) and np.allclose(scores.sum(axis=1), 1):
                scores = scores[:, 1]
            else:
                difference = scores[:, 1] - scores[:, 0]
                scores = 1 / (1 + np.exp(-np.clip(difference, -700, 700)))
        else:
            raise ValueError("Expected one or two class scores per image")
    elif scores.ndim != 1:
        raise ValueError("Expected a 1D or 2D score array")

    return (scores >= threshold).astype(int)


def classification_metrics(y_true, scores, threshold=0.5):
    """Return accuracy, precision, and recall for binary predictions.

    Return a dictionary with keys:
      n, positives, predicted_positives, accuracy, precision, recall
    """
    # TODO (Part 2): compute metrics from y_true and thresholded scores.
    y_true = np.asarray(y_true).reshape(-1).astype(int)
    predictions = scores_to_predictions(scores, threshold)
    if len(y_true) != len(predictions):
        raise ValueError("labels and predictions must be aligned")

    n = len(y_true)
    positives = int((y_true == 1).sum())
    predicted_positives = int((predictions == 1).sum())
    true_positives = int(((y_true == 1) & (predictions == 1)).sum())

    return {
        "n": n,
        "positives": positives,
        "predicted_positives": predicted_positives,
        "accuracy": float((predictions == y_true).mean()) if n else 0.0,
        "precision": true_positives / predicted_positives if predicted_positives else 0.0,
        "recall": true_positives / positives if positives else 0.0,
    }


def evaluate_by_group(y_true, scores, group_masks, threshold=0.5):
    """Compute overall metrics and metrics for each demographic group.

    `group_masks` should map names such as "female", "male", "dark_skin",
    and "light_skin" to boolean arrays aligned with `y_true`.
    """
    # TODO (Part 2): compute metrics overall and on each nonempty group mask.
    y_true = np.asarray(y_true).reshape(-1)
    scores = np.asarray(scores)
    results = {"overall": classification_metrics(y_true, scores, threshold)}
    for group, mask in group_masks.items():
        mask = np.asarray(mask, dtype=bool).reshape(-1)
        if len(mask) != len(y_true):
            raise ValueError("group mask length does not match labels")
        if mask.any():
            results[group] = classification_metrics(y_true[mask], scores[mask], threshold)
    return results


def recall_gap(group_metrics, group_names):
    """Return max recall minus min recall over the requested groups."""
    # TODO (Part 2): compute the recall gap over the named groups.
    return metric_gap(group_metrics, group_names, "recall")


def metric_gap(group_metrics, group_names, metric):
    """Return max metric value minus min metric value over the requested groups."""
    # TODO (Part 2): compute the requested metric gap over the named groups.
    values = [group_metrics[name][metric] for name in group_names]
    return float(max(values) - min(values)) if values else 0.0


def summarize_group_gaps(group_metrics, comparisons, metrics=("accuracy", "precision", "recall")):
    """Return metric gaps for named group comparisons.

    `comparisons` maps comparison names, such as "gender", to a list of group
    names, such as ["female", "male"].
    """
    # TODO (Part 2): compute each requested metric gap for each comparison.
    gaps = {}
    for name, groups in comparisons.items():
        gaps[name] = {}
        for metric in metrics:
            gaps[name][metric] = metric_gap(group_metrics, groups, metric)
    return gaps


def lowest_recall_group(group_metrics, group_names):
    """Return the group name with the lowest recall."""
    # TODO (Part 2): identify the group with the lowest recall.
    return min(group_names, key=lambda group: group_metrics[group]["recall"])


def category_target(object_labels, object_names, category):
    """Return a binary target for whether each image contains a category object."""
    # TODO (Part 2): combine object-label columns for the requested category.
    object_labels = np.asarray(object_labels)
    target = np.zeros(len(object_labels), dtype=bool)

    for i, name in enumerate(object_names):
        if str(name).lower() in COCO_CATEGORY_OBJECTS[category]:
            target = target | object_labels[:, i].astype(bool)

    return target.astype(int)


def object_recalls(object_labels, object_names, scores, category, threshold=0.5):
    """Return recall for each object inside a COCO category."""
    # TODO (Part 2): report per-object recall among images containing that object.
    object_labels = np.asarray(object_labels)
    predictions = scores_to_predictions(scores, threshold)
    recalls = {}
    for i, name in enumerate(object_names):
        name = str(name).lower()
        if name in COCO_CATEGORY_OBJECTS[category]:
            positives = object_labels[:, i].astype(bool)
            if positives.any():
                recalls[name] = float(predictions[positives].mean())
    return recalls


def category_recalls(category_labels, scores, threshold=0.5):
    """Return recall for each nonempty category label in arrays such as `val_sport_cat`."""
    # TODO (Part 2): report recall for each object category in the notebook labels.
    predictions = scores_to_predictions(scores, threshold)
    categories = {}
    for i, value in enumerate(category_labels):
        values = value if isinstance(value, (list, tuple, set, np.ndarray)) else [value]
        for name in values:
            if name is None or str(name).strip().lower() in ("", "none", "nan", "unsure"):
                continue
            categories.setdefault(str(name), []).append(i)
    recalls = {}
    for name, indices in sorted(categories.items()):
        recalls[name] = float(predictions[indices].mean())
    return recalls



def make_recall_reweighting(y_true, group_masks, target_group, weight=3.0):
    """Return sample weights for a simple pre-processing intervention.

    The intended intervention upweights positive examples in the group with
    the lowest baseline recall. Normalize weights to have mean 1.
    """
    # TODO (2b): implement the reweighting intervention.
    y_true = np.asarray(y_true).reshape(-1).astype(bool)
    mask = np.asarray(group_masks[target_group], dtype=bool).reshape(-1)
    if weight <= 0 or len(mask) != len(y_true):
        raise ValueError("invalid weight or group mask")
    weights = np.ones(len(y_true), dtype=float)
    weights[mask & y_true] = weight
    if len(weights):
        weights /= weights.mean()
    return weights


def soft_recall_difference(y_true, scores, group_masks, group_a, group_b, eps=1e-12):
    """Return the absolute soft-recall difference between two groups.

    Soft recall is the average positive-class score among true positive
    examples in a group. This helper mirrors the in-processing intervention in
    the notebook.
    """
    # TODO (2b loss): compute the soft-recall gap for group_a and group_b.
    if isinstance(scores, torch.Tensor):
        probabilities = scores.reshape(-1)
        y = torch.as_tensor(y_true, device=probabilities.device).reshape(-1).bool()
        a = y & torch.as_tensor(group_masks[group_a], device=probabilities.device).reshape(-1).bool()
        b = y & torch.as_tensor(group_masks[group_b], device=probabilities.device).reshape(-1).bool()

        recall_a = probabilities[a].sum() / (a.sum() + eps)
        recall_b = probabilities[b].sum() / (b.sum() + eps)
        return torch.abs(recall_a - recall_b)

    probabilities = np.asarray(scores)
    if probabilities.ndim == 2:
        if probabilities.shape[1] == 2:
            if np.all((probabilities >= 0) & (probabilities <= 1)) and np.allclose(probabilities.sum(axis=1), 1):
                probabilities = probabilities[:, 1]
            else:
                difference = probabilities[:, 1] - probabilities[:, 0]
                probabilities = 1 / (1 + np.exp(-np.clip(difference, -700, 700)))
        else:
            probabilities = probabilities[:, 0]

    y = np.asarray(y_true).reshape(-1).astype(bool)
    a = y & np.asarray(group_masks[group_a], dtype=bool).reshape(-1)
    b = y & np.asarray(group_masks[group_b], dtype=bool).reshape(-1)

    recall_a = probabilities[a].sum() / (a.sum() + eps)
    recall_b = probabilities[b].sum() / (b.sum() + eps)
    return float(abs(recall_a - recall_b))


def apply_group_thresholds(scores, group_masks, thresholds, default_threshold=0.5):
    """Return predictions using group-specific post-processing thresholds.

    `thresholds` maps group names such as "female" and "male" to thresholds.
    Examples not covered by those groups should use `default_threshold`.
    """
    # TODO (2b thresholding): apply group-specific thresholds to scores.
    scores = np.asarray(scores)
    predictions = scores_to_predictions(scores, default_threshold)
    for group, threshold in thresholds.items():
        mask = np.asarray(group_masks[group], dtype=bool).reshape(-1)
        predictions[mask] = scores_to_predictions(scores[mask], threshold)
    return predictions