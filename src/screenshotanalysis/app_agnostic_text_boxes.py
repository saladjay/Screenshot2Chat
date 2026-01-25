"""
App-agnostic text box utilities.

Shared helpers for app-agnostic demo logic: drawing, evaluation, filtering,
and speaker assignment heuristics for detected text boxes.
"""

from __future__ import annotations

import json
import os

import cv2
import numpy as np


def draw_boxes_by_speaker(image, boxes):
    speaker_colors = {
        "A": (0, 0, 255),
        "B": (255, 0, 0),
        "Unknown": (128, 128, 128),
    }
    for box in boxes:
        x_min, y_min, x_max, y_max = box.box.tolist()
        speaker = getattr(box, "speaker", "Unknown")
        color = speaker_colors.get(speaker, speaker_colors["Unknown"])
        image = cv2.rectangle(
            image,
            (int(x_min), int(y_min)),
            (int(x_max), int(y_max)),
            color,
            2,
        )
    return image


def save_detection_coords(coords_dir, image_path, boxes, metadata):
    os.makedirs(coords_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(coords_dir, f"{base_name}.json")
    groups = metadata.get("groups") if metadata else None
    groups_payload = None
    if groups:
        groups_payload = [[box.box.tolist() for box in group] for group in groups]
    metadata_payload = dict(metadata) if metadata else {}
    metadata_payload.pop("groups", None)
    payload = {
        "image": os.path.basename(image_path),
        "layout": metadata.get("layout"),
        "metadata": metadata_payload,
        "boxes": [box.box.tolist() for box in boxes],
        "groups": groups_payload,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return output_path


def compute_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def evaluate_against_gt(image_path, det_boxes, gt_dir="test_images_answer", iou_threshold=0.5):
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    gt_path = os.path.join(gt_dir, f"{base_name}.json")
    if not os.path.exists(gt_path):
        return None

    with open(gt_path, "r", encoding="utf-8") as f:
        gt_items = json.load(f)

    gt_boxes = [item["coordinates"] for item in gt_items if "coordinates" in item]
    det_coords = [box.box.tolist() for box in det_boxes]

    if not gt_boxes:
        return None

    matched_det = set()
    tp = 0
    for gt_box in gt_boxes:
        best_iou = 0.0
        best_idx = None
        for idx, det_box in enumerate(det_coords):
            if idx in matched_det:
                continue
            iou = compute_iou(gt_box, det_box)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        if best_idx is not None and best_iou >= iou_threshold:
            matched_det.add(best_idx)
            tp += 1

    precision = tp / len(det_coords) if det_coords else 0.0
    recall = tp / len(gt_boxes) if gt_boxes else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "gt_count": len(gt_boxes),
        "det_count": len(det_coords),
        "true_positive": tp,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "gt_path": gt_path,
    }


def select_layout_text_boxes(processor, layout_det_results):
    return processor._get_all_boxes_from_layout_det(layout_det_results, special_types=["text"])


def assign_speaker_by_center_x(boxes):
    if not boxes:
        return
    centers = [box.center_x for box in boxes]
    pivot = float(np.median(centers))
    for box in boxes:
        box.speaker = "A" if box.center_x <= pivot else "B"


def assign_speaker_by_edges(
    boxes,
    image_width,
    avatar_boxes=None,
    nickname=None,
    bin_size_ratio=0.02,
    min_count=3,
    min_ratio=0.25,
    left_edge_max_ratio=0.2,
    right_edge_min_ratio=0.8,
    avatar_max_gap_ratio=0.15,
    avatar_overlap_ratio=0.3,
):
    if not boxes:
        return
    xmins = []
    xmaxs = []
    for box in boxes:
        x_min, _, x_max, _ = box.box.tolist()
        xmins.append(x_min / image_width)
        xmaxs.append(x_max / image_width)

    def quantize(values):
        bins = {}
        for value in values:
            bin_id = int(round(value / bin_size_ratio))
            bins[bin_id] = bins.get(bin_id, 0) + 1
        return bins

    xmin_bins = quantize(xmins)
    xmax_bins = quantize(xmaxs)
    total = len(boxes)

    frequent_xmin_bins = {
        bin_id
        for bin_id, count in xmin_bins.items()
        if count >= min_count and count / total >= min_ratio
    }
    frequent_xmax_bins = {
        bin_id
        for bin_id, count in xmax_bins.items()
        if count >= min_count and count / total >= min_ratio
    }

    def bin_to_value(bin_id):
        return bin_id * bin_size_ratio

    left_bins = {
        bin_id
        for bin_id in frequent_xmin_bins
        if bin_to_value(bin_id) <= left_edge_max_ratio
    }
    right_bins = {
        bin_id
        for bin_id in frequent_xmax_bins
        if bin_to_value(bin_id) >= right_edge_min_ratio
    }

    for box in boxes:
        box.speaker = None

    if left_bins or right_bins:
        for box in boxes:
            x_min, _, x_max, _ = box.box.tolist()
            xmin_bin = int(round((x_min / image_width) / bin_size_ratio))
            xmax_bin = int(round((x_max / image_width) / bin_size_ratio))
            if xmin_bin in left_bins and xmax_bin in right_bins:
                continue
            if xmin_bin in left_bins:
                box.speaker = "A"
            elif xmax_bin in right_bins:
                box.speaker = "B"

    if left_bins and not right_bins and avatar_boxes:
        max_gap = image_width * avatar_max_gap_ratio
        for box in boxes:
            if box.speaker is not None:
                continue
            for avatar in avatar_boxes:
                if avatar.x_max > box.x_min:
                    continue
                gap = box.x_min - avatar.x_max
                if gap > max_gap:
                    continue
                inter_y = max(0, min(box.y_max, avatar.y_max) - max(box.y_min, avatar.y_min))
                min_h = min(box.height, avatar.height)
                if min_h <= 0:
                    continue
                if inter_y / min_h >= avatar_overlap_ratio:
                    box.speaker = "A"
                    break
            if box.speaker is None:
                box.speaker = "B"

    if any(box.speaker is None for box in boxes):
        assign_speaker_by_center_x(boxes)


def assign_speaker_by_avatar_order(
    layout_boxes,
    avatar_boxes,
    image_width,
    image_height,
    max_preceding_gap_ratio=0.15,
    min_vertical_overlap=0.2,
    drop_orphans=True,
):
    if not layout_boxes:
        return []
    if not avatar_boxes:
        return layout_boxes
    max_preceding_gap = image_height * max_preceding_gap_ratio
    sorted_avatars = sorted(avatar_boxes, key=lambda b: b.y_min)
    kept = []
    for box in layout_boxes:
        best_avatar = None
        best_dist = None
        for avatar in sorted_avatars:
            if avatar.y_min > box.y_min + max_preceding_gap:
                break
            if avatar.y_min > box.y_min:
                continue
            overlap_y = max(0, min(box.y_max, avatar.y_max) - max(box.y_min, avatar.y_min))
            min_h = min(box.height, avatar.height)
            if min_h > 0 and overlap_y / min_h < min_vertical_overlap:
                continue
            dist = abs(box.center_y - avatar.center_y)
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_avatar = avatar
        if best_avatar is None:
            if not drop_orphans:
                box.speaker = "Unknown"
                kept.append(box)
            continue
        box.speaker = "A" if best_avatar.center_x <= image_width / 2 else "B"
        kept.append(box)
    return kept


def assign_speaker_by_nearest_avatar(
    boxes,
    avatar_boxes,
    image_width,
    max_gap_ratio=0.25,
    min_vertical_overlap=0.3,
):
    if not boxes:
        return
    if not avatar_boxes:
        assign_speaker_by_center_x(boxes)
        return
    max_gap = image_width * max_gap_ratio
    for box in boxes:
        best_gap = None
        best_avatar = None
        for avatar in avatar_boxes:
            inter_y = max(0, min(box.y_max, avatar.y_max) - max(box.y_min, avatar.y_min))
            min_h = min(box.height, avatar.height)
            if min_h <= 0 or inter_y / min_h < min_vertical_overlap:
                continue
            if avatar.x_max <= box.x_min:
                gap = box.x_min - avatar.x_max
            elif avatar.x_min >= box.x_max:
                gap = avatar.x_min - box.x_max
            else:
                gap = 0
            if best_gap is None or gap < best_gap:
                best_gap = gap
                best_avatar = avatar
        if best_avatar is not None and best_gap is not None and best_gap <= max_gap:
            box.speaker = "A" if best_avatar.center_x <= image_width / 2 else "B"
        else:
            box.speaker = "B"


def filter_small_layout_boxes(boxes, min_height_ratio=0.6):
    if not boxes:
        return []
    heights = np.array([box.height for box in boxes if box.height > 0])
    if heights.size == 0:
        return boxes
    main_height = float(np.median(heights))
    min_height = main_height * min_height_ratio
    return [box for box in boxes if box.height >= min_height]


def detect_left_only_layout(
    boxes,
    image_width,
    bin_size_ratio=0.02,
    min_count=2,
    min_ratio=0.2,
    left_edge_max_ratio=0.2,
    right_edge_min_ratio=0.8,
    left_center_max_ratio=0.6,
    min_left_fraction=0.8,
):
    if not boxes:
        return False
    xmins = []
    xmaxs = []
    for box in boxes:
        x_min, _, x_max, _ = box.box.tolist()
        xmins.append(x_min / image_width)
        xmaxs.append(x_max / image_width)

    def quantize(values):
        bins = {}
        for value in values:
            bin_id = int(round(value / bin_size_ratio))
            bins[bin_id] = bins.get(bin_id, 0) + 1
        return bins

    xmin_bins = quantize(xmins)
    xmax_bins = quantize(xmaxs)
    total = len(boxes)
    frequent_xmin_bins = {
        bin_id
        for bin_id, count in xmin_bins.items()
        if count >= min_count and count / total >= min_ratio
    }
    frequent_xmax_bins = {
        bin_id
        for bin_id, count in xmax_bins.items()
        if count >= min_count and count / total >= min_ratio
    }

    def bin_to_value(bin_id):
        return bin_id * bin_size_ratio

    left_bins = {
        bin_id
        for bin_id in frequent_xmin_bins
        if bin_to_value(bin_id) <= left_edge_max_ratio
    }
    right_bins = {
        bin_id
        for bin_id in frequent_xmax_bins
        if bin_to_value(bin_id) >= right_edge_min_ratio
    }
    if not (bool(left_bins) and not right_bins):
        return False
    left_count = sum(1 for box in boxes if box.center_x <= image_width * left_center_max_ratio)
    return left_count / max(1, len(boxes)) >= min_left_fraction


def assign_speaker_by_nickname_in_layout(
    layout_boxes,
    text_boxes,
    image,
    nickname,
    text_rec,
    containment_ratio=0.6,
):
    if not layout_boxes or not nickname:
        return False
    assigned_any = False

    def box_area(box):
        return max(0.0, box.x_max - box.x_min) * max(0.0, box.y_max - box.y_min)

    for box in layout_boxes:
        candidates = []
        for text_box in text_boxes:
            inter_x1 = max(box.x_min, text_box.x_min)
            inter_y1 = max(box.y_min, text_box.y_min)
            inter_x2 = min(box.x_max, text_box.x_max)
            inter_y2 = min(box.y_max, text_box.y_max)
            inter_w = max(0, inter_x2 - inter_x1)
            inter_h = max(0, inter_y2 - inter_y1)
            inter_area = inter_w * inter_h
            text_area = box_area(text_box)
            if text_area <= 0:
                continue
            if inter_area / text_area >= containment_ratio:
                candidates.append(text_box)
        if not candidates:
            continue
        top_text_box = min(candidates, key=lambda b: b.y_min)
        x_min, y_min, x_max, y_max = top_text_box.box.tolist()
        text_image = image[int(y_min):int(y_max), int(x_min):int(x_max), :]
        if text_image.size == 0:
            continue
        text_output = text_rec.predict_text(text_image)
        if not text_output:
            continue
        text_value = text_output[0].get("rec_text", "")
        if text_value.startswith(nickname):
            box.speaker = "A"
            assigned_any = True
        else:
            box.speaker = "B"
            assigned_any = True
    return assigned_any


def filter_center_near_boxes(
    boxes,
    image_width,
    image_height,
    x_min_ratio=0.48,
    x_max_ratio=0.52,
    y_min_ratio=0.2,
    top_center_max_width_ratio=0.25,
):
    if not boxes:
        return []
    x_min = image_width * x_min_ratio
    x_max = image_width * x_max_ratio
    y_min = image_height * y_min_ratio
    filtered = []
    for box in boxes:
        if x_min <= box.center_x <= x_max and box.center_y > y_min:
            continue
        if x_min <= box.center_x <= x_max and box.center_y <= y_min:
            if box.width <= image_width * top_center_max_width_ratio:
                continue
        filtered.append(box)
    return filtered


def suppress_nested_boxes(boxes, overlap_ratio=0.85):
    if not boxes:
        return []
    kept = []

    def box_area(box):
        x_min, y_min, x_max, y_max = box.box.tolist()
        return max(0.0, x_max - x_min) * max(0.0, y_max - y_min)

    def overlap_over_small(box_a, box_b):
        ax1, ay1, ax2, ay2 = box_a.box.tolist()
        bx1, by1, bx2, by2 = box_b.box.tolist()
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter = inter_w * inter_h
        small_area = min(box_area(box_a), box_area(box_b))
        if small_area <= 0:
            return 0.0
        return inter / small_area

    boxes_sorted = sorted(boxes, key=box_area, reverse=True)
    for candidate in boxes_sorted:
        is_nested = False
        for keeper in kept:
            overlap_small = overlap_over_small(candidate, keeper)
            if overlap_small >= overlap_ratio and box_area(candidate) <= box_area(keeper):
                is_nested = True
                break
        if not is_nested:
            kept.append(candidate)
    return kept


def filter_by_frequent_edges(
    boxes,
    image_width,
    bin_size_ratio=0.02,
    min_count=2,
    min_ratio=0.2,
    left_edge_max_ratio=0.2,
    right_edge_min_ratio=0.8,
    keep_min_ratio=0.6,
):
    if not boxes:
        return []
    xmins = []
    xmaxs = []
    for box in boxes:
        x_min, _, x_max, _ = box.box.tolist()
        xmins.append(x_min / image_width)
        xmaxs.append(x_max / image_width)

    def quantize(values):
        bins = {}
        for value in values:
            bin_id = int(round(value / bin_size_ratio))
            bins[bin_id] = bins.get(bin_id, 0) + 1
        return bins

    xmin_bins = quantize(xmins)
    xmax_bins = quantize(xmaxs)
    total = len(boxes)

    frequent_xmin_bins = {
        bin_id
        for bin_id, count in xmin_bins.items()
        if count >= min_count and count / total >= min_ratio
    }
    frequent_xmax_bins = {
        bin_id
        for bin_id, count in xmax_bins.items()
        if count >= min_count and count / total >= min_ratio
    }

    def bin_to_value(bin_id):
        return bin_id * bin_size_ratio

    left_bins = {
        bin_id
        for bin_id in frequent_xmin_bins
        if bin_to_value(bin_id) <= left_edge_max_ratio
    }
    right_bins = {
        bin_id
        for bin_id in frequent_xmax_bins
        if bin_to_value(bin_id) >= right_edge_min_ratio
    }

    if not left_bins or not right_bins:
        return boxes

    filtered = []
    for box in boxes:
        x_min, _, x_max, _ = box.box.tolist()
        xmin_bin = int(round((x_min / image_width) / bin_size_ratio))
        xmax_bin = int(round((x_max / image_width) / bin_size_ratio))
        if xmin_bin in left_bins or xmax_bin in right_bins:
            filtered.append(box)
    if len(filtered) < len(boxes) * keep_min_ratio:
        return boxes
    return filtered
