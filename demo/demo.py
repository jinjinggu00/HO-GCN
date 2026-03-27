import argparse
import cv2
import mmcv
import numpy as np
import os
import os.path as osp
import shutil
import torch
import warnings
from scipy.optimize import linear_sum_assignment

from mmcv.parallel import collate, scatter
from pyskl.apis import init_recognizer

try:
    from pyskl.datasets.pipelines import Compose
except Exception as e:
    raise ImportError("Failed to import pyskl.datasets.pipelines.Compose. Please check your pyskl installation.") from e

try:
    from mmdet.apis import inference_detector, init_detector
except (ImportError, ModuleNotFoundError):
    def inference_detector(*args, **kwargs):
        pass

    def init_detector(*args, **kwargs):
        pass
    warnings.warn('Failed to import mmdet apis.')

try:
    from mmpose.apis import inference_top_down_pose_model, init_pose_model, vis_pose_result
except (ImportError, ModuleNotFoundError):
    def init_pose_model(*args, **kwargs):
        pass

    def inference_top_down_pose_model(*args, **kwargs):
        pass

    def vis_pose_result(*args, **kwargs):
        pass
    warnings.warn('Failed to import mmpose apis.')

try:
    import moviepy.editor as mpy
except ImportError:
    raise ImportError('Please install moviepy to enable output file')

FONTFACE = cv2.FONT_HERSHEY_DUPLEX
FONTSCALE = 0.6
FONTCOLOR = (255, 255, 255)
THICKNESS = 1
LINETYPE = 1

DEFAULT_NUM_PERSON = 2


def parse_args():
    parser = argparse.ArgumentParser(
        description='Per-person per-frame demo (with confidence + bbox top-right text) - fixed version'
    )
    parser.add_argument('video', help='input video')
    parser.add_argument('out_filename', help='output filename (mp4)')

    parser.add_argument('--config', default='/home/l3408/hcb/pyskl-main/config/HOGCN/b_esc.py')
    parser.add_argument('--checkpoint', default='/home/l3408/hcb/pyskl-main/work_dirs/best_top1_acc_epoch.pth')

    parser.add_argument('--det-config', default='demo/faster_rcnn_r50_fpn_1x_coco-person.py')
    parser.add_argument('--det-checkpoint',
                        default=('https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/'
                                 'faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth'))
    parser.add_argument('--pose-config', default='demo/hrnet_w32_coco_256x192.py')
    parser.add_argument('--pose-checkpoint',
                        default='https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth')

    parser.add_argument('--det-score-thr', type=float, default=0.9)
    parser.add_argument('--label-map', default='/home/l3408/hcb/pyskl-main/tools/data/label_map/esc.txt')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--short-side', type=int, default=480)

    # per-frame inference settings
    parser.add_argument('--clip-len', type=int, default=60,
                        help='clip length. If -1, will read from config UniformSample.clip_len (recommended)')
    parser.add_argument('--mode', type=str, default='tail', choices=['center', 'tail'],
                        help="tail is often better for per-frame (online-like) behaviour")
    parser.add_argument('--step', type=int, default=1,
                        help='run model every step frames, fill others with last result')

    parser.add_argument('--num-person-slot', type=int, default=DEFAULT_NUM_PERSON,
                        help='FormatGCNInput.num_person (if not found in config)')
    parser.add_argument('--min-kpt-score', type=float, default=0.05,
                        help='keypoint score threshold to build bbox from keypoints')
    parser.add_argument('--conf-thr', type=float, default=0.0,
                        help='only display label if conf >= this')

    args = parser.parse_args()
    return args


def frame_extraction(video_path, short_side):
    target_dir = osp.join('./tmp', osp.basename(osp.splitext(video_path)[0]))
    os.makedirs(target_dir, exist_ok=True)
    frame_tmpl = osp.join(target_dir, 'img_{:06d}.jpg')
    vid = cv2.VideoCapture(video_path)

    frames, frame_paths = [], []
    flag, frame = vid.read()
    cnt = 0
    new_h, new_w = None, None
    while flag:
        if new_h is None:
            h, w, _ = frame.shape
            new_w, new_h = mmcv.rescale_size((w, h), (short_side, np.Inf))
        frame = mmcv.imresize(frame, (new_w, new_h))
        frames.append(frame)
        frame_path = frame_tmpl.format(cnt + 1)
        frame_paths.append(frame_path)
        cv2.imwrite(frame_path, frame)
        cnt += 1
        flag, frame = vid.read()
    return frame_paths, frames


def detection_inference(args, frame_paths):
    model = init_detector(args.det_config, args.det_checkpoint, args.device)
    assert model is not None, 'Failed to build detection model.'
    assert model.CLASSES[0] == 'person', 'Detector must be trained on COCO person'

    results = []
    print('Performing Human Detection for each frame')
    prog_bar = mmcv.ProgressBar(len(frame_paths))
    for frame_path in frame_paths:
        result = inference_detector(model, frame_path)
        result = result[0][result[0][:, 4] >= args.det_score_thr]
        results.append(result)
        prog_bar.update()
    return results


def pose_inference(args, frame_paths, det_results):
    model = init_pose_model(args.pose_config, args.pose_checkpoint, args.device)
    ret = []
    print('Performing Human Pose Estimation for each frame')
    prog_bar = mmcv.ProgressBar(len(frame_paths))
    for f, d in zip(frame_paths, det_results):
        d = [dict(bbox=x) for x in list(d)]
        pose = inference_top_down_pose_model(model, f, d, format='xyxy')[0]
        ret.append(pose)
        prog_bar.update()
    return ret


def dist_ske(ske1, ske2):
    dist = np.linalg.norm(ske1[:, :2] - ske2[:, :2], axis=1) * 2
    diff = np.abs(ske1[:, 2] - ske2[:, 2])
    return np.sum(np.maximum(dist, diff))


def pose_tracking(pose_results, max_tracks=2, thre=30):
    tracks, num_tracks = [], 0
    num_joints = None

    for idx, poses in enumerate(pose_results):
        if len(poses) == 0:
            continue
        if num_joints is None:
            num_joints = poses[0].shape[0]

        track_proposals = [t for t in tracks if t['data'][-1][0] > idx - thre]
        n, m = len(track_proposals), len(poses)
        scores = np.zeros((n, m))

        for i in range(n):
            for j in range(m):
                scores[i][j] = dist_ske(track_proposals[i]['data'][-1][1], poses[j])

        if n > 0 and m > 0:
            row, col = linear_sum_assignment(scores)
            for r, c in zip(row, col):
                track_proposals[r]['data'].append((idx, poses[c]))

        if m > n:
            assigned_cols = set(col.tolist()) if (n > 0 and m > 0) else set()
            for j in range(m):
                if j not in assigned_cols:
                    num_tracks += 1
                    new_track = dict(track_id=num_tracks, data=[(idx, poses[j])])
                    tracks.append(new_track)

    if num_joints is None:
        return None, None

    tracks.sort(key=lambda x: -len(x['data']))
    result = np.zeros((max_tracks, len(pose_results), num_joints, 3), dtype=np.float16)
    for i, track in enumerate(tracks[:max_tracks]):
        for item in track['data']:
            fidx, pose = item
            result[i, fidx] = pose
    return result[..., :2], result[..., 2]


def build_window_indices(T, t, clip_len, mode):
    if mode == 'center':
        start = t - clip_len // 2
    else:  # tail
        start = t - clip_len + 1
    idxs = [0 if i < 0 else (T - 1 if i >= T else i) for i in range(start, start + clip_len)]
    return idxs


def build_clip_for_person(fake_anno, person_id, t, clip_len, mode, num_person_slot):
    
    kp_all = fake_anno['keypoint']        # (P, T, V, 2)
    ks_all = fake_anno['keypoint_score']  # (P, T, V)
    P, T, V, _ = kp_all.shape
    idxs = build_window_indices(T, t, clip_len, mode)

    clip_kp = np.zeros((num_person_slot, clip_len, V, 2), dtype=np.float16)
    clip_ks = np.zeros((num_person_slot, clip_len, V), dtype=np.float16)

    # slot0: target person
    if 0 <= person_id < P:
        clip_kp[0] = kp_all[person_id, idxs].copy()
        clip_ks[0] = ks_all[person_id, idxs].copy()

    # slot1: best other person in this window (most valid joints)
    if num_person_slot >= 2 and P >= 2:
        best_j, best_cnt = None, -1
        for j in range(P):
            if j == person_id:
                continue
            cnt = int((ks_all[j, idxs] > 0).sum())
            if cnt > best_cnt:
                best_cnt = cnt
                best_j = j
        if best_j is not None and best_cnt > 0:
            clip_kp[1] = kp_all[best_j, idxs].copy()
            clip_ks[1] = ks_all[best_j, idxs].copy()

    clip_anno = dict(
        keypoint=clip_kp,
        keypoint_score=clip_ks,
        total_frames=clip_len,
        start_index=0,
        img_shape=fake_anno['img_shape'],
        original_shape=fake_anno['original_shape'],
        label=-1,
        modality='Pose',
        frame_dir=''
    )
    return clip_anno


def make_min_test_pipeline(config, num_person_slot):
    
    ops = []
    for op in config.data.test.pipeline:
        t = op.get('type')
        if t in ['UniformSample', 'PoseDecode', 'DecompressPose']:
            continue
        if t in ['Collect', 'ToTensor']:
            continue
        ops.append(op)

    ops = [dict(op) for op in ops]
    for op in ops:
        if op.get('type') == 'FormatGCNInput':
            op['num_person'] = num_person_slot

    ops += [
        dict(type='Collect', keys=['keypoint'], meta_keys=[]),
        dict(type='ToTensor', keys=['keypoint'])
    ]
    return Compose(ops)


@torch.no_grad()
def infer_with_confidence(model, pipeline, clip_anno, device_str):
    
    data = pipeline(clip_anno)
    data = collate([data], samples_per_gpu=1)
    if 'cuda' in device_str:
        data = scatter(data, [device_str])[0]

    scores = model(return_loss=False, **data)
    if isinstance(scores, (list, tuple)):
        scores = scores[0]
    if isinstance(scores, np.ndarray):
        scores = torch.from_numpy(scores)
    if scores.dim() == 1:
        scores = scores.unsqueeze(0)

    row = scores[0]
    is_prob_like = (row.min() >= 0) and (row.max() <= 1) and (abs(float(row.sum()) - 1.0) < 1e-3)
    probs = scores if is_prob_like else torch.softmax(scores, dim=1)

    conf, pred = torch.max(probs, dim=1)
    return int(pred.item()), float(conf.item())


def bbox_from_keypoints(kp_xy, kp_score, min_score=0.05, pad=6):
    valid = kp_score >= min_score
    if not np.any(valid):
        return None
    pts = kp_xy[valid]
    x1 = float(np.min(pts[:, 0])) - pad
    y1 = float(np.min(pts[:, 1])) - pad
    x2 = float(np.max(pts[:, 0])) + pad
    y2 = float(np.max(pts[:, 1])) + pad
    return x1, y1, x2, y2


def draw_text_at_bbox_topright(frame, bbox, text):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    x2 = int(max(0, min(w - 1, x2)))
    y1 = int(max(0, min(h - 1, y1)))

    (tw, th), baseline = cv2.getTextSize(text, FONTFACE, FONTSCALE, THICKNESS)

    tx = x2 - tw
    ty = y1 - 5

    tx = max(0, min(w - tw - 1, tx))
    if ty - th < 0:
        ty = y1 + th + 5
        ty = min(h - 2, ty)

    bx1 = tx
    by1 = ty - th - baseline
    bx2 = tx + tw
    by2 = ty + baseline
    bx1 = max(0, bx1); by1 = max(0, by1); bx2 = min(w - 1, bx2); by2 = min(h - 1, by2)
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 0, 0), thickness=-1)
    cv2.putText(frame, text, (tx, ty), FONTFACE, FONTSCALE, FONTCOLOR, THICKNESS, LINETYPE)


def get_clip_len_from_config(config, fallback=-1):
    
    for split in ['test', 'val', 'train']:
        try:
            pipe = getattr(config.data, split).pipeline
        except Exception:
            continue
        for op in pipe:
            if op.get('type') == 'UniformSample' and 'clip_len' in op:
                return int(op['clip_len'])
    return fallback


def get_num_person_from_config(config, fallback=DEFAULT_NUM_PERSON):
    for split in ['test', 'val', 'train']:
        try:
            pipe = getattr(config.data, split).pipeline
        except Exception:
            continue
        for op in pipe:
            if op.get('type') == 'FormatGCNInput' and 'num_person' in op:
                return int(op['num_person'])
    return fallback


def main():
    args = parse_args()
    config = mmcv.Config.fromfile(args.config)

    
    num_person_slot = get_num_person_from_config(config, fallback=args.num_person_slot)

    
    if args.clip_len is None or int(args.clip_len) <= 0:
        cfg_clip_len = get_clip_len_from_config(config, fallback=100)
        args.clip_len = int(cfg_clip_len)

    print(f'[INFO] num_person_slot = {num_person_slot}')
    print(f'[INFO] clip_len        = {args.clip_len}')
    print(f'[INFO] mode/step       = {args.mode}/{max(1, args.step)}')

    # extract frames
    frame_paths, original_frames = frame_extraction(args.video, args.short_side)
    num_frame = len(frame_paths)
    if num_frame == 0:
        raise RuntimeError('No frames extracted from video.')
    h, w, _ = original_frames[0].shape

    # init recognizer
    model = init_recognizer(config, args.checkpoint, args.device)
    model.eval()

    # minimal pipeline
    pipeline = make_min_test_pipeline(config, num_person_slot)

    # label map
    label_map = [x.strip() for x in open(args.label_map, 'r', encoding='utf-8').readlines()]

    # detect -> pose
    det_results = detection_inference(args, frame_paths)
    torch.cuda.empty_cache()
    pose_results = pose_inference(args, frame_paths, det_results)
    torch.cuda.empty_cache()

    # build fake_anno with tracking
    fake_anno = dict(
        frame_dir='',
        label=-1,
        img_shape=(h, w),
        original_shape=(h, w),
        start_index=0,
        modality='Pose',
        total_frames=num_frame
    )

    tracking_inputs = [[pose['keypoints'] for pose in poses] for poses in pose_results]
    keypoint, keypoint_score = pose_tracking(tracking_inputs, max_tracks=num_person_slot)
    fake_anno['keypoint'] = keypoint
    fake_anno['keypoint_score'] = keypoint_score

    if fake_anno['keypoint'] is None:
        print('No keypoints found. Exit.')
        return

    P, T, V, _ = fake_anno['keypoint'].shape
    print(f'[INFO] tracked persons={P}, frames={T}, joints={V}')

    clip_len = int(args.clip_len)
    step = max(1, int(args.step))

    # per-person per-frame predictions
    per_person_label = [[''] * num_frame for _ in range(P)]
    per_person_conf = [[0.0] * num_frame for _ in range(P)]

    for p in range(P):
        last_lab = ''
        last_conf = 0.0
        for t in range(num_frame):
            if (t % step) == 0:
                clip_anno = build_clip_for_person(fake_anno, p, t, clip_len, args.mode, num_person_slot)
                pred_idx, conf = infer_with_confidence(model, pipeline, clip_anno, args.device)
                last_lab = label_map[pred_idx] if 0 <= pred_idx < len(label_map) else str(pred_idx)
                last_conf = conf
            per_person_label[p][t] = last_lab
            per_person_conf[p][t] = last_conf

    # save predictions
    out_txt = 'per_person_predictions_with_scores.txt'
    with open(out_txt, 'w', encoding='utf-8') as f:
        f.write('person_id\tframe\tlabel\tconf\n')
        for p in range(P):
            for t in range(num_frame):
                f.write(f'{p}\t{t}\t{per_person_label[p][t]}\t{per_person_conf[p][t]:.6f}\n')
    print(f'[INFO] Saved: {out_txt}')

    # visualize pose + label on bbox top-right
    pose_model = init_pose_model(args.pose_config, args.pose_checkpoint, args.device)
    vis_frames = [vis_pose_result(pose_model, frame_paths[i], pose_results[i]) for i in range(num_frame)]

    for t in range(num_frame):
        frame = vis_frames[t]
        for p in range(P):
            kp_xy = fake_anno['keypoint'][p, t]          # (V,2)
            kp_sc = fake_anno['keypoint_score'][p, t]   # (V,)
            bbox = bbox_from_keypoints(kp_xy, kp_sc, min_score=args.min_kpt_score, pad=6)
            if bbox is None:
                continue

            lab = per_person_label[p][t]
            conf = per_person_conf[p][t]
            if conf < args.conf_thr:
                continue

            text = f'{lab} {conf:.2f}'
            draw_text_at_bbox_topright(frame, bbox, text)

    # write output video
    vid = mpy.ImageSequenceClip([x[:, :, ::-1] for x in vis_frames], fps=24)
    vid.write_videofile(args.out_filename, remove_temp=True)

    # cleanup tmp frames
    tmp_frame_dir = osp.dirname(frame_paths[0])
    shutil.rmtree(tmp_frame_dir, ignore_errors=True)
    print('[INFO] Done.')


if __name__ == '__main__':
    main()

