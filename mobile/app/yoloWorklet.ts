/**
 * YOLOv8 TFLite postprocess (worklets / frame processor).
 * Assumes Ultralytics-style output: [1, 84, N] (NCHW) or [1, N, 84] (NHW).
 * Boxes are in 640×640 model space; mapped to full frame using center-crop (matches vision-camera-resize-plugin).
 */

export const YOLO_INPUT_SIZE = 640;
export const NUM_CLASSES = 80;

/** COCO class indices treated as path obstacles (aligned with edge/pi OBSTACLE_CLASSES). */
const OBSTACLE_MASK: boolean[] = (() => {
  const m = new Array<boolean>(NUM_CLASSES).fill(false);
  const set = (i: number) => {
    if (i >= 0 && i < NUM_CLASSES) m[i] = true;
  };
  [
    0, 1, 2, 3, 5, 7, 15, 16, 39, 56, 57, 58, 59, 60, 62, 63,
  ].forEach(set);
  return m;
})();

function iou(
  ax1: number,
  ay1: number,
  ax2: number,
  ay2: number,
  bx1: number,
  by1: number,
  bx2: number,
  by2: number,
): number {
  'worklet';
  const ix1 = Math.max(ax1, bx1);
  const iy1 = Math.max(ay1, by1);
  const ix2 = Math.min(ax2, bx2);
  const iy2 = Math.min(ay2, by2);
  const iw = Math.max(0, ix2 - ix1);
  const ih = Math.max(0, iy2 - iy1);
  const inter = iw * ih;
  const a = Math.max(0, ax2 - ax1) * Math.max(0, ay2 - ay1);
  const b = Math.max(0, bx2 - bx1) * Math.max(0, by2 - by1);
  const u = a + b - inter;
  return u > 0 ? inter / u : 0;
}

function mapModelBoxToFrame(
  x1m: number,
  y1m: number,
  x2m: number,
  y2m: number,
  fw: number,
  fh: number,
): [number, number, number, number] {
  'worklet';
  const cropSize = Math.min(fw, fh);
  const cropX = (fw - cropSize) / 2;
  const cropY = (fh - cropSize) / 2;
  const s = cropSize / YOLO_INPUT_SIZE;
  const x1 = x1m * s + cropX;
  const y1 = y1m * s + cropY;
  const x2 = x2m * s + cropX;
  const y2 = y2m * s + cropY;
  return [
    Math.max(0, Math.min(fw, x1)),
    Math.max(0, Math.min(fh, y1)),
    Math.max(0, Math.min(fw, x2)),
    Math.max(0, Math.min(fh, y2)),
  ];
}

export type YoloOutputLayout = 'nchw' | 'nhw';

export function parseYoloOutputLayout(shape: number[]): {
  layout: YoloOutputLayout;
  numAnchors: number;
} | null {
  if (shape.length !== 3 || shape[0] !== 1) return null;
  if (shape[1] === 84 && shape[2]! > 0) {
    return { layout: 'nchw', numAnchors: shape[2]! };
  }
  if (shape[2] === 84 && shape[1]! > 0) {
    return { layout: 'nhw', numAnchors: shape[1]! };
  }
  return null;
}

function readPred(
  out: Float32Array,
  layout: YoloOutputLayout,
  numAnchors: number,
  i: number,
): { cx: number; cy: number; w: number; h: number; cls: number; score: number } {
  'worklet';
  let cx: number;
  let cy: number;
  let w: number;
  let h: number;
  let bestCls = 0;
  let bestScore = -1;
  if (layout === 'nchw') {
    cx = out[0 * numAnchors + i];
    cy = out[1 * numAnchors + i];
    w = out[2 * numAnchors + i];
    h = out[3 * numAnchors + i];
    for (let k = 0; k < NUM_CLASSES; k++) {
      const sc = out[(4 + k) * numAnchors + i];
      if (sc > bestScore) {
        bestScore = sc;
        bestCls = k;
      }
    }
  } else {
    const b = i * 84;
    cx = out[b + 0];
    cy = out[b + 1];
    w = out[b + 2];
    h = out[b + 3];
    for (let k = 0; k < NUM_CLASSES; k++) {
      const sc = out[b + 4 + k];
      if (sc > bestScore) {
        bestScore = sc;
        bestCls = k;
      }
    }
  }
  return { cx, cy, w, h, cls: bestCls, score: bestScore };
}

const MAX_CAND = 120;
const MAX_OUT = 40;

/**
 * Returns flat buffer: [n, x1,y1,x2,y2,cls,score, ...] padded to MAX_OUT*6+1 (first slot = count).
 */
export function postprocessYoloToFlat(
  out: Float32Array,
  layout: YoloOutputLayout,
  numAnchors: number,
  fw: number,
  fh: number,
  confThreshold: number,
): Float32Array {
  'worklet';
  const candX1 = new Float32Array(MAX_CAND);
  const candY1 = new Float32Array(MAX_CAND);
  const candX2 = new Float32Array(MAX_CAND);
  const candY2 = new Float32Array(MAX_CAND);
  const candCls = new Int32Array(MAX_CAND);
  const candScore = new Float32Array(MAX_CAND);
  let n = 0;

  for (let i = 0; i < numAnchors; i++) {
    const p = readPred(out, layout, numAnchors, i);
    if (p.score < confThreshold || !OBSTACLE_MASK[p.cls]) continue;
    const x1m = p.cx - p.w / 2;
    const y1m = p.cy - p.h / 2;
    const x2m = p.cx + p.w / 2;
    const y2m = p.cy + p.h / 2;
    const [x1, y1, x2, y2] = mapModelBoxToFrame(x1m, y1m, x2m, y2m, fw, fh);
    if (n < MAX_CAND) {
      candX1[n] = x1;
      candY1[n] = y1;
      candX2[n] = x2;
      candY2[n] = y2;
      candCls[n] = p.cls;
      candScore[n] = p.score;
      n++;
    }
  }

  if (n === 0) {
    const buf = new Float32Array(1 + MAX_OUT * 6);
    buf[0] = 0;
    return buf;
  }

  // Sort indices by score (desc) — insertion sort for small n
  const ord = new Int32Array(n);
  for (let i = 0; i < n; i++) ord[i] = i;
  for (let i = 1; i < n; i++) {
    const key = ord[i];
    const ks = candScore[key];
    let j = i - 1;
    while (j >= 0 && candScore[ord[j]] < ks) {
      ord[j + 1] = ord[j];
      j--;
    }
    ord[j + 1] = key;
  }

  const keepX1 = new Float32Array(MAX_OUT);
  const keepY1 = new Float32Array(MAX_OUT);
  const keepX2 = new Float32Array(MAX_OUT);
  const keepY2 = new Float32Array(MAX_OUT);
  const keepCls = new Int32Array(MAX_OUT);
  const keepScore = new Float32Array(MAX_OUT);
  let kc = 0;
  const iouTh = 0.45;

  for (let t = 0; t < n && kc < MAX_OUT; t++) {
    const idx = ord[t];
    const x1 = candX1[idx];
    const y1 = candY1[idx];
    const x2 = candX2[idx];
    const y2 = candY2[idx];
    const cls = candCls[idx];
    const sc = candScore[idx];
    let skip = false;
    for (let u = 0; u < kc; u++) {
      if (keepCls[u] !== cls) continue;
      if (
        iou(x1, y1, x2, y2, keepX1[u], keepY1[u], keepX2[u], keepY2[u]) > iouTh
      ) {
        skip = true;
        break;
      }
    }
    if (skip) continue;
    keepX1[kc] = x1;
    keepY1[kc] = y1;
    keepX2[kc] = x2;
    keepY2[kc] = y2;
    keepCls[kc] = cls;
    keepScore[kc] = sc;
    kc++;
  }

  const buf = new Float32Array(1 + MAX_OUT * 6);
  buf[0] = kc;
  for (let i = 0; i < kc; i++) {
    const o = 1 + i * 6;
    buf[o + 0] = keepX1[i];
    buf[o + 1] = keepY1[i];
    buf[o + 2] = keepX2[i];
    buf[o + 3] = keepY2[i];
    buf[o + 4] = keepCls[i];
    buf[o + 5] = keepScore[i];
  }
  return buf;
}

export function maxObstacleAreaFraction(
  buf: Float32Array,
  fw: number,
  fh: number,
): number {
  const frameArea = fw * fh;
  if (frameArea <= 0) return 0;
  const count = Math.min(buf[0], MAX_OUT);
  let maxF = 0;
  for (let i = 0; i < count; i++) {
    const o = 1 + i * 6;
    const x1 = buf[o];
    const y1 = buf[o + 1];
    const x2 = buf[o + 2];
    const y2 = buf[o + 3];
    const bw = Math.max(0, x2 - x1);
    const bh = Math.max(0, y2 - y1);
    const f = (bw * bh) / frameArea;
    if (f > maxF) maxF = f;
  }
  return maxF;
}
