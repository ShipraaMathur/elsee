import { useRef, useState, useCallback, useEffect, useMemo } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ScrollView,
  Dimensions,
  Platform,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Audio } from 'expo-av';
import type { RecordingOptions } from 'expo-av/build/Audio/Recording.types';
import {
  AndroidOutputFormat,
  AndroidAudioEncoder,
  IOSOutputFormat,
  IOSAudioQuality,
} from 'expo-av/build/Audio/RecordingConstants';
import * as Speech from 'expo-speech';
import { useRouter } from 'expo-router';
import { useIsFocused } from '@react-navigation/native';
import {
  Camera,
  useCameraDevice,
  useCameraPermission,
  useFrameProcessor,
  runAtTargetFps,
} from 'react-native-vision-camera';
import { runOnJS } from 'react-native-reanimated';
import { useTensorflowModel } from 'react-native-fast-tflite';
import { useResizePlugin } from 'vision-camera-resize-plugin';
import {
  parseYoloOutputLayout,
  postprocessYoloToFlat,
  maxObstacleAreaFraction,
  YOLO_INPUT_SIZE,
} from './yoloWorklet';

const { width: W } = Dimensions.get('window');

const COCO_CLASSES: string[] = [
  'person',
  'bicycle',
  'car',
  'motorcycle',
  'airplane',
  'bus',
  'train',
  'truck',
  'boat',
  'traffic light',
  'fire hydrant',
  'stop sign',
  'parking meter',
  'bench',
  'bird',
  'cat',
  'dog',
  'horse',
  'sheep',
  'cow',
  'elephant',
  'bear',
  'zebra',
  'giraffe',
  'backpack',
  'umbrella',
  'handbag',
  'tie',
  'suitcase',
  'frisbee',
  'skis',
  'snowboard',
  'sports ball',
  'kite',
  'baseball bat',
  'baseball glove',
  'skateboard',
  'surfboard',
  'tennis racket',
  'bottle',
  'wine glass',
  'cup',
  'fork',
  'knife',
  'spoon',
  'bowl',
  'banana',
  'apple',
  'sandwich',
  'orange',
  'broccoli',
  'carrot',
  'hot dog',
  'pizza',
  'donut',
  'cake',
  'chair',
  'couch',
  'potted plant',
  'bed',
  'dining table',
  'toilet',
  'tv',
  'laptop',
  'mouse',
  'remote',
  'keyboard',
  'cell phone',
  'microwave',
  'oven',
  'toaster',
  'sink',
  'refrigerator',
  'book',
  'clock',
  'vase',
  'scissors',
  'teddy bear',
  'hair drier',
  'toothbrush',
];

const DEFAULT_CONF = Number(process.env.EXPO_PUBLIC_YOLO_CONF_THRESHOLD) || 0.25;
const CAUTION_AREA =
  process.env.EXPO_PUBLIC_YOLO_CAUTION_AREA_FRACTION != null
    ? Number(process.env.EXPO_PUBLIC_YOLO_CAUTION_AREA_FRACTION)
    : 0.25;

const SPEECH_DB = -38;
const LOUD_TICKS_TO_SWITCH_ASK = 4;
const CAUTION_DEBOUNCE_MS = 4500;

function getLiveSpeechRecordingOptions(): RecordingOptions {
  const iosWav: RecordingOptions['ios'] = {
    extension: '.wav',
    outputFormat: IOSOutputFormat.LINEARPCM,
    audioQuality: IOSAudioQuality.HIGH,
    sampleRate: 16000,
    numberOfChannels: 1,
    bitRate: 256000,
    linearPCMBitDepth: 16,
    linearPCMIsBigEndian: false,
    linearPCMIsFloat: false,
  };
  const iosM4a: RecordingOptions['ios'] = {
    extension: '.m4a',
    outputFormat: IOSOutputFormat.MPEG4AAC,
    audioQuality: IOSAudioQuality.MAX,
    sampleRate: 44100,
    numberOfChannels: 1,
    bitRate: 128000,
    linearPCMBitDepth: 16,
    linearPCMIsBigEndian: false,
    linearPCMIsFloat: false,
  };
  return {
    isMeteringEnabled: true,
    ios: Platform.OS === 'ios' ? iosWav : iosM4a,
    android: {
      extension: '.m4a',
      outputFormat: AndroidOutputFormat.MPEG_4,
      audioEncoder: AndroidAudioEncoder.AAC,
      sampleRate: 16000,
      numberOfChannels: 1,
      bitRate: 64000,
    },
    web: { mimeType: 'audio/webm', bitsPerSecond: 128000 },
  };
}

type YoloDetection = {
  label: string;
  confidence: number;
  area_fraction: number;
  bbox_xyxy: [number, number, number, number];
};

/** Map bbox from frame pixels to overlay coords (preview uses uniform scale). */
function mapBboxToOverlay(
  x1: number,
  y1: number,
  x2: number,
  y2: number,
  imgW: number,
  imgH: number,
  layoutW: number,
  layoutH: number,
): { left: number; top: number; width: number; height: number } {
  const scale = Math.min(layoutW / imgW, layoutH / imgH);
  const ox = (layoutW - imgW * scale) / 2;
  const oy = (layoutH - imgH * scale) / 2;
  return {
    left: ox + x1 * scale,
    top: oy + y1 * scale,
    width: (x2 - x1) * scale,
    height: (y2 - y1) * scale,
  };
}

const TFLITE_MODEL = require('../assets/models/yolov8n.tflite');

export default function LiveScreen() {
  const router = useRouter();
  const isFocused = useIsFocused();
  const { hasPermission, requestPermission } = useCameraPermission();
  const [facing, setFacing] = useState<'back' | 'front'>('back');
  const device = useCameraDevice(facing);
  const [isLiveMode, setIsLiveMode] = useState(false);
  const [statusMsg, setStatusMsg] = useState(
    'Press START for on-device YOLO and path checks',
  );
  const [cameraLayout, setCameraLayout] = useState<{ width: number; height: number } | null>(
    null,
  );
  const [yoloOverlay, setYoloOverlay] = useState<{
    frameW: number;
    frameH: number;
    detections: YoloDetection[];
  } | null>(null);

  const delegate = 'default';
  const tfPlugin = useTensorflowModel(TFLITE_MODEL, delegate);

  const outputMeta = useMemo(() => {
    if (tfPlugin.state !== 'loaded' || !tfPlugin.model) return null;
    const shape = tfPlugin.model.outputs[0]?.shape;
    if (!shape) return null;
    return parseYoloOutputLayout(shape);
  }, [tfPlugin.state, tfPlugin.model]);

  const inputKind = useMemo(() => {
    if (tfPlugin.state !== 'loaded' || !tfPlugin.model) return null;
    const t = tfPlugin.model.inputs[0];
    if (!t?.shape || t.shape.length !== 4) return 'unsupported' as const;
    const sh = t.shape;
    const last = sh[3];
    if (sh[1] === 3) return 'unsupported' as const;
    if (last === 3 && (t.dataType === 'float32' || t.dataType === 'float16'))
      return 'float32_nhwc' as const;
    if (last === 3 && t.dataType === 'uint8') return 'uint8_nhwc' as const;
    return 'unsupported' as const;
  }, [tfPlugin.state, tfPlugin.model]);

  const lastCautionAtRef = useRef(0);
  const liveVadRef = useRef<Audio.Recording | null>(null);
  const loudTicksRef = useRef(0);
  const liveNavigatingRef = useRef(false);

  const { resize } = useResizePlugin();

  const onDetectionsNative = useCallback(
    (flat: Float32Array, fw: number, fh: number) => {
      const count = Math.min(flat[0], 40);
      const dets: YoloDetection[] = [];
      const frameArea = fw * fh;
      for (let i = 0; i < count; i++) {
        const o = 1 + i * 6;
        const x1 = flat[o];
        const y1 = flat[o + 1];
        const x2 = flat[o + 2];
        const y2 = flat[o + 3];
        const cls = Math.round(flat[o + 4]);
        const conf = flat[o + 5];
        const bw = Math.max(0, x2 - x1);
        const bh = Math.max(0, y2 - y1);
        const area_fraction =
          frameArea > 0 ? (bw * bh) / frameArea : 0;
        const label =
          cls >= 0 && cls < COCO_CLASSES.length ? COCO_CLASSES[cls] : 'unknown';
        dets.push({
          label,
          confidence: conf,
          area_fraction,
          bbox_xyxy: [x1, y1, x2, y2],
        });
      }
      if (fw > 0 && fh > 0) {
        setYoloOverlay({ frameW: fw, frameH: fh, detections: dets });
      }

      const maxFrac = maxObstacleAreaFraction(flat, fw, fh);
      if (maxFrac >= CAUTION_AREA) {
        const now = Date.now();
        if (now - lastCautionAtRef.current >= CAUTION_DEBOUNCE_MS) {
          lastCautionAtRef.current = now;
          Speech.speak('caution', { rate: 1.05, pitch: 1.0 });
        }
      }
    },
    [],
  );

  const frameProcessor = useFrameProcessor(
    (frame) => {
      'worklet';
      const model = tfPlugin.model;
      const meta = outputMeta;
      if (model == null || meta == null) return;
      if (inputKind === 'unsupported' || inputKind == null) return;

      runAtTargetFps(5, () => {
        'worklet';
        const resized =
          inputKind === 'uint8_nhwc'
            ? resize(frame, {
                scale: { width: YOLO_INPUT_SIZE, height: YOLO_INPUT_SIZE },
                pixelFormat: 'rgb',
                dataType: 'uint8',
              })
            : resize(frame, {
                scale: { width: YOLO_INPUT_SIZE, height: YOLO_INPUT_SIZE },
                pixelFormat: 'rgb',
                dataType: 'float32',
              });

        const outputs = model.runSync([resized]);
        const out = outputs[0] as Float32Array;
        const flat = postprocessYoloToFlat(
          out,
          meta.layout,
          meta.numAnchors,
          frame.width,
          frame.height,
          DEFAULT_CONF,
        );
        runOnJS(onDetectionsNative)(flat, frame.width, frame.height);
      });
    },
    [tfPlugin.model, outputMeta, inputKind, resize, onDetectionsNative],
  );

  const stopLiveVadRecording = useCallback(async () => {
    const rec = liveVadRef.current;
    liveVadRef.current = null;
    if (!rec) return;
    try {
      rec.setOnRecordingStatusUpdate(null);
      await rec.stopAndUnloadAsync();
    } catch {
      /* ignore */
    }
  }, []);

  useEffect(() => {
    return () => {
      void stopLiveVadRecording();
    };
  }, [stopLiveVadRecording]);

  const goToAskFromVoice = useCallback(() => {
    if (liveNavigatingRef.current) return;
    liveNavigatingRef.current = true;
    Speech.stop();
    void stopLiveVadRecording();
    setIsLiveMode(false);
    setYoloOverlay(null);
    setStatusMsg('Press START for on-device YOLO and path checks');
    router.push('/ask');
    setTimeout(() => {
      liveNavigatingRef.current = false;
    }, 500);
  }, [router, stopLiveVadRecording]);

  useEffect(() => {
    if (!isLiveMode) {
      void stopLiveVadRecording();
      loudTicksRef.current = 0;
      return;
    }

    let cancelled = false;

    (async () => {
      const { status } = await Audio.requestPermissionsAsync();
      if (cancelled) return;
      if (status !== 'granted') {
        setStatusMsg(
          'Live: allow microphone in Settings to switch to Ask by speaking.',
        );
        return;
      }
      await Audio.setAudioModeAsync({
        allowsRecordingIOS: true,
        playsInSilentModeIOS: true,
      });
      await stopLiveVadRecording();
      loudTicksRef.current = 0;

      try {
        const { recording } = await Audio.Recording.createAsync(
          getLiveSpeechRecordingOptions(),
          (st) => {
            if (!st.isRecording || liveNavigatingRef.current) return;
            const db = st.metering ?? -160;
            if (db > SPEECH_DB) {
              loudTicksRef.current += 1;
              if (loudTicksRef.current >= LOUD_TICKS_TO_SWITCH_ASK) {
                goToAskFromVoice();
              }
            } else {
              loudTicksRef.current = 0;
            }
          },
          100,
        );
        if (cancelled) {
          try {
            await recording.stopAndUnloadAsync();
          } catch {
            /* ignore */
          }
          return;
        }
        liveVadRef.current = recording;
      } catch {
        setStatusMsg((prev) => `${prev} — Mic busy; tap Ask to speak.`);
      }
    })();

    return () => {
      cancelled = true;
      void stopLiveVadRecording();
    };
  }, [isLiveMode, goToAskFromVoice, stopLiveVadRecording]);

  useEffect(() => {
    if (tfPlugin.state === 'error') {
      const err = 'error' in tfPlugin ? tfPlugin.error : undefined;
      setStatusMsg(
        `Model: ${err?.message ?? 'failed to load'} — add a valid yolov8n.tflite under assets/models (Ultralytics export).`,
      );
    }
  }, [tfPlugin]);

  const startLiveMode = () => {
    if (tfPlugin.state !== 'loaded') {
      setStatusMsg('Wait for the YOLO model to finish loading.');
      return;
    }
    if (inputKind === 'unsupported') {
      setStatusMsg(
        'This TFLite model input shape is not supported. Use NHWC 640×640×3 (float32 or uint8).',
      );
      return;
    }
    if (outputMeta == null) {
      setStatusMsg('Unsupported YOLO output tensor layout.');
      return;
    }
    lastCautionAtRef.current = 0;
    liveNavigatingRef.current = false;
    setIsLiveMode(true);
    setStatusMsg(
      'Live — on-device YOLO · obstacle boxes · “caution” if large in frame. Speak → Ask.',
    );
  };

  const stopLiveMode = () => {
    setIsLiveMode(false);
    setYoloOverlay(null);
    setStatusMsg('Press START for on-device YOLO and path checks');
    Speech.stop();
    void stopLiveVadRecording();
  };

  const toggleCamera = () => {
    setFacing((f) => (f === 'back' ? 'front' : 'back'));
  };

  if (!hasPermission) {
    return (
      <SafeAreaView style={s.container}>
        <View style={s.permBox}>
          <Text style={s.permIcon}>📷</Text>
          <Text style={s.permTitle}>Camera Access Needed</Text>
          <Text style={s.permSub}>SeeForMe needs your camera for live preview</Text>
          <TouchableOpacity style={s.permBtn} onPress={() => void requestPermission()}>
            <Text style={s.permBtnText}>Grant Camera Permission</Text>
          </TouchableOpacity>
        </View>
      </SafeAreaView>
    );
  }

  if (device == null) {
    return (
      <SafeAreaView style={s.container}>
        <View style={s.permBox}>
          <Text style={s.permSub}>No camera device available.</Text>
        </View>
      </SafeAreaView>
    );
  }

  const modelReady = tfPlugin.state === 'loaded' && tfPlugin.model != null;
  const fp =
    isLiveMode && modelReady && inputKind !== 'unsupported' && outputMeta != null
      ? frameProcessor
      : undefined;

  return (
    <SafeAreaView style={s.container}>
      <View style={s.header}>
        <Text style={s.title}>SeeForMe</Text>
        <View style={[s.badge, isLiveMode && s.badgeActive]}>
          <View style={[s.dot, isLiveMode && s.dotActive]} />
          <Text style={[s.badgeText, isLiveMode && s.badgeTextActive]}>
            {isLiveMode ? 'LIVE' : 'IDLE'}
          </Text>
        </View>
      </View>

      <ScrollView contentContainerStyle={s.content} showsVerticalScrollIndicator={false}>
        <View
          style={s.cameraWrap}
          onLayout={(e) => {
            const { width, height } = e.nativeEvent.layout;
            if (width > 0 && height > 0) setCameraLayout({ width, height });
          }}
        >
          <Camera
            style={s.camera}
            device={device}
            isActive={isFocused}
            frameProcessor={fp}
            pixelFormat="yuv"
            photo={false}
            video={false}
            audio={false}
          />
          {isLiveMode && yoloOverlay && cameraLayout && yoloOverlay.frameW > 0 && (
            <View style={s.boxOverlay} pointerEvents="none">
              {yoloOverlay.detections.map((det, i) => {
                const [x1, y1, x2, y2] = det.bbox_xyxy;
                const rect = mapBboxToOverlay(
                  x1,
                  y1,
                  x2,
                  y2,
                  yoloOverlay.frameW,
                  yoloOverlay.frameH,
                  cameraLayout.width,
                  cameraLayout.height,
                );
                return (
                  <View
                    key={`${det.label}-${i}-${rect.left.toFixed(0)}`}
                    style={[
                      s.boxRect,
                      {
                        left: rect.left,
                        top: rect.top,
                        width: Math.max(2, rect.width),
                        height: Math.max(2, rect.height),
                      },
                    ]}
                  >
                    <Text style={s.boxLabel} numberOfLines={1}>
                      {det.label} {Math.round(det.confidence * 100)}%
                    </Text>
                  </View>
                );
              })}
            </View>
          )}
          <View style={s.liveVideoBadge}>
            <Text style={s.liveVideoBadgeText}>LIVE</Text>
            <Text style={s.liveVideoSub}>
              {modelReady ? 'On-device YOLO' : 'Loading model…'}
            </Text>
          </View>
          {isLiveMode && (
            <View style={s.askHint}>
              <Text style={s.askHintText}>Speak to switch to Ask</Text>
            </View>
          )}

          <TouchableOpacity style={s.flipBtn} onPress={toggleCamera}>
            <Text style={s.flipText}>🔄</Text>
          </TouchableOpacity>
        </View>

        <View style={s.statusBox}>
          <Text style={s.statusText}>{statusMsg}</Text>
        </View>

        <TouchableOpacity
          style={[s.mainBtn, isLiveMode && s.mainBtnStop]}
          onPress={isLiveMode ? stopLiveMode : startLiveMode}
          disabled={!modelReady && !isLiveMode}
        >
          <Text style={s.mainBtnText}>
            {isLiveMode ? '⏹  STOP' : '▶  START'}
          </Text>
        </TouchableOpacity>
      </ScrollView>
    </SafeAreaView>
  );
}

const s = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#080B12' },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 20,
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#1E2740',
  },
  title: { fontSize: 22, fontWeight: '800', color: '#00F5C4', letterSpacing: 1 },
  badge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    backgroundColor: '#0E1320',
    borderRadius: 20,
    paddingHorizontal: 12,
    paddingVertical: 5,
    borderWidth: 1,
    borderColor: '#1E2740',
  },
  badgeActive: { borderColor: '#00F5C4' },
  dot: { width: 7, height: 7, borderRadius: 4, backgroundColor: '#3A4260' },
  dotActive: { backgroundColor: '#00F5C4' },
  badgeText: { fontSize: 10, fontWeight: '700', letterSpacing: 2, color: '#3A4260' },
  badgeTextActive: { color: '#00F5C4' },

  content: { padding: 16, gap: 12 },

  cameraWrap: {
    width: '100%',
    height: W * 0.75,
    borderRadius: 14,
    overflow: 'hidden',
    borderWidth: 1,
    borderColor: '#1E2740',
    position: 'relative',
  },
  camera: { flex: 1 },
  boxOverlay: {
    ...StyleSheet.absoluteFillObject,
    zIndex: 2,
  },
  boxRect: {
    position: 'absolute',
    borderWidth: 2,
    borderColor: '#00F5C4',
    backgroundColor: 'rgba(0,245,196,0.08)',
  },
  boxLabel: {
    position: 'absolute',
    top: -18,
    left: 0,
    backgroundColor: 'rgba(0,0,0,0.75)',
    color: '#00F5C4',
    fontSize: 9,
    fontWeight: '700',
    paddingHorizontal: 4,
    paddingVertical: 2,
    borderRadius: 4,
    overflow: 'hidden',
    maxWidth: 120,
  },
  liveVideoBadge: {
    position: 'absolute',
    top: 10,
    left: 10,
    zIndex: 6,
    backgroundColor: 'rgba(0,0,0,0.55)',
    paddingHorizontal: 10,
    paddingVertical: 6,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: 'rgba(0,245,196,0.5)',
  },
  liveVideoBadgeText: {
    color: '#00F5C4',
    fontSize: 11,
    fontWeight: '900',
    letterSpacing: 2,
  },
  liveVideoSub: { color: '#9aa3b5', fontSize: 9, marginTop: 2 },
  askHint: {
    position: 'absolute',
    bottom: 16,
    alignSelf: 'center',
    zIndex: 6,
    backgroundColor: 'rgba(0,0,0,0.6)',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 20,
  },
  askHintText: { color: '#E8EDF5', fontSize: 11, fontWeight: '600' },
  flipBtn: {
    position: 'absolute',
    top: 10,
    right: 10,
    zIndex: 6,
    backgroundColor: 'rgba(0,0,0,0.5)',
    borderRadius: 20,
    padding: 8,
  },
  flipText: { fontSize: 18 },

  statusBox: {
    backgroundColor: '#0E1320',
    borderRadius: 10,
    padding: 12,
    borderWidth: 1,
    borderColor: '#1E2740',
  },
  statusText: { color: '#C8CDD8', fontSize: 13, textAlign: 'center' },

  mainBtn: {
    backgroundColor: '#00F5C4',
    borderRadius: 14,
    padding: 16,
    alignItems: 'center',
  },
  mainBtnStop: { backgroundColor: '#FF4D6D' },
  mainBtnText: { color: '#080B12', fontWeight: '800', fontSize: 16, letterSpacing: 0.5 },

  permBox: { flex: 1, alignItems: 'center', justifyContent: 'center', gap: 16, padding: 32 },
  permIcon: { fontSize: 56 },
  permTitle: { color: '#E8EDF5', fontSize: 20, fontWeight: '800' },
  permSub: { color: '#5A6580', fontSize: 14, textAlign: 'center' },
  permBtn: { backgroundColor: '#00F5C4', borderRadius: 12, padding: 14, paddingHorizontal: 28 },
  permBtnText: { color: '#080B12', fontWeight: '800', fontSize: 15 },
});
