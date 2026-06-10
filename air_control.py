"""
Air Control - touchless control app (dark theme, customizable, resizable).

Open the app, pick a feature, adjust settings with sliders, remap gestures, and
your settings persist between sessions (saved to air_control_settings.json).

Run:  python air_control.py
Requires: opencv-python, mediapipe==0.10.21, numpy, pyautogui, PyQt5
          (Air Writing also needs scikit-learn, joblib, char_model.joblib)
"""
import sys, os, json, math, time, platform, threading
import cv2, numpy as np, mediapipe as mp, pyautogui
from PyQt5 import QtCore, QtGui, QtWidgets

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0
SCREEN_W, SCREEN_H = pyautogui.size()


def resource_path(name):
    """Find a bundled resource whether running as a script or a PyInstaller .exe.
    PyInstaller unpacks bundled data to a temp folder exposed as sys._MEIPASS."""
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, name)


def user_data_path(name):
    """Writable location for settings - next to the .exe / script, not the
    read-only bundle temp dir."""
    if getattr(sys, "frozen", False):
        base = os.path.dirname(sys.executable)
    else:
        base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, name)


SETTINGS_PATH = user_data_path("air_control_settings.json")

WRIST = 0
THUMB_TIP, INDEX_TIP, MIDDLE_TIP = 4, 8, 12
INDEX_MCP = 5
TIPS = [4, 8, 12, 16, 20]
PIPS = [3, 6, 10, 14, 18]

DEFAULTS = {
    "alpha": 0.30,
    "click_on": 0.18,
    "scroll_speed": 12,
    "pen_thickness": 14,
    "game_gas": "right",
    "game_brake": "left",
    "gestures": {
        "click": "index_thumb",
        "drag": "middle_curl",
        "scroll": "two_fingers",
        "recognize": "open_palm",
        "clear": "fist",
    },
}
ACTIVE_LEFT, ACTIVE_RIGHT = 0.30, 0.70
ACTIVE_TOP, ACTIVE_BOTTOM = 0.25, 0.65
EDGE_MARGIN = 3
DEBOUNCE = 3

class Settings:
    def __init__(self, path=SETTINGS_PATH):
        self.path = path
        self.data = json.loads(json.dumps(DEFAULTS))   # deep copy
    def load(self):
        try:
            if os.path.exists(self.path):
                with open(self.path) as f:
                    loaded = json.load(f)
                for k, v in loaded.items():
                    if k == "gestures" and isinstance(v, dict):
                        self.data["gestures"].update(v)
                    else:
                        self.data[k] = v
        except Exception:
            self.data = json.loads(json.dumps(DEFAULTS))
        return self
    def save(self):
        try:
            with open(self.path, "w") as f:
                json.dump(self.data, f, indent=2)
        except Exception:
            pass

def fingers_up(lm):
    f = [1 if lm[THUMB_TIP].x < lm[THUMB_TIP - 1].x else 0]
    for tip, pip in zip(TIPS[1:], PIPS[1:]):
        f.append(1 if lm[tip].y < lm[pip].y else 0)
    return f

def hand_size(lm):
    return math.hypot(lm[INDEX_MCP].x - lm[WRIST].x, lm[INDEX_MCP].y - lm[WRIST].y) + 1e-6

def pinch_ratio(lm, a=INDEX_TIP, b=THUMB_TIP):
    return math.hypot(lm[a].x - lm[b].x, lm[a].y - lm[b].y) / hand_size(lm)

def map_to_screen(nx, ny):
    sx = np.interp(nx, (ACTIVE_LEFT, ACTIVE_RIGHT), (EDGE_MARGIN, SCREEN_W - EDGE_MARGIN))
    sy = np.interp(ny, (ACTIVE_TOP, ACTIVE_BOTTOM), (EDGE_MARGIN, SCREEN_H - EDGE_MARGIN))
    return sx, sy

# A gesture detector maps a gesture-name to a boolean test on the hand state.
def detect_gesture(name, f, total, ratio):
    if name == "index_thumb":   return ratio < 0.18
    if name == "middle_curl":   return ratio < 0.18 and f[2] == 0
    if name == "two_fingers":   return f[1] == 1 and f[2] == 1 and total == 2
    if name == "open_palm":     return total == 5
    if name == "fist":          return total == 0
    return False

class Worker(QtCore.QThread):
    frame_ready = QtCore.pyqtSignal(QtGui.QImage)
    status_ready = QtCore.pyqtSignal(str)
    draw_ready = QtCore.pyqtSignal(QtGui.QImage)   # screen-sized strokes for fullscreen

    def __init__(self, settings):
        super().__init__()
        self.settings = settings
        self.mode = None
        self.running = False
        self._lock = threading.Lock()
        self.ema_x = self.ema_y = None
        self.pen_candidate = False
        self.pen_count = 0
        self.click_state = False
        self.dragging = False
        self.canvas = None          # camera-sized buffer (used by recognizer)
        self.screen_canvas = None    # screen-sized buffer (shown fullscreen)
        self.prev_pt = None
        self.prev_screen_pt = None
        self.prev_scroll = None
        self.sentence = ""
        self.cooldown = 0
        self.model = None
        self.label_map = None
        self.game_key = None        # which key is currently held down (gas/brake)

    def set_mode(self, mode):
        with self._lock:
            self.mode = mode
            self.ema_x = self.ema_y = None
            self.pen_candidate = False
            self.pen_count = 0
            self.click_state = False
            if self.dragging:
                try: pyautogui.mouseUp()
                except Exception: pass
            self.dragging = False
            # release any held game key so it never gets stuck when leaving Game
            if self.game_key is not None:
                try: pyautogui.keyUp(self.game_key)
                except Exception: pass
            self.game_key = None
            self.prev_pt = None
            self.prev_screen_pt = None
            self.prev_scroll = None
            if self.canvas is not None:
                self.canvas[:] = 0
            if self.screen_canvas is not None:
                self.screen_canvas[:] = 0

    def ensure_model(self):
        if self.model is None:
            import joblib
            b = joblib.load(resource_path("char_model.joblib"))
            self.model = b["model"]; self.label_map = b["label_map"]

    def _smooth(self, x, y):
        a = self.settings.data["alpha"]
        if self.ema_x is None:
            self.ema_x, self.ema_y = x, y
        else:
            self.ema_x = a * x + (1 - a) * self.ema_x
            self.ema_y = a * y + (1 - a) * self.ema_y
        return self.ema_x, self.ema_y

    def _draw_screen(self, lm):
        """Draw the fingertip stroke onto the full-screen canvas (maps the
        fingertip across the whole monitor, not just the camera frame)."""
        if self.screen_canvas is None:
            return
        sx = int(np.clip(np.interp(lm[INDEX_TIP].x, (0.05, 0.95), (0, SCREEN_W)),
                         0, SCREEN_W - 1))
        sy = int(np.clip(np.interp(lm[INDEX_TIP].y, (0.05, 0.95), (0, SCREEN_H)),
                         0, SCREEN_H - 1))
        thick = max(2, int(self.settings.data["pen_thickness"] *
                           (SCREEN_W / 640.0)))
        if self.prev_screen_pt is not None:
            cv2.line(self.screen_canvas, self.prev_screen_pt, (sx, sy), 255, thick)
        self.prev_screen_pt = (sx, sy)

    @staticmethod
    def _preprocess(canvas):
        ys, xs = np.where(canvas > 0)
        if len(xs) == 0: return None
        x0,x1,y0,y1 = xs.min(),xs.max(),ys.min(),ys.max()
        crop = canvas[y0:y1+1, x0:x1+1]
        h,w = crop.shape; side=max(h,w)
        sq = np.zeros((side,side),np.uint8)
        sq[(side-h)//2:(side-h)//2+h,(side-w)//2:(side-w)//2+w]=crop
        small = cv2.resize(sq,(20,20),interpolation=cv2.INTER_AREA)
        img = np.zeros((28,28),np.uint8); img[4:24,4:24]=small
        return (img.astype(np.float32)/255.0).reshape(1,784)

    def run(self):
        self.running = True
        cap = (cv2.VideoCapture(0, cv2.CAP_DSHOW) if platform.system()=="Windows"
               else cv2.VideoCapture(0))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1,
                                         min_detection_confidence=0.7, min_tracking_confidence=0.5)
        du = mp.solutions.drawing_utils
        while self.running:
            ok, frame = cap.read()
            if not ok: continue
            frame = cv2.flip(frame, 1)
            h,w = frame.shape[:2]
            if self.canvas is None: self.canvas = np.zeros((h,w),np.uint8)
            if self.screen_canvas is None:
                self.screen_canvas = np.zeros((SCREEN_H, SCREEN_W), np.uint8)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable=False; res=hands.process(rgb); rgb.flags.writeable=True
            with self._lock: mode = self.mode
            status = "Idle" if mode is None else mode.title()
            if res.multi_hand_landmarks and mode is not None:
                hand = res.multi_hand_landmarks[0]
                du.draw_landmarks(frame, hand, mp.solutions.hands.HAND_CONNECTIONS)
                lm = hand.landmark
                try:
                    if mode=="mouse":   status=self._do_mouse(lm)
                    elif mode=="canvas": status=self._do_canvas(lm,w,h)
                    elif mode=="writing": status=self._do_writing(lm,w,h)
                    elif mode=="game": status=self._do_game(lm)
                except Exception as e:
                    status = "error: "+str(e)[:40]
            elif mode == "game" and self.game_key is not None:
                # no hand visible in Game mode -> release so the car coasts
                try: pyautogui.keyUp(self.game_key)
                except Exception: pass
                self.game_key = None
            if mode in ("canvas","writing") and self.canvas is not None:
                col = cv2.cvtColor(self.canvas, cv2.COLOR_GRAY2BGR); col[:,:,0]=0
                frame = cv2.addWeighted(frame,1.0,col,0.8,0)
                if mode=="writing":
                    cv2.rectangle(frame,(0,h-34),(w,h),(40,40,40),-1)
                    cv2.putText(frame,"> "+self.sentence[-30:],(8,h-10),
                                cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
                # Build a transparent ARGB overlay: accent-coloured strokes
                # where drawn, fully see-through everywhere else.
                sc = self.screen_canvas
                argb = np.zeros((SCREEN_H, SCREEN_W, 4), np.uint8)
                argb[..., 0] = 255   # B  -> accent (0,210,255) in B,G,R,A order
                argb[..., 1] = 210   # G
                argb[..., 2] = 0     # R
                argb[..., 3] = sc    # A = stroke intensity (0 = transparent)
                argb = np.ascontiguousarray(argb)
                simg = QtGui.QImage(argb.data, SCREEN_W, SCREEN_H, 4*SCREEN_W,
                                    QtGui.QImage.Format_ARGB32).copy()
                self.draw_ready.emit(simg)
            disp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            qimg = QtGui.QImage(disp.data,w,h,3*w,QtGui.QImage.Format_RGB888).copy()
            self.frame_ready.emit(qimg); self.status_ready.emit(status)
            if self.cooldown>0: self.cooldown-=1
        cap.release(); hands.close()

    def _do_mouse(self, lm):
        g = self.settings.data["gestures"]
        click_on = self.settings.data["click_on"]
        f = fingers_up(lm); total=sum(f); ratio=pinch_ratio(lm)
        drag_now = detect_gesture(g["drag"], f, total, ratio)
        click_now = detect_gesture(g["click"], f, total, ratio) and not drag_now
        scroll_now = detect_gesture(g["scroll"], f, total, ratio)
        # click debounce
        if click_now != self.pen_candidate:
            self.pen_candidate=click_now; self.pen_count=1
        else:
            self.pen_count+=1
            if self.pen_count>=DEBOUNCE: self.click_state=self.pen_candidate
        if scroll_now and not (ratio<click_on):
            cy=lm[INDEX_TIP].y
            if self.prev_scroll is not None:
                dy=self.prev_scroll-cy
                if abs(dy)>0.004: pyautogui.scroll(int(dy*self.settings.data["scroll_speed"]*100))
            self.prev_scroll=cy; return "Mouse: scroll"
        self.prev_scroll=None
        if drag_now:
            if not self.dragging: pyautogui.mouseDown(); self.dragging=True
            sx,sy=self._smooth(*map_to_screen(lm[INDEX_TIP].x,lm[INDEX_TIP].y))
            pyautogui.moveTo(float(np.clip(sx,EDGE_MARGIN,SCREEN_W-EDGE_MARGIN)),
                             float(np.clip(sy,EDGE_MARGIN,SCREEN_H-EDGE_MARGIN)))
            return "Mouse: dragging"
        if self.dragging: pyautogui.mouseUp(); self.dragging=False
        if self.click_state and click_now:
            if self.pen_count==DEBOUNCE: pyautogui.click(); return "Mouse: click"
            return "Mouse: pinch"
        sx,sy=self._smooth(*map_to_screen(lm[INDEX_TIP].x,lm[INDEX_TIP].y))
        pyautogui.moveTo(float(np.clip(sx,EDGE_MARGIN,SCREEN_W-EDGE_MARGIN)),
                         float(np.clip(sy,EDGE_MARGIN,SCREEN_H-EDGE_MARGIN)))
        return "Mouse: move"

    def _do_canvas(self, lm, w, h):
        g=self.settings.data["gestures"]; click_on=self.settings.data["click_on"]
        f=fingers_up(lm); total=sum(f); ratio=pinch_ratio(lm)
        if detect_gesture(g["clear"],f,total,ratio):
            self.canvas[:]=0; self.prev_pt=None
            if self.screen_canvas is not None: self.screen_canvas[:]=0
            self.prev_screen_pt=None
            return "Canvas: cleared"
        if f[1]==1 and f[2]==0 and ratio>=click_on:
            cx,cy=int(lm[INDEX_TIP].x*w),int(lm[INDEX_TIP].y*h)
            if self.prev_pt is not None:
                cv2.line(self.canvas,self.prev_pt,(cx,cy),255,self.settings.data["pen_thickness"])
            self.prev_pt=(cx,cy)
            self._draw_screen(lm)
            return "Canvas: drawing"
        self.prev_pt=None; self.prev_screen_pt=None; return "Canvas: pen up"

    def _do_writing(self, lm, w, h):
        self.ensure_model()
        g=self.settings.data["gestures"]; click_on=self.settings.data["click_on"]
        f=fingers_up(lm); total=sum(f); ratio=pinch_ratio(lm)
        if detect_gesture(g["clear"],f,total,ratio):
            self.canvas[:]=0; self.prev_pt=None
            if self.screen_canvas is not None: self.screen_canvas[:]=0
            self.prev_screen_pt=None
            return "Writing: cleared"
        if ratio<click_on and self.cooldown==0:
            self.sentence+=" "; pyautogui.typewrite(" "); self.cooldown=18; return "Writing: space"
        if f[1]==1 and f[2]==1 and total==2 and self.cooldown==0:
            if self.sentence: self.sentence=self.sentence[:-1]; pyautogui.press("backspace")
            self.cooldown=18; return "Writing: backspace"
        if detect_gesture(g["recognize"],f,total,ratio) and self.cooldown==0:
            vec=self._preprocess(self.canvas)
            if vec is not None:
                ch=self.label_map[int(self.model.predict(vec)[0])]
                self.sentence+=ch; pyautogui.typewrite(ch)
                self.canvas[:]=0; self.prev_pt=None
                if self.screen_canvas is not None: self.screen_canvas[:]=0
                self.prev_screen_pt=None
                self.cooldown=18; return "Writing: + "+ch
            return "Writing: nothing drawn"
        if f[1]==1 and f[2]==0 and ratio>=click_on:
            cx,cy=int(lm[INDEX_TIP].x*w),int(lm[INDEX_TIP].y*h)
            if self.prev_pt is not None:
                cv2.line(self.canvas,self.prev_pt,(cx,cy),255,self.settings.data["pen_thickness"])
            self.prev_pt=(cx,cy)
            self._draw_screen(lm)
            return "Writing: drawing"
        self.prev_pt=None; self.prev_screen_pt=None; return "Writing: pen up"

    def _do_game(self, lm):
        """Hill Climb Racing control: open palm = hold gas, fist = hold brake,
        anything else = coast (release). Keys are held across frames and only
        changed when the desired key changes - so the car drives smoothly and
        the key never gets stuck."""
        f = fingers_up(lm); total = sum(f)
        gas_key = self.settings.data.get("game_gas", "right")
        brake_key = self.settings.data.get("game_brake", "left")

        if total == 5:        desired = gas_key      # open palm -> accelerate
        elif total == 0:      desired = brake_key    # fist -> brake / reverse
        else:                 desired = None         # coast

        if desired != self.game_key:
            # release the old key, press the new one (only on change)
            if self.game_key is not None:
                try: pyautogui.keyUp(self.game_key)
                except Exception: pass
            if desired is not None:
                try: pyautogui.keyDown(desired)
                except Exception: pass
            self.game_key = desired

        if desired == gas_key:   return "Game: GAS"
        if desired == brake_key: return "Game: BRAKE"
        return "Game: coast"

    def stop(self):
        if self.game_key is not None:
            try: pyautogui.keyUp(self.game_key)
            except Exception: pass
            self.game_key = None
        self.running=False; self.wait(1000)

DARK_QSS = """
QWidget { background:#0e0e14; color:#e6e6ef; font-size:13px;
          font-family:'Segoe UI','Inter',sans-serif; }
QLabel#title { font-size:24px; font-weight:700; color:#e6e6ef;
               letter-spacing:0.5px; padding:2px 0 6px 0; }
QLabel#subtitle { color:#6b6b80; font-size:12px; padding-bottom:6px; }
QPushButton { background:#1a1a24; color:#cfcfe0; border:1px solid #24242f;
              border-radius:10px; padding:11px; font-weight:500; }
QPushButton:hover { background:#23232f; border-color:#33333f; }
QPushButton:checked { background:#00c2ff; color:#08080c; font-weight:700;
                      border-color:#00c2ff; }
QPushButton#stop { background:transparent; color:#e0556e;
                   border:1px solid #e0556e; font-weight:600; }
QPushButton#stop:hover { background:#e0556e; color:#fff; }
QComboBox { background:#15151d; border:1px solid #24242f; border-radius:8px;
            padding:6px 8px; color:#cfcfe0; }
QComboBox:hover { border-color:#33333f; }
QLineEdit { background:#15151d; border:1px solid #24242f; border-radius:8px;
            padding:6px 8px; color:#cfcfe0; }
QLineEdit:focus { border-color:#00c2ff; }
QComboBox QAbstractItemView { background:#15151d; color:#cfcfe0;
            selection-background-color:#00c2ff; selection-color:#08080c;
            border:1px solid #24242f; outline:none; }
QSlider::groove:horizontal { height:4px; background:#24242f; border-radius:2px; }
QSlider::sub-page:horizontal { background:#00c2ff; border-radius:2px; }
QSlider::handle:horizontal { background:#e6e6ef; width:14px; height:14px;
            margin:-6px 0; border-radius:7px; }
QSlider::handle:horizontal:hover { background:#00c2ff; }
QGroupBox { border:1px solid #1e1e28; border-radius:12px; margin-top:14px;
            padding:14px 12px 10px 12px; background:#121219; }
QGroupBox::title { subcontrol-origin:margin; left:14px; top:2px;
            color:#8a8aa0; font-weight:600; font-size:11px;
            text-transform:uppercase; letter-spacing:1px; }
QLabel#preview { background:#08080c; border:1px solid #1e1e28; border-radius:12px; }
QLabel#status { color:#6b6b80; font-size:12px; padding-top:4px; }
"""

GESTURE_CHOICES = ["index_thumb","middle_curl","two_fingers","open_palm","fist"]


class FullscreenDraw(QtWidgets.QWidget):
    """Transparent fullscreen drawing surface - a glass sheet over the desktop.
    Strokes are painted in the accent colour; the rest stays see-through so you
    can draw over whatever is on screen. Small camera preview in a corner; Esc
    or the button exits."""
    def __init__(self, on_exit):
        super().__init__()
        self.on_exit = on_exit
        self.setWindowTitle("Air Control - Overlay")
        # Frameless, always on top, and translucent so the desktop shows through.
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint |
                            QtCore.Qt.WindowStaysOnTopHint |
                            QtCore.Qt.Tool)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        # The overlay must NOT steal keyboard focus: when Writing types a
        # recognised character via pyautogui, that keystroke should go to the
        # user's real app (their document), not back into this window. If the
        # overlay held focus, typed keys (and stray Esc-like events) would land
        # here and could close it - which was disabling Writing on its own.
        self.setAttribute(QtCore.Qt.WA_ShowWithoutActivating, True)
        self.setFocusPolicy(QtCore.Qt.NoFocus)

        self._stroke_img = None     # QImage (grayscale mask of strokes)
        self._show_cam = True

        # camera preview (child label, sits on top of the glass)
        self.cam_label = QtWidgets.QLabel(self)
        self.cam_label.setStyleSheet(
            "background:#0d0d12; border:1px solid #2a2a3a; border-radius:10px;")
        self.cam_label.setFixedSize(240, 180)

        self.exit_btn = QtWidgets.QPushButton("Exit  (Esc)", self)
        self.exit_btn.setCursor(QtCore.Qt.PointingHandCursor)
        self.exit_btn.setStyleSheet(
            "QPushButton{background:#e0556e;color:#fff;font-weight:600;"
            "border:none;border-radius:10px;padding:10px 16px;}"
            "QPushButton:hover{background:#ec6a81;}")
        self.exit_btn.clicked.connect(self.on_exit)

        self.cam_btn = QtWidgets.QPushButton("Hide camera", self)
        self.cam_btn.setCursor(QtCore.Qt.PointingHandCursor)
        self.cam_btn.setStyleSheet(
            "QPushButton{background:rgba(30,30,46,200);color:#e6e6ef;"
            "border:1px solid #2a2a3a;border-radius:10px;padding:10px 16px;}"
            "QPushButton:hover{background:rgba(50,50,70,220);}")
        self.cam_btn.clicked.connect(self.toggle_cam)

    def toggle_cam(self):
        self._show_cam = not self._show_cam
        self.cam_label.setVisible(self._show_cam)
        self.cam_btn.setText("Hide camera" if self._show_cam else "Show camera")

    def resizeEvent(self, e):
        self.cam_label.move(self.width() - 260, 20)
        self.exit_btn.move(20, 20)
        self.cam_btn.move(20, 66)
        super().resizeEvent(e)

    @QtCore.pyqtSlot(QtGui.QImage)
    def set_strokes(self, qimg):
        # store the grayscale stroke mask and repaint the glass
        self._stroke_img = qimg
        self.update()

    @QtCore.pyqtSlot(QtGui.QImage)
    def set_camera(self, qimg):
        if not self._show_cam:
            return
        pix = QtGui.QPixmap.fromImage(qimg).scaled(
            self.cam_label.width(), self.cam_label.height(),
            QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self.cam_label.setPixmap(pix)

    def paintEvent(self, e):
        if self._stroke_img is None:
            return
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, True)
        # The image already has accent-coloured strokes on a transparent
        # background, so draw it straight onto the glass surface.
        painter.drawImage(self.rect(), self._stroke_img)

    # Note: no keyPressEvent here on purpose. The overlay never holds keyboard
    # focus (so typed characters reach the user's real app), which means Esc is
    # handled by a global shortcut on the control window instead - see AirControl.


class AirControl(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Air Control")
        self.setMinimumSize(440, 560)
        self.settings = Settings().load()
        self.active = None
        self.worker = Worker(self.settings)
        self.worker.frame_ready.connect(self.on_frame)
        self.worker.status_ready.connect(self.on_status)
        self.worker.start()
        self.fs = FullscreenDraw(on_exit=self.exit_fullscreen)
        self.worker.draw_ready.connect(self.fs.set_strokes)
        self.worker.frame_ready.connect(self.fs.set_camera)
        self._build_ui()
        self.setStyleSheet(DARK_QSS)

    def _build_ui(self):
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(20, 18, 20, 16)
        root.setSpacing(12)
        title = QtWidgets.QLabel("Air Control"); title.setObjectName("title")
        root.addWidget(title)
        subtitle = QtWidgets.QLabel("Touchless control \u2014 mouse, canvas, writing")
        subtitle.setObjectName("subtitle")
        root.addWidget(subtitle)

        # collapsible preview
        self.preview = QtWidgets.QLabel("camera starting...")
        self.preview.setObjectName("preview")
        self.preview.setMinimumHeight(240)
        self.preview.setAlignment(QtCore.Qt.AlignCenter)
        root.addWidget(self.preview, 1)
        self.preview_toggle = QtWidgets.QPushButton("Hide camera preview")
        self.preview_toggle.setCursor(QtCore.Qt.PointingHandCursor)
        self.preview_toggle.clicked.connect(self.toggle_preview)
        root.addWidget(self.preview_toggle)

        # feature buttons
        feat = QtWidgets.QHBoxLayout()
        feat.setSpacing(8)
        self.buttons = {}
        for key,label in [("mouse","Mouse"),("canvas","Canvas"),("writing","Writing"),("game","Game")]:
            b=QtWidgets.QPushButton(label); b.setCheckable(True); b.setMinimumHeight(46)
            b.setCursor(QtCore.Qt.PointingHandCursor)
            b.clicked.connect(lambda _,k=key:self.toggle(k))
            feat.addWidget(b); self.buttons[key]=b
        root.addLayout(feat)

        self.stop_btn=QtWidgets.QPushButton("Stop All"); self.stop_btn.setObjectName("stop")
        self.stop_btn.setMinimumHeight(38); self.stop_btn.setCursor(QtCore.Qt.PointingHandCursor)
        self.stop_btn.clicked.connect(self.stop_all)
        root.addWidget(self.stop_btn)

        # settings sliders
        sg = QtWidgets.QGroupBox("Settings"); sgl=QtWidgets.QFormLayout(sg)
        self.sliders={}
        def add_slider(key,lo,hi,scale,label):
            s=QtWidgets.QSlider(QtCore.Qt.Horizontal); s.setMinimum(lo); s.setMaximum(hi)
            s.setValue(int(self.settings.data[key]*scale))
            val=QtWidgets.QLabel(str(self.settings.data[key]))
            def changed(v,k=key,sc=scale,lbl=val):
                real=v/sc if sc!=1 else v
                self.settings.data[k]=real; lbl.setText(str(round(real,3))); self.settings.save()
            s.valueChanged.connect(changed)
            row=QtWidgets.QHBoxLayout(); w=QtWidgets.QWidget(); w.setLayout(row)
            row.addWidget(s); row.addWidget(val)
            sgl.addRow(label,w); self.sliders[key]=s
        add_slider("alpha",5,90,100,"Smoothing")
        add_slider("click_on",8,40,100,"Click sensitivity")
        add_slider("scroll_speed",4,40,1,"Scroll speed")
        add_slider("pen_thickness",4,30,1,"Pen thickness")
        root.addWidget(sg)

        # gesture remap
        rg=QtWidgets.QGroupBox("Gesture mapping"); rgl=QtWidgets.QFormLayout(rg)
        self.combos={}
        for action in ["click","drag","scroll","recognize","clear"]:
            c=QtWidgets.QComboBox(); c.addItems(GESTURE_CHOICES)
            c.setCurrentText(self.settings.data["gestures"][action])
            c.currentTextChanged.connect(lambda val,a=action:self.remap(a,val))
            rgl.addRow(action.title(), c); self.combos[action]=c
        root.addWidget(rg)

        # game keys - type which key palm (gas) and fist (brake) should press,
        # so Game mode works with any two-key game
        gk = QtWidgets.QGroupBox("Game keys"); gkl = QtWidgets.QFormLayout(gk)
        self.gas_edit = QtWidgets.QLineEdit(str(self.settings.data.get("game_gas","right")))
        self.brake_edit = QtWidgets.QLineEdit(str(self.settings.data.get("game_brake","left")))
        for edit, key in ((self.gas_edit,"game_gas"), (self.brake_edit,"game_brake")):
            edit.setMaxLength(12)
            edit.textChanged.connect(lambda val,k=key: self.set_game_key(k, val))
        gkl.addRow("Palm \u2192 gas key", self.gas_edit)
        gkl.addRow("Fist \u2192 brake key", self.brake_edit)
        hint = QtWidgets.QLabel("e.g. right / left, or d / a, or space, up, w")
        hint.setObjectName("status")
        gkl.addRow(hint)
        root.addWidget(gk)

        self.status=QtWidgets.QLabel("Status: idle"); self.status.setObjectName("status")
        root.addWidget(self.status)

        # Esc exits fullscreen drawing. This lives on the control window (not the
        # overlay, which never holds focus). ApplicationShortcut = works even when
        # the overlay is the visible window on top.
        esc = QtWidgets.QShortcut(QtGui.QKeySequence("Escape"), self)
        esc.setContext(QtCore.Qt.ApplicationShortcut)
        esc.activated.connect(self.exit_fullscreen)

    def toggle_preview(self):
        if self.preview.isVisible():
            self.preview.hide(); self.preview_toggle.setText("Show camera preview")
        else:
            self.preview.show(); self.preview_toggle.setText("Hide camera preview")

    def remap(self, action, value):
        self.settings.data["gestures"][action]=value; self.settings.save()

    def set_game_key(self, key, value):
        # store the typed key, lowercased and trimmed; ignore empty so we never
        # save a blank that would make a pedal do nothing
        v = value.strip().lower()
        if v:
            self.settings.data[key] = v
            self.settings.save()

    def toggle(self, key):
        self.active = None if self.active==key else key
        self.worker.set_mode(self.active)
        for k,b in self.buttons.items(): b.setChecked(k==self.active)
        # Canvas and Writing use the fullscreen drawing surface; Mouse does not.
        if self.active in ("canvas", "writing"):
            self.fs.showFullScreen()   # shown without activating (keeps focus off it)
        else:
            if self.fs.isVisible():
                self.fs.hide()

    def exit_fullscreen(self):
        # Only act if the drawing overlay is actually up. This prevents a stray
        # Esc (e.g. while in Mouse mode or idle) from doing anything.
        if not self.fs.isVisible():
            return
        self.fs.hide()
        if self.active in ("canvas", "writing"):
            self.active = None
            self.worker.set_mode(None)
            for b in self.buttons.values(): b.setChecked(False)
            self.status.setText("Status: stopped")
        self.activateWindow()

    def stop_all(self):
        self.active=None; self.worker.set_mode(None)
        for b in self.buttons.values(): b.setChecked(False)
        if self.fs.isVisible(): self.fs.hide()
        self.status.setText("Status: stopped")

    @QtCore.pyqtSlot(QtGui.QImage)
    def on_frame(self, qimg):
        if not self.preview.isVisible(): return
        pix=QtGui.QPixmap.fromImage(qimg).scaled(self.preview.width(),self.preview.height(),
            QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self.preview.setPixmap(pix)

    @QtCore.pyqtSlot(str)
    def on_status(self, text): self.status.setText("Status: "+text)

    def closeEvent(self, e):
        self.settings.save(); self.fs.close(); self.worker.stop(); e.accept()

def main():
    app=QtWidgets.QApplication(sys.argv)
    win=AirControl(); win.show(); sys.exit(app.exec_())

if __name__=="__main__":
    main()
