"""
UMACapture — scrolling screen-capture stitcher.
Python 3 / Tkinter / Pillow / mss / OpenCV
"""

import io
import tkinter as tk
from tkinter import messagebox, filedialog

import mss
import numpy as np
import cv2
from PIL import Image, ImageTk
import win32clipboard


# ---------------------------------------------------------------------------
# Constants / tuning knobs
# ---------------------------------------------------------------------------
MATCH_THRESHOLD = 0.85          # Step-4 template-match similarity threshold
MIN_OVERLAP_PX  = 20            # Minimum overlap rows to consider a valid match
MAX_OVERLAP_FRAC = 0.9          # Never accept overlap > 90 % of frame height
OVERLAY_ALPHA   = 0x55          # Dimming overlay transparency (0-255)


# ===========================================================================
# Region-selection overlay
# ===========================================================================
class RegionSelector:
    """
    Full-screen transparent overlay for click-and-drag region selection.
    Must be created on the main Tkinter thread; pass the app's root.
    """

    def __init__(self, parent_root, callback):
        self.callback = callback  # called with (x, y, w, h) or None on cancel

        with mss.MSS() as sct:
            monitors = sct.monitors   # [0] = all-in-one, [1..] = each monitor

        # Build a bounding box that covers all monitors
        left   = min(m["left"]                 for m in monitors[1:])
        top    = min(m["top"]                  for m in monitors[1:])
        right  = max(m["left"] + m["width"]   for m in monitors[1:])
        bottom = max(m["top"]  + m["height"]  for m in monitors[1:])
        self.ox, self.oy = left, top

        # Toplevel on the existing main root — no second Tk(), no threads
        self.win = tk.Toplevel(parent_root)
        self.win.overrideredirect(True)           # borderless
        # NOTE: do NOT use -fullscreen here; it conflicts with overrideredirect
        self.win.geometry(f"{right-left}x{bottom-top}+{left}+{top}")
        self.win.attributes("-alpha", 0.25)
        self.win.attributes("-topmost", True)
        self.win.configure(bg="black")
        self.win.lift()
        self.win.focus_force()

        self.canvas = tk.Canvas(self.win, cursor="crosshair",
                                bg="black", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.start_x = self.start_y = 0
        self.rect_id = None

        self.canvas.bind("<ButtonPress-1>",   self._on_press)
        self.canvas.bind("<B1-Motion>",       self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.win.bind("<Escape>", lambda e: self._finish(None))

    def _on_press(self, event):
        self.start_x, self.start_y = event.x, event.y
        if self.rect_id:
            self.canvas.delete(self.rect_id)

    def _on_drag(self, event):
        if self.rect_id:
            self.canvas.delete(self.rect_id)
        self.rect_id = self.canvas.create_rectangle(
            self.start_x, self.start_y, event.x, event.y,
            outline="red", width=2, fill="")

    def _on_release(self, event):
        x1 = min(self.start_x, event.x) + self.ox
        y1 = min(self.start_y, event.y) + self.oy
        x2 = max(self.start_x, event.x) + self.ox
        y2 = max(self.start_y, event.y) + self.oy
        w, h = x2 - x1, y2 - y1
        self._finish((x1, y1, w, h) if w > 5 and h > 5 else None)

    def _finish(self, region):
        self.win.destroy()
        self.callback(region)


# ===========================================================================
# Core image-processing functions
# ===========================================================================

def detect_identical_top_rows(frames: list[np.ndarray]) -> int:
    """
    Pixel-perfect scan: find how many rows from the top are identical
    across ALL frames.  Returns the count (may be 0).
    """
    if len(frames) < 2:
        return 0
    ref = frames[0]
    max_rows = ref.shape[0]
    identical = 0
    for row in range(max_rows):
        ref_row = ref[row]
        if all(np.array_equal(ref_row, f[row]) for f in frames[1:]):
            identical += 1
        else:
            break
    return identical


def find_overlap_rows(top_frame: np.ndarray, bottom_frame: np.ndarray) -> int:
    """
    Approximate template match: find how many rows of the BOTTOM of top_frame
    appear at the TOP of bottom_frame.

    Returns the number of overlapping rows (>= MIN_OVERLAP_PX),
    or -1 if no reliable match is found.
    """
    h = top_frame.shape[0]
    max_search = int(h * MAX_OVERLAP_FRAC)

    # Convert to grayscale for matching
    top_gray    = cv2.cvtColor(top_frame,    cv2.COLOR_RGB2GRAY)
    bottom_gray = cv2.cvtColor(bottom_frame, cv2.COLOR_RGB2GRAY)

    best_score  = -1.0
    best_overlap = -1

    # Try template sizes from large → small for robustness
    for overlap in range(max_search, MIN_OVERLAP_PX - 1, -1):
        template = top_gray[h - overlap : h]        # bottom slice of top frame
        search   = bottom_gray[0 : overlap]         # top slice of bottom frame

        if template.shape != search.shape:
            continue

        # Normalised cross-correlation over the whole slice
        result = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
        score  = float(result.max())

        if score > best_score:
            best_score   = score
            best_overlap = overlap

        # Early exit once score starts dropping significantly
        if best_score >= MATCH_THRESHOLD and score < best_score - 0.05:
            break

    if best_score >= MATCH_THRESHOLD and best_overlap >= MIN_OVERLAP_PX:
        return best_overlap
    return -1


def stitch_frames(frames: list[np.ndarray]) -> np.ndarray:
    """
    Full stitching pipeline (steps 3 → 5).
    Returns the stitched numpy image or raises ValueError on failure.
    """
    # --- Step 3: remove identical top region from frames 1..N ---
    identical_rows = detect_identical_top_rows(frames)
    cropped = [frames[0]] + [f[identical_rows:] for f in frames[1:]]

    # --- Step 4: detect and remove overlapping regions ---
    strips = [cropped[0]]
    for i in range(1, len(cropped)):
        overlap = find_overlap_rows(strips[-1], cropped[i])
        if overlap < 0:
            raise ValueError(
                f"Could not find a reliable overlap between frame {i} and "
                f"frame {i+1}.\n\nTry scrolling less between captures so "
                f"frames share more content."
            )
        # Keep only the non-overlapping bottom of the previous strip +
        # the non-overlapping portion of the current frame.
        # The previous strip is already trimmed; we trim the top of current.
        strips.append(cropped[i][overlap:])

    # --- Step 5: vertical concatenation ---
    return np.vstack(strips)


# ===========================================================================
# Clipboard helper
# ===========================================================================

def copy_image_to_clipboard(pil_image: Image.Image):
    """Copy a PIL image to the Windows clipboard as a DIB."""
    output = io.BytesIO()
    pil_image.convert("RGB").save(output, "BMP")
    data = output.getvalue()[14:]          # strip BMP file header
    output.close()

    win32clipboard.OpenClipboard()
    win32clipboard.EmptyClipboard()
    win32clipboard.SetClipboardData(win32clipboard.CF_DIB, data)
    win32clipboard.CloseClipboard()


# ===========================================================================
# Main application window
# ===========================================================================

class UMACaptureApp:
    THUMB_MAX = 120   # px — max height of per-frame thumbnails in status strip

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("UMACapture")
        self.root.minsize(600, 400)

        # State
        self.region       = None    # (x, y, w, h)
        self.frames_raw   = []      # list of PIL.Image captures
        self.result_pil   = None    # final stitched PIL.Image
        self._tk_result   = None    # PhotoImage reference (prevent GC)

        self._build_ui()
        self._update_buttons()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self):
        # ── Toolbar ────────────────────────────────────────────────────
        toolbar = tk.Frame(self.root, bd=1, relief=tk.RAISED, padx=4, pady=4)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        btn_cfg = dict(padx=8, pady=4)
        self.btn_select  = tk.Button(toolbar, text="Select Region",
                                     command=self._select_region,  **btn_cfg)
        self.btn_capture = tk.Button(toolbar, text="Capture",
                                     command=self._capture,        **btn_cfg)
        self.btn_stitch  = tk.Button(toolbar, text="Stitch",
                                     command=self._stitch,         **btn_cfg)
        self.btn_save    = tk.Button(toolbar, text="Save to File",
                                     command=self._save,           **btn_cfg)
        self.btn_reset   = tk.Button(toolbar, text="Reset",
                                     command=self._reset,          **btn_cfg)

        for btn in (self.btn_select, self.btn_capture, self.btn_stitch,
                    self.btn_save, self.btn_reset):
            btn.pack(side=tk.LEFT)

        # ── Status bar ─────────────────────────────────────────────────
        self.status_var = tk.StringVar(value="Ready — select a region to begin.")
        status_bar = tk.Label(self.root, textvariable=self.status_var,
                              bd=1, relief=tk.SUNKEN, anchor=tk.W, padx=6)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # ── Scrollable image canvas ────────────────────────────────────
        frame = tk.Frame(self.root)
        frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(frame, bg="#2b2b2b")
        vsb = tk.Scrollbar(frame, orient=tk.VERTICAL,   command=self.canvas.yview)
        hsb = tk.Scrollbar(frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        vsb.pack(side=tk.RIGHT,  fill=tk.Y)
        hsb.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.canvas.bind("<Configure>", self._on_canvas_resize)
        # Mouse-wheel scrolling
        self.canvas.bind("<MouseWheel>",
                         lambda e: self.canvas.yview_scroll(-1 * (e.delta // 120), "units"))

    # ------------------------------------------------------------------
    # Button state management
    # ------------------------------------------------------------------
    def _update_buttons(self):
        state = lambda ok: tk.NORMAL if ok else tk.DISABLED
        self.btn_capture["state"] = state(self.region is not None)
        self.btn_stitch ["state"] = state(len(self.frames_raw) >= 2)
        self.btn_save   ["state"] = state(self.result_pil is not None)

    def _set_status(self, msg: str):
        self.status_var.set(msg)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def _select_region(self):
        self.root.withdraw()
        self.root.update_idletasks()
        # Small delay so the main window is fully hidden before overlay appears
        self.root.after(120, self._open_selector)

    def _open_selector(self):
        RegionSelector(self.root, self._on_region_selected)

    def _on_region_selected(self, region):
        self.root.deiconify()
        if region:
            self.region = region
            x, y, w, h = region
            self._set_status(
                f"Region selected: {w}×{h} at ({x}, {y})  — "
                f"{len(self.frames_raw)} frame(s) captured."
            )
        else:
            self._set_status("Region selection cancelled.")
        self._update_buttons()

    def _capture(self):
        if not self.region:
            return
        x, y, w, h = self.region
        with mss.MSS() as sct:
            shot = sct.grab({"left": x, "top": y, "width": w, "height": h})
        img = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
        self.frames_raw.append(img)
        n = len(self.frames_raw)
        self._set_status(f"{n} frame{'s' if n != 1 else ''} captured.")
        self._update_buttons()

    def _stitch(self):
        if len(self.frames_raw) < 2:
            return

        self._set_status("Stitching…")
        self.root.update()

        try:
            np_frames = [np.array(f) for f in self.frames_raw]
            result_np = stitch_frames(np_frames)
            self.result_pil = Image.fromarray(result_np)
        except ValueError as e:
            messagebox.showerror("Stitch Failed", str(e))
            self._set_status("Stitch failed — see error dialog.")
            return

        # Copy to clipboard
        try:
            copy_image_to_clipboard(self.result_pil)
            clip_msg = "Copied to clipboard."
        except Exception as e:
            clip_msg = f"Clipboard copy failed: {e}"
            messagebox.showwarning("Clipboard Warning", clip_msg)

        # Show in canvas
        self._display_result()
        self._update_buttons()
        w, h = self.result_pil.size
        self._set_status(
            f"Stitched {len(self.frames_raw)} frames → {w}×{h} px. {clip_msg}"
        )

    def _save(self):
        if not self.result_pil:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG image", "*.png")],
            title="Save stitched image"
        )
        if path:
            self.result_pil.save(path, "PNG")
            self._set_status(f"Saved: {path}")

    def _reset(self):
        self.frames_raw  = []
        self.result_pil  = None
        self._tk_result  = None
        self.canvas.delete("all")
        self.canvas.configure(scrollregion=(0, 0, 0, 0))
        self._set_status(
            "Reset. "
            + (f"Region still set: {self.region[2]}×{self.region[3]}."
               if self.region else "Select a region to begin.")
        )
        self._update_buttons()

    # ------------------------------------------------------------------
    # Canvas display
    # ------------------------------------------------------------------
    def _display_result(self):
        self.canvas.delete("all")
        if not self.result_pil:
            return

        # Fit image width to canvas width if larger
        cw = self.canvas.winfo_width() or 600
        iw, ih = self.result_pil.size
        if iw > cw:
            scale  = cw / iw
            show_w = cw
            show_h = int(ih * scale)
        else:
            show_w, show_h = iw, ih

        display = self.result_pil.resize((show_w, show_h), Image.LANCZOS)
        self._tk_result = ImageTk.PhotoImage(display)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self._tk_result)
        self.canvas.configure(scrollregion=(0, 0, show_w, show_h))

    def _on_canvas_resize(self, event):
        if self.result_pil:
            self._display_result()

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    def run(self):
        self.root.mainloop()


# ===========================================================================
# Entry point
# ===========================================================================
if __name__ == "__main__":
    app = UMACaptureApp()
    app.run()
