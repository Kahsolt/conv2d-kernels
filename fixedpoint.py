#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/11/1

# find the fixed-point of well-known convolutional signal filters of 2D images

import tkinter as tk
import tkinter.ttk as ttk
import tkinter.messagebox as tkmsg
from PIL import Image, ImageTk
from collections import Counter
from traceback import format_exc, print_exc
import threading

import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

from modules.pgd import pgd_kernel_fixedpoint

WINDOW_TITLE    = 'conv2d fixedpoint'
IMAGE_MFX_SIZE  = 384
CONTROL_WIDTH   = 300
WINDOW_SIZE     = (IMAGE_MFX_SIZE*2+CONTROL_WIDTH+40, 600)
TX_WIDTH        = 40
TX_HEIGHT       = 10
HIST_FIG_SIZE   = (3.8, 2)
RESAMPLE_METHOD = Image.Resampling.NEAREST

assert IMAGE_MFX_SIZE < min(*WINDOW_SIZE)

KERNELS = {
  'embossing': torch.Tensor([
    [2,  0,  0],
    [0, -1,  0],
    [0,  0, -1],
  ]),
  'embossing 3x3': torch.Tensor([   # i.e. 45°浮雕
    [-1, -1,  0],
    [-1,  0,  1],
    [ 0,  1,  1],
  ]),
  'embossing 5x5': torch.Tensor([   # i.e. 45°浮雕
    [-1, -1, -1, -1,  0],
    [-1, -1, -1,  0,  1],
    [-1, -1,  0,  1,  1],
    [-1,  0,  1,  1,  1],
    [ 0,  1,  1,  1,  1],
  ]),

  'mean': torch.Tensor([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1],
  ]),
  'guassian 3x3': torch.Tensor([
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1],
  ]),
  'guassian 5x5': torch.Tensor([
    [1,  4,  7,  4, 1],
    [4, 16, 26, 16, 4],
    [7, 26, 41, 26, 7],
    [4, 16, 26, 16, 4],
    [1,  4,  7,  4, 1],
  ]),
  'motion blur': torch.Tensor([   # i.e. 运动模糊
    [1, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 1],
  ]),

  'sharpen': torch.Tensor([     # i.e. edge detect
    [-1, -1, -1],
    [-1,  9, -1],
    [-1, -1, -1],
  ]),
  'prewitt-x': torch.Tensor([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1],
  ]),
  'prewitt-y': torch.Tensor([   # i.e. 双边滤波
    [-1, -1, -1],
    [ 0,  0,  0],
    [ 1,  1,  1],
  ]),
  'sobel-x': torch.Tensor([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1],
  ]),
  'sobel-y': torch.Tensor([
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1],
  ]),
  'laplace': torch.Tensor([
    [ 0, -1,  0],
    [-1,  4, -1],
    [ 0, -1,  0],
  ]),
  'LoG (laplace of gaussian)': torch.Tensor([
    [ 0,  0, -1,  0,  0],
    [ 0, -1, -2, -1,  0],
    [-1, -2, 16, -2, -1],
    [ 0, -1, -2, -1,  0],
    [ 0,  0, -1,  0,  0],
  ]),

  'user_defined': None,
}
DEFAULT_KERNEL = 'mean'

to_tenosr = T.ToTensor()

class App:

  def __init__(self):
    self.kernel = None    # torch.Tensor; the conv2d kernel
    self.X      = None    # torch.Tensor; raw image tensor    [B=1, C=1, H, W]
    self.FX     = None    # torch.Tensor; out image tensor    [B=1, C=1, H, W]

    self.thr    = None

    self.setup_gui()
    self.init_workspace()

    try:
      self.wnd.mainloop()
    except KeyboardInterrupt:
      self.wnd.destroy()
    except: print_exc()

  def init_workspace(self):
    self.is_working = False
    self._change_kernel()


  def setup_gui(self):
    # window
    wnd = tk.Tk()
    W, H = wnd.winfo_screenwidth(), wnd.winfo_screenheight()
    w, h = WINDOW_SIZE
    wnd.geometry(f'{w}x{h}+{(W-w)//2}+{(H-h)//2}')
    wnd.resizable(False, False)
    wnd.title(WINDOW_TITLE)
    self.wnd = wnd

    # left: control
    frm1 = ttk.Frame(wnd)
    frm1.pack(side=tk.LEFT, anchor=tk.W, expand=tk.YES, fill=tk.Y)
    if True:
      frm11 = ttk.LabelFrame(frm1, text='Input Size')
      frm11.pack()
      if True:
        self.var_input_size = tk.IntVar(frm11, value=32)
        ent = ttk.Entry(frm11, textvariable=self.var_input_size)
        ent.pack()

      frm12 = ttk.LabelFrame(frm1, text='Kernel')
      frm12.pack()
      if True:
        self.var_kernel = tk.StringVar(frm12, value=DEFAULT_KERNEL)
        cb = ttk.Combobox(frm12, state='readonly', values=list(KERNELS.keys()), textvariable=self.var_kernel)
        cb.bind('<<ComboboxSelected>>', lambda evt: self._change_kernel())
        cb.pack()

        self.var_normalize = tk.BooleanVar(frm12, value=True)
        chk = ttk.Checkbutton(frm12, text='Force Normalize', variable=self.var_normalize)
        chk.pack()

        frm121 = ttk.LabelFrame(frm12, text='Edit Precision')
        frm121.pack()
        if True:
          self.var_n_prec = tk.IntVar(frm121, value=0)
          cb = ttk.Combobox(frm121, state='readonly', values=list(range(4)), textvariable=self.var_n_prec)
          cb.bind('<<ComboboxSelected>>', lambda evt: self._change_kernel())
          cb.pack()

        self.tx = tk.Text(frm12, height=TX_HEIGHT, width=TX_WIDTH)
        self.tx.pack()

      btn = ttk.Button(frm1, text='Go!', command=self._go)
      btn.focus()
      btn.pack()

    # right: display
    frm2 = ttk.Frame(wnd)
    frm2.pack(side=tk.RIGHT, anchor=tk.CENTER, expand=tk.YES, fill=tk.BOTH)
    if True:
      def _create_display_widgets(root, title):
        ''' Top: Image, Bottom: Hist'''

        frm = ttk.LabelFrame(root, text=title)
        frm.pack(side=tk.LEFT, anchor=tk.CENTER, expand=tk.YES, fill=tk.BOTH)
        if True:
          frm1 = ttk.Frame(frm)
          frm1.pack(side=tk.TOP, anchor=tk.CENTER)
          if True:
            lb_img = ttk.Label(frm1, image=None)
            lb_img.pack(expand=tk.YES, fill=tk.BOTH)

          frm2 = ttk.Frame(frm)
          frm2.pack(side=tk.BOTTOM, anchor=tk.CENTER)
          if True:
            fig, ax = plt.subplots()
            fig.set_size_inches(HIST_FIG_SIZE)
            fig.tight_layout()
            cvs = FigureCanvasTkAgg(fig, frm2)
            cvs.draw()
            cvs.get_tk_widget().pack(fill=tk.BOTH, expand=tk.YES)
        
        return lb_img, (ax, fig, cvs)

      self.lb_img_X,  (self.ax_X,  self.fig_X,  self.cvs_X)  = _create_display_widgets(frm2, 'X')
      self.lb_img_FX, (self.ax_FX, self.fig_FX, self.cvs_FX) = _create_display_widgets(frm2, 'FX')

  def _change_kernel(self):
    name = self.var_kernel.get()

    try:
      if name == 'user_defined':
        k_desc = self.tx.get('0.0', tk.END).strip()
        k_list = eval(k_desc)
        k_np = np.asarray(k_list).astype(np.float32)
        assert len(k_np.shape) == 2
        kernel = torch.from_numpy(k_np)
      else:
        def _fmt_kernel(kernel):
          n_prec = self.var_n_prec.get()
          if n_prec == 0:
            k_str = '\n'.join(['  [' + ', '.join([str(round(x)) for x in ln]) + '],' for ln in kernel])
          else:
            k_str = '\n'.join(['  [' + ', '.join([f'{x:.{n_prec}f}' for x in ln]) + '],' for ln in kernel])
          return f'[\n{k_str}\n]'

        kernel = KERNELS.get(name)
        k_np = kernel.detach().cpu().numpy()
        k_str = _fmt_kernel(k_np)
        self.tx.delete('0.0', tk.END)
        self.tx.insert('0.0', k_str)

      print(kernel)

      if self.var_normalize.get():
        w_sum = kernel.sum()
        if w_sum > 1e-5:
          kernel = kernel / w_sum

      self.kernel = kernel
    except:
      msg = format_exc()
      print(msg)
      tkmsg.showerror(msg)

  def _update_display_widgets(self, title:str):
    # collect widgets
    x      = getattr(self, title)
    lb_img = getattr(self, f'lb_img_{title}')
    ax     = getattr(self, f'ax_{title}')
    fig    = getattr(self, f'fig_{title}')
    cvs    = getattr(self, f'cvs_{title}')

    im_f = x.permute([0, 2, 3, 1]).squeeze().detach().cpu().numpy()    # [H, W, C=3] or [H, W]
    if title == 'DX':
      im_f_norm = (im_f - im_f.min()) / (im_f.mFX() - im_f.min())     # minmFX-norm
      img = Image.fromarray((im_f_norm * 255).astype(np.uint8))
    else:
      img = Image.fromarray((im_f * 255).astype(np.uint8))
    
    h, w = img.size
    if   h > w: size = (IMAGE_MFX_SIZE,          IMAGE_MFX_SIZE * w // h)
    elif w > h: size = (IMAGE_MFX_SIZE * h // w, IMAGE_MFX_SIZE)
    else:       size = (IMAGE_MFX_SIZE,          IMAGE_MFX_SIZE)
    img = img.resize(size, resample=RESAMPLE_METHOD)

    imgtk = ImageTk.PhotoImage(img)
    lb_img.imgtk = imgtk                  # avoid gc
    lb_img.config(image=imgtk)

    # draw hist
    im_i = (im_f * 255).astype(np.int32)
    bins = 256
    vrange = (0, 255)

    ax.cla()
    ax.hist(im_i.flatten(), bins=bins, range=vrange)
    fig.tight_layout(pad=0.1)

    if len(im_i.shape) == 3:
      def plot_ch(ch, color):
        cntr = Counter(im_i[:, :, ch].flatten())
        ax.plot([cntr.get(i, 0) for i in range(256)], color)

      plot_ch(0, 'r')
      plot_ch(1, 'g')
      plot_ch(2, 'b')

    cvs.draw()

  def _go(self):
    self._change_kernel()
    if self.kernel is None: return
    if self.is_working: return

    sz = self.var_input_size.get()
    if sz < min(*self.kernel.shape):
      tkmsg.showerror('Error', 'image size should be larger than kernel!')
      return

    def _task(app):
      X_init = torch.rand(size=[1, 1, sz, sz])
      atk_proc = pgd_kernel_fixedpoint(app.kernel, X_init, alpha=0.001, steps=3000, ret_every=100)
      for X, FX in atk_proc:
        app.X, app.FX = X, FX
        app._update_display_widgets('X')
        app._update_display_widgets('FX')

      # if task done, set idle
      app.is_working = False
      app.thr = None

    self.thr = threading.Thread(target=_task, args=(self,))
    self.thr.start()
    self.is_working = True

if __name__ == '__main__':
  App()
