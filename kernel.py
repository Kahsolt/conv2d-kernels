#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/10/30 

# inspect into the kernels (weights) of conv2d layer

import tkinter as tk
import tkinter.ttk as ttk
import tkinter.messagebox as tkmsg
import tkinter.filedialog as tkfdlg
from traceback import print_exc

import torch.nn.functional as F
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from modules.model import MODELS, get_model, get_first_conv2d_layer
from modules.util import kernel_norm

WINDOW_TITLE    = 'conv2d kernel'
WINDOW_SIZE     = (700, 600)
IMAGE_MAX_SIZE  = 512
RESAMPLE_METHOD = 'nearest'
HIST_FIG_SIZE   = (5, 5)
DEFAULT_MODEL   = MODELS[0]

assert IMAGE_MAX_SIZE < min(*WINDOW_SIZE)


class App:

  def __init__(self):
    self.cur_model = None    # str
    self.layer     = None    # nn.Conv2d; fisrt conv layer in model, [C_out=64, C_in=3, K_w, K_h]
    self.kernels   = None    # torch.Tensor

    self.setup_gui()
    self.init_workspace()

    try:
      self.wnd.mainloop()
    except KeyboardInterrupt:
      self.wnd.destroy()
    except: print_exc()

  def init_workspace(self):
    self._change_model()

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
      frm11 = ttk.LabelFrame(frm1, text='Model')
      frm11.pack()
      if True:
        self.var_model = tk.StringVar(frm11, value=DEFAULT_MODEL)
        cb = ttk.Combobox(frm11, state='readonly', values=MODELS, textvariable=self.var_model)
        cb.bind('<<ComboboxSelected>>', lambda evt: self._change_model())
        cb.pack()

        self.var_model_info = tk.StringVar(frm11, value='')
        lb = ttk.Label(frm11, textvariable=self.var_model_info, foreground='red')
        lb.pack()

        self.var_normalize = tk.BooleanVar(frm11, value=True)
        cb = ttk.Checkbutton(frm11, variable=self.var_normalize, text='Normalize', command=self._show)
        cb.pack(side=tk.LEFT)

      frm12 = ttk.LabelFrame(frm1, text='Channel')
      frm12.pack()
      if True:
        self.var_channel = tk.IntVar(frm12, value=-1)
        self.cb_channel = ttk.Combobox(frm12, state='readonly', values=-1, textvariable=self.var_channel)
        self.cb_channel.bind('<<ComboboxSelected>>', lambda evt: self._show())
        self.cb_channel.pack()

      frm13 = ttk.LabelFrame(frm1, text='Stats')
      frm13.pack(expand=tk.YES, fill=tk.BOTH)
      if True:
        self.vat_img_stats = tk.StringVar(frm13, '')
        lb = ttk.Label(frm13, textvariable=self.vat_img_stats)
        lb.pack()

    # right: display
    frm2 = ttk.Frame(wnd)
    frm2.pack(side=tk.RIGHT, anchor=tk.CENTER, expand=tk.YES, fill=tk.Y)
    if True:
      fig, ax = plt.subplots()
      ax.axis('off')
      fig.set_size_inches(HIST_FIG_SIZE)
      fig.tight_layout()
      cvs = FigureCanvasTkAgg(fig, frm2)
      cvs.draw()
      cvs.get_tk_widget().pack(fill=tk.BOTH, expand=tk.YES)

      self.ax = ax
      self.fig = fig
      self.cvs = cvs

  def _change_model(self):
    name = self.var_model.get()
    if name == self.cur_model: return

    try:
      model = get_model(name).eval()
      self.layer = get_first_conv2d_layer(model)
      self.kernels = self.layer.weight
      self.kernels.requires_grad = False
      C_out, C_in, _, _ = self.kernels.shape
      
      if C_in != 3:
        tkmsg.showerror('Error', 'in_channels of Conv2d must be 3 to be compatible with RGB, but got {C_in}')
        return

      self.var_model_info.set(f'found {C_out} filters')
      self.cb_channel.config(values=list(range(-1, C_out)))
      self._show()
      self.cur_model = name
    except:
      print_exc()
  
  def _update_stats(self, ch):
    k = self.kernels[ch]    # [C=3, H, W]
    C, H, W = k.shape

    info = [
      f'size: {W} x {H} x {C}',
      f'mean: {k.mean().item():.7f}',
      f'std:  {k.std().item():.7f}',
    ]
    self.vat_img_stats.set('\n'.join([f for f in info if f]))

  def _show(self):
    kernels = self.kernels
    if self.var_normalize.get():
      kernels = kernel_norm(kernels)
    
    ch = self.var_channel.get()
    if ch == -1:
      nrow = int(len(kernels) ** 0.5)
      x = make_grid(kernels, nrow=nrow)
    else:
      x = kernels[ch]
    self._update_stats(ch)

    c, h, w = x.shape
    if   h > w: shape = (IMAGE_MAX_SIZE,          IMAGE_MAX_SIZE * w // h)
    elif w > h: shape = (IMAGE_MAX_SIZE * h // w, IMAGE_MAX_SIZE)
    else:       shape = (IMAGE_MAX_SIZE,          IMAGE_MAX_SIZE)

    x.unsqueeze_(0)
    x = F.interpolate(x, shape, mode=RESAMPLE_METHOD)
    x.squeeze_(0)
    x = x.permute([1, 2, 0]).detach().cpu().numpy()

    self.ax.cla()
    self.ax.axis('off')
    self.ax.imshow(x)
    self.fig.tight_layout()
    self.cvs.draw()


if __name__ == '__main__':
  App()
