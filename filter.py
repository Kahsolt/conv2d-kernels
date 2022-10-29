#!/usr/bin/env pythonw3
# Author: Armit
# Create Time: 2022/10/28 

import os
import tkinter as tk
import tkinter.ttk as ttk
import tkinter.messagebox as tkmsg
import tkinter.filedialog as tkfdlg
from PIL import Image, ImageTk
from collections import Counter
from traceback import print_exc

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

from modules.model import MODELS, get_model, get_first_conv2d_layer
from modules.data import DATASETS, get_dataloader, normalize
from modules.env import device
from modules.pgd import pgd

WINDOW_TITLE    = 'conv2d filter'
WINDOW_SIZE     = (700, 600)
IMAGE_MAX_SIZE  = 512
CONTROL_WIDTH   = 140
HIST_FIG_SIZE   = (1.5, 1)
RESAMPLE_METHOD = Image.LANCZOS
NUM_CLASSES     = 1000

assert IMAGE_MAX_SIZE < min(*WINDOW_SIZE)

DEFAULT_MODEL   = MODELS[0]
DEFAULT_DATASET = DATASETS[-1]

avg_pool  = nn.AvgPool2d(kernel_size=2, stride=2).to(device)    # fix shape between original and feature map
rgb2grey  = T.Grayscale()
to_tenosr = T.ToTensor()


class App:

  def __init__(self):
    self.cur_model   = None    # str
    self.cur_dataset = None    # str
    self.cur_mode    = 'rgb'   # 'grey' | 'rgb'

    self.datagen  = None    # iter(DataLoader)
    self.model    = None    # nn.Module, the whole model
    self.layer    = None    # nn.Conv2d; fisrt conv layer in model                    [C_out=64, C_in=3, K_w, K_h]
    self.src      = None    # torch.Tensor; raw image tensor                          [B=1, C=3, H, W]
    self.tgt      = None    # torch.Tensor; truth label if available                  [B=1]
    self.tgt_atk  = None    # torch.Tensor; attack label if available                 [B=1]
    self.src_half = None    # torch.Tensor; raw image tensor half size                [B=1, C=3, H/2, W/2]
    self.src_grey = None    # torch.Tensor; raw image tensor half size (grey scale)   [B=1, C=1, H/2, W/2]
    self.out      = None    # torch.Tensor; output feature map tensor                 [B=1, C=64, H/2, W/2]
    self.prob     = None    # torch.Tensor; predicted probo distribution              [B=1, N_CLASS=1000]

    self.setup_gui()
    self.init_workspace()

    try:
      self.wnd.mainloop()
    except KeyboardInterrupt:
      self.wnd.destroy()
    except: print_exc()

  def init_workspace(self):
    self._change_dataset()
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

      frm12 = ttk.LabelFrame(frm1, text='Color Mode')
      frm12.pack()
      if True:
        frm121 = ttk.LabelFrame(frm12, text='Grey')
        frm121.pack()
        if True:
          def show_grey_fn(evt):
            self.cur_mode = 'grey'
            self._show()

          self.var_I = tk.IntVar(frm121, value=-1)
          self.cb_I = ttk.Combobox(frm121, state='readonly', values=-1, textvariable=self.var_I)
          self.cb_I.bind('<<ComboboxSelected>>', show_grey_fn)
          self.cb_I.pack(expand=tk.YES, fill=tk.X)
        
        frm122 = ttk.LabelFrame(frm12, text='RGB')
        frm122.pack()
        if True:
          frm1221 = ttk.Frame(frm122)
          frm1221.pack()
          if True:
            def show_rgb_fn(ch):
              self.cur_mode = 'rgb'
              self._show()
          
            self.var_R = tk.IntVar(frm1221, value=-1)
            self.cb_R = ttk.Combobox(frm1221, state='readonly', values=-1, textvariable=self.var_R, width=4)
            self.cb_R.bind('<<ComboboxSelected>>', lambda evt: show_rgb_fn('R'))
            self.cb_R.pack(side=tk.LEFT, expand=tk.NO)
            
            self.var_G = tk.IntVar(frm1221, value=-1)
            self.cb_G = ttk.Combobox(frm1221, state='readonly', values=-1, textvariable=self.var_G, width=4)
            self.cb_G.bind('<<ComboboxSelected>>', lambda evt: show_rgb_fn('G'))
            self.cb_G.pack(side=tk.LEFT, expand=tk.NO)
            
            self.var_B = tk.IntVar(frm1221, value=-1)
            self.cb_B = ttk.Combobox(frm1221, state='readonly', values=-1, textvariable=self.var_B, width=4)
            self.cb_B.bind('<<ComboboxSelected>>', lambda evt: show_rgb_fn('B'))
            self.cb_B.pack(side=tk.LEFT, expand=tk.NO)

        btn = ttk.Button(frm12, text='Reset', command=self._reset)
        btn.pack(expand=tk.YES, fill=tk.X)
      
      frm13 = ttk.LabelFrame(frm1, text='Data')
      frm13.pack()
      if True:
        frm131 = ttk.LabelFrame(frm13, text='Dataset')
        frm131.pack(expand=tk.YES, fill=tk.X)
        if True:
          self.var_dataset = tk.StringVar(frm131, value=DEFAULT_DATASET)
          cb = ttk.Combobox(frm131, state='readonly', values=DATASETS, textvariable=self.var_dataset)
          cb.bind('<<ComboboxSelected>>', lambda evt: self._change_dataset())
          cb.pack(expand=tk.YES, fill=tk.X)

          btn = ttk.Button(frm131, text='Next', command=self._next)
          btn.pack(expand=tk.YES, fill=tk.X)

        frm132 = ttk.LabelFrame(frm13, text='Local File')
        frm132.pack(expand=tk.YES, fill=tk.X)
        if True:
          self.var_fp_info = tk.StringVar(frm132, '')
          self.lb_fp_info = ttk.Label(frm132, textvariable=self.var_fp_info, wraplength=CONTROL_WIDTH, foreground='blue')

          btn = ttk.Button(frm132, text='Open..', command=self._open)
          btn.pack(expand=tk.YES, fill=tk.X)

      frm16 = ttk.LabelFrame(frm1, text='Attack')
      frm16.pack()
      if True:
        ATK_TGT = ['random', 'second prob', 'least prob']
        self.var_atk_tgt = tk.StringVar(frm16, value=ATK_TGT[0])
        cb = ttk.Combobox(frm16, values=ATK_TGT, textvariable=self.var_atk_tgt)
        cb.pack(expand=tk.YES, fill=tk.X)
        
        btn = ttk.Button(frm16, text='PGD Attack!', command=self._attack)
        btn.pack(expand=tk.YES, fill=tk.X)

      frm14 = ttk.LabelFrame(frm1, text='Stats')
      frm14.pack(expand=tk.YES, fill=tk.X)
      if True:
        self.vat_img_stats = tk.StringVar(frm14, '')
        lb = ttk.Label(frm14, textvariable=self.vat_img_stats)
        lb.pack()

      frm15 = ttk.LabelFrame(frm1, text='Hist')
      frm15.pack()
      if True:
        fig, ax = plt.subplots()
        fig.set_size_inches(HIST_FIG_SIZE)
        fig.tight_layout()
        cvs = FigureCanvasTkAgg(fig, frm15)
        cvs.draw()
        cvs.get_tk_widget().pack(fill=tk.BOTH, expand=tk.YES)

        self.ax = ax
        self.fig = fig
        self.cvs = cvs

    # right: display
    frm2 = ttk.Frame(wnd)
    frm2.pack(side=tk.RIGHT, anchor=tk.CENTER, expand=tk.YES, fill=tk.Y)
    if True:
      self.lb_img = ttk.Label(frm2, image=None)
      self.lb_img.pack(anchor=tk.CENTER, expand=tk.YES, fill=tk.BOTH)

  def _change_model(self):
    name = self.var_model.get()
    if name == self.cur_model: return

    try:
      self.model = get_model(name).eval()
      self.layer = get_first_conv2d_layer(self.model)
      W = self.layer.weight
      in_channels = W.shape[1]
      assert in_channels == 3     # must be compatible with RGB
      n_filters = W.shape[0]
      self.var_model_info.set(f'found {n_filters} filters')

      vals = list(range(-1, n_filters))
      self.cb_I.config(values=vals)
      self.cb_R.config(values=vals)
      self.cb_G.config(values=vals)
      self.cb_B.config(values=vals)

      self._forward()
      self.cur_model = name
    except:
      print_exc()
  
  def _change_dataset(self):
    name = self.var_dataset.get()
    if name == self.cur_dataset: return

    try:
      self.datagen = iter(get_dataloader(name))
      self._next()
      self.cur_dataset = name
    except:
      print_exc()

  def _mk_imgs(self, x, y=None):
    self.src = x
    self.src_half = avg_pool(self.src)
    self.src_grey = rgb2grey(self.src_half)
    self.tgt = y
  
  def _forward(self):
    if None in [self.layer, self.src]: return

    with torch.inference_mode():
      x_norm = normalize(self.src, self.cur_dataset)
      self.out = self.layer(x_norm)
      self.prob = F.softmax(self.model(x_norm), dim=-1)
    
    self._show()
  
  def _next(self):
    self.lb_fp_info.pack_forget()

    x, y = next(self.datagen)
    x, y = x.to(device), y.to(device)
    self.tgt_atk = None
    self._mk_imgs(x, y)
    self._forward()

  def _open(self):
    fp = tkfdlg.askopenfilename()
    if not fp or not os.path.exists(fp):
      tkmsg.showerror('Error', 'File not exists!')
      return

    self.var_fp_info.set(os.path.basename(fp))
    self.lb_fp_info.pack()

    img = Image.open(fp).convert('RGB')
    x = to_tenosr(img).unsqueeze_(0)
    self.tgt_atk = None
    self._mk_imgs(x)
    self._forward()

  def _reset(self):
    self.var_I.set(-1)
    self.var_R.set(-1)
    self.var_G.set(-1)
    self.var_B.set(-1)
    self.cur_mode = 'rgb'

    self._show()

  def _update_stats(self, img: Image):
    x = np.asarray(img, dtype=np.uint8)
    x_f = x / 255.0
    try:    H, W, C = x.shape
    except: (H, W), C = x.shape, 1

    prob = self.prob[0]
    info = [
      f'truth: {self.tgt.item()}' if self.tgt else None,
      f'attack: {self.tgt_atk.item()}' if self.tgt_atk else None,
      f'pred: {prob.argmax()} ({prob[prob.argmax()]:.2%})',
      f'size: {W} x {H} x {C}',
      f'mean: {x_f.mean():.7f}',
      f'std:  {x_f.std():.7f}',
    ]
    self.vat_img_stats.set('\n'.join([f for f in info if f]))

    self.ax.cla()
    self.ax.axis('off')
    self.ax.hist(x.flatten(), bins=256)
    self.fig.tight_layout(pad=0.1)

    if C == 3:
      def plot_ch(ch, color):
        cntr = Counter(x[:, :, ch].flatten())
        self.ax.plot([cntr.get(i, 0) for i in range(256)], color)

      plot_ch(0, 'r')
      plot_ch(1, 'g')
      plot_ch(2, 'b')

    self.cvs.draw()

  def _attack(self):
    atk_tgt = self.var_atk_tgt.get().strip()
    if atk_tgt.isdigit():
      atk_tgt = int(atk_tgt)
      if 0 <= atk_tgt < NUM_CLASSES:
        y = torch.LongTensor([atk_tgt])
      else:
        tkmsg.showerror('Error', f'wrong class id {atk_tgt}')
        return
    elif atk_tgt == 'random':
      y = torch.randint(0, NUM_CLASSES, [1])
    else:
      logits = self.model(normalize(self.src, self.cur_dataset))[0]
      if atk_tgt == 'second prob':
        logits[logits.argmax()] = logits.min() - 1
        y = logits.argmax().unsqueeze(0)
      elif atk_tgt == 'least prob':
        y = logits.argmin().unsqueeze(0)
    y = y.to(device)
    self.tgt_atk = y.to(device)

    x = pgd(self.model, self.src, self.tgt_atk, normalizer=lambda x: normalize(x, self.cur_dataset))
    self._mk_imgs(x, self.tgt)
    self._forward()

  def _show(self):
    if self.cur_mode == 'grey':
      f = self.var_I.get()
      if f == -1:
        x = self.src_grey                 # [B=1, C=1, H, W]
      else:
        x = self.out[:, f:f+1, :, :]      # [B=1, C=1, H, W]
    elif self.cur_mode == 'rgb':
      fR = self.var_R.get()
      if fR == -1: R = self.src_half[:, 0, :, :]    # [B=1, H, W]
      else:        R = self.out[:, fR, :, :]
      fG = self.var_G.get()
      if fG == -1: G = self.src_half[:, 1, :, :]
      else:        G = self.out[:, fG, :, :]
      fB = self.var_B.get()
      if fB == -1: B = self.src_half[:, 2, :, :]
      else:        B = self.out[:, fB, :, :]
      x = torch.stack([R, G, B], axis=1)            # [B=1, C=3, H, W] 
    else: return

    im = x.permute([0, 2, 3, 1]).squeeze().detach().cpu().numpy()    # [H, W, C=3] or [H, W]
    img = Image.fromarray((im * 255).astype(np.uint8))
    self._update_stats(img)

    h, w = img.size
    if h > w:   size = (IMAGE_MAX_SIZE, IMAGE_MAX_SIZE * w // h)
    elif w > h: size = (IMAGE_MAX_SIZE * h // w, IMAGE_MAX_SIZE)
    else:       size = (IMAGE_MAX_SIZE, IMAGE_MAX_SIZE)
    img = img.resize(size, resample=RESAMPLE_METHOD)

    imgtk = ImageTk.PhotoImage(img)
    self.lb_img.imgtk = imgtk           # avoid gc
    self.lb_img.config(image=imgtk)


if __name__ == '__main__':
  App()
