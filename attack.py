#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/10/28 

# pgd attack on model or single conv2d layer

import os
import tkinter as tk
import tkinter.ttk as ttk
import tkinter.messagebox as tkmsg
import tkinter.filedialog as tkfdlg
from PIL import Image, ImageTk
from collections import Counter
from traceback import format_exc, print_exc

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
from modules.pgd import ATK_METH, pgd, pgd_conv

WINDOW_TITLE    = 'conv2d attack'
IMAGE_MAX_SIZE  = 384
CONTROL_WIDTH   = 140
WINDOW_SIZE     = (IMAGE_MAX_SIZE*3+CONTROL_WIDTH+40, 670)
HIST_FIG_SIZE   = (3.8, 2)
RESAMPLE_METHOD = Image.Resampling.NEAREST
NUM_CLASSES     = 1000

assert IMAGE_MAX_SIZE < min(*WINDOW_SIZE)

ATK_TGT = ['random', 'second prob', 'least prob']

DEFAULT_MODEL    = MODELS  [0]
DEFAULT_TMODEL   = MODELS  [-1]
DEFAULT_DATASET  = DATASETS[-1]
DEFAULT_ATK_TGT  = ATK_TGT [0]
DEFAULT_ATK_METH = ATK_METH[0]

to_tenosr = T.ToTensor()


class App:

  def __init__(self):
    self.cur_model   = None    # str
    self.cur_dataset = None    # str
    self.cur_tmodel  = None    # str

    self.datagen  = None    # iter(DataLoader)
    self.model    = None    # nn.Module, the whole model for adv example gen
    self.layer    = None    # nn.Conv2d, the first Conv2d layer of self.model
    self.tmodel   = None    # nn.Module, the whole model for transfer attack test
    self.X        = None    # torch.Tensor; raw image tensor            [B=1, C=3, H, W]
    self.AX       = None    # torch.Tensor; adv image tensor            [B=1, C=3, H, W]
    self.DX       = None    # torch.Tensor; diff image tensor           [B=1, C=3, H, W]
    self.Y        = None    # torch.Tensor; truth label if available    [B=1]
    self.Y_atk    = None    # torch.Tensor; attack label if available   [B=1]
    self.pred_X   = None    # torch.Tensor; output logits               [B=1, NUM_CLASS]
    self.pred_AX  = None    # torch.Tensor; output logits               [B=1, NUM_CLASS]

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
    self._change_tmodel()

  def setup_gui(self):
    # window
    wnd = tk.Tk()
    W, H = wnd.winfo_screenwidth(), wnd.winfo_screenheight()
    w, h = WINDOW_SIZE
    wnd.geometry(f'{w}x{h}+{(W-w)//2}+{(H-h)//2}')
    wnd.resizable(False, False)
    wnd.title(WINDOW_TITLE)
    wnd.bind('<Return>', lambda evt: self._attack())
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
        lb = ttk.Label(frm11, textvariable=self.var_model_info, foreground='blue')
        lb.pack()

      frm12 = ttk.LabelFrame(frm1, text='Data')
      frm12.pack()
      if True:
        frm121 = ttk.LabelFrame(frm12, text='Dataset')
        frm121.pack(expand=tk.YES, fill=tk.X)
        if True:
          self.var_dataset = tk.StringVar(frm121, value=DEFAULT_DATASET)
          cb = ttk.Combobox(frm121, state='readonly', values=DATASETS, textvariable=self.var_dataset)
          cb.bind('<<ComboboxSelected>>', lambda evt: self._change_dataset())
          cb.pack(expand=tk.YES, fill=tk.X)

          btn = ttk.Button(frm121, text='Next', command=self._next)
          btn.pack(expand=tk.YES, fill=tk.X)

        frm122 = ttk.LabelFrame(frm12, text='Local File')
        frm122.pack(expand=tk.YES, fill=tk.X)
        if True:
          self.var_fp_info = tk.StringVar(frm122, '')
          self.lb_fp_info = ttk.Label(frm122, textvariable=self.var_fp_info, wraplength=CONTROL_WIDTH, foreground='blue')

          btn = ttk.Button(frm122, text='Open..', command=self._open)
          btn.pack(expand=tk.YES, fill=tk.X)

      frm13 = ttk.LabelFrame(frm1, text='Attack')
      frm13.pack()
      if True:
        frm130 = ttk.LabelFrame(frm13, text='method')
        frm130.pack(expand=tk.YES, fill=tk.X)
        if True:
          self.var_atk_meth = tk.StringVar(frm130, value=DEFAULT_ATK_METH)
          cb = ttk.Combobox(frm130, values=ATK_METH, textvariable=self.var_atk_meth)
          cb.pack(expand=tk.YES, fill=tk.X)

        frm131 = ttk.LabelFrame(frm13, text='target')
        frm131.pack(expand=tk.YES, fill=tk.X)
        if True:
          self.var_atk_tgt = tk.StringVar(frm131, value=DEFAULT_ATK_TGT)
          cb = ttk.Combobox(frm131, values=ATK_TGT, textvariable=self.var_atk_tgt)
          cb.pack(expand=tk.YES, fill=tk.X)
        
        frm132 = ttk.LabelFrame(frm13, text='eps')
        frm132.pack(expand=tk.YES, fill=tk.X)
        if True:
          self.var_atk_eps = tk.DoubleVar(frm132, value=0.03)
          ent = ttk.Entry(frm132, textvariable=self.var_atk_eps)
          ent.pack(expand=tk.YES, fill=tk.X)
        
        frm133 = ttk.LabelFrame(frm13, text='alpha')
        frm133.pack(expand=tk.YES, fill=tk.X)
        if True:
          self.var_atk_alpha = tk.DoubleVar(frm133, value=0.001)
          ent = ttk.Entry(frm133, textvariable=self.var_atk_alpha)
          ent.pack(expand=tk.YES, fill=tk.X)
        
        frm134 = ttk.LabelFrame(frm13, text='step')
        frm134.pack(expand=tk.YES, fill=tk.X)
        if True:
          self.var_atk_step = tk.IntVar(frm134, value=10)
          ent = ttk.Entry(frm134, textvariable=self.var_atk_step)
          ent.pack(expand=tk.YES, fill=tk.X)

        btn = ttk.Button(frm13, text='Attack!', command=self._attack)
        btn.focus()
        btn.pack(expand=tk.YES, fill=tk.X)

        frm135 = ttk.LabelFrame(frm13, text='Transfer Model')
        frm135.pack(expand=tk.YES, fill=tk.X)
        if True:
          self.var_tmodel = tk.StringVar(frm135, value=DEFAULT_TMODEL)
          cb = ttk.Combobox(frm135, state='readonly', values=MODELS, textvariable=self.var_tmodel)
          cb.bind('<<ComboboxSelected>>', lambda evt: self._change_tmodel())
          cb.pack(expand=tk.YES, fill=tk.X)
        
        self.var_tattack_info = tk.StringVar(frm13, value='')
        ttk.Label(frm13, textvariable=self.var_tattack_info, foreground='red').pack()

        btn = ttk.Button(frm13, text='Transfer Attack!', command=self._tattack)
        btn.pack(expand=tk.YES, fill=tk.X)

      frm14 = ttk.LabelFrame(frm1, text='Stats')
      frm14.pack(expand=tk.YES, fill=tk.X)
      if True:
        self.var_img_stats = tk.StringVar(frm14, '')
        lb = ttk.Label(frm14, textvariable=self.var_img_stats)
        lb.pack()

    # right: display
    frm2 = ttk.Frame(wnd)
    frm2.pack(side=tk.RIGHT, anchor=tk.CENTER, expand=tk.YES, fill=tk.Y)
    if True:
      def _create_display_widgets(root, title):
        ''' Top: Image, Bottom: Hist'''

        frm = ttk.LabelFrame(root, text=title)
        frm.pack(side=tk.LEFT, anchor=tk.CENTER, expand=tk.YES, fill=tk.Y)
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
            ax.axis('off')
            fig.set_size_inches(HIST_FIG_SIZE)
            fig.tight_layout()
            cvs = FigureCanvasTkAgg(fig, frm2)
            cvs.draw()
            cvs.get_tk_widget().pack(fill=tk.BOTH, expand=tk.YES)
        
        return lb_img, (ax, fig, cvs)

      self.lb_img_X,  (self.ax_X,  self.fig_X,  self.cvs_X)  = _create_display_widgets(frm2, 'X')
      self.lb_img_AX, (self.ax_AX, self.fig_AX, self.cvs_AX) = _create_display_widgets(frm2, 'AX')
      self.lb_img_DX, (self.ax_DX, self.fig_DX, self.cvs_DX) = _create_display_widgets(frm2, 'DX')

  def _change_model(self):
    name = self.var_model.get()
    if name == self.cur_model: return

    try:
      self.model = get_model(name).to(device).eval()
      self.layer = get_first_conv2d_layer(self.model)
      param_cnt = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
      self.var_model_info.set(f'total params: {param_cnt}')

      self._attack()
      self.cur_model = name
    except:
      print_exc()

  def _change_tmodel(self):
    name = self.var_tmodel.get()
    if name == self.cur_tmodel: return

    try:
      self.tmodel = get_model(name).to(device).eval()
      self.cur_tmodel = name
    except:
      msg = format_exc()
      print(msg)
      tkmsg.showerror(msg)

  def _change_dataset(self):
    name = self.var_dataset.get()
    if name == self.cur_dataset: return

    try:
      self.datagen = iter(get_dataloader(name))
      self._next()
      self.cur_dataset = name
    except:
      msg = format_exc()
      print(msg)
      tkmsg.showerror(msg)

  def _next(self):
    self.lb_fp_info.pack_forget()

    x, y = next(self.datagen)
    self.X = x.to(device)
    self.Y = y.to(device)
    self.Y_atk = None
    self._attack()

  def _open(self):
    fp = tkfdlg.askopenfilename()
    if not fp: return
    if not os.path.exists(fp):
      tkmsg.showerror('Error', 'File not exists!')
      return

    self.var_fp_info.set(os.path.basename(fp))
    self.lb_fp_info.pack()

    img = Image.open(fp).convert('RGB')
    self.X = to_tenosr(img).unsqueeze_(0).to(device)
    self.Y = self.Y_atk = None
    self._attack()

  def _attack(self):
    if None in [self.model, self.X]: return

    self.var_tattack_info.set('')

    with torch.inference_mode():
      self.pred_X = self.model(normalize(self.X, self.cur_dataset))

    atk_meth =  self.var_atk_meth.get()
    if atk_meth == 'pgd':
      # make target
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
        logits = self.pred_X[0].clone().detach()
        if atk_tgt == 'second prob':
          logits[logits.argmax()] = logits.min() - 1
          y = logits.argmax().unsqueeze(0)
        elif atk_tgt == 'least prob':
          y = logits.argmin().unsqueeze(0)
      self.Y_atk = y.to(device)

      self.AX = pgd(self.model, self.X, self.Y_atk, 
                    eps=self.var_atk_eps.get(),
                    alpha=self.var_atk_alpha.get(),
                    steps=self.var_atk_step.get(),
                    normalizer=lambda x: normalize(x, self.cur_dataset))
    elif atk_meth == 'pgd_conv':
      self.AX = pgd_conv(self.layer, self.X, 
                    eps=self.var_atk_eps.get(),
                    alpha=self.var_atk_alpha.get(),
                    steps=self.var_atk_step.get(),
                    normalizer=lambda x: normalize(x, self.cur_dataset))

    with torch.inference_mode():
      self.pred_AX = self.model(normalize(self.AX, self.cur_dataset))
      self.DX = self.AX - self.X

    self._show()

  def _tattack(self):
    if None in [self.model, self.tmodel, self.AX]: return

    with torch.inference_mode():
      pred_X_t  = self.tmodel(normalize(self.X, self.cur_dataset))
      prob_X_t  = F.softmax(pred_X_t[0], dim=-1)
      pred_X  = prob_X_t.argmax().item()

      pred_AX_t = self.tmodel(normalize(self.AX, self.cur_dataset))
      prob_AX_t = F.softmax(pred_AX_t[0], dim=-1)
      pred_AX = prob_AX_t.argmax().item()

    self.var_tattack_info.set(f'{pred_X}|{prob_X_t[pred_X]:.2%} → {pred_AX}|{prob_AX_t[pred_AX]:.2%}')

  def _show(self):
    def _update_display_widgets(title:str, x:torch.Tensor):
      # collect widgets
      lb_img = getattr(self, f'lb_img_{title}')
      ax     = getattr(self, f'ax_{title}')
      fig    = getattr(self, f'fig_{title}')
      cvs    = getattr(self, f'cvs_{title}')

      # draw image
      im_f = x.permute([0, 2, 3, 1]).squeeze().detach().cpu().numpy()    # [H, W, C=3] or [H, W]
      if title == 'DX':
        im_f_norm = (im_f - im_f.min()) / (im_f.max() - im_f.min())     # minmax-norm
        img = Image.fromarray((im_f_norm * 255).astype(np.uint8))
      else:
        img = Image.fromarray((im_f * 255).astype(np.uint8))

      h, w = img.size
      if   h > w: size = (IMAGE_MAX_SIZE,          IMAGE_MAX_SIZE * w // h)
      elif w > h: size = (IMAGE_MAX_SIZE * h // w, IMAGE_MAX_SIZE)
      else:       size = (IMAGE_MAX_SIZE,          IMAGE_MAX_SIZE)
      img = img.resize(size, resample=RESAMPLE_METHOD)

      imgtk = ImageTk.PhotoImage(img)
      lb_img.imgtk = imgtk                  # avoid gc
      lb_img.config(image=imgtk)

      # draw hist
      im_i = (im_f * 255).astype(np.int32)
      if title == 'DX':
        bins = im_i.max() - im_i.min() + 1 + 2
        vrange = (im_i.min() - 1, im_i.max() + 1)
      else:
        bins = 256
        vrange = (0, 255)

      ax.cla()
      #ax.axis('off')
      ax.hist(im_i.flatten(), bins=bins, range=vrange)
      fig.tight_layout(pad=0.1)

      if title != 'DX' and len(im_i.shape) == 3:
        def plot_ch(ch, color):
          cntr = Counter(im_i[:, :, ch].flatten())
          ax.plot([cntr.get(i, 0) for i in range(256)], color)

        plot_ch(0, 'r')
        plot_ch(1, 'g')
        plot_ch(2, 'b')

      cvs.draw()

    _update_display_widgets('X',  self.X)
    _update_display_widgets('AX', self.AX)
    _update_display_widgets('DX', self.DX)

    # stats
    B, C, H, W = self.X.shape

    prob_X  = F.softmax(self.pred_X[0],  dim=-1)
    pred_X = prob_X.argmax()
    prob_AX = F.softmax(self.pred_AX[0], dim=-1)
    pred_AX = prob_AX.argmax()
    info = [
      f'truth: {self.Y.item()}' if self.Y else None,
      f'pred_X: {pred_X}|{prob_X[pred_X]:.2%}',
      f'attack: {self.Y_atk.item()}',
      f'pred_AX: {pred_AX}|{prob_AX[pred_AX]:.2%}',
      f'Linf: {self.DX.abs().max():.4}',
      f'L2: {self.DX.square().sum().sqrt().mean():.4}',
      f'size: {W} x {H} x {C}',
    ]
    self.var_img_stats.set('\n'.join([f for f in info if f]))


if __name__ == '__main__':
  App()
