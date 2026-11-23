#!/usr/bin/env python3
"""Source-backed DeepSeek V4.1 circuit diagrams; standard-library SVG authoring."""
from __future__ import annotations
import argparse, gzip, hashlib, html, json, re, shutil
from pathlib import Path

INK='#20314c'; MUTED='#5c6d83'; BLUE='#2563b5'; TEAL='#087f87'; PURPLE='#7951ac'; GOLD='#b57413'
LIGHT={BLUE:'#edf4ff', TEAL:'#eaf8f6', PURPLE:'#f4effb', GOLD:'#fff6e6', INK:'#f3f5f8'}
ESC=lambda x:html.escape(str(x),quote=True)
REPO=Path('/Users/jiav/Documents/Codex/2026-11-11/z/outputs/dsv41-openbox')

class Drawing:
    def __init__(self,w,h,title):
        self.w=w; self.h=h; self.title=title; self.p=[]; self.nodes=[]; self.texts=[]
    def rect(self,x,y,w,h,fill='white',stroke='#d5deea',dash=False,r=14,sw=2):
        self.p.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"'+(' stroke-dasharray="10 7"' if dash else '')+'/>')
    def text(self,x,y,s,size=25,color=INK,bold=False,anchor='middle'):
        self.p.append(f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" fill="{color}" font-weight="{700 if bold else 400}">{ESC(s)}</text>')
        self.texts.append({'text':str(s),'x':x,'y':y,'size':size})
    def lines(self,x,y,lines,size=24,color=MUTED,step=34,anchor='middle'):
        for i,s in enumerate(lines):self.text(x,y+i*step,s,size,color,anchor=anchor)
    def line(self,pts,color=INK,dash=False,arrow=True,sw=2.8):
        path='M'+' L'.join(f'{x},{y}' for x,y in pts)
        self.p.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="{sw}" stroke-linejoin="round"'+(' stroke-dasharray="9 7"' if dash else '')+(f' marker-end="url(#{color[1:]})"' if arrow else '')+'/>')
    def down(self,x,y1,y2,color=INK,dash=False):self.line([(x,y1),(x,y2)],color,dash)
    def node(self,id,x,y,w,h,title,sub=(),color=BLUE,size=26,fill=None,source=''):
        self.p.append(f'<g id="{ESC(id)}" data-source="{ESC(source)}"><title>{ESC(source or title)}</title>')
        self.rect(x,y,w,h,fill or LIGHT[color],color,r=12)
        count=1+len(sub); step=31; first=y+h/2-(count-1)*step/2+size*.33
        self.text(x+w/2,first,title,size,color,True)
        self.lines(x+w/2,first+step,sub,23,color,step)
        self.p.append('</g>'); self.nodes.append({'id':id,'x':x,'y':y,'w':w,'h':h,'source':source})
    def circle(self,x,y,t='+',color=INK,r=22):
        self.p.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="white" stroke="{color}" stroke-width="2.8"/>')
        self.text(x,y+9,t,30,color,True)
    def panel(self,title,subtitle,color):
        self.rect(1,1,self.w-2,self.h-2,stroke=color,dash=True,r=18)
        self.text(30,52,title,35,color,True,'start')
        self.text(30,92,subtitle,23,MUTED,anchor='start')
    def body(self):return ''.join(self.p)
    def svg(self):
        markers=''.join(f'<marker id="{c[1:]}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto"><path d="M0 0 L10 5 L0 10 Z" fill="{c}"/></marker>' for c in [INK,BLUE,TEAL,PURPLE,GOLD])
        return f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" viewBox="0 0 {self.w} {self.h}"><title>{ESC(self.title)}</title><desc>Logical dimensions, zero-based layers, source symbols in node tooltips. Solid arrows carry data; dashed lines denote control, coefficients or detail expansion.</desc><defs>{markers}</defs><rect width="100%" height="100%" fill="white"/><g font-family="Arial Unicode MS, Hiragino Sans GB, sans-serif">{self.body()}</g></svg>'

def vision():
    d=Drawing(1080,1320,'视觉编码器与 Aligner')
    d.panel('01  视觉编码器 → 语言嵌入','图像独立编码；32 个双向 Vision Transformer block',TEAL)
    d.node('v-patch',210,130,660,75,'RGB 14 × 14 patch → Linear',('[P, 588] → [P, 1024]',),TEAL,source='vision.py:DeepseekV41PatchEmbed.forward')
    d.rect(95,255,890,875,stroke=TEAL,dash=True)
    d.text(125,294,'Vision Block ×32',27,TEAL,True,'start')
    d.node('v-norm1',300,320,480,55,'RMSNorm',color=TEAL)
    d.down(540,205,320,TEAL)
    d.node('v-qkv',270,420,540,70,'Linear Wqkv → split Q / K / V',('1024 → 3 × 1024；16 heads × 64',),TEAL,size=25)
    d.down(540,375,420,TEAL)
    d.node('v-ropeq',130,540,230,60,'2D RoPE (Q)',color=TEAL,size=25)
    d.node('v-ropek',425,540,230,60,'2D RoPE (K)',color=TEAL,size=25)
    d.node('v-value',720,540,230,60,'V 不做 RoPE',color=TEAL,size=25)
    for c in [245,540,835]:d.line([(540,490),(540,510),(c,510),(c,540)],TEAL)
    d.node('v-attn',235,650,610,75,'双向 Attention → Linear Wo',('[P, 16, 64] → [P, 1024]',),TEAL,source='vision.py:DeepseekV41VisionAttention.forward')
    for c in [245,540,835]:d.line([(c,600),(c,625),(540,625),(540,650)],TEAL)
    d.circle(540,775,color=TEAL);d.down(540,725,753,TEAL)
    d.line([(540,230),(115,230),(115,775),(518,775)],TEAL)
    d.text(115,748,'x',23,TEAL)
    d.node('v-norm2',300,825,480,55,'RMSNorm',color=TEAL);d.down(540,797,825,TEAL)
    d.node('v-mlp',210,980,660,80,'Linear 1024 → 2 × 2816 → SwiGLU',('SiLU(gate) × up → Linear 2816 → 1024',),TEAL,size=24,source='vision.py:DeepseekV41VisionMLP.forward')
    d.down(540,880,980,TEAL);d.circle(540,1100,color=TEAL);d.down(540,1060,1078,TEAL)
    d.line([(540,800),(925,800),(925,1100),(562,1100)],TEAL)
    d.text(925,975,'残差',23,TEAL)
    d.node('v-outnorm',210,1140,660,55,'32 层后 RMSNorm → 3 × 3 空间合并',color=TEAL,size=25)
    d.down(540,1122,1140,TEAL)
    d.node('v-align',130,1230,820,65,'Linear 9216 → 5120 → GELU → Linear 5120',('[M, 5120]；M = ceil(Hp/3) × ceil(Wp/3)',),TEAL,size=24,source='vision.py:DeepseekV41Aligner.forward')
    d.down(540,1195,1230,TEAL)
    return d

def engram():
    d=Drawing(1080,1100,'Engram 条件记忆内部计算')
    d.panel('02  Engram：检索 → 逐流门控 → 注入','仅 L1、L14；在 Decoder 的 attention mHC pre 之前执行',PURPLE)
    d.node('e-token',65,140,440,85,'历史 token / lookback',('2 / 3 / 4-gram；排除图像区域',),PURPLE,size=25)
    d.node('e-hash',65,270,440,85,'n-gram hash → 表行索引',('3 种 n-gram × 8 heads',),PURPLE,size=25)
    d.down(285,225,270,PURPLE)
    d.node('e-lookup',65,400,440,85,'Embedding lookup → concat',('3 × 8 × 256 = 6144 channels',),PURPLE,size=25)
    d.down(285,355,400,PURPLE)
    d.node('e-wkv',65,530,440,85,'Linear Wkv 6144 → 25600',('K: 4 × 5120；V: 5120',),PURPLE,size=25,source='engram/layer.py:AscendEngram.forward')
    d.down(285,485,530,PURPLE)
    d.node('e-hidden',590,140,420,85,'当前四路残差 H',('[N, 4, 5120]',),PURPLE,size=25)
    d.node('e-key',590,400,420,85,'每流 K + learned q/k weights',('与 H 做 RMS 归一化内积',),PURPLE,size=23)
    d.line([(505,572),(550,572),(550,442),(590,442)],PURPLE)
    d.line([(800,225),(800,400)],PURPLE)
    d.node('e-gate',590,570,420,115,'signed sqrt → sigmoid',('s = normalized weighted dot / √5120','g = σ(sign(s) × √|s|) → [N, 4]'),PURPLE,size=24,source='engram/common.py:engram_gate')
    d.down(800,485,570,PURPLE)
    d.node('e-mask',590,740,420,65,'token mask → 无效 token 的 g=0',color=PURPLE,size=23)
    d.down(800,685,740,PURPLE)
    d.circle(540,890,'×',PURPLE,r=25)
    d.line([(285,615),(285,890),(515,890)],PURPLE)
    d.text(285,853,'V 广播到 4 路',23,PURPLE)
    d.line([(800,805),(800,890),(565,890)],PURPLE)
    d.circle(540,990,color=PURPLE);d.down(540,915,968,PURPLE)
    d.line([(1010,182),(1040,182),(1040,990),(562,990)],PURPLE)
    d.text(930,955,'H 旁路',23,PURPLE)
    d.lines(540,1055,['H′ = H + g[..., None] × V[:, None, :] → [N, 4, 5120]'],24,PURPLE)
    return d

def hc():
    d=Drawing(1080,1040,'mHC 内部计算与延迟系数交接')
    d.panel('03  mHC：四路 residual 怎样进出子层','Attention 与 MoE 各用一套参数；非普通的 x + f(x)',BLUE)
    d.node('h-input',270,140,540,75,'H ∈ [N, 4, 5120]',('前一子层传来的 pre_mix ∈ [N, 4]',),BLUE)
    d.node('h-collapse',55,300,445,110,'按传入 pre_mix 合并四路',('x = Σᵢ pre_mixᵢ × Hᵢ','x ∈ [N, 5120] → RMSNorm'),BLUE,size=24)
    d.node('h-coeff',580,300,445,110,'H flatten → RMS / Linear',('20480 → 24 个系数','4 pre + 4 post + 4 × 4 comb'),BLUE,size=24)
    d.line([(540,215),(540,250),(277,250),(277,300)],BLUE)
    d.line([(540,215),(540,250),(802,250),(802,300)],BLUE)
    d.node('h-sub',55,490,445,80,'Attention 或 MoE',('y ∈ [N, 5120]',),BLUE)
    d.down(277,410,490,BLUE)
    d.node('h-sk',580,490,445,145,'约束系数 / Sinkhorn ×20',('post ∈ [N, 4]','comb ∈ [N, 4, 4]','next_pre ∈ [N, 4]'),BLUE,size=24,source='model.py:DeepseekV41DecoderLayer.hc_pre; npu_hc_pre_v3')
    d.down(802,410,490,BLUE)
    d.node('h-post',155,710,770,100,'mHC post：混合 residual + 广播子层输出',('H′ⱼ = Σᵢ combᵢⱼ × Hᵢ + postⱼ × y','H′ ∈ [N, 4, 5120]'),BLUE,size=25,source='model.py:DeepseekV41DecoderLayer.hc_post')
    d.line([(277,570),(277,650),(370,650),(370,710)],BLUE)
    d.line([(802,635),(802,670),(710,670),(710,710)],BLUE)
    d.line([(810,177),(1050,177),(1050,760),(925,760)],BLUE)
    d.text(1050,675,'H',23,BLUE)
    d.node('h-next',155,900,770,85,'H′ 与 next_pre 一起交给下一子层',('Attention 产生的 attn_pre → MoE；ffn_pre → 下一层',),BLUE,size=25)
    d.down(540,810,900,BLUE)
    d.line([(1025,547),(1060,547),(1060,942),(925,942)],PURPLE,True)
    d.text(1000,867,'next_pre',22,PURPLE)
    return d

def trunk():
    d=Drawing(1100,3580,'输入到输出与单层完整执行顺序')
    d.panel('主干：输入 → 40 层 → 输出','N = 当前展开的 token 数；所有尺寸均为逻辑尺寸',INK)
    x=205;w=690;c=550
    seq=[('t-token',135,85,'文本 token → Token Embedding',['vocab 129280 → [N, 5120]']),
         ('t-merge',300,85,'在 image placeholder 位置替换 embedding',['← 视觉 Aligner 输出 [M, 5120]']),
         ('t-expand',465,85,'repeat 到 4 路 mHC residual',['H₀: [N, 4, 5120]；pre_mix = [1,0,0,0]'])]
    for id,y,h,t,s in seq:d.node(id,x,y,w,h,t,s,INK,size=26)
    d.down(c,220,300);d.down(c,385,465)
    d.rect(60,650,980,2260,stroke=BLUE,dash=True,r=18)
    d.text(95,700,'Decoder Lℓ ×40   ℓ=0…39',32,BLUE,True,'start')
    d.lines(c,755,['L0–1: SWA  |  L2–19: C2 + SWA  |  L20–39: C1 + SWA'],22,BLUE)
    d.node('t-engram',x,825,w,95,'若 ℓ∈{1,14}：先注入 Engram',('其他层直接通过；H 保持 [N,4,5120]',),PURPLE,size=26)
    d.down(c,550,825)
    flow=[('t-hpre-a',1010,95,'① mHC pre (Attention)',('输入 pre_mix → x [N,5120]；产生 attn 系数',),BLUE),
          ('t-rms-a',1190,65,'② RMSNorm (ε = 1e−20)',(),BLUE),
          ('t-attn',1350,100,'③ 本层 Q + 本层 SWA + 共享长程 KV',('Attention 详见右侧 A；输出 [N,5120]',),BLUE),
          ('t-hpost-a',1540,100,'④ mHC post (Attention)',('Attention 输出 + 原 H + post / comb → Hₐ',),BLUE),
          ('t-hpre-f',1765,95,'⑤ mHC pre (FFN)',('使用 attn_pre；产生 ffn_post / comb / ffn_pre',),GOLD),
          ('t-rms-f',1945,85,'⑥ RMSNorm + FP32 router 输入',('专家计算与路由共享同一次归一化结果',),GOLD),
          ('t-moe',2140,100,'⑦ MoE：6 routed experts + 1 shared',('专家内部 SwiGLU 与路由权重见右侧 B',),GOLD),
          ('t-hpost-f',2335,100,'⑧ mHC post (FFN)',('MoE 输出 + Hₐ + ffn_post / comb → Hℓ₊₁',),GOLD)]
    prev=920
    for id,y,h,t,s,col in flow:
        d.node(id,x,y,w,h,t,s,col,size=26);d.down(c,prev,y);prev=y+h
    d.line([(x,1057),(130,1057),(130,1590),(x,1590)],BLUE)
    d.text(140,1480,'H 残差旁路',22,BLUE,anchor='start')
    d.line([(x,1812),(130,1812),(130,2385),(x,2385)],GOLD)
    d.text(140,2285,'Hₐ 残差旁路',22,GOLD,anchor='start')
    d.line([(895,1057),(970,1057),(970,1812),(895,1812)],PURPLE,True)
    d.text(985,1680,'attn_pre',22,PURPLE,anchor='end')
    d.node('t-state',x,2560,w,100,'交给下一层：Hℓ₊₁ 与 ffn_pre',('[N,4,5120] + [N,4]；每层独立 Q 与 SWA',),INK,size=25)
    d.down(c,2435,2560)
    d.line([(895,2610),(1010,2610),(1010,810),(895,810)],INK,False)
    d.text(c,2770,'重复框只代表执行顺序；KV／索引共享见下方层图',24,MUTED)
    d.node('t-collapse',x,3020,w,85,'40 层后：按最后 ffn_pre collapse',('Σᵢ pre_mixᵢ × Hᵢ → [N,5120]',),INK,size=26)
    d.down(c,2660,3020)
    d.node('t-finalnorm',x,3170,w,65,'最终 RMSNorm',(),INK,size=27)
    d.down(c,3105,3170)
    d.node('t-lmhead',x,3310,w,85,'LM Head 5120 → 129280 logits',('Embedding 与 LM Head 不绑定权重',),INK,size=26)
    d.down(c,3235,3310)
    d.node('t-sample',x,3470,w,65,'采样 / 输出下一个 token',(),INK,size=27)
    d.down(c,3395,3470)
    return d

def attention():
    d=Drawing(1440,2110,'Attention 投影、压缩、Indexer、缓存与输出')
    d.panel('A  Attention：由投影一路追到输出','64 Q heads ×512；1 组共享 KV；部分 RoPE=64 / 非 RoPE=448',BLUE)
    d.text(720,150,'归一化子层输入 x [N,5120]',28,BLUE,True)
    d.line([(720,170),(720,195),(240,195),(240,225)],BLUE)
    d.line([(720,170),(720,225)],BLUE)
    d.line([(720,195),(1190,195),(1190,225)],BLUE)
    q=[('a-qa',225,75,'Wq_a 5120 → 1280',('[N,1280] = q latent',)),('a-qnorm',340,60,'RMSNorm(q latent)',()),('a-qb',440,75,'Wq_b 1280 → 64 × 512',('[N,64,512]',)),('a-qrope',555,85,'部分 RoPE（后 64 维）',('其余 448 维不旋转',))]
    prev=None
    for id,y,h,t,s in q:
        d.node(id,45,y,390,h,t,s,BLUE,size=24,source='model.py:init_attention_projections; attention/dsa_v41.py:_project_q')
        if prev:d.down(240,prev,y,BLUE)
        prev=y+h
    kv=[('a-wkv',225,75,'Wkv 5120 → 512',('[N,1,512]',)),('a-kvnorm',340,60,'RMSNorm(KV)',()),('a-kvrope',440,75,'部分 RoPE（后 64 维）',('与本层 Q 使用对应位置',)),('a-swa',555,85,'写入本层 SWA cache',('各层独有；window = 128',))]
    prev=None
    for id,y,h,t,s in kv:
        d.node(id,525,y,390,h,t,s,BLUE,size=24,source='attention/dsa_v41.py:_project_kv; _write_swa_cache')
        if prev:d.down(720,prev,y,BLUE)
        prev=y+h
    d.node('a-c2',1000,225,390,165,'C2：仅 L2 / L8 / L14',('x→Wkv, Wgate：各 5120→512','每 2 token 按通道 softmax 池化','RMSNorm → latent [Tc,512]'),PURPLE,size=24,source='compressor.py:pool_projected; compressor_triton.py:compressor_ref')
    d.node('a-c1',1000,440,390,100,'C1：仅 L20',('Wkv 5120→512 → RMSNorm','逐 token latent；不压缩长度'),PURPLE,size=25,source='compressor.py:forward')
    d.line([(1190,195),(1410,195),(1410,420),(1195,420),(1195,440)],PURPLE)
    d.node('a-latent',1000,590,390,65,'源层 latent（尚未 RoPE）',color=PURPLE,size=24)
    d.line([(1195,390),(970,390),(970,622),(1000,622)],PURPLE)
    d.down(1195,540,590,PURPLE)
    d.node('a-long',1000,715,390,95,'部分 RoPE → 长程 KV cache',('消费者直接读 source cache','C2: Tc≈⌊T/2⌋；C1: Tc=T'),PURPLE,size=23)
    d.down(1195,655,715,PURPLE)
    d.rect(35,865,1370,650,stroke=PURPLE,dash=True)
    d.text(65,907,'Indexer：只在 index source 层算 Top-K，消费者复用索引',27,PURPLE,True,'start')
    d.node('a-iq',65,965,375,95,'q latent → Wq_b(index)',('1280 → 32 heads ×128','后 64 维做 RoPE'),PURPLE,size=24,source='indexer.py:project_query; apply_query_rope')
    d.line([(435,370),(475,370),(475,940),(252,940),(252,965)],PURPLE)
    d.node('a-ik',535,965,375,95,'源 latent → Wk → RMSNorm',('512 → 128 → 部分 RoPE','写入 source index-K cache'),PURPLE,size=23,source='indexer.py:update_keys')
    d.line([(1000,622),(940,622),(940,940),(722,940),(722,965)],PURPLE)
    d.node('a-iw',1005,965,365,95,'x → weights_proj',('5120 → 32 个 head 权重','× (1/√128) × (1/√32)'),PURPLE,size=24,source='indexer.py:project_weights')
    d.text(1187,930,'来自当前层 x',22,PURPLE)
    d.down(1187,938,965,PURPLE)
    d.node('a-iscore',265,1130,910,80,'量化 Q / K → 加权索引打分（causal）',('32 index heads 的 QK score → 候选 / Top-K',),PURPLE,size=25,source='indexer.py:select_projected')
    for x in [252,722,1187]:d.line([(x,1060),(x,1095),(720,1095),(720,1130)],PURPLE)
    d.node('a-ipolicy',90,1270,790,110,'L2 / 8 / 14 / 20：直接搜索；L20 同时产候选',('L24 / 28 / 32 / 36：只在 L20 候选中重选','候选 2048 blocks ×8 tokens → 最终 Top-512 位置'),PURPLE,size=24)
    d.down(720,1210,1270,PURPLE)
    d.node('a-topk',965,1280,390,95,'共享 Top-K index buffer',('[N,512]，不是 Attention 输出','其余层不重新跑 Indexer'),PURPLE,size=23)
    d.line([(880,1325),(965,1325)],PURPLE)
    d.text(720,1470,'index source = {2,8,14,20,24,28,32,36}；index-K 始终来自对应 KV source',22,PURPLE)
    d.node('a-sparse',275,1600,890,130,'一个 SparseFlashMLA / QSMLA 注意力计算',('Q + 本层 SWA KV + source 长程 KV[Top-512] + learned sink','两类 KV 共同参与注意力归一化；不是两份输出相加'),BLUE,size=26,source='attention/dsa_v41.py:_forward_attention')
    d.line([(240,640),(20,640),(20,1665),(275,1665)],BLUE)
    d.line([(720,640),(720,835),(1430,835),(1430,1755),(920,1755),(920,1730)],BLUE)
    d.line([(1390,762),(1415,762),(1415,1560),(1040,1560),(1040,1600)],PURPLE)
    d.line([(1160,1375),(1160,1540),(810,1540),(810,1600)],PURPLE)
    d.node('a-oa',225,1810,990,85,'分成 8 组 → Wo_a：每组 4096 → rank 1024',('[N,64,512] → [N,8,1024] → concat 8192',),BLUE,size=26)
    d.down(720,1730,1810,BLUE)
    d.node('a-ob',225,1955,990,85,'Wo_b 8192 → 5120 → 返回 mHC post',('[N,5120]；8 groups ×1024 是输出低秩瓶颈',),BLUE,size=26)
    d.down(720,1895,1955,BLUE)
    d.text(720,2082,'物理 cache 依硬件选择；950DT/A5：SWA FP8 + BF16 scale，长程 KV FP4 + BF16 scale',21,MUTED)
    return d

def moe():
    d=Drawing(1440,1410,'MoE 路由、专家内部与聚合')
    d.panel('B  MoE：路由计算与专家内部 SwiGLU','40 层均有 MoE；384 routed / Top-6 + 1 shared；中间维度 2304',GOLD)
    d.node('m-input',420,135,600,75,'归一化 x [N,5120]',('FP32 副本用于 router',),GOLD,size=27)
    d.line([(720,210),(720,245),(340,245),(340,285)],GOLD)
    d.line([(720,245),(1100,245),(1100,285)],GOLD)
    d.node('m-router',70,285,575,85,'Router Linear 5120 → 384',('sqrtsoftplus → scores + bias 选 Top-6',),GOLD,size=25,source='model.py:DeepseekV41MoE.__init__; forward')
    d.node('m-shared',840,285,520,85,'1 个 shared expert：每 token 都算',('不占 routed Top-6 名额',),GOLD,size=25)
    d.node('m-weight',70,430,575,95,'Top-6 权重归一化 → dispatch',('wᵢ = scoreᵢ / Σ_selected score','384 个专家中仅激活 6 个'),GOLD,size=25)
    d.down(357,370,430,GOLD)
    d.node('m-routed',70,590,575,90,'6 个 selected experts 并行',('每个专家使用下面同一 SwiGLU 结构',),GOLD,size=25)
    d.down(357,525,590,GOLD)
    d.rect(710,440,670,620,stroke=GOLD,dash=True)
    d.text(745,483,'单个专家：shared / routed 均适用',25,GOLD,True,'start')
    d.node('m-gate',735,550,280,85,'Gate Linear',('5120 → 2304',),GOLD,size=25)
    d.node('m-up',1060,550,280,85,'Up Linear',('5120 → 2304',),GOLD,size=25)
    d.line([(1100,370),(1100,510),(875,510),(875,550)],GOLD)
    d.line([(1100,510),(1200,510),(1200,550)],GOLD)
    d.node('m-silu',735,700,280,60,'SiLU + clamp',(),GOLD,size=25)
    d.down(875,635,700,GOLD)
    d.circle(1040,830,'×',GOLD,r=25)
    d.line([(875,760),(875,830),(1015,830)],GOLD)
    d.line([(1200,635),(1200,830),(1065,830)],GOLD)
    d.node('m-down',800,920,480,85,'Down Linear 2304 → 5120',('SwiGLU limit = 10；返回 [N,5120]',),GOLD,size=25,source='model.py:DeepseekV41MLP.forward')
    d.down(1040,855,920,GOLD)
    d.node('m-sum',70,845,575,100,'按路由权重加权求和，再 ×1.5',('y_route = 1.5 × Σᵢ∈Top6 wᵢ Expertᵢ(x)','各 expert 输出 [N,5120]'),GOLD,size=24)
    d.down(357,680,845,GOLD)
    d.circle(720,1140,color=GOLD,r=28)
    d.line([(357,945),(357,1140),(692,1140)],GOLD)
    d.line([(1040,1005),(1040,1140),(748,1140)],GOLD)
    d.node('m-out',350,1225,740,75,'y = y_route + y_shared → mHC post',('[N,5120]',),GOLD,size=26)
    d.down(720,1168,1225,GOLD)
    d.text(720,1364,'源码还支持 token→expert 的 hash 路由；归档 config 未声明 num_hash_layers，无法确认分层范围。',21,MUTED)
    return d

def topology():
    d=Drawing(2220,650,'逐层 KV 和 Indexer 复用图')
    d.panel('04  跨层复用：KV 生产层与索引生产层不同','每一行都是该组层执行自己的 Attention；共享缓存或索引不等于跳过层',PURPLE)
    cols=[75,410,710,1075,1510,1930]
    heads=['层号 (0 基)','层数 / 类型','长程 KV source','Index Top-K source','搜索策略','Engram']
    for x,t in zip(cols,heads):d.text(x,150,t,23,PURPLE,True,'start')
    groups=[('L0–1','2 / SWA only','—','—','无长程分支','L1'),('L2–7','6 / C2','L2','L2','全域 → Top-512','—'),('L8–13','6 / C2','L8','L8','全域 → Top-512','—'),('L14–19','6 / C2','L14','L14','全域 → Top-512','L14'),('L20–23','4 / C1','L20','L20','Top-512 + 2048 候选块','—'),('L24–27','4 / C1','L20','L24','L20 候选内 → Top-512','—'),('L28–31','4 / C1','L20','L28','L20 候选内 → Top-512','—'),('L32–35','4 / C1','L20','L32','L20 候选内 → Top-512','—'),('L36–39','4 / C1','L20','L36','L20 候选内 → Top-512','—')]
    for i,row in enumerate(groups):
        y=175+i*45
        if i%2==0:d.rect(35,y-2,2150,43,'#f6f3fb','#f6f3fb',r=3,sw=0)
        for x,t in zip(cols,row):d.text(x,y+28,t,24,INK,anchor='start')
    d.text(75,625,'2 + 18 + 20 = 40 层；43 个 compress_ratios 仅前 40 项属于主干；候选块 ID ≠ Top-K token 位置。',23,PURPLE,anchor='start')
    return d

def mtp():
    d=Drawing(1480,650,'MTP DSpark 可选执行分支')
    d.panel('05  MTP / DSpark：独立的三层草稿分支','结构存在；归档服务未启用投机解码，本次只核对静态路径',PURPLE)
    d.node('s-target',45,145,665,85,'主干 target states [37,38,39] → concat',('3 ×5120 → main_proj 15360→5120 → RMSNorm',),PURPLE,size=24)
    d.node('s-cache',825,145,610,85,'预计算每个 draft layer 的上下文 SWA KV',('packed Wkv → KV norm → RoPE → 各层 cache',),PURPLE,size=23)
    d.line([(710,187),(825,187)],PURPLE)
    d.node('s-embed',45,305,310,90,'draft token / noise',('Embedding → mHC ×4',),PURPLE,size=24)
    for i,x in enumerate([420,765,1110]):
        d.node(f's-layer{i}',x,305,285,90,f'mtp.{i} / L{40+i}',('SWA + MoE 128 / Top-3',),PURPLE,size=24)
    d.line([(355,350),(420,350)],PURPLE)
    d.line([(705,350),(765,350)],PURPLE);d.line([(1050,350),(1110,350)],PURPLE)
    d.line([(1130,230),(1130,270),(562,270),(562,305)],PURPLE,True)
    d.text(950,255,'各 draft 层读上下文 KV',22,PURPLE)
    d.node('s-head',420,460,975,100,'mHC collapse → RMSNorm → LM Head / Markov rank 256',('token logits + Markov bias；confidence head → sigmoid','dspark_block_size = 5 是配置项，不代表 5 个 MTP 层'),PURPLE,size=25)
    d.line([(1252,395),(1252,460)],PURPLE)
    d.text(45,620,'target_layer_ids 的取样点按 runner 约定；实际取进入对应层的 residual mean。',22,MUTED,anchor='start')
    return d

def verify_facts(repo):
    data=json.loads((repo/'model/architecture/facts.json').read_text())
    for k,v in data['source_identity'].items():
        b=(repo/v['path']).read_bytes()
        assert hashlib.sha256(b).hexdigest()==v['sha256'],k
        if v.get('compression')=='gzip':assert hashlib.sha256(gzip.decompress(b)).hexdigest()==v['decompressed_sha256'],k
    config=json.loads((repo/'model/config.json').read_text()); t=config['text_config']; v=config['vision_config']
    for key,value in {'hidden_size':5120,'num_hidden_layers':40,'num_attention_heads':64,'head_dim':512,'q_lora_rank':1280,'o_lora_rank':1024,'o_groups':8,'n_routed_experts':384,'n_shared_experts':1,'num_experts_per_tok':6,'moe_intermediate_size':2304,'hc_mult':4,'index_n_heads':32,'index_head_dim':128,'index_topk':512,'candidate_topk_blocks':2048,'candidate_block_size':8,'engram_max_ngram_size':4,'engram_n_heads':8,'engram_head_dim':256,'num_nextn_predict_layers':3}.items():
        assert t[key]==value,(key,t[key],value)
    assert t['kv_source_layer_ids']==[2,8,14,20]
    assert t['index_source_layer_ids']==[2,8,14,20,24,28,32,36]
    assert t['engram_layer_ids']==[1,14]
    assert t['compress_ratios'][:40]==[0]*2+[2]*18+[1]*20
    assert (v['hidden_size'],v['num_hidden_layers'],v['num_attention_heads'],v['intermediate_size'],v['patch_size'],v['downsample_ratio'])==(1024,32,16,2816,14,3)
    return data

def make(out):
    out.mkdir(parents=True,exist_ok=True)
    pieces={'vision':vision(),'engram':engram(),'mhc':hc(),'decoder':trunk(),'attention':attention(),'moe':moe(),'layer-sharing':topology(),'mtp':mtp()}
    positions={'vision':(70,240),'engram':(70,1620),'mhc':(70,2780),'decoder':(1230,240),'attention':(2410,240),'moe':(2410,2410),'layer-sharing':(70,3880),'mtp':(2370,3880)}
    master=Drawing(3920,4740,'DeepSeek V4.1 Flash 1017 详细模型结构与执行流程')
    master.text(70,82,'DeepSeek V4.1 Flash',60,INK,True,'start')
    master.text(70,140,'模型内部执行流程 · 源码审计版',34,BLUE,anchor='start')
    master.text(3850,75,'40 layers  /  hidden 5120  /  E384 Top-6 + 1 shared',29,BLUE,True,'end')
    master.text(3850,130,'Flash-1017 · dequantized BF16 · 2026-11-21',25,MUTED,anchor='end')
    master.text(70,199,'阅读顺序：中央追主干；右上追 Attention；右下追 MoE；左侧追视觉 / Engram / mHC；底部查逐层共享。',27,MUTED,anchor='start')
    for k,d in pieces.items():
        x,y=positions[k];master.p.append(f'<g transform="translate({x},{y})" id="panel-{k}">{d.body()}</g>')
        (out/f'{k}-detail.svg').write_text(d.svg())
    # Across-panel detail links use outer gutters, not text-bearing routes.
    master.line([(1150,1503),(1190,1503),(1190,582),(1435,582)],TEAL)
    master.line([(1150,2670),(1210,2670),(1210,1112),(1435,1112)],PURPLE)
    master.line([(2125,1640),(2365,1640),(2365,1905),(2685,1905)],BLUE,True)
    master.line([(2125,2430),(2350,2430),(2350,2582),(2830,2582)],GOLD,True)
    master.rect(70,4585,3780,112,'#f3f6fa','#f3f6fa',r=12)
    master.text(100,4625,'实线 = 数据流；虚线 = 系数传递 / 模块展开 / 可选路径。P=图像 patch 数，M=合并后视觉 token 数，Tc=长程 cache 长度。',24,MUTED,anchor='start')
    master.text(100,4665,'完整 E384 与 E32 验证范围分开；Engram 完整模型开启 / E32 关闭。配置上下文 1,048,576，本次未新增 NPU 实测。',24,MUTED,anchor='start')
    (out/'model-architecture.svg').write_text(master.svg())
    qa={'panels':{},'master_size':[master.w,master.h]}
    for k,d in pieces.items():
        bad=[n['id'] for n in d.nodes if n['x']<0 or n['y']<0 or n['x']+n['w']>d.w or n['y']+n['h']>d.h]
        assert not bad,(k,bad)
        qa['panels'][k]={'dimensions':[d.w,d.h],'nodes':len(d.nodes),'texts':len(d.texts)}
    (out/'layout-inventory.json').write_text(json.dumps(qa,ensure_ascii=False,indent=2)+'\n')
    return pieces

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=Path(__file__).resolve().parent);p.add_argument('--repo',type=Path);a=p.parse_args()
    if a.repo:verify_facts(a.repo)
    make(a.out)
