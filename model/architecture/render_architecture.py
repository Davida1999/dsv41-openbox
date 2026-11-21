#!/usr/bin/env python3
"""Render the audited DSV4.1 architecture without importing any model code."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import html
import json
from pathlib import Path
import unicodedata

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
WIDTH, HEIGHT = 2600, 2540
INK = '#182c49'
MUTED = '#566b84'
BLUE = '#275daa'
TEAL = '#087e83'
PURPLE = '#7450a2'
AMBER = '#a76915'


def verify_facts(data):
    for identity in data['source_identity'].values():
        raw = (REPO / identity['path']).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == identity['sha256'], identity['path']
        if identity.get('compression') == 'gzip':
            assert hashlib.sha256(gzip.decompress(raw)).hexdigest() == identity['decompressed_sha256']
    config = json.loads((REPO / 'model/config.json').read_text())
    for fact in data['facts']:
        if not fact['name'].startswith(('text.', 'vision.')):
            continue
        source = fact['sources'][0]['json_pointer'].split('/')[1:]
        value = config
        for key in source:
            value = value[key]
        assert value == fact['value'], fact['name']


class Diagram:
    def __init__(self):
        self.parts = [f'''<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">
<title>DeepSeek V4.1 Flash 开箱模型结构图</title>
<desc>依据归档配置、权重索引及固定版本建模源码绘制；主干40层，独立3层MTP，逻辑维度与实测范围分别标注。</desc>
<defs>
<marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto"><path d="M0 0 L10 5 L0 10 Z" fill="{INK}"/></marker>
<marker id="teal" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto"><path d="M0 0 L10 5 L0 10 Z" fill="{TEAL}"/></marker>
<marker id="purple" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto"><path d="M0 0 L10 5 L0 10 Z" fill="{PURPLE}"/></marker>
</defs>
<rect width="2600" height="2540" fill="#f5f8fc"/>
<g font-family="Arial Unicode MS, Noto Sans CJK SC, Hiragino Sans GB, sans-serif">''']

    def rect(self, x, y, w, h, fill='white', stroke='#d6dfeb', radius=18, dash=False, sw=2):
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"'+(' stroke-dasharray="10 7"' if dash else '')+'/>')

    def text(self, x, y, value, size=24, fill=INK, weight=400, anchor='start', max_width=None):
        if max_width:
            units = sum(1 if unicodedata.east_asian_width(ch) in 'WF' else .58 for ch in str(value))
            size = min(size, max_width / max(units, 1))
        self.parts.append(f'<text x="{x}" y="{y}" font-size="{size:.2f}" fill="{fill}" font-weight="{weight}" text-anchor="{anchor}">{html.escape(str(value))}</text>')

    def lines(self, x, y, lines, size=24, step=36, fill=MUTED, max_width=None):
        for i, line in enumerate(lines):
            self.text(x, y + step*i, line, size, fill, max_width=max_width)

    def arrow(self, path, color=INK, dash=False, width=3):
        marker = {TEAL: 'teal', PURPLE: 'purple'}.get(color, 'arrow')
        self.parts.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="{width}" marker-end="url(#{marker})"'+(' stroke-dasharray="10 7"' if dash else '')+'/>')

    def panel(self, x, y, w, h, title, color=BLUE):
        self.rect(x, y, w, h, stroke=color, sw=2)
        self.rect(x, y, w, 74, fill=color, stroke=color, radius=18)
        self.rect(x+1, y+39, w-2, 34, fill=color, stroke=color, radius=0)
        self.text(x+28, y+48, title, 31, 'white', 700, max_width=w-56)

    def node(self, x, y, w, h, title, lines=(), color=BLUE, fill='#eef4ff'):
        self.rect(x, y, w, h, fill, color)
        self.text(x+w/2, y+39, title, 29, color, 700, 'middle', w-34)
        for i, line in enumerate(lines):
            self.text(x+w/2, y+77+33*i, line, 23, MUTED, anchor='middle', max_width=w-36)

    def finish(self):
        return '\n'.join(self.parts + ['</g></svg>'])


def build(data):
    f = {row['name']: row['value'] for row in data['facts']}
    t = lambda key: f['text.' + key]
    v = lambda key: f['vision.' + key]
    assert t('num_hidden_layers') == 40 and t('compress_ratios')[:2] == [0, 0]
    assert f['topology.backbone_compress_counts'] == {'0': 2, '2': 18, '1': 20}
    d = Diagram()
    d.text(70, 94, 'DeepSeek V4.1 Flash', 61, weight=700)
    d.text(2530, 94, '模型开箱结构图', 42, BLUE, 700, 'end')
    d.text(72, 143, 'Flash-1017 · 完整反量化 BF16 模型 · 配置 / 索引 / 建模源码交叉核对', 27, MUTED)
    d.text(2530, 143, '2026-11-21', 25, MUTED, anchor='end')
    d.rect(70, 183, 2460, 68, '#e6edf7', stroke='#e6edf7')
    d.text(100, 226, f"主干 {t('num_hidden_layers')} 层  |  hidden {t('hidden_size'):,}  |  MoE {t('n_routed_experts')} / Top-{t('num_experts_per_tok')} + {t('n_shared_experts')} shared  |  ViT {v('num_hidden_layers')} 层  |  MTP {t('num_nextn_predict_layers')} 层", 28, BLUE, 700)
    d.text(2500, 226, '0 基层号 · 逻辑维度', 24, MUTED, anchor='end')

    # Main language path.
    d.node(850, 300, 900, 120, '文本 token → Token Embedding', [f"vocab {t('vocab_size'):,}  →  [N, {t('hidden_size'):,}]"])
    d.node(850, 475, 900, 100, '图文 embedding 合并', ['在图像占位符位置替换为 image-span embeddings'])
    d.node(850, 630, 900, 100, 'mHC 残差流展开 ×4', ['[N, 5120] → [N, 4, 5120] · Sinkhorn 20 次'])
    d.arrow('M1300 420 V470')
    d.arrow('M1300 575 V625')
    d.arrow('M1300 730 V780')
    d.panel(850, 785, 900, 930, '40 × Decoder Layer · 按层执行', BLUE)
    d.lines(890, 900, [
        'L0–1   × 2：SWA only；compress_ratio = 0',
        'L2–19  ×18：SWA + 稀疏长程；compress_ratio = 2',
        'L20–39 ×20：SWA + 稀疏长程；compress_ratio = 1',
    ], size=25, step=43, max_width=820)
    d.text(1300, 1050, '每层内部数据流', 25, BLUE, 700, 'middle')
    d.node(925, 1090, 750, 110, 'mHC pre → RMSNorm → Attention  A', ['SWA / DSA 按本层拓扑执行；hidden = 5120'])
    d.node(925, 1235, 750, 64, 'mHC post · 残差混合', [], fill='#f4f7fc')
    d.node(925, 1340, 750, 110, 'mHC pre → RMSNorm → MoE  B', ['384 路由专家选 6 + 1 个共享专家'])
    d.node(925, 1485, 750, 64, 'mHC post · 残差混合', [], fill='#f4f7fc')
    d.arrow('M1300 1200 V1230')
    d.arrow('M1300 1299 V1335')
    d.arrow('M1300 1450 V1480')
    d.lines(890, 1605, ['mHC 残差流保持 [N, 4, 5120]；系数按源码延迟接续',
                         '文本 RMSNorm ε = 1e−20；mHC ε = 1e−6',
                         'Engram 仅在 L1 / L14 的 Decoder 前注入'], size=24, step=35, max_width=820)
    d.node(850, 1790, 900, 100, 'mHC collapse', ['四路残差流合并 → [N, 5120]'])
    d.node(850, 1950, 900, 135, '最终 RMSNorm → LM Head', ['5120 → 129280 logits', 'Embedding 与 LM Head 不绑定权重'])
    d.node(850, 2160, 900, 100, '自回归采样 / 输出 token', ['配置上下文上限 1,048,576；实测窗口见下方范围'])
    d.arrow('M1300 1715 V1785')
    d.arrow('M1300 1890 V1945')
    d.arrow('M1300 2085 V2155')

    # Vision branch.
    d.panel(70, 300, 720, 680, '01  视觉输入 · ViT + Aligner', TEAL)
    d.node(110, 405, 640, 100, '图片 → Patch Embedding', ['14 × 14 RGB patch：588 → 1024'], TEAL, '#eaf7f5')
    d.node(110, 555, 640, 145, '32 × Vision Transformer', ['16 heads × 64；双向注意力 + 2D RoPE', 'RMSNorm → Attention / SwiGLU 残差'], TEAL, '#eaf7f5')
    d.node(110, 750, 640, 135, '3 × 3 空间合并 → Aligner', ['9 × 1024 = 9216 → 5120', 'GELU → 5120；按图像网格补齐'], TEAL, '#eaf7f5')
    d.arrow('M430 505 V550', TEAL)
    d.arrow('M430 700 V745', TEAL)
    d.lines(110, 930, ['视觉 RMSNorm ε = 1e−6；图像 token 配置上限 1024'], size=22, max_width=640)
    d.arrow('M750 817 H820 V525 H845', TEAL)

    # Engram branch.
    d.panel(70, 1050, 720, 515, '02  Engram · 条件记忆', PURPLE)
    d.lines(105, 1170, ['历史 token → n-gram hash → embedding lookup',
                      'max n-gram = 4；8 heads × 256',
                      '两表条目：384,006,168 / 384,016,682'], size=24, step=43, max_width=650)
    d.node(110, 1305, 640, 100, '记忆检索 → 更新 mHC 残差流', ['完整模型：CPU offload / DP shared memory'], PURPLE, '#f3eef9')
    d.lines(105, 1470, ['注入位置：L1、L14；发生在该层 Attention 之前',
                      'E32 受测副本关闭 Engram；不沿用完整模型结论'], size=23, step=40, max_width=650)
    d.arrow('M750 1355 H810 V1122 H920', PURPLE)

    # Explicit sharing topology, without drawing consumers as independent producers.
    d.panel(70, 1640, 720, 620, '03  跨层 KV / Indexer 复用', PURPLE)
    d.text(105, 1760, '长程 KV source → consumer layers', 26, PURPLE, 700)
    d.lines(105, 1807, ['L2  → L2–7       L8  → L8–13',
                      'L14 → L14–19     L20 → L20–39'], size=26, step=46, max_width=650)
    d.text(105, 1930, 'Indexer source → consumer layers', 26, PURPLE, 700)
    d.lines(105, 1977, ['L2→2–7；L8→8–13；L14→14–19',
                      'L20→20–23；L24→24–27；L28→28–31',
                      'L32→32–35；L36→36–39'], size=25, step=42, max_width=650)
    d.lines(105, 2130, ['L20 生成候选；L24+ 的 Indexer 在候选中筛选',
                      '候选 2048 blocks × 8 tokens；最终 Top-K 512',
                      '0 基层号；每层仍维护自身的 SWA'], size=23, step=42, max_width=650)

    # Attention detail.
    d.panel(1810, 300, 720, 1000, 'A  注意力展开 · SWA + DSA', BLUE)
    d.lines(1845, 420, ['Q：5120 → rank 1280 → 64 heads × 512',
                       'KV：5120 → 1 head × 512；RoPE 64 / 非RoPE 448'], size=24, step=43, max_width=650)
    d.arrow('M2170 488 V515 H1997 V540')
    d.arrow('M2170 515 H2342 V540')
    d.node(1845, 545, 305, 145, 'SWA', ['window = 128', '每层独立本地 KV'], BLUE)
    d.node(2190, 545, 305, 145, '稀疏长程 DSA', ['L2–39', 'Indexer Top-K = 512'], BLUE)
    d.arrow('M1997 690 V735 H2170 V775')
    d.arrow('M2342 690 V735 H2170')
    d.node(1845, 785, 650, 110, 'Attention + sink → 分组低秩 O 投影', ['8 groups × rank 1024 → hidden 5120'])
    d.lines(1845, 960, ['KV 与索引仅由 source 层生产，其余层复用',
                       '完整 source / consumer 拓扑见左下 03'], size=24, step=40, max_width=650)
    d.rect(1845, 1055, 650, 195, '#fff7e8', '#e7c88d')
    d.text(1870, 1096, '950DT / A5 的实际 packed KV 格式', 25, AMBER, 700, max_width=600)
    d.lines(1870, 1140, ['SWA：FP8 值 + BF16 scale',
                       '长程 KV：FP4 值 + BF16 scale',
                       'BF16 权重 / CLI dtype 不等于全 BF16 cache'], size=23, step=38, max_width=600)
    d.arrow('M1675 1145 H1805', BLUE, dash=True)

    # MoE detail.
    d.panel(1810, 1370, 720, 400, 'B  MoE · 路由与共享专家', AMBER)
    d.node(1845, 1470, 405, 105, '384 routed / Top-6', ['sqrtsoftplus；Top-K 权重归一化'], AMBER, '#fff7e8')
    d.node(2280, 1470, 215, 105, '1 shared', ['独立共享支路'], AMBER, '#fff7e8')
    d.arrow('M2047 1575 V1608 H2143')
    d.arrow('M2387 1575 V1608 H2197')
    d.rect(2148, 1586, 44, 44, 'white', AMBER, radius=22)
    d.text(2170, 1618, '+', 32, AMBER, anchor='middle')
    d.lines(1845, 1660, ['每专家 SwiGLU：5120 → 2304 → 5120',
                       '路由输出缩放 1.5；共享专家不计入 Top-6',
                       '全部 40 层均存在 MoE 权重'], size=24, step=39, max_width=650)
    d.arrow('M1675 1395 H1805', INK, dash=True)

    # Structurally present but unvalidated speculative path.
    d.panel(1810, 1840, 720, 420, 'C  MTP / DSpark · 可选草稿支路', PURPLE)
    for x, title in [(1845, 'mtp.0'), (2070, 'mtp.1'), (2295, 'mtp.2')]:
        d.node(x, 1950, 200, 72, title, [], PURPLE, '#f3eef9')
    d.arrow('M2045 1986 H2065', PURPLE)
    d.arrow('M2270 1986 H2290', PURPLE)
    d.lines(1845, 2070, ['3 个串行草稿层；128 路由专家 / Top-3',
                       'target layers = [37,38,39]；block size = 5',
                       'Markov rank 256 + confidence head'], size=24, step=39, max_width=650)
    d.text(2170, 2225, '结构存在；归档服务未启用 / 未验收投机解码', 23, PURPLE, 700, 'middle', 650)
    d.arrow('M1750 1670 H1780 V1986 H1805', PURPLE, dash=True)

    # Evidence scope and legend.
    d.rect(70, 2330, 2460, 160, '#e6edf7', '#e6edf7')
    d.text(100, 2370, '阅读与证据边界', 27, BLUE, 700)
    d.lines(100, 2410, ['实线：数据流；虚线：模块展开 / 可选支路。N 为当前扁平 token 数；全部尺寸为逻辑尺寸，未标成并行分片。',
                       '完整模型：40层 / E384 / Engram 开启；RL与1M专项：40层 / E32 / Engram 关闭。',
                       '配置上限 1,048,576；完整模型归档服务窗口 40,960 / 73,728；E32 已测 1,000,000 输入。本图不新增 NPU 验证结论。'],
            size=23, step=32, max_width=2390)
    d.text(70, 2521, '依据：facts.json · 配置 SHA256 '+data['source_identity']['config']['sha256'][:16]+'… · vLLM-Ascend 28e9e97 + 已归档运行补丁', 20, MUTED)
    d.text(2530, 2521, 'MindSpeed-Agent / model-rl-case-guard 0.30.0', 20, MUTED, anchor='end')
    return d.finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--svg-only', action='store_true')
    parser.add_argument('--png-width', type=int, default=2600)
    parser.add_argument('--font-file', action='append', default=[])
    args = parser.parse_args()
    data = json.loads((HERE / 'facts.json').read_text())
    verify_facts(data)
    svg = build(data)
    (HERE / 'model-architecture.svg').write_text(svg, encoding='utf-8')
    if not args.svg_only:
        import resvg_py
        defaults = ['/Library/Fonts/Arial Unicode.ttf', '/System/Library/Fonts/Hiragino Sans GB.ttc']
        fonts = args.font_file or [p for p in defaults if Path(p).is_file()]
        png = resvg_py.svg_to_bytes(svg_string=svg, width=args.png_width, font_files=fonts)
        (HERE / 'model-architecture.png').write_bytes(png)
    print('Verified source hashes and rendered architecture in', HERE)


if __name__ == '__main__':
    main()
