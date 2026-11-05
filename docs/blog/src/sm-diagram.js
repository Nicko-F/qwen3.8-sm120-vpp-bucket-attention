/* Original SM120 schematic. Geometry is illustrative; memory paths follow the kernels. */
'use strict';
(() => {
  const escape = value => String(value).replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
  const operations = {
    norm: {
      name: ['Residual / RMSNorm / NVFP4', '残差 / RMSNorm / NVFP4'],
      input: ['x + residual + γ', 'x + residual + γ'],
      value: ['BF16 y · 5120 values', 'BF16 y · 每行 5120 个值'],
      arithmetic: ['Add → RMSNorm', '残差相加 → RMSNorm'],
      pack: ['group16 → NVFP4', '每 16 个值一组'],
      shared: ['10 warp partials → warp 0', '10 个 warp 求和，由 warp 0 汇总'],
      bytes: ['10 KiB / direction / token', '每个 token 读、写各 10 KiB'],
      output: ['FP4 + scales; BF16 residual', 'FP4 和缩放因子、BF16 残差'],
      next: ['Next: NVFP4 GEMM', '下游：NVFP4 GEMM'],
      kept: ['BF16 rounding in RF → FP4 packing', '在寄存器内舍入到 BF16，再打包成 FP4'],
    },
    front: {
      name: ['QK Norm / RoPE / KV append', 'QK Norm / RoPE / KV 写入'],
      input: ['QKV projection + γ', 'QKV 投影 + γ'],
      value: ['BF16 K · 256 values / head', 'BF16 K · 每头 256 个值'],
      arithmetic: ['QK Norm → RoPE', 'QK Norm → RoPE'],
      pack: ['group16 → NVFP4', '每 16 个值一组'],
      shared: ['4 warp partials → warp 0', '4 个 warp 求和，由 warp 0 汇总'],
      bytes: ['512 B / direction / K head', '每个 K 头读、写各 512 B'],
      output: ['Q / gate: BF16; K/V: FP4 + scales', 'Q 和门控值用 BF16，K/V 用 FP4 加缩放因子'],
      next: ['Next: paged XQA', '下游：分页 XQA'],
      kept: ['K: Norm → RoPE → pack; V: pack', 'K：Norm → RoPE → 打包；V：直接打包'],
    },
    tail: {
      name: ['Attention output / gate / FP8', '注意力输出 / 门控 / FP8'],
      input: ['Attention result + gate', '注意力结果和门控值'],
      value: ['BF16 O · 1536 values / rank', 'BF16 O · 每卡 1536 个值'],
      arithmetic: ['Normalize / final merge', '归一化 / 最终合并'],
      pack: ['Gate → E4M3', '门控 → E4M3'],
      shared: ['', ''],
      bytes: ['3 KiB / direction / token / rank', '每卡每 token 读、写各 3 KiB'],
      output: ['8 E4M3 values / 8 B store', '一次写出 8 个 E4M3 值（8 B）'],
      next: ['Next: FP8 output projection', '下游：FP8 输出投影'],
      kept: ['Final writer: BF16 gate → FP8 packing', '合并后做 BF16 门控，再打包成 FP8'],
    },
  };
  const canvas = document.createElement('canvas');
  const context = canvas.getContext('2d');

  window.renderSmDiagram = (host, kind, fused, step) => {
    const operation = operations[kind];
    if (!host || !operation) return;
    const chinese = document.documentElement.lang.startsWith('zh');
    const pick = pair => pair[chinese ? 1 : 0];
    const local = (en, zh) => chinese ? zh : en;
    const phase = Math.max(0, Math.min(2, Number(step) || 0));
    const width = Math.floor(host.clientWidth || host.getBoundingClientRect().width || 800);
    if (width < 1) return;
    const compact = width < 680;
    const id = `sm-${kind}`;
    const family = getComputedStyle(host).fontFamily;
    const measure = (text, size) => {
      if (!context) return Array.from(text).length * size * .65;
      context.font = `${size}px ${family}`;
      return context.measureText(text).width;
    };
    function wrap(value, available, size) {
      const tokens = String(value).match(/[\u3400-\u9fff]|[^\s\u3400-\u9fff]+|\s+/g) || [];
      const lines = [];
      let line = '';
      for (const token of tokens) {
        if (measure(line + token, size) <= available) { line += token; continue; }
        if (line.trim()) lines.push(line.trim());
        line = token.trimStart();
        while (measure(line, size) > available && line.length > 1) {
          let end = 1;
          while (end < line.length && measure(line.slice(0, end + 1), size) <= available) end++;
          lines.push(line.slice(0, end));
          line = line.slice(end);
        }
      }
      if (line.trim()) lines.push(line.trim());
      return lines;
    }
    const text = (x, y, value, available, cls = 'sm-label', anchor = 'start') => {
      const size = cls === 'sm-eyebrow' ? 12 : ['sm-heading', 'sm-phase'].includes(cls) ? 16 : 14;
      const lines = wrap(value, available, size);
      return `<text x="${x}" y="${y}" class="${cls}" text-anchor="${anchor}">${lines.map((line, i) => `<tspan x="${x}" dy="${i ? size + 4 : 0}">${escape(line)}</tspan>`).join('')}</text>`;
    };
    const rect = (x, y, w, h, cls = 'sm-unit', radius = 5) => `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${radius}" class="${cls}"/>`;
    const route = (d, color, active = true, extra = '') => `<path d="${d}" class="sm-route sm-${color}${active ? ' is-current' : ''} ${extra}" marker-end="url(#${id}-${color})"/>`;
    let drawing = '';

    // Layout uses the host's CSS width as SVG units, keeping labels at their CSS size.
    const rfValue = kind === 'tail' && fused && phase === 0 ? local('Final O + loaded gate', '最终结果 O 和门控值') : pick(phase === 0 ? operation.input : phase === 1 ? operation.value : operation.output);
    const sm = {x: 8, y: 72, w: compact ? width - 16 : width - 210};
    const part = {x: compact ? 38 : 24, y: compact ? 160 : 174, w: compact ? width - 76 : sm.w - 32};
    const schedulerHeight = 18 + wrap(local('Warp scheduler / dispatch', 'warp 调度与发射'), part.w - 38, 14).length * 18;
    const rf = {x: part.x + 12, y: part.y + 15 + schedulerHeight + 18, w: part.w - 24};
    rf.h = 48 + wrap(rfValue, rf.w - 20, 14).length * 18;
    const gap = 10;
    const available = part.w - 24;
    const half = (available - gap) / 2;
    const unit = (x, y, w, h) => ({x, y, w, h, cx: x + w / 2, right: x + w, bottom: y + h});
    const start = part.x + 12;
    const cuda = unit(start, rf.y + rf.h + 28, compact ? half : (available - 3 * gap) * .32, compact ? 156 : 144);
    const tensor = unit(cuda.right + gap, cuda.y, compact ? half : (available - 3 * gap) * .28, cuda.h);
    const sfu = unit(compact ? start : tensor.right + gap, compact ? cuda.bottom + 16 : cuda.y, compact ? half : (available - 3 * gap) * .18, compact ? 112 : cuda.h);
    const ldst = unit(sfu.right + gap, sfu.y, compact ? half : (available - 3 * gap) * .22, sfu.h);
    part.h = ldst.bottom + 18 - part.y;
    const sharedTitle = compact && kind !== 'tail' ? local('L1 / shared', 'L1 / 共享内存') : local('L1 cache / shared · 128 KB', 'L1 缓存 / 共享内存 · 128 KB');
    const sharedBody = compact && kind !== 'tail' ? `128 KB · ${pick(operation.shared)}` : pick(operation.shared);
    const blockHeight = (title, body, w) => wrap(title, w - 20, 16).length * 20 + (body ? 10 + wrap(body, w - 20, 14).length * 18 : 0);
    const shared = {x: part.x, y: part.y + part.h + 18, w: part.w, h: blockHeight(sharedTitle, sharedBody, part.w) + 24};
    sm.h = shared.y + shared.h + 20 - sm.y;
    const memoryDetail = phase === 1 && fused ? '' : kind === 'tail' && fused && phase === 0 ? local('BF16 gate input', 'BF16 门控输入') : phase === 1 ? pick(operation.value) : phase === 0 ? pick(operation.input) : pick(operation.output);
    const memoryNote = phase === 1 && !fused ? pick(operation.bytes) : phase === 2 ? pick(operation.next) : local('Global address space', '全局地址空间');
    const memoryBody = [memoryDetail, memoryNote].filter(Boolean).join(' · ');
    const memory = {x: compact ? 38 : width - 180, w: compact ? width - 76 : 164};
    memory.h = blockHeight('GPU L2 cache', memoryBody, memory.w) + 28;
    const gddr = {x: memory.x, w: memory.w, h: blockHeight('GDDR7', local('Device VRAM', '设备显存'), memory.w) + 28};
    memory.y = compact ? sm.y + sm.h + 64 : sm.y + (sm.h - memory.h - 64 - gddr.h) / 2;
    gddr.y = memory.y + memory.h + 64;
    const title = `${pick(operation.name)} · ${local(fused ? 'Fused path' : 'Native path', fused ? '融合路径' : '原生路径')}`;
    const stages = [local('01 · Load into registers', '01 · 加载到寄存器'), local(fused ? '02 · Compute and pack in SM' : '02 · Materialize BF16, then reload', fused ? '02 · SM 内计算并打包' : '02 · BF16 写出，再读回'), local('03 · Write the consumer format', '03 · 写出下游所需格式')];
    drawing += text(8, 18, local('RTX BLACKWELL · ONE SM', 'RTX BLACKWELL · 单个 SM'), width - 16, 'sm-eyebrow');
    drawing += text(8, 42, stages[phase], width - 16, 'sm-phase');
    drawing += rect(sm.x, sm.y, sm.w, sm.h, 'sm-boundary', 8);
    drawing += text(sm.x + 14, sm.y + 23, local('Streaming Multiprocessor', '流式多处理器 SM'), sm.w - 28, 'sm-heading');
    const miniGap = compact ? 5 : 9;
    const miniWidth = (sm.w - 28 - miniGap * 3) / 4;
    for (let index = 0; index < 4; index++) {
      const x = sm.x + 14 + index * (miniWidth + miniGap);
      drawing += rect(x, 108, miniWidth, 34, `sm-partition${index === 0 ? ' is-expanded' : ''}`, 3);
      drawing += text(x + miniWidth / 2, 131, `SMSP ${index}`, miniWidth - 8, 'sm-small', 'middle');
    }
    drawing += `<path d="M${sm.x + 14 + miniWidth / 2} 142 V152 H${part.x + 12} V${part.y}" class="sm-expansion"/>`;
    drawing += rect(part.x, part.y, part.w, part.h, 'sm-expanded', 6);
    drawing += rect(part.x + 12, part.y + 15, part.w - 24, schedulerHeight, 'sm-unit');
    drawing += text(part.x + part.w / 2, part.y + 15 + 10 + 14, local('Warp scheduler / dispatch', 'warp 调度与发射'), part.w - 38, 'sm-label', 'middle');
    drawing += rect(rf.x, rf.y, rf.w, rf.h, `sm-register${phase === 1 && fused ? ' is-current' : ''}`);
    drawing += text(rf.x + rf.w / 2, rf.y + 23, local('Register file · RF', '寄存器组（RF）'), rf.w - 20, 'sm-heading', 'middle');
    drawing += text(rf.x + rf.w / 2, rf.y + 45, rfValue, rf.w - 20, 'sm-small', 'middle');
    const stripeWidth = (rf.w - 28) / 8;
    for (let stripe = 0; stripe < 8; stripe++) drawing += rect(rf.x + 10 + stripe * (stripeWidth + 1), rf.y + rf.h - 12, stripeWidth - 2, 5, 'sm-register-stripe', 1);

    const cell = (box, title, subtitle, cls = '', key = '') => {
      let node = rect(box.x, box.y, box.w, box.h, `sm-unit ${cls}`);
      const top = box.y + (box.h - blockHeight(title, subtitle, box.w)) / 2;
      node += text(box.cx, top + 16, title, box.w - 20, 'sm-heading', 'middle');
      if (subtitle) node += text(box.cx, top + wrap(title, box.w - 20, 16).length * 20 + 10 + 14, subtitle, box.w - 20, 'sm-small', 'middle');
      drawing += `<g class="sm-node"${key ? ` data-sm-node="${key}"` : ''}>${node}</g>`;
    };
    drawing += rect(cuda.x, cuda.y, cuda.w, cuda.h, `sm-unit${phase === 1 ? ' is-compute' : ''}`);
    drawing += text(cuda.cx, cuda.y + 25, compact ? 'CUDA' : local('CUDA lanes', 'CUDA 通道'), cuda.w - 16, 'sm-heading', 'middle');
    // Thirty-two FP32 lanes per partition; the tiny matrix below is one Tensor Core.
    for (let lane = 0; lane < 32; lane++) {
      const tile = (cuda.w - 22) / 8;
      drawing += rect(cuda.x + 8 + (lane % 8) * tile, cuda.y + 44 + Math.floor(lane / 8) * 6, tile - 2, 4, 'sm-cuda-lane', 0);
    }
    const compactArithmetic = {norm: 'RMSNorm', front: 'Norm / RoPE', tail: local('Final merge', '最终归约')}[kind];
    const arithmetic = compact ? compactArithmetic : pick(operation.arithmetic);
    const pack = compact && kind !== 'tail' ? local('NVFP4 group16', '每组 16 个值') : pick(operation.pack);
    const computeLabel = phase === 0 ? arithmetic : fused || phase === 2 ? pack : arithmetic;
    drawing += text(cuda.cx, cuda.y + 89, local('32 lanes', '32 个通道'), cuda.w - 16, 'sm-small', 'middle');
    drawing += text(cuda.cx, cuda.y + 115, computeLabel, cuda.w - 16, 'sm-small', 'middle');
    drawing += rect(tensor.x, tensor.y, tensor.w, tensor.h, 'sm-unit');
    drawing += text(tensor.cx, tensor.y + 25, 'Tensor Core', tensor.w - 16, 'sm-heading', 'middle');
    const matrixSize = Math.min(36, tensor.w - 24);
    for (let tile = 0; tile < 12; tile++) {
      drawing += rect(tensor.cx - matrixSize / 2 + tile % 4 * matrixSize / 4, tensor.y + 61 + Math.floor(tile / 4) * 6, matrixSize / 4 - 2, 4, 'sm-tensor-tile', 0);
    }
    drawing += text(tensor.cx, tensor.y + 113, local('5th gen · MMA', '第五代 · MMA'), tensor.w - 16, 'sm-small', 'middle');
    cell(sfu, 'SFU', local('Reciprocal / exp', '倒数 / 指数'), '', 'sfu');
    const loadStoreState = phase === 1 && fused && kind === 'tail' ? '' : `is-${phase === 0 ? 'load' : phase === 2 ? 'output' : fused ? 'compute' : 'native'}`;
    cell(ldst, 'LD/ST', local('Global / shared', '全局 / 共享'), loadStoreState, 'ldst');

    cell({...shared, cx:shared.x + shared.w / 2}, sharedTitle, sharedBody, `sm-shared${fused && phase === 1 && kind !== 'tail' ? ' is-current' : ''}`);

    // Global traffic goes directly between LD/ST and L2, beside the shared-memory block.
    const port = compact ? ldst.y + 28 : ldst.y + 36;
    const corridor = compact ? width - 18 : width - 194;
    const memoryIn = memory.y + 28;
    const loadRegister = compact ? `M${ldst.right} ${port} H${part.x + part.w - 3} V${rf.y + 28} H${rf.x + rf.w}` : `M${ldst.cx} ${ldst.y} V${rf.y + rf.h}`;
    const storeRegister = compact ? `M${rf.x + rf.w} ${rf.y + 42} H${part.x + part.w - 8} V${port + 12} H${ldst.right}` : `M${ldst.cx + 8} ${rf.y + rf.h} V${ldst.y}`;
    if (phase === 0) drawing += route(loadRegister, 'load');
    if (phase === 1 && !fused) drawing += route(loadRegister, 'native') + route(storeRegister, 'native');
    if (phase === 1 && fused && kind !== 'tail') drawing += route(loadRegister, 'fused') + route(storeRegister, 'fused');
    if (phase === 2) drawing += route(storeRegister, 'output');
    drawing += route(`M${cuda.cx} ${rf.y + rf.h} V${cuda.y}`, phase === 1 && !fused ? 'native' : 'fused', phase === 1);
    if (phase === 1) drawing += route(`M${cuda.cx + 10} ${cuda.y} V${rf.y + rf.h}`, fused ? 'fused' : 'native');
    if (phase === 0) {
      drawing += route(`M${memory.x + (compact ? memory.w : 0)} ${memoryIn} H${corridor} V${port} H${ldst.right}`, 'load');
    } else if (phase === 1 && !fused) {
      drawing += route(`M${ldst.right} ${port} H${corridor} V${memoryIn} H${memory.x + (compact ? memory.w : 0)}`, 'native');
      drawing += route(`M${memory.x + (compact ? memory.w : 0)} ${memoryIn + 28} H${corridor - 10} V${port + 24} H${ldst.right}`, 'native');
    } else if (phase === 2) {
      drawing += route(`M${ldst.right} ${port} H${corridor} V${memoryIn} H${memory.x + (compact ? memory.w : 0)}`, 'output');
    }
    if (kind !== 'tail') {
      drawing += route(`M${ldst.cx - 4} ${ldst.bottom} V${shared.y}`, 'fused', fused && phase === 1);
      drawing += route(`M${ldst.cx + 4} ${shared.y} V${ldst.bottom}`, 'fused', fused && phase === 1);
    }

    drawing += text(memory.x + memory.w / 2, memory.y - 15, local('Outside this SM', 'SM 之外'), memory.w, 'sm-small', 'middle');
    cell({...memory, cx:memory.x + memory.w / 2}, 'GPU L2 cache', memoryBody, 'sm-memory', 'l2');
    cell({...gddr, cx:gddr.x + gddr.w / 2}, 'GDDR7', local('Device VRAM', '设备显存'), 'sm-memory', 'gddr');
    const center = memory.x + memory.w / 2;
    drawing += route(phase === 0 ? `M${center} ${gddr.y} V${memory.y + memory.h}` : `M${center} ${memory.y + memory.h} V${gddr.y}`, phase === 0 ? 'load' : phase === 1 && !fused ? 'native' : 'output', false, `is-conditional${phase === 1 && fused ? ' is-context' : ''}`);
    if (phase === 1 && !fused) drawing += route(`M${center - 6} ${gddr.y} V${memory.y + memory.h}`, 'native', false, 'is-conditional');
    drawing += text(center + 10, memory.y + memory.h + 26, local('Miss / eviction', '未命中 / 驱逐'), memory.w / 2 - 14, 'sm-small');

    const caption = phase === 0 ? kind === 'tail' && fused ? local('Gate: L2 → LD/ST → RF; final O is already on chip', '门控值从 L2 读入寄存器，最终结果 O 已在片上') : local('Inputs → L2 → LD/ST → RF', '输入 → L2 → LD/ST → RF') : phase === 1 ? fused ? pick(operation.kept) : `${pick(operation.bytes)} · ${local('BF16 store + reload', 'BF16 写出 + 读回')}` : `${pick(operation.output)} · ${pick(operation.next)}`;
    const footerY = Math.max(sm.y + sm.h, gddr.y + gddr.h) + 30;
    const legend = [['load', local('Input load', '输入加载')], ['native', local('BF16 round trip', 'BF16 往返')], ['fused', local('RF packing', 'RF 内打包')], ['output', local('Output store', '写出结果')]];
    const columns = compact ? 2 : 4;
    const legendRowHeight = compact ? 42 : 24;
    legend.forEach(([color, label], index) => {
      const x = 8 + index % columns * (width - 16) / columns;
      const y = footerY + Math.floor(index / columns) * legendRowHeight;
      drawing += `<path d="M${x} ${y - 4} h18" class="sm-route sm-${color} is-current"/>`;
      drawing += text(x + 25, y, label, (width - 16) / columns - 30, 'sm-small');
    });
    const height = footerY + (compact ? legendRowHeight * 2 : legendRowHeight);
    const defs = ['load', 'native', 'fused', 'output'].map(color => `<marker id="${id}-${color}" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M0 0 L8 4 L0 8 Z" class="sm-arrow-${color}"/></marker>`).join('');
    const description = `${stages[phase]}. ${local('Four SMSPs; one expanded. Packing uses CUDA lanes. L1/shared is a separate cooperation path. GDDR7 traffic is conditional on L2 misses or evictions.', '图中展开了四个执行分区（SMSP）中的一个。CUDA 执行单元负责打包，线程可通过共享内存交换数据。L2 未命中或需要写回时才会访问 GDDR7。')} ${caption}`;
    host.innerHTML = `<svg class="sm-svg" viewBox="0 0 ${width} ${height}" role="img" aria-labelledby="${id}-title ${id}-desc" xmlns="http://www.w3.org/2000/svg"><title id="${id}-title">${escape(title)}</title><desc id="${id}-desc">${escape(description)}</desc><defs>${defs}</defs>${drawing}</svg>`;
  };
})();
