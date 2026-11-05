/* Rank-local data topology: all three streams are shown concurrently, without steps. */
'use strict';
(() => {
  const escape = value => String(value).replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
  const canvas = document.createElement('canvas');
  const context = canvas.getContext('2d');

  window.renderRankTopology = host => {
    if (!host) return;
    const width = Math.floor(host.clientWidth || host.getBoundingClientRect().width || 800);
    if (width < 1) return;
    const narrow = width < 760;
    const local = (en, zh) => document.documentElement.lang.startsWith('zh') ? zh : en;
    const receiveTitle = local('H2D Stream', '接收流 H2D');
    const computeTitle = local('Compute Stream', '计算流');
    const sendTitle = local('D2H Stream', '发送流 D2H');
    const computeBody = local('S1 / S6 · serial local stages', 'S1 / S6 · 本地阶段串行');
    const hostTitle = local('Host proxy thread', '主机代理线程');
    const hostBody = local('Event · READY / ACK', '事件 · READY / ACK');
    const slotTitle = index => local(`Slot ${index}`, `槽 ${index}`);
    const id = `rt-${(host.id || 'rank').replace(/[^a-zA-Z0-9_-]/g, '-')}`;
    const family = getComputedStyle(host).fontFamily;
    const measure = (value, size) => {
      if (!context) return [...value].length * size * .7;
      context.font = `${size === 16 ? '600 ' : ''}${size}px ${family}`;
      return context.measureText(value).width;
    };
    const wrap = (value, available, size) => {
      const lines = [];
      let line = '';
      for (const word of String(value).match(/[\u3400-\u9fff]|[^\s\u3400-\u9fff]+|\s+/g) || []) {
        if (measure(line + word, size) <= available) { line += word; continue; }
        if (line.trim()) lines.push(line.trim());
        line = word.trimStart();
        while (measure(line, size) > available && line.length > 1) {
          let end = 1;
          while (end < line.length && measure(line.slice(0, end + 1), size) <= available) end++;
          lines.push(line.slice(0, end));
          line = line.slice(end);
        }
      }
      if (line.trim()) lines.push(line.trim());
      return lines;
    };
    const text = (x, y, value, available, heading = false, anchor = 'middle') => {
      const size = heading ? 16 : 14;
      return `<text x="${x}" y="${y}" class="rt-${heading ? 'heading' : 'label'}" text-anchor="${anchor}">${wrap(value, available, size).map((line, index) => `<tspan x="${x}" dy="${index ? size + 4 : 0}">${escape(line)}</tspan>`).join('')}</text>`;
    };
    const frame = (x, y, w, h, cls = '') => `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="5" class="rt-frame ${cls}"/>`;
    const wire = (d, role, control = false) => `<path d="${d}" class="rt-wire rt-${role}${control ? ' rt-control' : ''}" marker-end="url(#${id}-${role})"/>`;
    const heightFor = (title, body, w) => Math.max(78, 24 + wrap(title, w - 20, 16).length * 20 + 8 + wrap(body, w - 20, 14).length * 18);
    const box = (x, y, w, h, key, title, body, cls = '') => {
      const titleLines = wrap(title, w - 20, 16).length;
      const total = titleLines * 20 + (body ? 8 + wrap(body, w - 20, 14).length * 18 : 0);
      const top = y + (h - total) / 2;
      return `<g class="rt-node" data-rank-node="${key}">${frame(x, y, w, h, cls)}${text(x + w / 2, top + 16, title, w - 20, true)}${body ? text(x + w / 2, top + titleLines * 20 + 8 + 14, body, w - 20) : ''}</g>`;
    };
    const geometry = (x, y, w, h) => ({x, y, w, h, cx: x + w / 2, cy: y + h / 2, right: x + w, bottom: y + h});
    const sideWidth = narrow ? (width - 24) / 2 : 126;
    const gpu = {x: narrow ? 8 : 150, y: narrow ? 148 : 60, w: narrow ? width - 16 : width - 300};
    const inner = narrow ? gpu.w - 48 : gpu.w - 32;
    const columns = inner - 36;
    const receiveWidth = narrow ? inner : columns * .38;
    const sourceWidth = narrow ? inner : columns * .35;
    const leftWidth = narrow ? inner : columns * .27;
    const left = gpu.x + (narrow ? 24 : 16);
    const middle = narrow ? left : left + leftWidth + 18;
    const right = narrow ? left : middle + receiveWidth + 18;
    const copy = local('GPU copy engine', 'GPU 复制引擎');
    const h2d = geometry(left, narrow ? 194 : 146, leftWidth, heightFor(receiveTitle, copy, leftWidth));
    const receiveRow = Math.max(36, 12 + wrap(local('R0 · current input', '槽 0 · 当前输入'), receiveWidth - 40, 14).length * 18);
    const receive = geometry(middle, narrow ? h2d.bottom + 34 : 112, receiveWidth, 64 + receiveRow * 3 + 12);
    const compute = geometry(middle, receive.bottom + 36, receiveWidth, heightFor(computeTitle, computeBody, receiveWidth));
    const sourceRow = Math.max(58, heightFor(slotTitle(0), local('Earlier output', '较早输出'), sourceWidth - 20));
    const source = geometry(right, narrow ? compute.bottom + 36 : 290, sourceWidth, 64 + sourceRow * 2 + 8 + 12);
    const d2h = geometry(right, narrow ? source.bottom + 36 : 146, sourceWidth, heightFor(sendTitle, copy, sourceWidth));
    gpu.h = (narrow ? d2h.bottom : Math.max(source.bottom, compute.bottom)) + 20 - gpu.y;
    const hostWidth = narrow ? inner : Math.min(360, gpu.w - 32);
    const hostBox = geometry((width - hostWidth) / 2, gpu.y + gpu.h + 44, hostWidth, heightFor(hostTitle, hostBody, hostWidth));
    const endpointBody = [local('Forward · S0 → S1', '去程 · S0 → S1'), local('Return · S5 → S6', '回程 · S5 → S6'), local('Forward · S1 → S2', '去程 · S1 → S2'), local('Return · S6 → S7', '回程 · S6 → S7')];
    const endpointHeight = Math.max(...endpointBody.map(body => heightFor('GPU0', body, sideWidth)));
    const inputs = narrow ? [geometry(8, 16, sideWidth, endpointHeight), geometry(width - sideWidth - 8, 16, sideWidth, endpointHeight)] : [geometry(8, 132, sideWidth, endpointHeight), geometry(8, 354, sideWidth, endpointHeight)];
    const outputs = narrow ? [geometry(8, hostBox.bottom + 36, sideWidth, endpointHeight), geometry(width - sideWidth - 8, hostBox.bottom + 36, sideWidth, endpointHeight)] : [geometry(width - sideWidth - 8, 132, sideWidth, endpointHeight), geometry(width - sideWidth - 8, 354, sideWidth, endpointHeight)];
    let drawing = frame(gpu.x, gpu.y, gpu.w, gpu.h, 'rt-gpu');
    drawing += text(gpu.x + 14, gpu.y + 25, 'GPU1 / RANK 1', gpu.w - 100, true, 'start');
    drawing += text(gpu.x + gpu.w - 14, gpu.y + 25, 'S1 · S6', 70, false, 'end');

    // Data links retain the original fold on desktop; the narrow layout follows it vertically.
    if (narrow) {
      drawing += wire(`M${inputs[0].cx} ${inputs[0].bottom} V${gpu.y - 12} H${gpu.x + 12} V${h2d.cy} H${h2d.x}`, 'receive');
      drawing += wire(`M${inputs[1].cx} ${inputs[1].bottom} V${gpu.y - 12} H${gpu.x + gpu.w - 12} V${h2d.cy} H${h2d.right}`, 'receive');
      drawing += wire(`M${h2d.cx} ${h2d.bottom} V${receive.y}`, 'receive');
      drawing += wire(`M${receive.cx} ${receive.bottom} V${compute.y}`, 'compute');
      drawing += wire(`M${compute.cx} ${compute.bottom} V${source.y - 18} H${source.x - 8} V${source.y + 60 + sourceRow + 8 + sourceRow / 2} H${source.x + 10}`, 'compute');
      drawing += wire(`M${source.right - 10} ${source.y + 60 + sourceRow / 2} H${source.right + 8} V${d2h.y - 18} H${d2h.cx} V${d2h.y}`, 'send');
      drawing += wire(`M${d2h.x} ${d2h.cy} H${gpu.x + 12} V${outputs[0].y - 16} H${outputs[0].cx} V${outputs[0].y}`, 'send');
      drawing += wire(`M${d2h.right} ${d2h.cy} H${gpu.x + gpu.w - 12} V${outputs[1].y - 16} H${outputs[1].cx} V${outputs[1].y}`, 'send');
      drawing += wire(`M${hostBox.cx} ${hostBox.y} V${gpu.y + gpu.h}`, 'control', true);
    } else {
      inputs.forEach(endpoint => { drawing += wire(`M${endpoint.right} ${endpoint.cy} H${gpu.x - 7} V${h2d.cy} H${h2d.x}`, 'receive'); });
      drawing += wire(`M${h2d.right} ${h2d.cy} H${receive.x}`, 'receive');
      drawing += wire(`M${receive.cx} ${receive.bottom} V${compute.y}`, 'compute');
      drawing += wire(`M${compute.right} ${compute.cy} H${source.x - 9} V${source.y + 60 + sourceRow + 8 + sourceRow / 2} H${source.x + 10}`, 'compute');
      drawing += wire(`M${source.right - 10} ${source.y + 60 + sourceRow / 2} H${source.right + 8} V${d2h.bottom + 18} H${d2h.cx} V${d2h.bottom}`, 'send');
      outputs.forEach(endpoint => { drawing += wire(`M${d2h.right} ${d2h.cy} H${gpu.x + gpu.w + 7} V${endpoint.cy} H${endpoint.x}`, 'send'); });
      [h2d.cx, compute.cx, d2h.cx].forEach((target, index) => { drawing += wire(`M${hostBox.x + hostBox.w * (index + 1) / 4} ${hostBox.y} V${gpu.y + gpu.h + 20} H${target} V${gpu.y + gpu.h}`, 'control', true); });
    }
    inputs.forEach((endpoint, index) => { drawing += box(endpoint.x, endpoint.y, endpoint.w, endpoint.h, `from-gpu${index ? 2 : 0}`, `GPU${index ? 2 : 0}`, endpointBody[index]); });
    outputs.forEach((endpoint, index) => { drawing += box(endpoint.x, endpoint.y, endpoint.w, endpoint.h, `to-gpu${index ? 0 : 2}`, `GPU${index ? 0 : 2}`, endpointBody[index + 2]); });
    drawing += box(h2d.x, h2d.y, h2d.w, h2d.h, 'h2d', receiveTitle, copy, 'rt-stream rt-receive');
    drawing += box(compute.x, compute.y, compute.w, compute.h, 'compute', computeTitle, computeBody, 'rt-stream rt-compute');
    drawing += box(d2h.x, d2h.y, d2h.w, d2h.h, 'd2h', sendTitle, copy, 'rt-stream rt-send');
    drawing += frame(receive.x, receive.y, receive.w, receive.h, 'rt-arena rt-receive');
    drawing += text(receive.cx, receive.y + 25, local('Receive', '接收缓冲'), receive.w - 20, true);
    drawing += text(receive.cx, receive.y + 47, local('3 shared slots', '共享 3 个槽'), receive.w - 20);
    [local('R0 · current input', '槽 0 · 当前输入'), local('R1 · prefetch', '槽 1 · 预取输入'), local('R2 · prefetch', '槽 2 · 预取输入')].forEach((label, index) => {
      const y = receive.y + 60 + index * receiveRow;
      drawing += `<g data-rank-slot="receive-${index}">${frame(receive.x + 10, y, receive.w - 20, receiveRow - 5, index ? 'rt-slot rt-receive' : 'rt-slot rt-compute')}${text(receive.cx, y + (receiveRow - 5 - wrap(label, receive.w - 40, 14).length * 18) / 2 + 14, label, receive.w - 40)}</g>`;
    });
    drawing += frame(source.x, source.y, source.w, source.h, 'rt-arena rt-send');
    drawing += text(source.cx, source.y + 25, local('Source', '输出缓冲'), source.w - 20, true);
    drawing += text(source.cx, source.y + 47, local('2 shared slots', '共享 2 个槽'), source.w - 20);
    [local('Earlier output', '较早输出'), local('Current output', '当前输出')].forEach((label, index) => {
      const y = source.y + 60 + index * (sourceRow + 8);
      drawing += `<g data-rank-slot="source-${index}">${box(source.x + 10, y, source.w - 20, sourceRow, `source-slot-${index}`, slotTitle(index), label, `rt-slot rt-${index ? 'compute' : 'send'}`)}</g>`;
    });
    drawing += box(hostBox.x, hostBox.y, hostBox.w, hostBox.h, 'host-proxy', hostTitle, hostBody, 'rt-host');
    const title = local('GPU1 rank topology: shared buffers and three concurrent CUDA streams', 'GPU1 Rank 拓扑：共享缓冲与三条可并行的 CUDA Stream');
    const description = local('Forward input arrives from GPU0; return input arrives from GPU2. H2D fills three shared receive slots, Compute executes S1 or S6, and two source slots feed D2H toward GPU2 or GPU0. Solid links show data dependencies, while dashed links connect host control. The three streams can overlap work on different chunks.', 'Forward 输入来自 GPU0，Return 输入来自 GPU2。H2D 写入共享的三个 Receive 槽，Compute 执行 S1 或 S6，两个 Source 槽经 D2H 发往 GPU2 或 GPU0。实线表示数据依赖，虚线连接 Host 控制；三条 Stream 可重叠处理不同 Chunk。');
    const defs = ['receive', 'compute', 'send', 'control'].map(role => `<marker id="${id}-${role}" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0 L8 4 L0 8 Z" class="rt-arrow rt-${role}"/></marker>`).join('');
    const top = narrow ? 0 : 40;
    const height = Math.max(hostBox.bottom, ...outputs.map(endpoint => endpoint.bottom)) + 16 - top;
    host.innerHTML = `<svg class="rank-topology-svg" viewBox="0 ${top} ${width} ${height}" role="img" aria-labelledby="${id}-title ${id}-desc" xmlns="http://www.w3.org/2000/svg"><title id="${id}-title">${escape(title)}</title><desc id="${id}-desc">${escape(description)}</desc><defs>${defs}</defs>${drawing}</svg>`;
  };
})();
