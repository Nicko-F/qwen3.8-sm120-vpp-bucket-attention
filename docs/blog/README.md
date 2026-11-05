# Qwen3.8 27B · 千问3.8 27B

**[Download English HTML](https://github.com/Nicko-F/qwen3.8-sm120-vpp-bucket-attention/raw/refs/heads/main/docs/blog/index.html) · [下载中文 HTML](https://github.com/Nicko-F/qwen3.8-sm120-vpp-bucket-attention/raw/refs/heads/main/docs/blog/zh-CN.html)**

Follow a prefill chunk through four RTX 5090 GPUs, redistribute a mixed-length decode batch across CTAs, and step through three fused CUDA data paths. The diagrams have manual controls. Measured charts use the project's [benchmark CSVs](../../benchmarks).

在 Prefill 章可以追踪单个 chunk 的折返路径，在 Decode 章可以点选 CTA 查看其负责的 KV 区间，在 Fusion 章可以逐步对照三个算子的输入、处理和写出。

Open either HTML file in a browser. Keep both files in the same folder to switch languages at the current section. The diagrams, charts and themes work offline.

双击任一 HTML 即可离线阅读。将两个文件保存在同一目录，点击页首的 **English / 中文** 即可切换对应章节。

![Chinese edition on desktop, with the three optimization results beside the title](preview.png)

## Edit the article

The two article bodies live in [English](src/content.en.html) and [Chinese](src/content.html). Both pages share the [interactive models](src/blog.js), [diagram labels](src/messages.json), [stylesheet](src/styles.css) and benchmark data. See the [source notes](SOURCES.md) for the model assumptions and implementation links.

Rebuild both pages from the repository root with the Python standard library.

```bash
python docs/blog/build.py
```

With Playwright and Chromium available, run the optional interactive checks with `node docs/blog/tests/browser_smoke.cjs`. Set `BLOG_BROWSER` to use an existing Chromium executable.
