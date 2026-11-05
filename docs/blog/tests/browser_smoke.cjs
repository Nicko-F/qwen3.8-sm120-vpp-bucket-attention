// Optional browser checks: Node.js, Playwright and Chromium.
const assert = require('node:assert/strict');
const path = require('node:path');
const { pathToFileURL } = require('node:url');
const { chromium } = require('playwright');

function checkModels() {
    const {taskPlan,nativePlan,schedule}=BlogModel;
    let seed=913;
    function rand(){seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed/4294967296;}
    function check(ok,msg){if(!ok)throw Error(msg);}
    for(let k=0;k<500;k++){
        const b=1+Math.floor(rand()*32), c=b+Math.floor(rand()*(171-b));
        const lens=Array.from({length:b},()=>Math.floor(rand()*131073));
        const p=taskPlan(lens,c),n=nativePlan(lens,c);
        check(p.loads.reduce((a,b)=>a+b,0)===p.total,'bucket work');
        check(n.loads.reduce((a,b)=>a+b,0)===p.total,'native work');
        check(p.tasks.length<=c+b-1,'task capacity');
        check(p.active<=c,'CTA bound');
        p.loads.forEach(x=>check(x<=p.budget,'budget'));
        p.requests.forEach((r,i)=>{
            let end=0;
            p.tasks.slice(r.begin,r.begin+r.count).forEach(t=>{
                check(t.request===i && t.begin===end && t.end>t.begin,'contiguous work');
                end=t.end;
            });
            check(end===p.counts[i],'request coverage');
        });
    }
    for(const n of [1,2,3,7,8,16,24,32]) for(const g of [0,.12,.24]){
        const c=schedule(n,g,'cpp'),v=schedule(n,g,'vpp');
        check(Math.abs(c.work-v.work)<1e-7,'compute work');
        check(Math.abs(c.finish-(c.work/4+3*(1+g*(n-1))))<1e-7,'CPP recurrence');
        for(const p of [c,v]){
            for(let r=0;r<4;r++){
                let end=0;
                p.nodes.filter(x=>x.rank===r).forEach(x=>{check(x.start>=end-1e-8,'stream ordering');end=x.end;});
            }
            p.nodes.forEach(x=>{if(x.stage)check(x.start>=p.get(x.chunk,x.stage-1).end-1e-8,'dependency');});
        }
    }
    return {bucket_cases:500,pipeline_cases:24};
}

async function stepThrough(page, selector) {
  const buttons = page.locator(selector + ' button');
  for (let i = 0; i < await buttons.count(); i++) {
    await buttons.nth(i).click();
    assert.equal(await buttons.nth(i).getAttribute('aria-pressed'), 'true');
  }
}

function smLabelAlignment(svg) {
  const misplaced = [];
  for (const node of svg.querySelectorAll('[data-sm-node]')) {
    const box = node.querySelector('rect').getBoundingClientRect();
    const labels = [...node.querySelectorAll('text')].map(el => el.getBoundingClientRect());
    const left = Math.min(...labels.map(r => r.left));
    const right = Math.max(...labels.map(r => r.right));
    const top = Math.min(...labels.map(r => r.top));
    const bottom = Math.max(...labels.map(r => r.bottom));
    const offsetX = (left + right - box.left - box.right) / 2;
    const offsetY = (top + bottom - box.top - box.bottom) / 2;
    if (Math.abs(offsetX) > 2 || Math.abs(offsetY) > 5 ||
        left < box.left || right > box.right || top < box.top || bottom > box.bottom) {
      misplaced.push([node.dataset.smNode, Math.round(offsetX), Math.round(offsetY)]);
    }
  }
  return misplaced;
}

function readingGeometry() {
  const rect = element => element.getBoundingClientRect();
  const main = rect(document.querySelector('main'));
  const mismatches = [];
  const columns = document.querySelectorAll(
    'main > section, .chapter-heading, .chapter > .prose, .chapter figure, .deep-dive, .details-body > .prose'
  );
  for (const element of columns) {
    const box = rect(element);
    if (!box.width) continue;
    if (Math.abs(box.left - main.left) > 2 || Math.abs(box.right - main.right) > 2) {
      mismatches.push(element.id || element.className);
    }
  }
  for (const element of document.querySelectorAll('.chapter-heading h2, .chapter .prose > h3, .chapter figcaption, .figure-header h4')) {
    if (!element.getClientRects().length) continue;
    const range = document.createRange();
    range.selectNodeContents(element);
    const firstLine = range.getClientRects()[0];
    if (firstLine && Math.abs(firstLine.left - main.left) > 2) mismatches.push(element.textContent.slice(0, 30));
  }
  for (const element of document.querySelectorAll('.equation math, .sm-architecture, .fusion-mode button, .hero-results strong')) {
    const bounds = rect(element);
    if (!bounds.width) continue;
    if (bounds.left < main.left - 1 || bounds.right > main.right + 1) mismatches.push(element.textContent.slice(0, 35));
  }
  const hero = document.querySelector('.hero');
  if (hero.scrollWidth > hero.clientWidth + 1) mismatches.push('hero content');
  const chart = document.querySelector('#load-chart');
  const rail = rect(document.querySelector('#toc'));
  return {
    mismatches,
    chartOverflow: chart.scrollWidth - chart.clientWidth,
    bodyOverflow: document.documentElement.scrollWidth - innerWidth,
    railOutside: !rail.width || rail.left >= main.right,
  };
}

async function checkPage(browser, file, lang, otherPage) {
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 }, reducedMotion: 'reduce' });
  const errors = [];
  const network = [];
  page.on('pageerror', error => errors.push(error.message));
  page.on('request', req => { if (/^https?:/.test(req.url())) network.push(req.url()); });
  const url = pathToFileURL(path.resolve(__dirname, '..', file));
  try {
    await page.goto(url.href);
    assert.equal(await page.locator('html').getAttribute('lang'), lang);
    assert.equal(await page.locator('h1').innerText(), lang === 'en' ? 'Accelerating Qwen3.8 27B Inference on SM120' : '在 SM120 上加速千问3.8 27B 推理');
    assert.equal(await page.locator('svg[role=img]').count(), 10);
    assert.equal(await page.locator('.fusion-widget .sm-architecture svg').count(), 3);
    assert.equal(await page.locator('.reference-list li').count(), 16);
    await page.evaluate(checkModels);
    await page.evaluate(() => document.querySelectorAll('details').forEach(d => d.open = true));
    for (const selector of ['#pipe-count', '#pipe-growth', '#load-preset', '#bucket-preset']) {
      const select = page.locator(selector);
      for (const value of await select.locator('option').evaluateAll(options => options.map(o => o.value))) {
        await select.selectOption(value);
      }
    }
    await page.locator('#bucket-selected').selectOption('169');
    for (const holder of ['pipe', 'rank', 'dma', 'bucket', 'merge']) {
      await stepThrough(page, '#' + holder + '-steps');
    }
    for (const kind of ['norm', 'front', 'tail']) {
      for (const mode of ['before', 'after']) {
        await page.locator('#' + kind + '-widget button[data-mode=' + mode + ']').click();
        const widget = page.locator('#' + kind + '-widget');
        const diagram = widget.locator('.sm-architecture svg');
        assert.match(await diagram.textContent(), /GDDR7/);
        assert.equal(await widget.locator('button[data-mode=' + mode + ']').getAttribute('aria-pressed'), 'true');
        await stepThrough(page, '#' + kind + '-widget .fusion-steps');
      }
    }
    await page.locator('#prefill-metric [data-value=absolute]').click();
    assert.match(await page.locator('#prefill-chart').innerText(), /3\.936 s/);
    await page.locator('#prefill-metric [data-value=relative]').click();
    assert.match(await page.locator('#prefill-chart').innerText(), /100\.0%/);

    const panel = page.locator('.trace-panel').first();
    const traceColors = () => panel.evaluate(el => ({
      background: getComputedStyle(el).backgroundColor,
      lane: getComputedStyle(el.querySelector('rect')).fill,
      label: getComputedStyle(el.querySelector('.track-label')).fill,
    }));
    await page.emulateMedia({ colorScheme: 'light' });
    const lightTrace = await traceColors();
    await page.emulateMedia({ colorScheme: 'dark' });
    await page.waitForFunction(() => document.documentElement.dataset.theme === 'dark');
    const darkTrace = await traceColors();
    for (const key of ['lane', 'label']) assert.notEqual(lightTrace[key], darkTrace[key], key + ' should follow the theme');
    await page.locator('#theme-toggle').click();
    assert.deepEqual(await traceColors(), lightTrace);
    for (const width of [320, 390, 650, 768, 1024, 1120, 1440, 1850, 1920]) {
      await page.setViewportSize({ width, height: 900 });
      for (const svg of await page.locator('.sm-architecture svg').all()) {
        assert.equal(await svg.locator('[data-sm-node]').count(), 4);
        assert.deepEqual(await svg.evaluate(smLabelAlignment), [], file + ' SM labels at ' + width);
      }
      const geometry = await page.evaluate(readingGeometry);
      const context = file + ' at ' + width + 'px';
      assert.deepEqual(geometry.mismatches, [], context + ' reading column');
      assert.ok(geometry.bodyOverflow <= 1, context + ' page overflow');
      assert.ok(geometry.chartOverflow <= 1, context + ' CTA chart overflow');
      assert.ok(geometry.railOutside, context + ' contents overlap');
    }
    await page.setViewportSize({ width: 390, height: 844 });
    await page.locator('#menu-toggle').click();
    assert.equal(await page.locator('#menu-toggle').getAttribute('aria-expanded'), 'true');
    assert.equal(await page.locator('#article').evaluate(element => element.inert), true);
    await page.keyboard.press('Escape');
    assert.equal(await page.locator('#menu-toggle').evaluate(element => element === document.activeElement), true);
    await page.locator('#menu-toggle').click();
    await page.setViewportSize({ width: 1440, height: 900 });
    await page.waitForFunction(() => !document.getElementById('article').inert);
    await page.setViewportSize({ width: 390, height: 844 });
    await page.locator('#menu-toggle').click();
    await page.locator('#toc a[href="#decode"]').click();
    assert.equal(await page.locator('#menu-toggle').getAttribute('aria-expanded'), 'false');
    await page.locator('#language-switch').click();
    assert.equal(new URL(page.url()).pathname.split('/').pop(), otherPage);
    assert.equal(new URL(page.url()).hash, '#decode');
    assert.equal(await page.locator('.equation').count(), 5);
    assert.equal(await page.locator('.equation small').count(), 0);
    assert.equal(await page.locator('.equation munderover').count(), 1);
    assert.deepEqual(errors, []);
    assert.deepEqual(network, []);
    console.log(lang + ': ten diagrams, model checks, both themes, nine widths and language navigation passed');
  } finally {
    await page.close();
  }
}

(async () => {
  const browser = await chromium.launch({ executablePath: process.env.BLOG_BROWSER });
  try {
    await checkPage(browser, 'index.html', 'en', 'zh-CN.html');
    await checkPage(browser, 'zh-CN.html', 'zh-CN', 'index.html');
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
