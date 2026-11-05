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
    assert.equal(await page.locator('h1').innerText(), lang === 'en' ? 'Qwen3.8 27B' : '千问3.8 27B');
    assert.equal(await page.locator('svg[role=img]').count(), 10);
    assert.equal(await page.locator('.reference-list li').count(), 15);
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
        await stepThrough(page, '#' + kind + '-widget .fusion-steps');
      }
    }
    await page.locator('#prefill-metric [data-value=absolute]').click();
    assert.match(await page.locator('#prefill-chart').innerText(), /3\.936 s/);
    await page.locator('#prefill-metric [data-value=relative]').click();
    assert.match(await page.locator('#prefill-chart').innerText(), /100\.0%/);

    const initialTheme = await page.locator('html').getAttribute('data-theme');
    await page.locator('#theme-toggle').click();
    assert.notEqual(await page.locator('html').getAttribute('data-theme'), initialTheme);
    for (const width of [320, 360, 390, 650, 768, 981, 1024, 1440, 1920]) {
      await page.setViewportSize({ width, height: 900 });
      assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), file + ' overflow at ' + width);
    }
    await page.setViewportSize({ width: 390, height: 844 });
    await page.locator('#menu-toggle').click();
    assert.equal(await page.locator('#menu-toggle').getAttribute('aria-expanded'), 'true');
    await page.locator('#toc a[href="#decode"]').click();
    assert.equal(await page.locator('#menu-toggle').getAttribute('aria-expanded'), 'false');
    await page.locator('#language-switch').click();
    assert.equal(new URL(page.url()).pathname.split('/').pop(), otherPage);
    assert.equal(new URL(page.url()).hash, '#decode');
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
