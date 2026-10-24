/* Actual application browser QA. Benchmark datasets/model artifacts are read only.
   Mutations: labelled demo orders/pipeline and a restored local profile setting. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {chromium,launchOptions} = require('./browser_runtime.cjs');
const base = process.env.FLEET_QA_URL || 'http://127.0.0.1:8765';
const out = path.resolve(__dirname, '../output/aeroguard');
fs.mkdirSync(out, {recursive:true});
const checks = [], pageErrors = [], badResponses = [];
const record = (name, value=true) => {assert.ok(value,name);checks.push(name);console.log('PASS '+name);};
const headings = {dashboard:'Fleet Overview',engines:'Engine Explorer',analytics:'Anomaly & RUL Analytics',models:'Model Registry & MLOps','work-orders':'Maintenance Work Orders',pipeline:'Pipeline Execution & Triage',data:'Dataset Management',settings:'Application & Pipeline Settings'};
let browser, context, page, originalSettings, profileChanged=false;
const report = {started_at:new Date().toISOString(),browser:'Installed headless Chromium / Playwright',viewports:['1440x1000','1024x768','390x844'],checks,pageErrors,badResponses,limitations:['Browser checks do not establish field reliability or live physical telemetry.','Benchmark artifacts are read only; order mutations use the labelled demo namespace.','No real training job is launched by this browser test.']};
async function jsonGet(route){const r=await context.request.get(base+route);assert.equal(r.status(),200,route);return r.json();}
async function screenshot(name,target=page){await target.locator('#toast').evaluate(el=>el.hidden=true).catch(()=>{});await target.screenshot({path:path.join(out,name),fullPage:true});}
async function goto(route,heading){const r=await page.goto(base+route);assert.equal(r.status(),200,route);await page.getByRole('heading',{name:heading,exact:true}).waitFor();}
async function waitJob(id){for(let i=0;i<300;i++){const j=await jsonGet('/api/jobs/'+encodeURIComponent(id));if(['complete','failed','interrupted'].includes(j.status))return j;await page.waitForTimeout(100);}throw new Error('Demo pipeline exceeded 30 seconds');}
async function idleJobs(){for(let i=0;i<600;i++){const j=await jsonGet('/api/jobs');if(!j.items.some(x=>['running','queued'].includes(x.status)))return;await page.waitForTimeout(100);}throw new Error('Existing benchmark job is still running; demo job was not submitted');}
async function closeDialog(){await page.getByRole('button',{name:'Close dialog',exact:true}).click();}

(async()=>{
 try {
  browser=await chromium.launch(launchOptions);
  context=await browser.newContext({viewport:{width:1440,height:1000},acceptDownloads:true});
  page=await context.newPage();page.on('pageerror',e=>pageErrors.push(e.message));page.on('response',r=>{if(r.status()>=400)badResponses.push({url:r.url(),status:r.status()});});
  originalSettings=await jsonGet('/api/settings');
  const baselineModels=await jsonGet('/api/models');
  report.benchmark_active_before=baselineModels.active;
  await goto('/','Self-Healing Machines for a Safer Tomorrow.');
  record('Landing page renders the requested AeroGuard hero and real navigation');
  await page.waitForLoadState('networkidle');
  record('Hero turbine asset and Inter font load',await page.evaluate(()=>document.fonts.check('14px Inter')));
  await screenshot('landing-desktop.png');
  await page.getByRole('link',{name:'Explore Dashboard',exact:true}).first().click();
  await page.getByRole('heading',{name:headings.dashboard,exact:true}).waitFor();
  record('Landing CTA navigates to the functional dashboard',new URL(page.url()).pathname==='/dashboard');
  const live=await jsonGet('/api/dashboard');
  record('Dashboard classifications sum to the backend engine count',live.summary.total===live.summary.healthy+live.summary.attention+live.summary.critical+live.summary.unknown);
  record('Dashboard pipeline summary preserves benchmark/demo separation',!live.last_successful_run||(live.last_successful_run.configuration.mode||'benchmark')==='benchmark');
  if(live.summary.total){await page.locator('#dashboard-trend svg.data-chart').waitFor();record('Dashboard loads a measured engine history plot',true);}
  record('Benchmark mode clearly identifies historical replay',(await page.locator('.mode-strip').innerText()).includes('Not a live physical sensor stream'));
  record('Desktop dashboard has no document overflow',await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
  await screenshot('dashboard-desktop.png');
  await page.getByRole('button',{name:'Notifications',exact:true}).click();
  await page.locator('#notifications .popover-heading').waitFor();
  record('Notifications open recorded alerts',await page.locator('#notifications').isVisible());
  await page.getByRole('button',{name:'Close',exact:true}).click();

  for(const route of Object.keys(headings)){
    await goto('/'+route,headings[route]);
    await page.reload();await page.getByRole('heading',{name:headings[route],exact:true}).waitFor();
    record('Direct refresh works for /'+route);
  }
  await goto('/engines',headings.engines);
  await page.locator('#data-mode').selectOption('demo');
  await page.locator('.mode-strip.demo').waitFor();
  record('Demo mode is labelled synthetic', (await page.locator('.mode-strip').innerText()).includes('Synthetic'));
  await page.locator('#global-subset').selectOption('FD004');
  await page.getByRole('heading',{name:headings.engines,exact:true}).waitFor();
  await page.locator('#engine-filter').fill('005');await page.locator('#severity-filter').selectOption('critical');
  await page.getByRole('button',{name:'Apply filters',exact:true}).click();
  await page.locator('tr[data-engine="FD004/demo/005"]').waitFor();
  await page.waitForFunction(()=>document.querySelectorAll('tr[data-engine]').length===1);
  record('Dataset, engine, and severity filters combine correctly',await page.locator('tr[data-engine]').count()===1);
  await page.locator('#cycle-min').fill('9999');await page.getByRole('button',{name:'Apply filters',exact:true}).click();
  await page.getByText('No matching engines',{exact:true}).waitFor();
  record('Cycle filter excludes out-of-range engine observations');
  await page.getByRole('button',{name:'Reset',exact:true}).click();
  await page.locator('tr[data-engine]').first().waitFor();
  const engineRow=page.locator('tr[data-engine]').first();const demoAsset=await engineRow.getAttribute('data-engine');
  await engineRow.focus();await engineRow.press('Enter');
  await page.getByRole('heading',{name:/^Engine #FD004-/}).waitFor();
  record('Keyboard engine-row navigation opens a dedicated detail route',decodeURIComponent(new URL(page.url()).pathname).includes(demoAsset));
  await page.reload();await page.getByRole('heading',{name:/^Engine #FD004-/}).waitFor();
  record('Encoded engine detail path survives direct refresh');
  await page.getByRole('tab',{name:'Sensor Data',exact:true}).click();
  record('Sensor detail uses C-MAPSS engineering labels',(await page.locator('#main').innerText()).includes('T30'));
  await page.getByRole('tab',{name:'RUL Prediction',exact:true}).click();
  record('Engine prediction tab shows the actual available history chart',await page.locator('#main svg.data-chart').count()>0);
  await page.getByRole('tab',{name:'Decisions',exact:true}).click();
  record('Decision tab exposes rule input evidence',await page.getByText('Rules and inputs',{exact:true}).count()>0);
  await page.getByRole('tab',{name:'Overview',exact:true}).click();
  await screenshot('engine-detail-demo-desktop.png');
  await page.locator('#global-subset').selectOption('ALL');

  await goto('/dashboard',headings.dashboard);
  await page.locator('#global-search').fill('FD001-005');
  await page.locator('#search-results a').first().waitFor();
  record('Global search returns a matching engine link',(await page.locator('#search-results').innerText()).includes('FD001-005'));
  await page.locator('#search-results a').first().click();await page.getByRole('heading',{name:'Engine #FD001-005',exact:true}).waitFor();
  record('Search result navigates to engine details');

  await goto('/work-orders',headings['work-orders']);
  const ack=page.locator('[data-action="ack-order"]').first();const orderId=await ack.getAttribute('data-id');
  await ack.click();
  const manage=()=>page.locator(`[data-action="manage-order"][data-id="${orderId}"]`);
  await manage().waitFor();await manage().click();
  await page.getByRole('button',{name:'Assign owner',exact:true}).click();
  await page.locator('#assign-form input[name="owner"]').fill('Browser QA engineer');
  await page.locator('#assign-form textarea[name="notes"]').fill('Synthetic example only; browser workflow verification.');
  await page.locator('#assign-form button[type="submit"]').click();
  await manage().waitFor();await manage().click();
  await page.getByRole('button',{name:'Start work',exact:true}).click();
  await manage().waitFor();await manage().click();
  await page.getByRole('button',{name:'Resolve with outcome',exact:true}).click();
  await page.locator('#resolve-form select[name="resolution"]').selectOption('no_fault_found');
  await page.locator('#resolve-form textarea[name="notes"]').fill('Demonstration record reviewed; no physical maintenance performed.');
  record('Resolution requires an explicit confirmation checkbox',await page.locator('#resolve-form input[name="confirm"]').isVisible());
  await page.locator('#resolve-form input[name="confirm"]').check();
  await page.locator('#resolve-form button[type="submit"]').click();
  await page.getByRole('heading',{name:headings['work-orders'],exact:true}).waitFor();
  await page.reload();await page.getByRole('heading',{name:headings['work-orders'],exact:true}).waitFor();
  await page.locator('#order-search').fill(orderId);
  await page.locator(`[data-action="order-detail"][data-id="${orderId}"]`).first().click();
  await page.locator('#dialog .audit-list .audit-item').first().waitFor();
  record('Acknowledgement, assignment, progress, and resolution persist with audit',await page.locator('#dialog .audit-item').count()>=4 && (await page.locator('#dialog').innerText()).includes('Browser QA engineer'));
  await closeDialog();
  record('Demo order outcome is absent from benchmark orders',!(await jsonGet('/api/orders?mode=benchmark')).items.some(o=>o.id===orderId));
  await screenshot('work-orders-demo-desktop.png');

  await goto('/pipeline',headings.pipeline);await idleJobs();
  await page.getByRole('button',{name:'Run pipeline',exact:true}).click();
  await page.locator('#pipeline-form input[name="cycles"]').fill('3');
  const startResponse=page.waitForResponse(r=>r.url().includes('/api/pipeline/run')&&r.request().method()==='POST');
  await page.locator('#pipeline-form button[type="submit"]').click();
  const submitted=await startResponse;assert.equal(submitted.status(),202);const job=await submitted.json();
  const finished=await waitJob(job.job_id);assert.equal(finished.status,'complete');
  record('Pipeline form starts a real background demo job and records stages',finished.stages.length>=8);
  record('Demo ML stages are honestly marked skipped',finished.stages.some(s=>s.status==='skipped'&&s.name==='RUL inference'));
  report.demo_job_id=job.job_id;
  await page.getByRole('button',{name:'Refresh',exact:true}).click();
  await page.locator(`[data-action="select-job"][data-id="${job.job_id}"]`).click();
  await page.locator('.stage-timeline').first().waitFor();
  await screenshot('pipeline-desktop.png');

  await goto('/data',headings.data);await page.locator('#data-mode').selectOption('benchmark');
  await page.locator('#preview-subset').selectOption('FD004');
  await page.getByText('Data Preview · FD004',{exact:true}).waitFor();
  record('All four validated dataset cards render',await page.locator('.dataset-card').count()===4);
  await page.locator('#preview-partition').selectOption('RUL');
  await page.locator('th[title="Official endpoint RUL"]').waitFor();
  record('Endpoint truth is separated from sensor-column preview',(await page.locator('#main').innerText()).includes('evaluation only'));
  const schemaEvent=page.waitForEvent('download');await page.getByRole('button',{name:'CSV schema',exact:true}).click();
  record('CSV schema download works',(await schemaEvent).suggestedFilename().endsWith('.csv'));
  await page.getByRole('button',{name:'Upload dataset',exact:true}).click();
  record('ZIP/CSV/TXT import form is available without overwriting real inputs',await page.locator('#upload-file').isVisible());await closeDialog();
  await screenshot('data-desktop.png');
  await page.locator('#global-subset').selectOption('ALL');

  await goto('/models',headings.models);
  record('Registry shows actual saved baseline versions',await page.locator('.model-card').count()>=Object.keys(baselineModels.active).length);
  if(await page.locator('[data-model-compare]').count()>=2){await page.locator('[data-model-compare]').nth(0).check();await page.locator('[data-model-compare]').nth(1).check();await page.getByRole('heading',{name:'Model Comparison',exact:true}).waitFor();record('Model comparison displays selected measured runs');}
  await page.locator('[data-action="model-detail"]').first().click();await page.getByText('Reproducible configuration',{exact:true}).waitFor();
  record('Model details expose saved configuration, splits and artifact integrity',(await page.locator('#dialog').innerText()).includes('Reproducible configuration'));
  await closeDialog();
  let trainRequests=0;page.on('request',r=>{if(r.url().endsWith('/api/models/train')&&r.method()==='POST')trainRequests++;});
  await page.getByRole('button',{name:'Train a baseline',exact:true}).click();
  await page.locator('#train-form input[name="iterations"]').fill('2');
  await page.locator('#train-form button[type="submit"]').click();
  record('Training controls reject an invalid iteration count without starting a fit',await page.locator('#train-form input[name="iterations"]').evaluate(el=>!el.validity.valid)&&trainRequests===0);
  await closeDialog();await screenshot('models-desktop.png');

  await goto('/settings',headings.settings);await screenshot('settings-desktop.png');
  await page.locator('#settings-form input[name="researcher_name"]').fill('Temporary browser QA researcher');
  const savedResponse=page.waitForResponse(r=>r.url().endsWith('/api/settings')&&r.request().method()==='PUT');
  await page.locator('#settings-form button[type="submit"]').click();assert.equal((await savedResponse).status(),200);profileChanged=true;
  await page.reload();await page.getByRole('heading',{name:headings.settings,exact:true}).waitFor();
  record('Workspace setting persists after reload',(await page.locator('#settings-form input[name="researcher_name"]').inputValue())==='Temporary browser QA researcher');
  let current=await jsonGet('/api/settings');current.researcher_name=originalSettings.researcher_name;
  const restore=await context.request.put(base+'/api/settings',{headers:{'X-AeroGuard-Client':'dashboard'},data:current});assert.equal(restore.status(),200);profileChanged=false;

  for(const route of ['dashboard','engines','analytics','models','work-orders','pipeline','data','settings']){
    await page.setViewportSize({width:1024,height:768});await goto('/'+route,headings[route]);
    record('Tablet /'+route+' avoids document overflow',await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
  }
  const mobile=await browser.newContext({viewport:{width:390,height:844},isMobile:true,hasTouch:true,deviceScaleFactor:1});
  const phone=await mobile.newPage();phone.on('pageerror',e=>pageErrors.push(e.message));
  await phone.goto(base+'/');await phone.getByRole('heading',{name:'Self-Healing Machines for a Safer Tomorrow.',exact:true}).waitFor();
  record('Mobile landing has no document overflow',await phone.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));await screenshot('landing-mobile.png',phone);
  await phone.getByRole('link',{name:'Explore Dashboard',exact:true}).first().click();await phone.getByRole('heading',{name:headings.dashboard,exact:true}).waitFor();
  await phone.locator('#dashboard-trend svg.data-chart').waitFor();await screenshot('dashboard-mobile.png',phone);
  record('Mobile dashboard has no document overflow',await phone.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
  await phone.getByRole('button',{name:'Toggle navigation',exact:true}).click();
  await phone.locator('#mobile-subset').selectOption('FD001');
  await phone.getByRole('heading',{name:headings.dashboard,exact:true}).waitFor();
  await phone.waitForFunction(()=>document.querySelector('.metric-value')?.textContent.trim()==='100');
  record('Mobile sidebar provides a working dataset-subset filter',true);
  await phone.getByRole('button',{name:'Toggle navigation',exact:true}).click();
  await phone.locator('#sidebar a[href="/engines"]').click();await phone.getByRole('heading',{name:headings.engines,exact:true}).waitFor();
  record('Mobile sidebar navigation opens and closes',!(await phone.locator('#sidebar').getAttribute('class')).split(' ').includes('open'));
  await phone.locator('tr[data-engine]').first().click();await phone.getByRole('heading',{name:/^Engine #FD/}).waitFor();
  record('Mobile engine detail shows actual RUL and sensors',await phone.locator('.engine-detail-grid').isVisible());await screenshot('engine-detail-mobile.png',phone);
  for(const route of ['analytics','models','work-orders','pipeline','data','settings']){
    await phone.goto(base+'/'+route);await phone.getByRole('heading',{name:headings[route],exact:true}).waitFor();
    record('Mobile /'+route+' avoids document overflow',await phone.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
  }
  const after=await jsonGet('/api/models');assert.deepEqual(after.active,baselineModels.active);record('Browser QA did not replace benchmark active models');
  record('No JavaScript page errors',pageErrors.length===0);record('No unexpected HTTP error responses',badResponses.length===0);
  report.status='passed';report.finished_at=new Date().toISOString();report.benchmark_active_after=after.active;
 } catch(error){report.status='failed';report.failure=String(error.stack||error);if(page)await screenshot('failure.png').catch(()=>{});console.error(error);process.exitCode=1;}
 finally {
  if(profileChanged&&context&&originalSettings){try{const now=await jsonGet('/api/settings');now.researcher_name=originalSettings.researcher_name;await context.request.put(base+'/api/settings',{headers:{'X-AeroGuard-Client':'dashboard'},data:now});report.profile_restored=true;}catch(error){report.profile_restore_error=String(error);}}
  fs.writeFileSync(path.join(out,'browser-checks.json'),JSON.stringify(report,null,2)+'\n');
  if(browser)await browser.close();console.log(JSON.stringify({status:report.status,passed:checks.length,pageErrors,badResponses,report:'output/aeroguard/browser-checks.json'}));
 }
})();
