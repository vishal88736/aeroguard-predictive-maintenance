/* Real local CPU workflows on an explicitly synthetic, disposable workspace. */
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const {chromium,launchOptions}=require('./browser_runtime.cjs');
const base=process.env.MLOPS_QA_URL;
const fixture=JSON.parse(fs.readFileSync(process.env.MLOPS_QA_FIXTURE,'utf8'));
assert.equal(fixture.mlops_fixture,true);
const output=path.resolve('output/aeroguard/mlops');fs.mkdirSync(output,{recursive:true});
const checks=[],errors=[];
const pass=(label,value=true)=>{assert.ok(value,label);checks.push(label);console.log('PASS '+label);};
let browser;
(async()=>{
 try{
  browser=await chromium.launch(launchOptions);
  const context=await browser.newContext({viewport:{width:1440,height:1000}});
  await context.addInitScript(()=>{localStorage.setItem('aeroguard-subset','FD001');localStorage.setItem('aeroguard-mode','benchmark');});
  const page=await context.newPage();page.on('pageerror',e=>errors.push(e.message));
  const get=async p=>(await context.request.get(base+'/api'+p)).json();
  const waitJob=async(kind='lifecycle',versionId=null)=>{await page.waitForURL('**/pipeline');for(let i=0;i<300;i++){const jobs=(await get('/jobs')).items;const selected=jobs.find(j=>j.kind===kind&&(!versionId||j.configuration.version_id===versionId));if(selected&&!['queued','running'].includes(selected.status)){assert.equal(selected.status,'complete',selected.error);await page.reload();await page.locator('#mlops-lifecycle').waitFor();return (await get('/jobs/'+selected.id));}await page.waitForTimeout(100);}throw new Error('MLOps job timeout');};
  const upload=async(file,label,cohort)=>{await page.locator('[data-mlops-action="upload"]').click();const form=page.locator('#mlops-upload-form');await form.locator('[name="cohort"]').fill(cohort);await form.locator('[name="label_status"]').selectOption(label);await form.locator('[name="file"]').setInputFiles(file);await form.locator('[name="provenance"]').fill('Isolated synthetic UI workflow fixture; not NASA measurements.');await form.getByRole('button',{name:'Save immutable version'}).click();await page.locator('#dialog').waitFor({state:'hidden'});const versions=(await get('/mlops')).versions;return versions.find(v=>v.namespace===cohort);};
  await page.goto(base+'/models');await page.locator('#mlops-lifecycle').waitFor();
  pass('Original AeroGuard sidebar remains',await page.locator('#sidebar').count()===1);
  pass('Same blue/orange/white palette',await page.evaluate(()=>{const s=getComputedStyle(document.documentElement);return s.getPropertyValue('--bg').trim().toUpperCase()==='#F4F8FD'&&s.getPropertyValue('--orange').trim().toUpperCase()==='#FF8A24';}));
  pass('Fresh MLOps registry honestly starts empty',(await get('/mlops')).registry.length===0);
  await page.locator('[data-mlops-action="bootstrap"]').click();await waitJob('mlops_bootstrap');
  let state=await get('/mlops');
  pass('Initialize action creates real MLflow model and champion alias',state.registry.length===1&&state.registry[0].aliases.champion==='1');
  pass('Frozen evaluation protocol visible',state.protocols.length===1);
  const observed=await upload(fixture.observed,'observed_prefix','ui_observed');
  pass('Upload creates an immutable observed-prefix version',Boolean(observed));
  await page.locator(`[data-mlops-action="run-version"][data-id="${observed.id}"]`).click();
  let job=await waitJob('lifecycle',observed.id);
  pass('Observed batch runs actual inference without training',job.result.status==='waiting_labels'&&job.result.inference.processed_cycles>0);
  pass('Observed batch did not create a neural/CPU challenger',(await get('/mlops')).registry.length===1);
  await page.goto(base+'/engines');await page.locator('#engine-filter').fill('ui_observed');await page.getByRole('button',{name:'Apply filters',exact:true}).click();
  await page.locator('[data-engine*="/incoming/ui_observed/"]').first().waitFor();
  pass('Incoming engines appear in existing explorer',(await page.locator('tbody').innerText()).includes('UPLOADED BATCH'));
  await page.locator('[data-engine*="/incoming/ui_observed/"]').first().click();await page.waitForURL('**/engines/**');await page.locator('[role="tab"]').first().waitFor();
  await page.getByRole('heading',{level:1}).waitFor();
  pass('Incoming engine detail uses existing application',!(await page.locator('#main').innerText()).includes('Workspace unavailable'));
  await page.goto(base+'/pipeline');await page.locator('#mlops-lifecycle').waitFor();
  const complete=await upload(fixture.complete,'complete_run_to_failure','ui_training');
  await page.locator(`[data-mlops-action="run-version"][data-id="${complete.id}"]`).click();job=await waitJob('lifecycle',complete.id);
  pass('New labelled dataset trains and registers an actual challenger',Boolean(job.result.run_id)&&job.result.registry_version==='2');
  pass('Worse challenger rejected using measured development results',job.result.status==='rejected');
  state=await get('/mlops');
  pass('Rejected candidate retains original champion',state.registry.every(r=>r.aliases.champion==='1'));
  await page.locator('[data-mlops-action="scheduler"]').click();
  await page.locator('#mlops-settings-form [name="enabled"]').check();
  await page.locator('#mlops-settings-form [name="interval_seconds"]').fill('5');
  await page.getByRole('button',{name:'Save scheduler settings'}).click();await page.locator('#dialog').waitFor({state:'hidden'});
  pass('Scheduler settings persist',(await get('/mlops')).settings.enabled);
  const automatic=await upload(fixture.observed,'observed_prefix','ui_scheduled');
  for(let i=0;i<250;i++){state=await get('/mlops');if(state.versions.find(v=>v.id===automatic.id)?.processing_status==='waiting_labels')break;await page.waitForTimeout(100);}
  pass('Built-in scheduler automatically processes new version',state.versions.find(v=>v.id===automatic.id)?.processing_status==='waiting_labels');
  const bad=await upload(fixture.damaged,'observed_prefix','ui_damaged');
  for(let i=0;i<250;i++){state=await get('/mlops');if(state.versions.find(v=>v.id===bad.id)?.processing_status==='review_required')break;await page.waitForTimeout(100);}
  pass('Invalid quality enters persistent exception review',state.versions.find(v=>v.id===bad.id)?.processing_status==='review_required');
  await page.reload();await page.locator('[data-mlops-action="review"]').first().click();await page.locator('#mlops-review-form [name="notes"]').fill('Reviewed synthetic missing-channel fixture; no training labels or promotion approval inferred.');await page.getByRole('button',{name:'Record resolution'}).click();await page.locator('#dialog').waitFor({state:'hidden'});
  pass('Human exception resolution is audited',(await get('/mlops')).reviews.some(r=>r.status==='resolved'));
  for(const route of ['models','pipeline','data','settings'])for(const width of [1440,1024,390]){await page.setViewportSize({width,height:1000});await page.goto(base+'/'+route);await page.locator('#mlops-lifecycle').waitFor();pass(`${route} preserves responsive layout at ${width}px`,await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));if((route==='pipeline'&&width!==1024)||(route==='models'&&width===1440))await page.screenshot({path:path.join(output,`${route}-${width}.png`),fullPage:true});}
  pass('No browser JavaScript errors',errors.length===0);
 }catch(error){console.error(error);errors.push(error.message);process.exitCode=1;}
 finally{fs.writeFileSync(path.join(output,'report.json'),JSON.stringify({scope:'Isolated synthetic CPU/MLflow software verification; no NASA benchmark or GPU performance claim',checks,errors},null,2));if(browser)await browser.close();}
})();
