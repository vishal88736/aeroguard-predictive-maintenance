/* Software integration tests in fresh headless Chromium. Research imports use ONLY
   research_qa_server.py's temporary, explicitly labelled fixture environment. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {chromium,launchOptions} = require('./browser_runtime.cjs');
const live = process.env.FLEET_QA_URL || 'http://127.0.0.1:8765';
const isolated = process.env.RESEARCH_QA_URL || 'http://127.0.0.1:8766';
const fixture = process.env.RESEARCH_QA_FIXTURE;
assert.ok(fixture && fs.existsSync(fixture),'Provide the isolated fixture ZIP printed by research_qa_server.py');
const out = path.resolve(__dirname,'../output/aeroguard');
const checks=[], errors=[], report={started_at:new Date().toISOString(),checks,errors,
  scope:'Live website theme/pending states, plus isolated fixture import/results UI. No GPU training or actual neural explanations executed.'};
const record = (name,condition=true)=>{assert.ok(condition,name);checks.push(name);console.log('PASS '+name);};
let browser;
(async()=>{
 try{
  browser=await chromium.launch(launchOptions);
  const liveContext=await browser.newContext({viewport:{width:1440,height:1000},acceptDownloads:true});
  const page=await liveContext.newPage();page.on('pageerror',error=>errors.push(error.message));
  await page.goto(live+'/models');await page.getByRole('heading',{name:'Transformer · KAN · BiLSTM',exact:true}).waitFor();
  record('Selected paper architecture is prominent in the real model registry');
  record('Pending research state has no fabricated neural metrics',await page.getByRole('heading',{name:'No completed GPU results yet',exact:true}).count()===1);
  record('Reference/extension validation rule is explained',(await page.locator('.research-intro').innerText()).includes('ties or performs worse on validation'));
  record('SHAP and LIME are the final displayed workflow stage',(await page.locator('.research-flow>div').last().innerText()).includes('SHAP & LIME, last'));
  const palette=await page.evaluate(()=>{const style=getComputedStyle(document.documentElement);return Object.fromEntries(['--bg','--card','--blue','--orange','--text','--border'].map(name=>[name,style.getPropertyValue(name).trim().toUpperCase()]));});
  record('Exact requested palette is present',palette['--bg']==='#F4F8FD'&&palette['--blue']==='#3478F6'&&palette['--orange']==='#FF8A24'&&palette['--text']==='#172A46'&&palette['--border']==='#E1EAF4');
  record('Panel surfaces are white',await page.locator('.research-intro').evaluate(el=>getComputedStyle(el).backgroundColor==='rgb(255, 255, 255)'));
  record('Primary action is orange',await page.locator('.research-intro .button.primary').first().evaluate(el=>getComputedStyle(el).backgroundColor==='rgb(255, 138, 36)'));
  for(const width of [1440,1024,390]){
    await page.setViewportSize({width,height:1000});await page.reload();await page.locator('#research-title').waitFor();
    record('Research registry avoids horizontal overflow at '+width+'px',await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
    await page.screenshot({path:path.join(out,'research-registry-'+width+'.png'),fullPage:true});
  }
  await page.setViewportSize({width:1440,height:1000});
  const notebookResponse=await liveContext.request.get(live+'/downloads/kan-notebook');
  const notebook=JSON.parse(await notebookResponse.text());
  record('GPU notebook download is valid and unexecuted',notebook.nbformat===4&&notebook.cells.filter(c=>c.cell_type==='code').every(c=>c.execution_count===null&&c.outputs.length===0));
  const requirements=await (await liveContext.request.get(live+'/downloads/kan-requirements')).text();
  record('Downloaded environment includes real SHAP/LIME dependencies',requirements.includes('shap==')&&requirements.includes('lime==')&&requirements.includes('torch=='));
  record('Research architecture diagram is served',(await liveContext.request.get(live+'/documents/research.svg')).status()===200);

  const context=await browser.newContext({viewport:{width:1440,height:1000},acceptDownloads:true});
  const test=await context.newPage();test.on('pageerror',error=>errors.push(error.message));
  await test.goto(isolated+'/models');await test.locator('#research-title').waitFor();
  await test.locator('#research-bundle').setInputFiles(fixture);
  await test.getByRole('button',{name:'Import training bundle',exact:true}).click();
  await test.getByRole('heading',{name:'Research Model Results',exact:true}).waitFor();
  record('Actual upload form imports a validated ZIP into the isolated fixture app');
  record('Fixture research source stays visibly labelled',(await test.locator('#main').innerText()).includes('Isolated contract fixture; no benchmark training'));
  record('Research detail identifies its own model source',await test.getByText('GPU research export',{exact:true}).count()>=1&&await test.locator('#data-mode').count()===0);
  record('SHAP contributions and LIME coefficients render as distinct panels',await test.getByRole('heading',{name:'SHAP contributions',exact:true}).count()===1&&await test.getByRole('heading',{name:'LIME coefficients',exact:true}).count()===1);
  record('Explanation bars have per-feature values',await test.locator('.attribution-row').count()===42);
  record('Held-out fidelity and SHAP residual are visible',(await test.locator('#main').innerText()).includes('LIME held-out R²')&&(await test.locator('#main').innerText()).includes('SHAP residual'));
  record('Exported charts use correct engine/epoch axes',(await test.locator('#main').innerText()).includes('Test engine ID')&&(await test.locator('#main').innerText()).includes('Epoch'));
  await test.getByText('Explanation provenance and numerical evidence',{exact:true}).click();
  record('Explanation provenance expands actual method/background evidence',(await test.locator('.research-evidence').innerText()).includes('Fit engine 1, cycle 15'));
  await test.locator('#research-engine-select').selectOption('2');
  await test.getByText('No explanation was exported for this engine and cycle.',{exact:false}).waitFor();
  record('Engine picker updates the route and shows a missing explanation honestly',new URL(test.url()).searchParams.get('engine')==='2');
  await test.getByText('Run this saved model on an observed prefix',{exact:true}).click();
  record('Missing PyTorch disables inference with an explicit setup message',await test.getByRole('button',{name:'Predict from saved weights',exact:true}).isDisabled()&&(await test.locator('.research-inference').innerText()).includes('PyTorch'));
  await test.locator('#research-engine-select').selectOption('1');await test.getByRole('heading',{name:'SHAP contributions',exact:true}).waitFor();
  for(const width of [1440,1024,390]){
    await test.setViewportSize({width,height:1000});await test.reload();await test.getByRole('heading',{name:'Research Model Results',exact:true}).waitFor();
    record('Research result/SHAP/LIME view avoids overflow at '+width+'px',await test.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
    await test.screenshot({path:path.join(out,'research-fixture-results-'+width+'.png'),fullPage:true});
  }
  const status=await (await context.request.get(isolated+'/api/dashboard')).json();
  record('Research import never fabricates CPU fleet or changes active triage models',status.summary.total===0&&Object.keys(status.active).length===0);
  record('No JavaScript errors in research workflows',errors.length===0);
  report.status='passed';report.finished_at=new Date().toISOString();
 }catch(error){report.status='failed';report.failure=String(error.stack||error);console.error(error);process.exitCode=1;}
 finally{fs.writeFileSync(path.join(out,'research-browser-checks.json'),JSON.stringify(report,null,2)+'\n');if(browser)await browser.close();console.log(JSON.stringify({status:report.status,passed:checks.length,errors}));}
})();
