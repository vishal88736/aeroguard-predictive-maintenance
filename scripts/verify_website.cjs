/* Local browser QA. Use an isolated temporary Application for the optional ZIP fixture.
   No packages are installed. Override PLAYWRIGHT_MODULE / CHROMIUM_EXECUTABLE if needed. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {chromium,launchOptions} = require('./browser_runtime.cjs');
const base = process.env.FLEET_QA_URL || 'http://127.0.0.1:8765';
const output = path.resolve(__dirname, '../output/website');
const checks = [], errors = [];
const check = (name, condition) => {assert.ok(condition, name);checks.push(name);};
fs.mkdirSync(output, {recursive:true});

(async()=>{
  const browser = await chromium.launch(launchOptions);
  try {
    const context = await browser.newContext({viewport:{width:1440,height:1000},acceptDownloads:true});
    const page = await context.newPage();
    page.on('pageerror', error=>errors.push(error.message));
    const navigate = async(route, heading) => {
      await page.locator(`#nav a[href="#/${route}"]`).click();
      await page.getByRole('heading', {name:heading,exact:true}).waitFor();
    };
    const response = await page.goto(base);
    await page.getByRole('heading',{name:'A clearer view of engine health.'}).waitFor();
    check('Website serves successfully', response.status()===200);
    check('Dataset mode starts empty and demo is opt-in',await page.locator('#mode-banner').isHidden() && await page.getByText('No evaluated engines yet',{exact:true}).isVisible());
    await page.screenshot({path:path.join(output,'dataset-overview-desktop.png'),fullPage:true});
    await page.locator('#mode-control').click();
    await page.getByText('Showing 8 of 32 engines',{exact:true}).waitFor();
    check('Demo is explicitly labelled synthetic', (await page.locator('#mode-banner').innerText()).includes('synthetic'));
    await page.screenshot({path:path.join(output,'demo-overview-desktop.png'),fullPage:true});
    check('Desktop has no page overflow',await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));

    await navigate('engines','Every engine, in context.');
    await page.getByText('Showing 32 of 32 engines',{exact:true}).waitFor();
    await page.locator('#subset-select').selectOption('FD004');
    await page.getByText('Showing 8 of 8 engines',{exact:true}).waitFor();
    await page.locator('#engine-search').fill('005');
    check('Dataset and engine search filters work',await page.locator('#engine-rows tr[data-asset]').count()===1);
    await page.locator('#status-filter').selectOption('HIGH');
    check('Status filter excludes nonmatching engines',await page.locator('#engine-rows tr[data-asset]').count()===0);
    await page.locator('#status-filter').selectOption('CRITICAL');
    const row=page.locator('#engine-rows tr[data-asset]');await row.focus();await row.press('Enter');
    await page.locator('#asset-content .dialog-panel').first().waitFor();
    check('Keyboard opens an engine and two history plots',await page.locator('#asset-dialog').isVisible() && await page.locator('#asset-dialog svg[role=img]').count()===2);
    await page.screenshot({path:path.join(output,'engine-detail-desktop.png'),fullPage:true});
    await page.keyboard.press('Escape');
    await page.locator('#subset-select').selectOption('ALL');
    await page.getByText('Showing 4 of 32 engines',{exact:true}).waitFor();

    await navigate('orders','Give every alert a next step.');
    const orderRow=page.locator('tbody tr').filter({has:page.getByRole('button',{name:'Acknowledge',exact:true})}).first();
    const orderId=(await orderRow.locator('td strong').first().textContent()).trim();
    await orderRow.getByRole('button',{name:'Acknowledge',exact:true}).click();
    let savedRow=page.locator('tbody tr').filter({hasText:orderId});await savedRow.getByText('Acknowledged',{exact:true}).waitFor();
    await savedRow.getByRole('button',{name:'Resolve',exact:true}).click();
    await page.locator('#resolution').selectOption('no_fault_found');
    await page.getByRole('button',{name:'Resolve order',exact:true}).click();
    await page.getByText('Maintenance outcome recorded.',{exact:true}).waitFor();
    await page.reload();await page.getByRole('heading',{name:'Give every alert a next step.'}).waitFor();
    await page.getByRole('button',{name:'All orders',exact:true}).click();
    savedRow=page.locator('tbody tr').filter({hasText:orderId});await savedRow.getByText('Resolved',{exact:true}).waitFor();
    check('Work-order acknowledgement and explicit outcome persist after reload', (await savedRow.innerText()).includes('no fault found'));
    await page.screenshot({path:path.join(output,'work-orders-desktop.png'),fullPage:true});
    await page.locator('#mode-control').click();await page.getByText('Nothing in this queue',{exact:true}).waitFor();
    check('Demo outcomes do not create dataset-mode orders',await page.locator('tbody tr').count()===0);

    await navigate('datasets','Four subsets. One training pipeline.');
    check('All four dataset cards render',await page.locator('.dataset-card').count()===4);
    if(process.env.FLEET_BROWSER_FIXTURE){
      const imported=page.waitForResponse(r=>r.url().includes('/api/import')&&r.request().method()==='POST');
      await page.locator('#dataset-upload').setInputFiles(process.env.FLEET_BROWSER_FIXTURE);
      check('Isolated synthetic ZIP imports successfully',(await imported).status()===200);
      await page.getByText('Dataset imported and validated. No training was started.',{exact:true}).waitFor();
      await page.locator('.dataset-card .inline-error').first().waitFor();
      check('Non-NASA fixture counts are disclosed',await page.locator('.dataset-card .inline-error').count()===4);
    }
    await page.screenshot({path:path.join(output,'datasets-desktop.png'),fullPage:true});
    const downloaded=page.waitForEvent('download');await page.locator('a[href="/downloads/notebook"]').last().click();
    const notebook=await downloaded;const downloadedPath=await notebook.path();
    check('Notebook download is an actual unexecuted notebook',JSON.parse(fs.readFileSync(downloadedPath,'utf8')).cells.filter(c=>c.cell_type==='code').every(c=>c.execution_count===null&&c.outputs.length===0));

    await navigate('training','From sensor data to trained models.');
    await page.getByRole('button',{name:'Configure training',exact:true}).click();
    check('Missing training dependencies block a start with clear guidance',await page.getByRole('button',{name:'Start training',exact:true}).isDisabled() && (await page.locator('#training-content').innerText()).includes('requirements.txt'));
    await page.keyboard.press('Escape');
    await page.screenshot({path:path.join(output,'training-desktop.png'),fullPage:true});
    await navigate('models','A home for every experiment.');
    check('Registry contains no fabricated trained scores',await page.getByText('Your model registry is ready',{exact:true}).isVisible());
    await navigate('architecture','Research, connected to a working system.');
    await page.waitForFunction(()=>[...document.querySelectorAll('.architecture-grid img')].every(img=>img.complete&&img.naturalWidth>0));
    check('Both research architecture diagrams load',await page.locator('.architecture-grid img').count()===2);
    await page.screenshot({path:path.join(output,'architecture-desktop.png'),fullPage:true});

    const mobile=await browser.newContext({viewport:{width:390,height:844},isMobile:true,hasTouch:true,deviceScaleFactor:1});
    const phone=await mobile.newPage();phone.on('pageerror',error=>errors.push(error.message));
    await phone.goto(base);await phone.getByRole('heading',{name:'A clearer view of engine health.'}).waitFor();
    await phone.locator('#mode-control').click();await phone.getByText('Showing 8 of 32 engines',{exact:true}).waitFor();
    check('Mobile overview does not overflow the viewport',await phone.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
    await phone.screenshot({path:path.join(output,'demo-overview-mobile.png'),fullPage:true});
    await phone.locator('#mobile-menu').click();await phone.locator('#nav a[href="#/engines"]').click();
    await phone.getByRole('heading',{name:'Every engine, in context.'}).waitFor();
    check('Mobile navigation opens and closes',await phone.locator('#mobile-menu').getAttribute('aria-expanded')==='false');
    await phone.locator('#engine-search').fill('FD001/demo/004');
    await phone.locator('#engine-rows tr[data-asset]').click();
    await phone.locator('#asset-content .dialog-panel').first().waitFor();
    check('Mobile engine dialog fits the screen',(await phone.locator('#asset-dialog').boundingBox()).width<=390);
    await phone.screenshot({path:path.join(output,'engine-detail-mobile.png'),fullPage:true});
    await phone.getByRole('button',{name:'Close engine details',exact:true}).click();
    for(const route of ['datasets','training','models','orders','architecture']){
      await phone.locator('#mobile-menu').click();await phone.locator(`#nav a[href="#/${route}"]`).click();
      await phone.locator('h1').waitFor();
      await phone.waitForFunction(()=>!document.querySelector('#main .loading-state'));
      check(`Mobile ${route} has no page overflow`,await phone.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
    }
    check('No JavaScript page errors',errors.length===0);
    const report={timestamp:new Date().toISOString(),browser:'Installed Chromium, headless Playwright',viewports:['1440x1000','390x844'],checks,errors,synthetic_import_fixture:Boolean(process.env.FLEET_BROWSER_FIXTURE),limitations:['No trained neural bundles or CUDA runtime executed','ZIP fixture is synthetic and confined to the isolated QA Application','No deployment or external CMMS delivery tested']};
    fs.writeFileSync(path.join(output,'browser-checks.json'),JSON.stringify(report,null,2)+'\n');
    console.log(JSON.stringify({passed:checks.length,errors,report:'output/website/browser-checks.json'},null,2));
  } finally {await browser.close();}
})().catch(error=>{console.error(error);process.exit(1);});
