/* Check buildless browser modules and development scripts without installing tools. */
const fs = require('node:fs');
const path = require('node:path');
const {spawnSync} = require('node:child_process');
const root = path.resolve(__dirname,'..');
const modules = ['frontend/app.js','frontend/research.js','frontend/charts.js','frontend/vendor/icons.js','web/app.js'];
const scripts = ['scripts/browser_runtime.cjs','scripts/browser_aeroguard.cjs','scripts/browser_research.cjs','scripts/verify_website.cjs'];
for (const file of modules) {
  const result = spawnSync(process.execPath,['--input-type=module','--check'],{input:fs.readFileSync(path.join(root,file)),encoding:'utf8'});
  if(result.status!==0){process.stderr.write(result.stderr);process.exit(result.status||1);}
}
for (const file of scripts) {
  const result = spawnSync(process.execPath,['--check',path.join(root,file)],{encoding:'utf8'});
  if(result.status!==0){process.stderr.write(result.stderr);process.exit(result.status||1);}
}
console.log(`JavaScript syntax passed for ${modules.length+scripts.length} files.`);
