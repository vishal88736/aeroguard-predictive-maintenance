/* Standard local Playwright install, or an explicitly selected existing runtime. */
const fs = require('node:fs');
let browserModule;
try {
  browserModule = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
} catch (error) {
  throw new Error('Browser checks need npm ci and a Playwright Chromium install, or PLAYWRIGHT_MODULE pointing to an existing installation.', {cause:error});
}
const executablePath = process.env.CHROMIUM_EXECUTABLE || (fs.existsSync('/usr/bin/chromium') ? '/usr/bin/chromium' : undefined);
module.exports = {
  chromium: browserModule.chromium,
  launchOptions: {headless:true, ...(executablePath ? {executablePath} : {}), args:['--no-sandbox']}
};
