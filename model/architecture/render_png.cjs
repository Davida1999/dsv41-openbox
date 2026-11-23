// Use a locally installed @resvg/resvg-js. Does not install packages or fonts.
const fs = require('fs');
const path = require('path');
const modulePath = process.env.DSV41_RESVG_MODULE || '@resvg/resvg-js';
const { Resvg } = require(modulePath);
const dir = path.resolve(process.argv[2] || __dirname);
const options = {font: {fontFiles: ['/Library/Fonts/Arial Unicode.ttf', '/System/Library/Fonts/Hiragino Sans GB.ttc'].filter(p => fs.existsSync(p)), loadSystemFonts: true}};
for (const file of fs.readdirSync(dir).filter(f => f.endsWith('.svg'))) {
  const svg = fs.readFileSync(path.join(dir, file));
  const rendered = new Resvg(svg, options).render();
  fs.writeFileSync(path.join(dir, file.replace(/\.svg$/, '.png')), rendered.asPng());
  if (file === 'model-architecture.svg') {
    const preview = new Resvg(svg, {...options, fitTo: {mode: 'width', value: 1568}}).render();
    fs.writeFileSync(path.join(dir, 'model-architecture-preview.png'), preview.asPng());
  }
  console.log(`${file} -> PNG ${rendered.width}x${rendered.height}`);
}
