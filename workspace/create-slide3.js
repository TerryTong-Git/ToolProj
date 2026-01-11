const pptxgen = require('pptxgenjs');
const path = require('path');

// Import html2pptx from the skills directory
const html2pptx = require(path.join(__dirname, '../.claude/skills/pptx/scripts/html2pptx.js'));

async function createSlide() {
    const pptx = new pptxgen();
    pptx.layout = 'LAYOUT_16x9';
    pptx.title = 'Three Arms Comparison';

    // Convert HTML to slide
    const htmlFile = path.join(__dirname, 'slide3-simplified.html');
    await html2pptx(htmlFile, pptx);

    // Save the presentation
    const outputPath = path.join(__dirname, 'slide3-simplified.pptx');
    await pptx.writeFile({ fileName: outputPath });
    console.log('Created:', outputPath);
}

createSlide().catch(console.error);
