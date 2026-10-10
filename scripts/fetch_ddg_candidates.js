// Fetch up to 50 DuckDuckGo image-search candidates per English word, ONCE
// per word (not per language) - the image is shared across every language's
// question about that word, same as the original pipeline design. Query is
// "<word> <descriptor> Africa", where the descriptor is a short category noun
// (fruit, animal, vehicle, ...) that helps ground ambiguous/brand-colliding
// words ("Apple fruit Africa" finds apples, not Apple Music).
//
// DDG blocks plain HTTP scraping with a bot-detection challenge, so this
// drives a real headless browser instead, which executes their JS like an
// actual visitor would. Running many hundreds of queries from one IP in a
// short window gets that IP rate-limited for hours - run this from a
// different network than one that's already been hammered.
//
// Usage: node fetch_ddg_candidates.js [--limit N] [--concurrency N]
// Reads data/bu200.json, writes data/ddg_candidates_africa.json:
//   { "<english word>": { "query": "...", "category": "...", "images": [...] } }
'use strict';
const fs = require('fs');
const path = require('path');
const puppeteer = require('puppeteer-core');

const ROOT = path.resolve(__dirname, '..');
const EXCLUDED_CATEGORIES = new Set(['days of the week', 'numbers', 'colors', 'body parts']);
const CHROME_PATH = process.env.CHROME_PATH || '/usr/bin/chromium-browser';

// Default descriptor per BU 200 category - a short, natural category noun,
// not the literal category name (which is sometimes too broad to be a search
// term, e.g. "food and drink" covering both fruit and utensil-less dishes).
const CATEGORY_DESCRIPTOR = {
  animals: 'animal', clothing: 'clothing', cooking: 'cooking',
  farm: 'farm', 'food and drink': 'food', house: 'home',
  'natural world': 'nature', professions: 'profession', transportation: 'vehicle',
  // "places" and "people" words (market, doctor's patient, child, ...) are
  // unambiguous enough on their own; no descriptor added for those.
};

// Per-word overrides where the category default is too broad or wrong for
// that specific word - mainly the "food and drink" category, which mixes
// fruit, vegetables, grains and drinks under one label, and a few school
// objects that aren't really "supplies".
const WORD_DESCRIPTOR = {
  Apple: 'fruit', Banana: 'fruit', 'Baobab fruit': 'fruit', Grape: 'fruit',
  Guava: 'fruit', Kolanut: 'fruit', Lemon: 'fruit', Mango: 'fruit',
  Orange: 'fruit', Peach: 'fruit', Pineapple: 'fruit', Plantain: 'fruit',
  Watermelon: 'fruit', 'Cola nut': 'nut',
  Yam: 'vegetable', Cocoyam: 'vegetable', Onion: 'vegetable',
  Beans: 'crop', Corn: 'crop', Millet: 'crop', Rice: 'crop',
  Water: 'drink', Beer: 'drink', Coffee: 'drink',
  Book: 'school supply', Chalk: 'school supply', Eraser: 'school supply',
  Paper: 'school supply', Pen: 'school supply', Pencil: 'school supply',
};

function parseArgs() {
  const limitIdx = process.argv.indexOf('--limit');
  const limit = limitIdx >= 0 ? parseInt(process.argv[limitIdx + 1], 10) : null;
  const concIdx = process.argv.indexOf('--concurrency');
  const concurrency = concIdx >= 0 ? parseInt(process.argv[concIdx + 1], 10) : 4;
  return { limit, concurrency };
}

async function fetchCandidates(page, query, max = 50) {
  const url = `https://duckduckgo.com/?q=${encodeURIComponent(query)}&iax=images&ia=images`;
  await page.goto(url, { waitUntil: 'networkidle2', timeout: 30000 });
  await new Promise((r) => setTimeout(r, 1800));
  const countThumbs = () => document.querySelectorAll('img[src*="/iu/?u="]').length;
  // -1 so a genuine "0 results so far" on the first check doesn't look like
  // "stopped growing" and bail before ever scrolling.
  let lastCount = -1;
  for (let i = 0; i < 12; i++) {
    const n = await page.evaluate(countThumbs);
    if (n >= max || n === lastCount) break;
    lastCount = n;
    await page.evaluate(() => window.scrollBy(0, 2000));
    await new Promise((r) => setTimeout(r, 900));
  }
  const urls = await page.evaluate((max) => {
    // Only real search-result thumbnails ("/iu/?u=<encoded source url>"),
    // never site favicons ("/ip3/<domain>.ico"), which are also <img> tags.
    const imgs = Array.from(document.querySelectorAll('img[src*="/iu/?u="]'));
    const seen = new Set();
    const out = [];
    for (const img of imgs) {
      if ((img.naturalWidth || img.width) < 80) continue;
      if (seen.has(img.src)) continue;
      seen.add(img.src);
      out.push(img.src);
      if (out.length >= max) break;
    }
    return out;
  }, max);
  return urls;
}

async function main() {
  const { limit, concurrency } = parseArgs();
  const bu200 = JSON.parse(fs.readFileSync(path.join(ROOT, 'data/bu200.json'), 'utf8'));
  let words = bu200.words.filter((w) => !EXCLUDED_CATEGORIES.has(w.category));
  // One entry per distinct English word (bu200.json can repeat a word across
  // near-duplicate entries); category/descriptor come from the first hit.
  const seenWords = new Set();
  words = words.filter((w) => (seenWords.has(w.english) ? false : seenWords.add(w.english)));
  if (limit) words = words.slice(0, limit);

  const outPath = path.join(ROOT, 'data/ddg_candidates_africa.json');
  const existing = fs.existsSync(outPath) ? JSON.parse(fs.readFileSync(outPath, 'utf8')) : {};
  const todo = words.filter((w) => !existing[w.english] || existing[w.english].images.length === 0);
  console.log(`${words.length} words total, ${words.length - todo.length} already done, ${todo.length} to fetch`);

  const browser = await puppeteer.launch({ executablePath: CHROME_PATH, args: ['--no-sandbox'] });
  let cursor = 0;
  let done = 0;

  async function worker() {
    const page = await browser.newPage();
    await page.setUserAgent(
      'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36'
    );
    while (cursor < todo.length) {
      const w = todo[cursor++];
      const descriptor = WORD_DESCRIPTOR[w.english] || CATEGORY_DESCRIPTOR[w.category];
      const query = descriptor ? `${w.english} ${descriptor} Africa` : `${w.english} Africa`;
      try {
        const images = await fetchCandidates(page, query, 50);
        existing[w.english] = { query, category: w.category, images };
        done++;
        console.log(`  [${done}/${todo.length}] ${w.english} <- "${query}" : ${images.length} images`);
      } catch (e) {
        console.error(`  ERROR ${w.english} ("${query}"): ${e.message}`);
      }
      fs.writeFileSync(outPath, JSON.stringify(existing, null, 2));
    }
    await page.close();
  }

  await Promise.all(Array.from({ length: concurrency }, worker));
  await browser.close();
  console.log(`wrote ${outPath}`);
}

main();
