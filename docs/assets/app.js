(() => {
  'use strict';

  const languages = {
    en: { name: 'English', native: 'English' },
    de: { name: 'German', native: 'Deutsch' },
    fr: { name: 'French', native: 'Français' },
    ja: { name: 'Japanese', native: '日本語' },
    ko: { name: 'Korean', native: '한국어' },
    zh_sichuan: { name: 'Sichuanese', native: '四川话', textLang: 'zh-CN', isDialect: true }
  };
  const modelNames = { reference: 'Reference prompt', m5: 'M5-TTS', moss: 'MOSS-TTS-Nano' };
  const samples = Array.isArray(window.M5_SAMPLES) ? window.M5_SAMPLES : [];
  const multilingualSamples = samples.filter(sample => languages[sample.language] && !languages[sample.language].isDialect);
  const dialectSamples = samples.filter(sample => languages[sample.language]?.isDialect);
  const sampleContainer = document.getElementById('audio-samples');
  const dialectContainer = document.getElementById('dialect-samples');
  const tabs = [...document.querySelectorAll('[data-language]')];
  const playbackGroups = [
    { container: sampleContainer, status: document.getElementById('playback-status') },
    { container: dialectContainer, status: document.getElementById('dialect-playback-status') }
  ];
  const toast = document.getElementById('toast');
  let currentLanguage = 'en';
  let toastTimeout;

  function element(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function icon(name) {
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.classList.add('icon');
    svg.setAttribute('aria-hidden', 'true');
    const use = document.createElementNS('http://www.w3.org/2000/svg', 'use');
    use.setAttribute('href', `#icon-${name}`);
    svg.append(use);
    return svg;
  }

  function sampleId(sample) {
    return `sample-${sample.language}-${String(sample.selection_rank).padStart(2, '0')}`;
  }

  function announce(message) {
    window.clearTimeout(toastTimeout);
    toast.textContent = message;
    toast.classList.add('visible');
    toastTimeout = window.setTimeout(() => toast.classList.remove('visible'), 3000);
  }

  function updatePlaybackStatus() {
    playbackGroups.forEach(({ container, status }) => {
      const playing = [...container.querySelectorAll('audio')].find(audio => !audio.paused && !audio.ended);
      status.classList.toggle('is-playing', Boolean(playing));
      status.replaceChildren();
      if (playing) {
        status.textContent = `Playing ${playing.dataset.modelName} · ${playing.dataset.sampleLabel}`;
      } else {
        status.append(element('span', 'idle-dot'), document.createTextNode('Ready to listen'));
      }
    });
  }

  function audioCell(sample, model) {
    const cell = element('div', `audio-cell${model === 'm5' ? ' ours' : ''}`);
    const header = element('div', 'audio-cell-heading');
    const name = element('h4', '', modelNames[model]);
    name.id = `${sampleId(sample)}-${model}-label`;
    header.append(name);
    const badge = model === 'reference' ? 'Voice prompt' : model === 'm5' ? 'Ours' : 'Baseline';
    header.append(element('span', 'model-badge', badge));
    const duration = model === 'reference' ? sample.reference_duration : sample[model].duration;
    if (Number.isFinite(duration)) header.append(element('span', 'duration', `${duration.toFixed(1)} s`));
    const audio = element('audio');
    audio.controls = true;
    audio.preload = 'none';
    audio.src = sample.local_audio[model];
    audio.dataset.modelName = modelNames[model];
    audio.dataset.sampleLabel = `${languages[sample.language].name} / ${String(sample.selection_rank).padStart(2, '0')}`;
    audio.setAttribute('aria-label', `${modelNames[model]}, ${languages[sample.language].name}, sample ${sample.selection_rank}`);
    audio.textContent = 'Your browser does not support audio playback.';
    audio.addEventListener('play', () => {
      document.querySelectorAll('.sample-card audio').forEach(other => {
        if (other !== audio && !other.paused) other.pause();
      });
      cell.classList.add('playing');
      updatePlaybackStatus();
    });
    for (const event of ['pause', 'ended']) {
      audio.addEventListener(event, () => {
        cell.classList.remove('playing');
        updatePlaybackStatus();
      });
    }
    audio.addEventListener('error', () => {
      if (cell.querySelector('.audio-error')) return;
      const error = element('p', 'audio-error', 'Audio could not be loaded. ');
      const link = element('a', '', 'Open WAV file');
      link.href = sample.local_audio[model];
      link.target = '_blank';
      link.rel = 'noopener';
      error.append(link);
      cell.append(error);
      updatePlaybackStatus();
    });
    cell.append(header, audio);
    return cell;
  }

  function sampleCard(sample) {
    const language = languages[sample.language];
    const rank = String(sample.selection_rank).padStart(2, '0');
    const card = element('article', 'sample-card');
    card.id = sampleId(sample);
    card.setAttribute('aria-labelledby', `${card.id}-title`);
    const header = element('div', 'sample-card-header');
    const heading = element('div', 'sample-heading');
    const title = element('h3', '', `${language.name} · Sample ${rank}`);
    title.id = `${card.id}-title`;
    heading.append(element('span', 'sample-number', rank), title);
    if (sample.language !== 'en') heading.append(element('span', 'native-label', language.native));
    const permalink = element('button', 'icon-button');
    permalink.type = 'button';
    permalink.title = 'Copy sample link';
    permalink.setAttribute('aria-label', `Copy link to ${language.name} sample ${rank}`);
    permalink.append(icon('link'));
    permalink.addEventListener('click', async () => {
      const url = new URL(window.location.href);
      url.hash = card.id;
      try {
        if (!navigator.clipboard) throw new Error('Clipboard unavailable');
        await navigator.clipboard.writeText(url.href);
        announce('Sample link copied');
      } catch {
        window.location.hash = card.id;
        announce('Sample opened. Copy the link from your address bar.');
      }
    });
    header.append(heading, permalink);
    const target = element('div', 'target-text');
    const text = element('p', '', sample.target);
    text.lang = language.textLang || sample.language;
    target.append(element('span', 'target-label', 'TARGET TEXT'), text);
    const audioColumns = element('div', 'audio-columns');
    const availableModels = ['reference', 'm5', 'moss'].filter(model => sample.local_audio[model]);
    audioColumns.style.setProperty('--audio-column-count', availableModels.length);
    for (const model of availableModels) audioColumns.append(audioCell(sample, model));
    const prompt = element('div', 'target-text prompt-text');
    const promptText = typeof sample.prompt_text === 'string' ? sample.prompt_text.trim() : '';
    const missingTranscript = sample.prompt_text_status === 'not_displayed' ? 'Reference transcript not displayed.' : 'Reference transcript not provided.';
    const transcript = element('p', promptText ? '' : 'transcript-missing', promptText || missingTranscript);
    transcript.lang = promptText ? language.textLang || sample.language : 'en';
    prompt.append(element('span', 'target-label', 'PROMPT TEXT · REFERENCE AUDIO'), transcript);
    card.append(header, prompt, target, audioColumns);
    if (sample.flags.includes('时长相差超过2倍')) {
      card.append(element('p', 'sample-warning', 'Listening note: the two generated clips differ in duration by more than 2×.'));
    }
    return card;
  }

  function showLanguage(language) {
    if (!tabs.some(tab => tab.dataset.language === language)) return;
    sampleContainer.querySelectorAll('audio').forEach(audio => audio.pause());
    currentLanguage = language;
    tabs.forEach(tab => {
      const selected = tab.dataset.language === language;
      tab.setAttribute('aria-selected', String(selected));
      tab.tabIndex = selected ? 0 : -1;
    });
    const visibleSamples = multilingualSamples.filter(sample => language === 'all' || sample.language === language);
    sampleContainer.replaceChildren(...visibleSamples.map(sampleCard));
    if (!visibleSamples.length) {
      sampleContainer.append(element('p', 'notice', 'Audio examples are unavailable. Please check that assets/samples.js is included with this page.'));
    }
    document.getElementById('audio-panel').setAttribute('aria-labelledby', `tab-${language}`);
    document.getElementById('sample-count').textContent = `${language === 'all' ? 'All languages' : languages[language].name} · ${visibleSamples.length} listening examples`;
    updatePlaybackStatus();
  }

  tabs.forEach((tab, index) => {
    tab.addEventListener('click', () => {
      if (currentLanguage === tab.dataset.language) return;
      showLanguage(tab.dataset.language);
      try {
        history.pushState(null, '', `#examples-${currentLanguage}`);
      } catch {
        window.location.hash = `examples-${currentLanguage}`;
      }
    });
    tab.addEventListener('keydown', event => {
      let target;
      if (event.key === 'ArrowRight') target = (index + 1) % tabs.length;
      if (event.key === 'ArrowLeft') target = (index - 1 + tabs.length) % tabs.length;
      if (event.key === 'Home') target = 0;
      if (event.key === 'End') target = tabs.length - 1;
      if (target === undefined) return;
      event.preventDefault();
      tabs[target].focus();
      tabs[target].click();
    });
  });

  function routeHash(shouldScroll) {
    const hash = window.location.hash.slice(1);
    const languageMatch = /^examples-(en|de|fr|ja|ko|all)$/.exec(hash);
    const sample = samples.find(item => sampleId(item) === hash);
    if (hash === 'examples-zh_sichuan') {
      if (shouldScroll) requestAnimationFrame(() => document.getElementById('dialect-examples').scrollIntoView());
    } else if (languageMatch) {
      if (currentLanguage !== languageMatch[1]) showLanguage(languageMatch[1]);
      if (shouldScroll) requestAnimationFrame(() => document.getElementById('examples').scrollIntoView());
    } else if (sample) {
      if (!languages[sample.language].isDialect && currentLanguage !== sample.language && currentLanguage !== 'all') showLanguage(sample.language);
      if (shouldScroll) requestAnimationFrame(() => document.getElementById(hash)?.scrollIntoView());
    }
  }

  dialectContainer.replaceChildren(...dialectSamples.map(sampleCard));
  document.getElementById('dialect-count').textContent = `Sichuanese · ${dialectSamples.length} listening examples`;
  if (!dialectSamples.length) {
    dialectContainer.append(element('p', 'notice', 'Dialect examples are unavailable. Please check that assets/samples.js is included with this page.'));
  }
  showLanguage('en');
  routeHash(true);
  window.addEventListener('hashchange', () => routeHash(true));

  const dialog = document.getElementById('figure-dialog');
  const expandedFigure = document.getElementById('expanded-figure');
  document.querySelectorAll('[data-zoom]').forEach(button => {
    button.addEventListener('click', () => {
      expandedFigure.src = button.dataset.zoom;
      expandedFigure.alt = button.closest('figure').querySelector('img').alt;
      document.getElementById('dialog-caption').textContent = button.dataset.caption;
      document.getElementById('original-figure').href = button.dataset.zoom;
      dialog.showModal();
      document.body.classList.add('modal-open');
    });
  });
  document.getElementById('close-figure').addEventListener('click', () => dialog.close());
  dialog.addEventListener('click', event => {
    const bounds = dialog.getBoundingClientRect();
    if (event.target === dialog && (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom)) dialog.close();
  });
  dialog.addEventListener('close', () => document.body.classList.remove('modal-open'));

  const backToTop = document.querySelector('.back-to-top');
  const navLinks = [...document.querySelectorAll('.top-nav a')];
  const anchors = ['abstract', 'tokenizer', 'architecture', 'examples', 'dialect-examples']
    .map(id => document.getElementById(id))
    .filter(Boolean);
  let scrollPending = false;
  function updateNavigation() {
    backToTop.classList.toggle('visible', window.scrollY > 650);
    let activeId = '';
    anchors.forEach(section => {
      if (section.getBoundingClientRect().top <= 170) activeId = section.id;
    });
    if (activeId === 'architecture') activeId = 'tokenizer';
    navLinks.forEach(link => {
      const active = link.getAttribute('href') === `#${activeId}`;
      link.classList.toggle('active', active);
      if (active) link.setAttribute('aria-current', 'location');
      else link.removeAttribute('aria-current');
    });
    scrollPending = false;
  }
  window.addEventListener('scroll', () => {
    if (scrollPending) return;
    scrollPending = true;
    requestAnimationFrame(updateNavigation);
  }, { passive: true });
  updateNavigation();
})();
