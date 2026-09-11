import './Dossier.css';
import { ROSTER, otherCharacter, type CharacterId } from '../entities/Character';
import type { MatchMode } from '../game/GameState';
import { CONSEQUENCE_TIERS, FIGHTERS, COMPARE_ROWS, type FighterContent } from './dossierContent';

type LockInHandler = (mode: MatchMode, p1: CharacterId, p2: CharacterId) => void;

export type DossierHandle = {
  showLoading: (progress: number) => void;
  showSelect: () => void;
  destroy: () => void;
};

const portraitBackground = (id: CharacterId): string => {
  const p = ROSTER[id].portrait;
  return `background-image: url('${p.src}'); background-position: -${p.sx}px -${p.sy}px; background-size: auto;`;
};

const el = <K extends keyof HTMLElementTagNameMap>(
  tag: K,
  className?: string,
  html?: string
): HTMLElementTagNameMap[K] => {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (html !== undefined) node.innerHTML = html;
  return node;
};

const buildFighterSection = (fighter: FighterContent, side: 'left' | 'right'): HTMLElement => {
  const section = el(
    'section',
    `dossier-fighter${fighter.id === 'heinrich' ? ' heinrich' : ''}`
  );
  section.dataset.side = side;
  section.dataset.in = 'false';

  const portrait = el('div', 'dossier-fighter__portrait');
  portrait.dataset.tag = fighter.tag;
  portrait.setAttribute('style', portraitBackground(fighter.id));

  const text = el('div', 'dossier-fighter__text');
  text.append(
    el('h2', 'dossier-fighter__name', fighter.name),
    el('p', 'dossier-fighter__archetype', fighter.archetype),
    el('p', 'dossier-fighter__bio', fighter.bio),
    el('p', 'dossier-fighter__quote', fighter.quote)
  );

  const readout = el('div', 'dossier-readout');
  for (const stat of fighter.stats) {
    const cell = el('div', 'dossier-readout__cell');
    cell.append(el('div', 'dossier-readout__k', stat.k), el('div', 'dossier-readout__v', stat.v));
    readout.append(cell);
  }
  text.append(readout);

  const system = el('div', 'dossier-fighter__system');
  system.innerHTML = `<strong>No preloaded moveset.</strong> ${fighter.system}`;
  text.append(system);

  section.append(portrait, text);
  return section;
};

const buildCompareTable = (): HTMLTableElement => {
  const table = el('table', 'dossier-table');
  const thead = el('thead');
  const headRow = el('tr');
  ['Property', FIGHTERS.bhuru.name, FIGHTERS.heinrich.name].forEach((label, i) => {
    const th = el('th', undefined, label);
    th.scope = 'col';
    headRow.append(th);
    if (i > 0) return;
  });
  thead.append(headRow);

  const tbody = el('tbody');
  for (const [label, a, b] of COMPARE_ROWS) {
    const row = el('tr');
    const th = el('th', undefined, label);
    th.scope = 'row';
    row.append(th, el('td', 'a', a), el('td', 'b', b));
    tbody.append(row);
  }

  table.append(thead, tbody);
  return table;
};

export const mountDossier = (onLockIn: LockInHandler): DossierHandle => {
  const app = document.getElementById('app');
  if (!app) throw new Error('#app not found');

  const root = el('div', undefined);
  root.id = 'dossier-root';
  root.style.setProperty('--bhuru', ROSTER.bhuru.color);
  root.style.setProperty('--heinrich', ROSTER.heinrich.color);

  // Boot strip / loading readout -------------------------------------------
  const boot = el('div', 'dossier-boot');
  const bootDot = el('span', 'dossier-boot__dot');
  const bootLabel = el('span', undefined, 'INITIALIZING ROSTER…');
  const bootBar = el('div', 'dossier-boot__bar');
  const bootBarFill = el('div', 'dossier-boot__bar-fill');
  bootBar.append(bootBarFill);
  const bootPct = el('span', 'dossier-boot__pct', '0%');
  boot.append(bootDot, bootLabel, bootBar, bootPct);

  // Hero ---------------------------------------------------------------------
  const hero = el('header', 'dossier-hero');
  hero.append(
    el('div', 'dossier-hero__case', 'CASE FILE #47,301 — ROSTER DOSSIER, VOL. 1'),
    (() => {
      const h1 = el('h1', 'dossier-hero__title');
      h1.append(
        el('span', 'a', 'BHURU'),
        el('span', 'a', 'SUKURIN'),
        el('span', 'versus', 'v.'),
        el('span', 'b', 'HEINRICH')
      );
      return h1;
    })(),
    (() => {
      const p = el('p', 'dossier-hero__sub');
      p.innerHTML =
        'One law governs every match: energy cannot be created, only paid for. ' +
        "Pick the fighter who pays it your way — <em>you cannot pick again</em>.";
      return p;
    })()
  );

  // The Law --------------------------------------------------------------------
  const law = el('section', 'dossier-section dossier-law');
  const lawBody = el('div', 'dossier-law__body');
  lawBody.append(
    el('div', 'dossier-label', 'GOVERNING PHYSICS'),
    el('h2', 'dossier-heading', 'The thermodynamic punishment system'),
    (() => {
      const p = el('p');
      p.innerHTML =
        'Reality here runs on strict conservation. Break entropy — dodge a hit ' +
        "you shouldn't, land a throw you haven't earned — and the universe " +
        'collects. <strong>Both fighters violate the law differently</strong>, ' +
        'which is the entire reason to read past this page before you pick one.';
      return p;
    })(),
    el(
      'p',
      undefined,
      'Bhuru-sukurin borrows from information: he sees outcomes before they ' +
        'happen and pays for it in fractured attention. Heinrich borrows from ' +
        'momentum: he commits fully to force and pays for it when that force ' +
        'is denied a target.'
    )
  );
  const tiers = el('div', 'dossier-tiers');
  for (const tier of CONSEQUENCE_TIERS) {
    const row = el('div', 'dossier-tier');
    row.append(el('span', 'dossier-tier__n', tier.n), el('span', 'dossier-tier__t', tier.t));
    tiers.append(row);
  }
  law.append(lawBody, tiers);

  // Fighter sections + VS rule ---------------------------------------------------
  const bhuruSection = buildFighterSection(FIGHTERS.bhuru, 'left');
  const vsRule = el('div', 'dossier-vs-rule');
  vsRule.append(
    el('span', undefined, 'NEITHER FIGHTER SHIPS WITH COMBOS'),
    el('span', 'dossier-vs-rule__line'),
    el('span', undefined, 'YOUR MATCHES WRITE THEIR MOVESET')
  );
  const heinrichSection = buildFighterSection(FIGHTERS.heinrich, 'right');

  // Comparison --------------------------------------------------------------------
  const compare = el('section', 'dossier-section');
  compare.append(
    el('div', 'dossier-label', 'SIDE BY SIDE'),
    el('h2', 'dossier-heading', 'Same law, opposite debt'),
    buildCompareTable()
  );

  // Permanence callout ---------------------------------------------------------------
  const permanence = el('section', 'dossier-section dossier-permanence');
  const permBody = el('div', 'dossier-permanence__body');
  permBody.append(
    el('div', 'dossier-label', 'BEFORE YOU CHOOSE'),
    el('h2', 'dossier-heading', 'Frame selection is not a choice you revisit'),
    el(
      'p',
      undefined,
      'In this universe, Bhuru-sukurin discovers that every "choice" is really ' +
        'a predetermined frame pulled from a fixed set — the feeling of ' +
        'deciding arrives after the outcome is already fixed.'
    ),
    el(
      'p',
      undefined,
      'Your fighter select works the same way. Everything below is reversible ' +
        'until the moment you lock it in. After that, the roster behind you ' +
        'closes — no swapping, no second main, no way back.'
    )
  );
  const permFlag = el('div', 'dossier-permanence__flag');
  permFlag.append(
    el('h3', undefined, 'Irreversible'),
    el(
      'p',
      undefined,
      'One fighter, permanently. Your matches, movesets, and progress are tied ' +
        'to that choice from here on.'
    )
  );
  permanence.append(permBody, permFlag);

  // Select -----------------------------------------------------------------------------
  const select = el('section', 'dossier-select');
  select.append(el('div', 'dossier-label', 'FINAL STEP'), el('h2', 'dossier-heading', 'Choose your fighter'));

  let mode: MatchMode | null = null;
  let picked: CharacterId | null = null;
  let confirming = false;
  let locked = false;

  const modeGrid = el('div', 'dossier-mode-grid');
  modeGrid.setAttribute('role', 'radiogroup');
  modeGrid.setAttribute('aria-label', 'Choose match mode');
  const modeButtons: Record<MatchMode, HTMLButtonElement> = {} as Record<MatchMode, HTMLButtonElement>;
  const modeDefs: Array<{ id: MatchMode; title: string; desc: string }> = [
    { id: '1P_VS_AI', title: '1 Player', desc: 'vs. AI opponent (takes the other fighter automatically)' },
    { id: '2P', title: '2 Players', desc: 'Same keyboard — P1 picks first, P2 picks the remaining fighter' },
  ];
  for (const def of modeDefs) {
    const btn = el('button', 'dossier-mode');
    btn.type = 'button';
    btn.setAttribute('role', 'radio');
    btn.append(el('div', 'dossier-mode__title', def.title), el('div', 'dossier-mode__desc', def.desc));
    btn.addEventListener('click', () => {
      mode = def.id;
      picked = null;
      confirming = false;
      render();
    });
    modeButtons[def.id] = btn;
    modeGrid.append(btn);
  }

  const pickGrid = el('div', 'dossier-select-grid');
  pickGrid.setAttribute('role', 'radiogroup');
  pickGrid.setAttribute('aria-label', 'Choose your fighter');
  const pickButtons: Record<CharacterId, HTMLButtonElement> = {} as Record<CharacterId, HTMLButtonElement>;
  (['bhuru', 'heinrich'] as CharacterId[]).forEach((id) => {
    const f = FIGHTERS[id];
    const btn = el('button', 'dossier-pick');
    btn.type = 'button';
    btn.setAttribute('role', 'radio');
    btn.style.setProperty('--pick-accent', f.color);
    const radioLabel = el('div', 'dossier-pick__radio', '[ SELECT ]');
    btn.append(
      el('div', 'dossier-pick__name', f.name),
      el('div', 'dossier-pick__tag', `${f.archetype}`),
      radioLabel
    );
    btn.addEventListener('click', () => {
      picked = id;
      confirming = false;
      render();
    });
    pickButtons[id] = btn;
    pickGrid.append(btn);
  });

  const cta = el('div', 'dossier-cta');
  const ctaText = el('div', 'dossier-cta__text');
  const lockBtn = el('button', 'dossier-lock-btn', 'LOCK IN');
  lockBtn.type = 'button';
  lockBtn.addEventListener('click', () => {
    if (locked || !mode || !picked) return;
    if (!confirming) {
      confirming = true;
      render();
      return;
    }
    locked = true;
    const p1 = picked;
    const p2 = otherCharacter(p1);
    render();
    onLockIn(mode, p1, p2);
  });
  cta.append(ctaText, lockBtn);

  const hint = el(
    'div',
    'dossier-hint',
    'Keyboard: press 1 for ONE PLAYER, 2 for TWO PLAYERS, then 1/2 to pick a fighter.'
  );

  select.append(modeGrid, pickGrid, cta, hint);

  const render = (): void => {
    for (const [id, btn] of Object.entries(modeButtons) as Array<[MatchMode, HTMLButtonElement]>) {
      btn.dataset.active = String(mode === id);
      btn.setAttribute('aria-checked', String(mode === id));
    }
    const modeChosen = mode !== null;
    pickGrid.style.opacity = modeChosen ? '1' : '0.4';
    pickGrid.style.pointerEvents = modeChosen ? 'auto' : 'none';

    for (const [id, btn] of Object.entries(pickButtons) as Array<[CharacterId, HTMLButtonElement]>) {
      const active = picked === id;
      btn.dataset.active = String(active);
      btn.setAttribute('aria-checked', String(active));
      btn.disabled = locked;
      const radio = btn.querySelector('.dossier-pick__radio');
      if (radio) radio.textContent = active ? '[ SELECTED ]' : '[ SELECT ]';
    }

    if (locked && picked) {
      ctaText.innerHTML = `Fighter locked: <strong>${FIGHTERS[picked].name}</strong>. Starting match…`;
      lockBtn.dataset.ready = 'false';
      lockBtn.disabled = true;
    } else if (!mode) {
      ctaText.textContent = 'Select a match mode above to continue.';
      lockBtn.dataset.ready = 'false';
      lockBtn.disabled = true;
    } else if (!picked) {
      ctaText.textContent = 'Select a fighter above to continue.';
      lockBtn.dataset.ready = 'false';
      lockBtn.disabled = true;
    } else if (confirming) {
      ctaText.innerHTML = `Confirm: <strong>${FIGHTERS[picked].name}</strong>, permanently. This cannot be undone.`;
      lockBtn.dataset.ready = 'true';
      lockBtn.dataset.confirming = 'true';
      lockBtn.textContent = 'CONFIRM — NO WAY BACK';
      lockBtn.disabled = false;
      lockBtn.style.setProperty('--pick-accent', FIGHTERS[picked].color);
    } else {
      ctaText.innerHTML = `Fighter selected: <strong>${FIGHTERS[picked].name}</strong>. Locking in is final.`;
      lockBtn.dataset.ready = 'true';
      lockBtn.dataset.confirming = 'false';
      lockBtn.textContent = 'LOCK IN';
      lockBtn.disabled = false;
      lockBtn.style.setProperty('--pick-accent', FIGHTERS[picked].color);
    }
  };
  render();

  root.append(boot, hero, law, bhuruSection, vsRule, heinrichSection, compare, permanence, select);
  app.append(root);

  // Scroll reveal via IntersectionObserver (framework-free, mirrors the
  // reference component's own no-dependency fallback path) ---------------------
  const revealTargets = [bhuruSection, heinrichSection];
  const observer = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        if (entry.isIntersecting) {
          (entry.target as HTMLElement).dataset.in = 'true';
          observer.unobserve(entry.target);
        }
      }
    },
    { threshold: 0.2 }
  );
  revealTargets.forEach((target) => observer.observe(target));

  // Keyboard fallback so 1/2 keys keep working exactly as before ------------------
  let selectVisible = false;
  const onKeydown = (e: KeyboardEvent): void => {
    if (!selectVisible || locked) return;
    if (!mode) {
      if (e.key === '1') { mode = '1P_VS_AI'; render(); }
      else if (e.key === '2') { mode = '2P'; render(); }
      return;
    }
    if (e.key === '1' || e.key === '2') {
      const choice: CharacterId = e.key === '1' ? 'bhuru' : 'heinrich';
      if (mode === '2P' && picked && choice === picked) return;
      picked = choice;
      render();
    } else if (e.key === 'Enter' && picked) {
      lockBtn.click();
    }
  };
  window.addEventListener('keydown', onKeydown);

  const showLoading = (progress: number): void => {
    const pct = Math.round(Math.max(0, Math.min(1, progress)) * 100);
    bootBarFill.style.width = `${pct}%`;
    bootPct.textContent = `${pct}%`;
    bootLabel.textContent = pct >= 100 ? 'TWO SUBJECTS FOUND' : 'INITIALIZING ROSTER…';
    bootDot.classList.toggle('armed', pct >= 100);
  };

  const showSelect = (): void => {
    selectVisible = true;
    bootBarFill.style.width = '100%';
    bootPct.textContent = 'READY';
    bootLabel.textContent = 'FRAME SELECTION IS PERMANENT';
    bootDot.classList.add('armed');
    select.dataset.lockedOut = 'false';
  };

  const destroy = (): void => {
    observer.disconnect();
    window.removeEventListener('keydown', onKeydown);
    root.remove();
  };

  select.dataset.lockedOut = 'true';

  return { showLoading, showSelect, destroy };
};
