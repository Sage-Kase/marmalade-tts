# espeak-ng / kitten-daemon English pronunciation audit

Probe: `python3 tools/ph_probe.py` in `/home/max/coding/marmalade-tts-cli`
(espeak-ng via phonemizer + `fix_en_phonemes`, i.e. the exact string the model sees).
Target accent: General American. Dialect variation (e.g. `niche` = /nɪtʃ/) not flagged.
Known-and-being-fixed `biweekly` / `bimonthly` / `biyearly` excluded.

All "suggested text" values were re-run through the probe; the IPA shown for them is
the actual probe output, not a guess.

## 2. Items probed per category

| Category | File | Items |
|---|---|---|
| Prefixed / compound words (bi-, tri-, re-, pre-, co-, anti-, de-, un-, non-, semi-, multi-, micro-, over-, under-, out-) | `prefix.txt` | 502 |
| Tech / software vocabulary | `tech.txt` | 388 |
| Brand & product names | `brands.txt` | 225 |
| Modern / internet words | `modern.txt` | 135 |
| Common irregulars, loanwords, place names | `irregular.txt` | 337 |
| Inflections (plural / possessive / -ed / -ing) of tricky stems | `inflect.txt` | 215 |
| Numbers, numerals, units, symbols | `numbers.txt` | 115 |
| First names & surnames from many cultures | `names.txt` | 321 |
| Medical / science / astronomy terms | `sci.txt` | 387 |
| Food words | `food.txt` | 319 |
| Heteronym-in-context sentences | `het.txt` | 208 |
| Punctuation / structural passages | `passages.txt` | 15 |
| Respelling verification runs | `fix1`–`fix5.txt` | ~430 |
| **Total probed** | | **~3600** |

## 3. Patterns found

1. **Heteronym disambiguation is a two-word trigger heuristic, not grammar.** This is the
   single biggest source of HIGH errors. (Correction, 2026-08-21 research pass: espeak does
   carry `$verb/$noun/$past` variants for ~100 words, selected only when one of ~180
   closed-class marker words — `the/a/my` → noun, `to/will/I` → verb, `have/was` → past —
   sits one or two words left. Imperatives, "Please X", open-class subjects, and bare past
   tense never fire it; `lead/minute/resume/bass/sow/subject/excuse` have no second entry
   at all. ~43% of minimal pairs come out right.)
   Noun/verb stress pairs that *do* work (`object`, `convict`, `rebel`, `suspect`, `progress`,
   `increase`, `console`, `content`, `desert`, `separate`, `appropriate`, `deliberate`,
   `elaborate`, `moderate`, `escort`, `export`, `extract`, `upset`, `permit`, `refuse`,
   `graduate`, `house`, `buffet`, `polish`) are ones espeak has hard-coded rules for. Everything
   else (`read`, `lead`, `wind`, `wound`, `bass`, `record`, `close`, `dove`, `minute`, `bow`,
   `sow`, `does`, `sewer`, `tier`, `axes`, `bases`, `slough`, `use`, `excuse`, `diffuse`,
   `compact`, `import`, `estimate`, `alternate`, `advocate`, `associate`, `duplicate`,
   `invalid`, `entrance`, `present`, `decrease`, `transfer`, `conflict`, `insult`, `subject`,
   `contract`, `moped`, `number`, `resume`, `Nice`, `Reading`) is a coin flip that lands wrong
   about half the time.
2. **The `bi-` prefix collapses to /bɪ/ before a `w`, `m`, `y` or `l` stem** — `biweekly`,
   `bimonthly`, `biyearly`, `bilayer`, `bimorph`. Before other letters it is correctly /baɪ/
   (`bicameral`, `bisect`, `bimodal`, `bipedal`). The same class of failure hits `tri-`
   (`triennial`, `trivalent`, `triphasic`, `trilateral`, `trifecta`) but there the failure is
   stress/vowel, not the prefix vowel.
3. **`co-` before a consonant-initial stem is often read as a syllable of the stem**, giving
   "cow-" or "koh/kaw-": `coworking`→"cow-erking", `cowrite`→"cow-rite", `comingle`→"cummingle",
   `cosign`→"kah-sign", `cofactor`→"kah-factor", `coroutine`→"cor-outine", `copayment`.
4. **`pre-`/`de-` before an s/w/vowel stem mis-syllabifies**: `presale`→"pre-ZALE" (voiced z),
   `prestage`→"PRESS-taj", `premix`→"PREM-ix", `deescalate`→"DEES-calate" (syllable lost),
   `deinstall`→"DAY-nstall", `dewater`→"doo-water", `reupload`→"ROO-pload" (the `re-` is eaten).
5. **A hyphen is the single most reliable text-level fix.** Inserting a hyphen at the morpheme
   boundary forces espeak to phonemize each part separately and fixes the great majority of
   compound/prefix errors (`bi-layer`, `co-working`, `de-escalate`, `re-upload`, `pre-mix`,
   `smart-home`, `e-sports`, `reg-ex`, `go-routine`, `ex-org`, `macro-phage`, `lypo-protein`).
   Cost: it usually produces two primary stresses (`bˈaɪlˈeɪɚ`), which is a minor prosodic
   artefact, not a wrong phoneme. Where a single stress matters, an un-hyphenated respelling
   (`makedir`, `mayven`, `deeno`, `onnix`, `numpie`) is better.
6. **Compound-boundary `th` is read as /θ/**: `smarthome` → `smˈɑːɹθoʊm` ("smar-thome"). Same
   class as `hothouse`, `pothole` risks. Fix: hyphenate.
7. **Initialisms are guessed inconsistently.** Some are correctly spelled out (`XML`, `HTTP`,
   `pnpm`, `VMware`, `ffmpeg`, `nginx`→"engine X" — impressive), but others are wrongly read as
   words: `CLI`→"cly", `CI`→"sigh", `ORM`→"orm", `XOR`→"zor", `xorg`→"zorg", `ECDSA`→"ekdsuh",
   `DJI`→"jai", `OVH`→"ov", `IMAP`→"im-app", `chmod`→"C-H-mod", `mkdir`→"em-kay-dire".
8. **`x` word-initial is always /z/**, which breaks Chinese pinyin and some names: `Xiaomi`,
   `Xian`, `Xu`, `Xiang` (all should be /ʃ/). `Xiaoming` also produces a malformed doubled
   schwa `zˈaɪəəmɪŋ`.
9. **Polish digraphs `sz`/`cz`/`rz` are not mapped** — they surface as literal letter clusters:
   `Agnieszka`→`ˈæɡnɪszkə`, `Tomasz`→`tˈɑːmæsz`. Same for Portuguese/Brazilian `nh`
   (`caipirinha`→`kˈeɪpɪɹˌɪnhə`).
10. **Word-final `-e` in Italian/Japanese/Spanish loanwords is silenced** instead of /eɪ/ or /i/:
    `penne`→"pen", `farfalle`→"far-fahl", `linguine`→"lin-gwine", `tagliatelle`, `pappardelle`,
    `ceviche`→"sevitch", `elote`, `halloumi`, `Miele`→"meel", `Kwame`→"kwaym",
    `Guadalupe`, `Salvatore`, `Giuseppe`.
11. **Phonemes that may be outside the model's inventory** appear in a handful of outputs and
    should be treated as structural bugs, not just wrong vowels:
    - nasal vowels: `croissant`→`kwˈɑːsɑ̃`, `denouement`→`deɪnˈuːmɔ̃`
    - velar fricative `x`: `Ahmed`→`ˈæxmɛd`
    - plain `r` (not `ɹ`): `Ruairi`→`rjˈuːɛɹi`, `Katarzyna`→`kˌæɾɚrzˈiːnə`, `arrhythmia`→`ɚrhˈɪθmiə`
    - bare `h` mid-cluster: `arrhythmia`, `caipirinha`
    - doubled/geminate segments: `focussed`→`fˈoʊkəsst`, `onnx`→`ˈɑːŋŋks`, `za'atar`→`zˈææɾɚ`,
      `coxswain`→`kˈɑːkssweɪn`, `Xiaoming`→`zˈaɪəəmɪŋ`
    - stress anomalies: two primary stress marks in one word (`Zephaniah`→`zɛfˈeɪnˈaɪə`,
      `e-mail`, `co-operate`, `pancreatitis`) or **no** primary stress at all (`Redis`→`ɹᵻdiz`,
      `nano`→`nˌænoʊ`).
12. **Numbers/symbols: the decimal point is never verbalized.** `0.5` → `zˈiəɹoʊ.fˈaɪv`,
    `3.14159` → `θɹˈiː.fˈɔːɹtiːn...` — the "." is left as a literal character with no "point".
    Also: thousands separators are mis-grouped (`1,000,000` → "one, zero zero zero, zero zero
    zero"), currency symbols are spoken *before* the amount (`$19.99` → "dollar nineteen ninety
    nine", `£10` → "pound ten"), `<` is silently deleted, month abbreviations are not expanded
    (`Aug.` → "awg"), and Roman numerals are garbage (`MCMXCIX` → `məkmkssˈɪks`, `Ⅳ` →
    "letter two one seven three").
13. **Hyphens inside path/slug tokens are deleted rather than treated as boundaries**:
    `marmalade-tts-cli` → `mˈɑːɹmɐlˌeɪdtˌiːtˌiːˈɛsklˈaɪ` (words run together). Ironically the
    same hyphen that *fixes* a plain word *breaks* a slug, because espeak first strips it.
14. **Intervocalic flapping is applied before a stressed vowel** where it shouldn't be:
    `pytorch`→`pˈaɪɾɔːɹtʃ`, `Saturn`→`sˈæɾɜːn`, `subtle`→`sˈʌɾəl` (the last is fine).
15. **Japanese/Korean romanization gets a spurious /j/ or /aɪ/**: `umami`→"you-mami",
    `Sakura`→"sa-KYUR-a", `furikake`→"fyoor-", `Fukuoka`→"fyoo-kyoo-", `mochi`→"mo-CHIGH",
    `kimchi`→"kim-CHIGH", `Ryu`→"RYE-oo", `Ji-woo`→"JYE-woo".

---

## 1. Findings table

Sorted HIGH first, then MEDIUM. "Suggested text" is a verified respelling; the parenthesised
IPA after it is the probe's actual output for that respelling.

### HIGH — clearly wrong

| Word / sentence | espeak IPA | Expected (GA) | Suggested text (verified probe output) |
|---|---|---|---|
| **Heteronyms (all HIGH — no context disambiguation)** ||||
| I read the book last night. | `aɪ ɹˈiːd ðə bˈʊk...` | `ɹˈɛd` | `I red the book last night.` (`ɹˈɛd`) |
| They read the minutes ... yesterday. | `ɹˈiːd` | `ɹˈɛd` | `red` (`ɹˈɛd`) |
| The lead pipe was corroded. | `lˈiːd` | `lˈɛd` | `The led pipe was corroded.` (`lˈɛd`) |
| The paint contains lead. | `lˈiːd` | `lˈɛd` | `contains led` (`lˈɛd`) |
| Wind the clock before bed. | `wˈɪnd` | `wˈaɪnd` | `Wined the clock before bed.` (`wˈaɪnd`) |
| She wound the bandage around his arm. | `wˈuːnd` | `wˈaʊnd` | `wowned` (`wˈaʊnd`) |
| The wound tape came loose. | `wˈuːnd` | `wˈaʊnd` | `wowned` (`wˈaʊnd`) |
| He caught a large bass in the river. | `bˈeɪs` | `bˈæs` | `bas` (`bˈæs`) |
| Please record the meeting. | `ɹˈɛkɚd` | `ɹᵻkˈɔːɹd` | `re-cord` (`ɹˌiːkˈɔːɹd`) |
| Close the door behind you. | `klˈoʊs` | `klˈoʊz` | `Cloze the door...` (`klˈoʊz`) |
| She dove into the pool. | `dˈʌv` | `dˈoʊv` | `dohv` (`dˈoʊv`) |
| The differences are minute. | `mˈɪnɪt` | `maɪnˈuːt` | `my-newt` (`maɪnˈuːt`) |
| A minute amount of dust remained. | `mˈɪnɪt` | `maɪnˈuːt` | `my-newt` (`maɪnˈuːt`) |
| Nice is in southern France. | `nˈaɪs` | `nˈiːs` | `Neece` (`nˈiːs`) |
| Reading is a town in England. | `ɹˈiːdɪŋ` | `ɹˈɛdɪŋ` | `Redding` (`ɹˈɛdɪŋ`) |
| The graph has two axes. | `ˈæksᵻz` | `ˈæksiːz` | `ax-eez` (`ˈæksˈiːz`) |
| The bases of the argument are weak. | `bˈeɪsᵻz` | `bˈeɪsiːz` | `bay-seez` (`bˈeɪsˈiːz`) |
| Snakes slough their skin. | `slˈaʊ` | `slˈʌf` | `sluff` (`slˈʌf`) |
| Two does grazed in the field. | `dˈʌz` | `dˈoʊz` | `dohz` (`dˈoʊz`) |
| The sewer stitched the hem. | `sˈuːɚ` | `sˈoʊɚ` | `soh-er` (`sˈoʊˈɜː`) |
| He is a habitual tier of knots. | `tˈɪɹ` | `tˈaɪɚ` | `tie-er` (`tˈaɪˈɜːɹ`) |
| The resume was long. | `ɹᵻzˈuːm` | `ɹˈɛzəmeɪ` | `rez-oo-may` (`ɹˈɛzˈuːmˈeɪ`) |
| The bow of the ship creaked. | `bˈoʊ` | `bˈaʊ` | `bough` (`bˈaʊ`) |
| He took a bow after the show. | `bˈoʊ` | `bˈaʊ` | `bau` (`bˈaʊ`) |
| The sow had six piglets. | `sˈoʊ` | `sˈaʊ` | `sau` (`sˈaʊ`) |
| He moped around all day. | `mˈoʊpɛd` | `mˈoʊpt` | `mohpt` (`mˈoʊpt`) |
| My hand felt number after the ice. | `nˈʌmbɚɹ` | `nˈʌmɚ` | `nummer` (`nˈʌmɚɹ`) |
| Use the correct use. | `jˈuːs` (verb) | `jˈuːz` | `Yooz the correct use.` (`jˈuːz`) |
| Please excuse the delay. | `ɛkskjˈuːs` | `ɪkskjˈuːz` | `ex-cuze` (`ˈɛkskjˈuːz`) |
| The diffuse light was soft. | `dᵻfjˈuːz` | `dɪfjˈuːs` | `dif-fuce` (`dˈɪffjˈuːs`) |
| A compact car. | `kəmpˈækt` | `kˈɑːmpækt` | `com-pact` (`kˈɑːmpˈækt`) |
| The import tax rose. | `ɪmpˈɔːɹt` | `ˈɪmpɔːɹt` | `im-port` (`ˈɪmpˈɔːɹt`) |
| Please estimate the cost. | `ˈɛstᵻmət` | `ˈɛstᵻmeɪt` | `estimayt` (`ˈɛstᵻmˌeɪt`) |
| The signals alternate. | `ɔːltˈɜːnət` | `ˈɔːltɚneɪt` | `alter-nate` (`ˈɔltɚnˈeɪt`) |
| The advocate spoke. | `ˈædvəkˌeɪt` | `ˈædvəkət` | `advocut` (`ˈædvəkˌʌt`) |
| The associate arrived. | `ɐsˈoʊsɪˌeɪt` | `ɐsˈoʊʃiət` | `associut` (`ɐsˈoʊsɪˌʌt`) |
| A duplicate copy. | `dˈuːplᵻkˌeɪt` | `dˈuːplᵻkət` | `duplicut` (`dˈuːplɪkˌʌt`) |
| The invalid needed care. | `ɪnvˈælɪd` | `ˈɪnvəlɪd` | *no clean respelling found;* `in-vuh-lid` → `ɪnvˈʌlˈɪd` (stress still wrong) |
| The songs entrance the crowd. | `ˈɛntɹəns` | `ɛntɹˈæns` | `en-trance` (`ˈɛntɹˈæns`) |
| Let me present the findings. | `pɹˈɛzənt` | `pɹɪzˈɛnt` | `pre-zent` (`pɹˈiːzˈɛnt`) |
| Sales will decrease again. | `dˈiːkɹiːs` | `dᵻkɹˈiːs` | `de-crease` (`dəkɹˈiːs`) |
| Please transfer the funds. | `tɹˈænsfɜː` | `tɹænsfˈɜː` | `trans-fer` (`tɹˈænzfˈɜː`) |
| Their schedules conflict. | `kˈɑːnflɪkt` | `kənflˈɪkt` | `con-flict` (`kˈɑːnflˈɪkt`) |
| Do not insult him. | `ˈɪnsʌlt` | `ɪnsˈʌlt` | `in-sult` (`ɪnsˈʌlt`) |
| They will subject him to a test. | `sˈʌbdʒɛkt` | `səbdʒˈɛkt` | `sub-ject` (`sˈʌbdʒˈɛkt`) |
| Muscles contract and relax. | `kˈɑːntɹækt` | `kəntɹˈækt` | `con-tract` (`kˈɑːntɹˈækt`) |
| They produce fresh produce daily. | 2nd `pɹədˈuːs` | `pɹˈoʊduːs` | write the noun as `pro-duce`/`prohduce` |
| **Prefix / compound** ||||
| bilayer | `bˈɪleɪɚ` | `baɪlˈeɪɚ` | `bi-layer` (`bˈaɪlˈeɪɚ`) |
| bimorph | `bˈɪmɔːɹf` | `bˈaɪmɔːɹf` | `bi-morph` (`bˈaɪmˈɔːɹf`) |
| triennial | `tɹˈaɪnɪəl` | `tɹaɪˈɛnɪəl` | `tri-ennial` (`tɹˈaɪˈɛnɪəl`) |
| trivalent | `tɹˈaɪvələnt` | `tɹaɪvˈeɪlənt` | `tri-valent` (`tɹˈaɪvˈeɪlənt`) |
| triphasic | `tɹɪfˈæzɪk` | `tɹaɪfˈeɪzɪk` | `tri-phasic` (`tɹˈaɪfˈæzɪk`) — vowel still `æ`, partial fix |
| trilateral | `tɹˈaɪlɐɾɚɹəl` | `tɹaɪlˈæɾɚɹəl` | `tri-lateral` (`tɹˈaɪlˈæɾɚɹəl`) |
| reupload | `ɹˈuːploʊd` | `ɹiːʌplˈoʊd` | `re-upload` (`ɹˌiːˈʌploʊd`) |
| presale | `pɹɪzˈeɪl` | `pɹˈiːseɪl` | `pre-sale` (`pɹˈiːsˈeɪl`) |
| prestage | `pɹˈɛsteɪdʒ` | `pɹiːstˈeɪdʒ` | `pre-stage` (`pɹˈiːstˈeɪdʒ`) |
| premix | `pɹˈɛmɪks` | `pɹˈiːmɪks` | `pre-mix` (`pɹˈiːmˈɪks`) |
| coworking | `kˈaʊɜːkɪŋ` | `kˈoʊwɜːkɪŋ` | `co-working` (`kˈoʊwˈɜːkɪŋ`) |
| cowrite / cowriter | `kˈaʊɹaɪt` / `kˈaʊɹaɪɾɚ` | `koʊɹˈaɪt` | `co-write` / `co-writer` (`kˈoʊɹˈaɪt`) |
| comingle | `kˈʌmɪŋəl` | `koʊmˈɪŋɡəl` | `co-mingle` (`kˈoʊmˈɪŋɡəl`) |
| copayment | `kˈɑːpeɪmənt` | `kˈoʊpeɪmənt` | `co-payment` (`kˈoʊpˈeɪmənt`) |
| cosign | `kˈɑːsaɪn` | `koʊsˈaɪn` | `co-sign` (`kˈoʊsˈaɪn`) |
| cofactor | `kˈɑːfæktɚ` | `kˈoʊfæktɚ` | `co-factor` (`kˈoʊfˈæktɚ`) |
| coproduce | `kˈoʊpɹədʒˌuːs` | `koʊpɹədˈuːs` | `co-produce` (`kˈoʊpɹədˈuːs`) |
| coroutine / goroutine | `kˈɔːɹuːtˌiːn` / `ɡˈɔːɹuːtˌiːn` | `koʊɹˈuːtiːn` / `ɡˈoʊɹuːtiːn` | `co-routine` (`kˈoʊɹuːtˈiːn`), `go-routine` (`ɡˌoʊɹuːtˈiːn`) |
| deescalate | `dˈiːskɐlˌeɪt` | `diːˈɛskəleɪt` | `de-escalate` (`diːˈɛskɐlˌeɪt`) |
| deinstall | `dˈeɪnstɔːl` | `diːɪnstˈɔːl` | `de-install` (`diːɪnstˈɔːl`) |
| dewater | `dˈuːɔːɾɚ` | `diːwˈɔːɾɚ` | `dee-water` (`dˈiːwˈɔːɾɚ`) |
| **Tech** ||||
| regex / regexp | `ɹᵻdʒˈɛks` | `ɹˈɛdʒɛks` | `reg-ex` (`ɹˈɛdʒˈɛks`) |
| Postgres | `pˈoʊstɡɚz` | `pˈoʊstɡɹɛs` | `postgress` (`pˈoʊstɡɹɛs`) |
| Redis | `ɹᵻdiz` *(no primary stress)* | `ɹˈɛdɪs` | `red-iss` (`ɹˈɛdˈɪs`) |
| CLI | `klˈaɪ` | `sˌiːˌɛlˈaɪ` | `C-L-I` (`sˈiːˈɛlˈaɪ`) |
| CI | `sˈaɪ` | `sˌiːˈaɪ` | `C-I` |
| ORM | `ˈɔːɹm` | `ˌoʊɑːɹˈɛm` | `O-R-M` |
| XOR | `zˈɔːɹ` | `ˈɛksɔːɹ` | `ex-or` (`ˈɛksɔːɹ`) |
| xorg | `zˈɔːɹɡ` | `ˈɛksɔːɹɡ` | `ex-org` (`ˈɛksˈɔːɹɡ`) |
| ECDSA | `ˈɛkdsə` | spelled out | `E C D S A` (`ˈiː sˈiː dˈiː ˈɛs ˈeɪ`) |
| DeFi | `də fˌaɪ` | `dˈiːfaɪ` | `Dee-Fi` (`dˈiːfˌaɪ`) |
| IMAP | `ɪmˈæp` | `ˈaɪmæp` | `eye-map` (`ˈaɪmˈæp`) |
| IPv6 | `ˈɪpv sˈɪks` | `ˌaɪpiːviːsˈɪks` | `I P v six` (`aɪ pˈiː vˈiː sˈɪks`) |
| CIDR | `sˈɪdɚ` | `sˈaɪdɚ` | `cider` (`sˈaɪdɚ`) |
| chmod | `sˌiːˈeɪtʃmˈɑːd` | `tʃˈɑːmɑːd` | `cha-mod` (`tʃˈɑːmˈɑːd`) |
| mkdir | `ˌɛmkˈeɪdˈaɪɚ` | `mˈeɪkdɪɹ` | `makedir` (`mˈeɪkdɪɹ`) |
| rmdir | `ˌɑːɹɹˈɛmdˈaɪɚ` | `ˌɑːɹɛmdɪɹ` | `r m dir` (`ˈɑːɹ ˈɛm dˈaɪɚ`) — partial |
| devops | `dɪvˈɑːps` | `dˈɛvɑːps` | `dev-ops` (`dˈɛvˈɑːps`) |
| prettier (the tool) | `pɹˈɪɾiɚ` | `pɹˈɛɾiɚ` | *no clean fix;* `pret-ee-er` → `pɹˈɛtˈiːˈɜː` |
| Deno | `dᵻnˈoʊ` | `dˈiːnoʊ` | `deeno` (`dˈiːnoʊ`) |
| Maven | `mˈævən` | `mˈeɪvən` | `mayven` (`mˈeɪvən`) |
| virtualenv | `vˈɜːtʃuːˌeɪlənv` | `vˈɜːtʃuəl ɛnv` | `virtual env` |
| memoize / memoization | `mˈɛmɔɪz` | `mˈɛmoʊaɪz` | `memo-ize` (`mˈɛmoʊˈaɪz`) |
| idempotent | `ˈaɪdmpoʊtənt` *(no vowel in `dm`)* | `aɪdˈɛmpətənt` | `eye-dempotent` (`ˈaɪdˈɛmpoʊtənt`) |
| async | `ɐsˈɪŋk` | `ˈeɪsɪŋk` | `aysync` (`ˈaɪsɪŋk`) |
| enum | `ɪnˈʌm` | `ˈiːnʌm` | `ee-num` (`ˈiːnˈʌm`) |
| deserialize | `dɪzˈɪɹiəlˌaɪz` | `diːsˈɪɹiəlaɪz` | `dee-serialize` (`dˈiːsˈɪɹiəlˌaɪz`) |
| onnx / onnxruntime | `ˈɑːŋŋks` *(doubled ŋ)* | `ˈɑːnɪks` | `onnix` (`ˈɑːnɪks`) or `O N N X` |
| numpy | `nˈʌmpi` | `nˈʌmpaɪ` | `numpie` (`nˈʌmpaɪ`) |
| scipy | `sˈaɪpi` | `sˈaɪpaɪ` | `sci-pie` (`sˈaɪpˈaɪ`) |
| jupyter | `dʒˈʌpaɪɾɚ` | `dʒˈuːpɪɾɚ` | `jupiter` (`dʒˈuːpɪɾɚ`) |
| phonemizer | `fˈoʊnmaɪzɚ` | `fˈoʊniːmaɪzɚ` | `phoneme-izer` (`fˈoʊniːmˈaɪzɚ`) |
| **Modern / internet** ||||
| smarthome | `smˈɑːɹθoʊm` | `smˈɑːɹthoʊm` | `smart-home` (`smˈɑːɹthˈoʊm`) |
| esports | `ɛspˈɔːɹts` | `ˈiːspɔːɹts` | `e-sports` (`ˈiːspˈɔːɹts`) |
| vlogger | `vˈiːlˈɔɡɚ` ("V-logger") | `vlˈɑːɡɚ` | *no clean fix;* `vee-logger` unchanged |
| umami | `juːmˈɑːmi` | `uːmˈɑːmi` | `oomami` (`uːmˈɑːmi`) |
| kombucha | `kˈɑːmbʌtʃə` | `kɑːmbˈuːtʃə` | `kom-boocha` (`kˈɑːmbˈuːtʃə`) |
| charcuterie | `tʃɑːɹkjˈuːɾɚɹi` | `ʃɑːɹkˈuːɾɚɹi` | `shar-cooteree` (`ʃˈɑːɹkˈuːɾɚɹˌiː`) |
| patreon | `pˈætɹɪən` | `pˈeɪtɹiɑːn` | `paytreon` (`pˈeɪtɹɪən`) |
| emoticon | `ɪmˈɑːɾɪkən` | `ɪmˈoʊɾɪkɑːn` | `emoh-ticon` (`ɪmˈoʊtˈɪkən`) |
| **Brands** ||||
| Xiaomi | `zˌaɪəˈoʊmi` | `ʃaʊmˈiː` | `show-mee` (`ʃˈoʊmˈiː`) — best available |
| Huawei | `hjˈuːɐwˌeɪ` | `wˈɑːweɪ` | `wahway` (`wˈɑːweɪ`) |
| Vimeo | `vˈaɪmɪˌoʊ` | `vˈɪmioʊ` | `vimmeeoh` (`vˈɪmiːˌoʊ`) |
| Vercel | `vˈɜːsəl` | `vɚsˈɛl` | `Ver-sell` (`vˈɜːsˈɛl`) |
| Linode | `lˈɪnoʊd` | `lˈaɪnoʊd` | `Lie-node` (`lˈaɪnˈoʊd`) |
| OVH | `ˈɑːv` | `ˌoʊviːˈeɪtʃ` | `O V H` |
| DJI | `dʒˈaɪ` | `ˌdiːdʒeɪˈaɪ` | `D J I` |
| Asics | `ɐsˈɪks` | `ˈeɪsɪks` | `aysicks` (`ˈaɪsɪks`) |
| Patagonia | `pætˈæɡənˌiə` | `pˌæɾəɡˈoʊniə` | `patagoania` (`pˌæɾɐɡˈoʊniə`) |
| Bugatti | `bjuːɡˈæɾi` | `buːɡˈɑːɾi` | `boogotti` (`buːɡˈɑːɾi`) |
| Renault | `ɹᵻnˈɔlt` | `ɹənˈoʊ` | `renoh` (`ɹᵻnˈoʊ`) |
| Miele | `mˈiːl` | `mˈiːlə` | `meeaylah` (`mˈiːeɪlə`) |
| Mattel | `mˈæɾəl` | `mətˈɛl` | `Ma-tell` (`mˈɑːtˈɛl`) |
| Cartier | `kˈɑːɹɾiɚ` | `kɑːɹtiˈeɪ` | `carteeay` (`kˈɑːɹɾiːˌeɪ`) |
| Bose | `bˈoʊs` | `bˈoʊz` | `Boze` |
| **Irregulars / loanwords / places** ||||
| coup / coups | `kˈuːp` / `kˈuːps` | `kˈuː` / `kˈuːz` | `koo` (`kˈuː`) |
| gunwale | `ɡˈʌnweɪl` | `ɡˈʌnəl` | `gunnel` (`ɡˈʌnəl`) |
| coxswain | `kˈɑːkssweɪn` *(doubled s)* | `kˈɑːksən` | `coxun` (`kˈɑːksʌn`) |
| pterodactyl | `tˈɛɹədˌæktaɪl` | `tˌɛɹədˈæktɪl` | `terra-dactil` (`tˈɛɹədˈæktɪl`) |
| choleric | `koʊlˈɛɹɪk` | `kˈɑːlɚɹɪk` | `collerick` (`kˈɑːlɚɹˌɪk`) |
| archipelago | `ˌɑːɹkɪpɪlˈeɪɡoʊ` | `ˌɑːɹkəpˈɛləɡoʊ` | `arki-pell-a-go` (`ˈɑːɹkipˈɛlɐɡˈoʊ`) |
| naive / naivete | `naɪˈiːv` | `nɑːˈiːv` | `nigh-eve` (`nˈaɪˈiːv`) — partial |
| chorale | `kˈɔːɹeɪl` | `kəɹˈæl` | `kor-al` (`kˈɔːɹˈæl`) |
| gnocchi | `nˈɑːkaɪ` | `nˈjɔːki` | `nokey` (`nˈoʊki`) — closest found |
| prosciutto | `pɹəsɪˈʌɾoʊ` | `pɹəʃˈuːɾoʊ` | `proshooto` (`pɹəʃˈuːɾoʊ`) |
| biscotti | `baɪskˈɑːɾi` | `bɪskˈɑːɾi` | `biscotty` (`bˈɪskɑːɾi`) |
| tortilla | `tɔːɹtˈiːɐ` | `tɔːɹtˈiːjə` | `torteeya` (`tˈɔːɹɾiːjə`) |
| quesadilla | `kˌeɪsədˈiːə` | `kˌeɪsədˈiːjə` | `kaysadeeya` (`kˈeɪsɐdˌiːjə`) |
| quinoa | `kwɪnˈoʊə` | `kˈiːnwɑː` | `keen-wah` (`kˈiːnwˈɑː`) |
| acai | `ɐkˈaɪ` | `ˌɑːsaɪˈiː` | `ahsighee` (`ˈɑːsaɪˌiː`) |
| mochi | `mˈɑːtʃaɪ` | `mˈoʊtʃi` | `mohchee` (`mˈoʊtʃiː`) |
| udon | `jˈuːdɑːn` | `ˈuːdoʊn` | `oodon` (`ˈuːdɑːn`) |
| edamame | `ˈɛdɐmˌeɪm` *(syllable lost)* | `ˌɛdəmˈɑːmeɪ` | `edamahmay` (`ˈɛdɐmˌɑːmeɪ`) |
| sriracha | `sɹɜːɹˈɑːtʃə` | `sɪɹˈɑːtʃə` | `sir-rocha` (`sˌɜːɹˈɑːtʃə`) |
| banh mi | `bˈæn mˈaɪ` | `bˈɑːn mˈiː` | `bahn mee` (`bˈɑːn mˈiː`) |
| Wichita | `wˈɪtʃᵻtˌɑːɹ` *(spurious final r)* | `wˈɪtʃɪtɔː` | `witchitaw` (`wˈɪtʃɪtˌɔː`) |
| Poughkeepsie | `pˈoʊkiːpsi` | `pəkˈɪpsi` | `puh-kipsee` (`pˈʌkˈɪpsiː`) |
| Schenectady | `ʃˈɛnɪktˌædi` | `skənˈɛktədi` | `skuh-nectady` (`skˈʌnˈɛktædi`) |
| Bicester | `baɪsˈɛstɚ` | `bˈɪstɚ` | `Bister` (`bˈɪstɚ`) |
| Southwark | `sˈaʊθwɔːɹk` | `sˈʌðɚk` | `suth-uck` (`sˈʌθˈʌk`) |
| Marylebone | `mˈɛɹaɪlbˌoʊn` | `mˈɑːɹləbən` | `marlabun` (`mˈɑːɹlɐbˌʌn`) |
| Alnwick | `ɐlnwˈɪk` | `ˈænɪk` | `Annick` (`ˈænɪk`) |
| Holborn | `hˈɑːlbɔːɹn` | `hˈoʊbən` | `Hoban` |
| Bruges | `bɹˈuːdʒᵻz` | `bɹˈuːʒ` | `Broozh` (`bɹˈuːʒ`) |
| Tbilisi | `tˈiːbaɪlˈɪsi` | `təbɪlˈiːsi` | `tbileesee` (`tˈiːbaɪlˈiːsiː`) — partial |
| Xian | `zˈaɪən` | `ʃiːˈɑːn` | `sheeahn` (`ʃˈiːɑːn`) |
| Kyoto | `kaɪˈoʊɾoʊ` | `kiˈoʊɾoʊ` | `keeohtoh` (`kˈiːoʊtˌoʊ`) |
| Hokkaido | `həkˈeɪdoʊ` | `hɑːkˈaɪdoʊ` | `hoh-kye-doh` (`hˈoʊkˈaɪdˈoʊ`) |
| Fukuoka | `fjˌuːkjuːˈoʊkə` | `ˌfuːkuːˈoʊkə` | `fookoooka` (`fˌʊkuːˈoʊkə`) |
| Nagoya | `næɡˈɔɪə` | `nɑːɡˈoʊjə` | *no clean fix found* |
| Karachi | `kɚɹˈɑːɹtʃi` *(spurious r)* | `kəɹˈɑːtʃi` | `karahchee` (`kˈæɹɑːtʃˌiː`) |
| croissant | `kwˈɑːsɑ̃` *(nasal — likely out of inventory)* | `kɹwɑːsˈɑːnt` | `kruh-sont` |
| denouement | `deɪnˈuːmɔ̃` *(nasal)* | `deɪnuːmˈɑːn` | `daynoomahn` |
| **Inflections** ||||
| focussed | `fˈoʊkəsst` *(geminate)* | `fˈoʊkəst` | `focused` (`fˈoʊkəst`) |
| crocheted | `kɹoʊʃˈeɪᵻd` *(extra syllable)* | `kɹoʊʃˈeɪd` | `crochade` |
| sauteed / sauteing | `sˈɔːɾiːd` / `sˈɔːɾeɪɪŋ` | `soʊtˈeɪd` | `soh-tayed` |
| parqueted | `pˈɑːɹkeɪᵻd` | `pɑːɹkˈeɪd` | `parkayed` |
| debuts | `deɪbjˈuːs` | `deɪbjˈuːz` | `debuze` |
| chateaux | `ʃˈeɪɾɔːks` | `ʃætˈoʊz` | `shattohz` |
| psyches | `sˈaɪkiːᵻz` | `sˈaɪkiz` | `sykees` |
| concerti | `kənsˈɜːɾi` | `kəntʃˈɛɹti` | `con-chairtee` |
| theses | `θəsˈiːz` | `θˈiːsiːz` | `thee-seez` |
| fungi | `fˈʌŋɡi` | `fˈʌndʒaɪ` | `fun-guy` |
| alumni | `ɐlˈʌmni` | `əlˈʌmnaɪ` | `alum-nigh` |
| **Numbers / symbols** ||||
| `0.5` | `zˈiəɹoʊ.fˈaɪv` | `zˈiəɹoʊ pˈɔɪnt fˈaɪv` | write `zero point five` |
| `3.14159` | `θɹˈiː.fˈɔːɹtiːn θˈaʊzənd...` | digits after the point | write `three point one four one five nine` |
| `1,000,000` | `wˈʌn,zˈiəɹoʊzˈiəɹoʊ zˈiəɹoʊ,...` | `wˈʌn mˈɪliən` | write `one million` |
| `$19.99` | `dˈɑːlɚ nˈaɪntiːn.nˈaɪnti nˈaɪn` | `nˈaɪntiːn dˈɑːlɚz nˈaɪnti nˈaɪn` | write `nineteen dollars ninety-nine` |
| `£10`, `€5`, `¥100` | `pˈaʊnd tˈɛn` (symbol first) | `tˈɛn pˈaʊndz` | write the words out |
| `p < 0.05` | `<` **silently deleted** | `lˈɛs ðən` | write `less than` |
| `Aug.` | `ˈɔːɡ` | `ˈɔːɡəst` | write `August` |
| `MCMXCIX` | `məkmkssˈɪks` | `nˌaɪntiːn nˈaɪnɾi nˈaɪn` | write the number |
| `Ⅳ` (U+2163) | `lˌɛɾɚtˈuːwˈʌnsˈɛvənθɹˈiː` | `fˈɔːɹ` | use ASCII `IV`/`four` |
| `marmalade-tts-cli` in a path | `mˈɑːɹmɐlˌeɪdtˌiːtˌiːˈɛsklˈaɪ` | separate words | hyphens in slugs are deleted, not honoured |
| **Names** ||||
| Niamh | `nˈaɪəm` | `nˈiːv` | `Neev` |
| Grainne | `ɡɹˈeɪn` | `ɡɹˈɑːnjə` | `Grahnya` |
| Eoin | `ˈiːəˌɪn` | `ˈoʊɪn` | `Owen` |
| Cian | `sˈaɪən` | `kˈiːən` | `Kee-an` |
| Tadhg | `tˈædɡ` | `tˈaɪɡ` | `Tyge` |
| Sian | `sˈaɪən` | `ʃˈɑːn` | `Shahn` |
| Mhairi | `ˈɛmhˈɛɹi` | `vˈɑːɹi` | `Vahree` |
| Farquhar | `fˈɑːɹkwəhˌɑːɹ` | `fˈɑːɹkɚ` | `Farker` |
| Joaquin | `dʒˈoʊkwɪn` | `wɑːkˈiːn` | `Wah-keen` |
| Ximena | `zˈaɪmnə` | `hɪmˈeɪnə` | `Hi-mayna` |
| Guillermo | `ɡˈɪlɚmˌoʊ` | `ɡiːjˈɛɹmoʊ` | `Gee-yairmo` |
| Guadalupe | `ɡwˈɑːdɐlˌuːp` | `ˌɡwɑːdəlˈuːpeɪ` | `gwada-loopay` |
| Pilar | `pˈɪlɚ` | `piːlˈɑːɹ` | `Pee-lar` |
| Anais | `ˈænaɪz` | `ˌɑːnɑːˈiːs` | `ah-nah-ees` |
| Margaux | `mˈɑːɹɡɔːks` | `mɑːɹɡˈoʊ` | `Margo` |
| Giuseppe | `dʒˈɪjuːsˌɛp` | `dʒuːsˈɛpeɪ` | `joo-seppay` |
| Salvatore | `sˈælvæɾɚ` | `ˌsælvətˈɔːɹeɪ` | `salva-toray` |
| Johannes | `dʒˈoʊhænz` | `joʊhˈɑːnəs` | `yo-hahnes` |
| Jurgen / Jorgen | `dʒˈɜːdʒən` | `jˈɜːɡən` | `Yurgen` |
| Wojciech | `wˈɑːdʒsaɪtʃ` | `vˈɔɪtʃɛk` | `Voycheck` |
| Grzegorz | `dʒˌiːˈɑːɹzˈɛɡɔːɹz` | `ɡʒˈɛɡɔːʃ` | `Gzhegosh` |
| Tomasz | `tˈɑːmæsz` *(literal `sz`)* | `tˈɔːmɑːʃ` | `Tomash` |
| Agnieszka | `ˈæɡnɪszkə` *(literal `sz`)* | `æɡnjˈɛʃkə` | `ag-nyeshka` |
| Zbigniew | `zˈiːbˈɪɡnjuː` | `zbˈɪɡnjɛf` | `Zbignyef` |
| Andrzej | `ˈændəzˌɛdʒ` | `ˈɑːndʒeɪ` | `Ahn-jay` |
| Natalya | `nˈeɪɾɐlɪə` | `nətˈɑːljə` | `na-tahlya` |
| Ahmed | `ˈæxmɛd` *(velar `x` — likely out of inventory)* | `ˈɑːmɛd` | `Ahmed` → write `Ah-med` |
| Wei | `wˈaɪ` | `wˈeɪ` | `Way` |
| Xiang | `zjˈæŋ` | `ʃjˈɑːŋ` | `Shyahng` |
| Xu | `zˈuː` | `ʃˈuː` | `Shoo` |
| Qian | `kˈaɪən` | `tʃjˈɛn` | `Chyen` |
| Xiaoming | `zˈaɪəəmɪŋ` *(doubled schwa)* | `ʃaʊmˈɪŋ` | `Show-ming` |
| Takeshi | `tˈeɪkʃi` | `tɑːkˈɛʃi` | `ta-kesshee` |
| Sakura | `sækjˈʊɹɹə` | `sɑːkˈʊɹə` | `sa-koora` |
| Ryu | `ɹˈaɪuː` | `ɹjˈuː` | `Ryoo` |
| Ji-woo | `dʒˈaɪwˈuː` | `dʒˈiːwuː` | `Jee-woo` |
| Kwame | `kwˈeɪm` | `kwˈɑːmeɪ` | `Kwah-may` |
| Zephaniah | `zɛfˈeɪnˈaɪə` *(2 primary stresses)* | `ˌzɛfənˈaɪə` | `zeffa-nigh-a` |
| Melchizedek | `mˈɛltʃaɪzdˌɛk` | `mɛlkˈɪzədɛk` | `mel-kizzedek` |
| Nebuchadnezzar | `nˈɛbətʃˌædnɪzˌɑːɹ` | `ˌnɛbjəkədnˈɛzɚ` | `nebbyu-kad-nezzer` |
| Barnabas | `bɑːɹnˈɑːbəz` | `bˈɑːɹnəbəs` | `Barna-bus` |
| Jemima | `dʒˈɛmɪmə` | `dʒəmˈaɪmə` | `Je-my-ma` |
| Magdalena | `mˈæɡdeɪlnə` *(syllable lost)* | `ˌmæɡdəlˈeɪnə` | `magda-layna` |
| **Medical / science** ||||
| omeprazole | `ˈoʊmpɹɐzˌoʊl` *(syllable lost)* | `oʊmˈɛpɹəzoʊl` | `oh-meprazole` (`ˈoʊmˈɛpɹɐzˌoʊl`) |
| prednisone | `pɹɪdnˈɪsoʊn` | `pɹˈɛdnɪsoʊn` | `pred-nisone` (`pɹˈɛdnˈɪsoʊn`) |
| norepinephrine | `nˈɔːɹpaɪnfɹˌiːn` | `ˌnɔːɹɛpɪnˈɛfɹɪn` | `nor-epinephrine` (`nˈɔːɹˈɛpɪnˌɛfɹiːn`) |
| levothyroxine | `lˈɛvəθˌɪɹəksˌaɪn` | `ˌliːvoʊθaɪɹˈɑːksiːn` | `leevo-thy-roxine` |
| metastasis | `mˌɛɾəstˈɑːsiz` | `mətˈæstəsɪs` | `metass-tasis` (`mˈɛɾæstˈæsɪz`) — partial |
| emphysema | `ɛmfˈaɪsmə` | `ˌɛmfɪsˈiːmə` | `emfi-seema` (`ɛmfisˈiːmə`) |
| edema | `ˈɛdᵻmə` | `ɪdˈiːmə` | `ee-deema` (`ˈiːdˈiːmə`) |
| cecum | `sˈɛkəm` | `sˈiːkəm` | `seekum` (`sˈiːkəm`) |
| duodenum | `dˈuːoʊdnəm` | `ˌduːədˈiːnəm` | `doooh-deenum` (`dˈuːoʊdˈiːnəm`) |
| trachea | `tɹɐkˈiə` | `tɹˈeɪkiə` | `tray-kia` (`tɹˈeɪkˈiə`) |
| alveoli | `ˈælvɪˌɑːli` | `ælvˈiːəlaɪ` | `alveeohlye` (`ˈælviːˌoʊlaɪ`) |
| amygdala | `ˌæmɪɡdˈɑːlə` | `əmˈɪɡdələ` | `uh-migdala` (`ˈʌmɪɡdˈɑːlə`) |
| meninges | `mˈɛnɪndʒᵻz` | `mənˈɪndʒiːz` | `meninjeez` (`mˈɛnɪndʒˌiːz`) |
| phalanges | `fˈælændʒᵻz` | `fəlˈændʒiːz` | `falanjeez` (`fˈælɐndʒˌiːz`) |
| arrhythmia | `ɚrhˈɪθmiə` *(plain `r`+`h`; also `θ` for `ð`)* | `əɹˈɪðmiə` | `a-rithmia` |
| adenosine | `ˈædənˌɑːsaɪn` | `ədˈɛnəsiːn` | `a-den-oh-seen` (`ˈeɪdˈɛnˈoʊsˈiːn`) |
| ligase | `lˈɪɡeɪs` | `lˈaɪɡeɪs` | `lygase` (`lˈaɪɡeɪs`) |
| macrophage | `mˈækɹəfɪdʒ` | `mˈækɹəfeɪdʒ` | `macro-phage` (`mˈækɹoʊfˈeɪdʒ`) |
| bilirubin | `baɪlˈɜːɹuːbˌɪn` | `ˌbɪlɪɹˈuːbɪn` | `billy-roobin` (`bˈɪliɹˈuːbɪn`) |
| triglyceride | `tɹˈɪɡlɪsɚɹˌaɪd` | `tɹaɪɡlˈɪsəɹaɪd` | `try-glyceride` (`tɹˈaɪɡlˈɪsɚɹˌaɪd`) |
| lipoprotein | `lˈɪpəpɹˌoʊtiːn` | `ˌlaɪpoʊpɹˈoʊtiːn` | `lypo-protein` (`lˈaɪpoʊpɹˈoʊtiːn`) |
| chlorophyll | `klˌɔːɹoʊfˈɪl` | `klˈɔːɹəfɪl` | `klorofill` (`klˈɔːɹəfˌɪl`) |
| covalent | `kˈoʊvələnt` | `koʊvˈeɪlənt` | `co-valent` (`kˈoʊvˈeɪlənt`) |
| electronegativity | `ᵻlˌɛktɹoʊŋɡɐtˈɪvᵻɾi` *(spurious ŋɡ)* | `ᵻlˌɛktɹoʊnɛɡətˈɪvᵻɾi` | `electro-negativity` |
| peroxisome | `pˈɛɹəksisˌʌm` | `pəɹˈɑːksɪsoʊm` | `per-oxisome` |
| Antares | `ˈæntɛɹz` | `æntˈɛɹiːz` | `An-tarries` (`æntˈæɹiz`) |
| Aldebaran | `ˈɔːldɪbˌæɹən` | `ældˈɛbəɹən` | `al-debaran` (`ˈældᵻbˈæɹən`) |
| Ophiuchus | `ˈɑːfɪˌʌtʃəs` | `ˌɑːfiˈjuːkəs` | `oh-fee-yookus` (`ˈoʊfˈiːjˈuːkəs`) |
| Io | `ˈiːoʊ` | `ˈaɪoʊ` | `Eye-oh` |
| etiology | `iːtˈɪələdʒi` | `ˌiːtiˈɑːlədʒi` | `etee-ology` |
| **Food** ||||
| creme brulee | `kɹˈiːm bɹˈuːliː` | `kɹˈɛm bɹuːlˈeɪ` | `krem broo-lay` (`kɹˈɛm bɹˈuːlˈeɪ`) |
| cassoulet | `kˈæsoʊlɪt` | `ˌkæsuːlˈeɪ` | `cassoo-lay` (`kˈæsuːlˈeɪ`) |
| consomme | `kənsˈɑːm` | `ˌkɑːnsəmˈeɪ` | `consoh-may` (`kənsˈoʊmˈeɪ`) |
| bechamel | `bˈɛtʃeɪməl` | `ˌbeɪʃəmˈɛl` | `baysha-mel` (`bˈeɪʃəmˈɛl`) |
| veloute | `vˈɛlaʊt` | `vəluːtˈeɪ` | `velloo-tay` |
| julienne | `dʒˈuːliən` | `ˌdʒuːliˈɛn` | `jooly-en` (`dʒˈuːliˈɛn`) |
| brunoise | `bɹˈʌnɔɪs` | `bɹuːnwˈɑːz` | `broon-wahz` |
| mise en place | `mɪsˈiː ˈɛn plˈeɪs` | `mˈiːz ɑːn plˈɑːs` | `meez on plahss` |
| sous vide | `sˈuːz vˈaɪd` | `sˈuː vˈiːd` | `soo veed` (`sˈuː vˈiːd`) |
| flambe | `flˈæmb` | `flɑːmbˈeɪ` | `flom-bay` (`flˈɑːmbˈeɪ`) |
| penne | `pˈɛn` | `pˈɛneɪ` | `pennay` (`pˈɛneɪ`) |
| linguine | `lˈɪŋɡwaɪn` | `lɪŋɡwˈiːni` | `lin-gweenee` (`lˈɪŋɡwˈiːniː`) |
| farfalle | `fˈɑːɹfɔːl` | `fɑːɹfˈɑːleɪ` | `far-fahlay` (`fˈɑːɹfˈɑːleɪ`) |
| tagliatelle | `tˌæɡlɪɐtˈɛl` | `ˌtɑːljətˈɛleɪ` | `tal-ya-telly` (`tˈæljɐtˈɛli`) |
| pappardelle | `pˌæpɑːɹdˈɛl` | `ˌpɑːpɑːɹdˈɛleɪ` | `pappar-delay` |
| focaccia | `foʊkˈæksiə` | `foʊkˈɑːtʃə` | `fo-cotcha` (`fˈoʊkˈɑːtʃə`) |
| ciabatta | `sˈaɪəbˌæɾə` | `tʃəbˈɑːɾə` | `cha-botta` (`tʃˈɑːbˈɑːɾə`) |
| panettone | `pˈænɪʔˌn̩` *(glottal collapse)* | `ˌpænɪtˈoʊneɪ` | `panettohnay` (`pˈænɪtˌoʊneɪ`) |
| osso buco | `ˈɑːsoʊ bjˈuːkoʊ` *(spurious j)* | `ˈɑːsoʊ bˈuːkoʊ` | `osso booko` |
| puttanesca | `pˈʌteɪnskə` | `ˌpuːtənˈɛskə` | `poota-nesca` |
| ceviche | `sˈɛvɪtʃ` | `səvˈiːtʃeɪ` | `seveechay` (`sˈɛviːtʃˌeɪ`) |
| elote | `ᵻlˈoʊt` | `ɛlˈoʊteɪ` | `eh-lohtay` |
| jamon | `dʒˈæmən` | `hɑːmˈoʊn` | `hah-mohn` |
| caipirinha | `kˈeɪpɪɹˌɪnhə` *(literal `nh`)* | `ˌkaɪpɪɹˈiːnjə` | `kypi-reenya` |
| tzatziki | `tˈiːzˈætsɪki` | `tsɑːtsˈiːki` | `tsat-seeky` (`tsˈætsˈiːki`) |
| spanakopita | `spˌænɐkəpˈiːɾə` | `ˌspænəkˈoʊpɪtə` | `spana-kopita` |
| halloumi | `hˈælaʊmi` | `həlˈuːmi` | `ha-loomee` (`hˈɑːlˈuːmiː`) |
| feta | `fˈiːɾə` | `fˈɛɾə` | `fetta` (`fˈɛɾə`) |
| tagine | `tˈædʒɪn` | `tɑːʒˈiːn` | `ta-zheen` (`tˈɑːʒˈiːn`) |
| za'atar | `zˈææɾɚ` *(doubled `æ`)* | `zˈɑːtɑːɹ` | `zah-tar` |
| turmeric | `tɜːmˈɛɹɪk` | `tˈɜːmɚɹɪk` | `turmerick` (`tˈɜːmɚɹˌɪk`) |
| kimchi | `kˈɪmtʃaɪ` | `kˈɪmtʃi` | `kimchee` (`kˈɪmtʃiː`) |
| bibimbap | `baɪbˈɪmbæp` | `bˈiːbɪmbɑːp` | `bee-bim-bop` (`bˈiːbˈɪmbˈɑːp`) |
| japchae | `dʒˈæpkiː` | `dʒˈɑːptʃeɪ` | `jop-chay` |
| tteokbokki | `tˈiːɾɪˈɑːkbɑːki` | `tˈɔːkbɔːki` | `tawk-bokki` |
| miso | `mɪsˈoʊ` | `mˈiːsoʊ` | `meeso` (`mˈiːsoʊ`) |
| wagyu | `wˈædʒɪˌuː` | `wˈɑːɡjuː` | `wag-yoo` (`wˈæɡjˈuː`) |
| takoyaki | `tˌækɔɪˈæki` | `ˌtɑːkoʊjˈɑːki` | `tako-yaki` (`tˈɑːkoʊjˈæki`) |
| furikake | `fjˈʊɹɹɪkˌeɪk` | `ˌfʊɹikˈɑːkeɪ` | `foory-kakay` (`fˈɔːɹikˈækeɪ`) |
| poutine | `pˈaʊtiːn` | `puːtˈiːn` | `poo-teen` (`pˈuːtˈiːn`) |
| parfait | `pˈɑːɹfeɪt` *(final t voiced)* | `pɑːɹfˈeɪ` | `parfay` (`pˈɑːɹfeɪ`) |
| praline | `pɹˈeɪlaɪn` | `pɹˈɑːliːn` | `prahleen` (`pɹˈɑːliːn`) |
| endive | `ɛndˈɪv` | `ˈɛndaɪv` | `en-dive` (`ˈɛndˈaɪv`) |
| frisee | `fɹˈɪsiː` | `fɹiːzˈeɪ` | `free-zay` (`fɹˈiːzˈeɪ`) |
| jicama | `dʒɪkˈɑːmə` | `hˈiːkəmə` | `hee-kama` (`hˈiːkˈɑːmə`) |
| radicchio | `ɹædˈɪkɪˌoʊ` | `ɹədˈiːkioʊ` | `ra-deekio` (`ɹˈɑːdˈiːkɪˌoʊ`) |
| courgette | `kɜːdʒˈɛt` | `kʊɹʒˈɛt` | `koor-zhet` |

### MEDIUM — arguably wrong (stress, minor vowel, or contested)

| Word / sentence | espeak IPA | Expected (GA) | Suggested text |
|---|---|---|---|
| trimester | `tɹˈaɪmɛstɚ` | `tɹaɪmˈɛstɚ` | `tri-mester` |
| triathlon | `tɹˈaɪæθlən` | `tɹaɪˈæθlɑːn` | `tri-athlon` |
| trifecta | `tɹˈaɪfɛktə` | `tɹaɪfˈɛktə` | `tri-fecta` (`tɹˈaɪfˈɛktə`) |
| debug | `dˈiːbˌʌɡ` | `diːbˈʌɡ` | `de-bug` |
| prerequisite | `pɹˌiːɹˈɛkwɪsˌɪt` | `pɹɪɹˈɛkwəzɪt` | — |
| overture | `ˈoʊvɚtjˌʊɹ` | `ˈoʊvɚtʃɚ` | `overchur` |
| underling | `ˈʌndɜːlɪŋ` | `ˈʌndɚlɪŋ` | — |
| outfield / outgoing / outreach / outsource / outtake | verb-stressed | noun-stressed | hyphenate or respell |
| multivariate | `mˌʌltɪvˈɛɹɪˌeɪt` | `ˌmʌltɪvˈɛɹiət` | — |
| kubectl | `kjˈuːbɛktəl` | `kjuːb kəntɹˈoʊl` | `kube control` |
| systemctl | `sˈɪstəmktəl` | `sˈɪstəm kəntɹˈoʊl` | `system control` |
| JSON | `dʒˈeɪsˈɑːn` | `dʒˈeɪsən` | `jayson` |
| PostgreSQL | `pˈoʊstɡɚɹ ˌɛskjˌuːˈɛl` | `pˈoʊstɡɹɛs kjuː ɛl` | `postgress Q L` |
| Ubuntu | `uːbˈuːntuː` | `ʊbˈʊntuː` | — |
| Manjaro | `mændʒˈæɹoʊ` | `mændʒˈɑːɹoʊ` | `man-jaro` |
| vite | `vˈaɪt` | `vˈiːt` | `veet` |
| tsconfig | `tˈiːskˈɑːnfɪɡ` | `tˌiːˌɛs kˈɑːnfɪɡ` | `T S config` |
| PyPI | `pˈaɪ pˈaɪ` | `pˈaɪ pˌiːˈaɪ` | `pie P I` |
| Laravel | `lˈæɹævəl` | `ˌlæɹəvˈɛl` | `lara-vell` |
| BIOS | `bˈaɪoʊz` | `bˈaɪoʊs` | `bye-oss` |
| backend / endpoint | verb-stressed | noun-stressed | hyphenate |
| gitignore | `ɡˈɪɾɪɡnˌɔːɹ` | `ɡˈɪt ɪɡnˈɔːɹ` | `git ignore` |
| submodule | `sˈʌbmədʒˌuːl` | `sˈʌbmɑːdʒuːl` | `sub-module` |
| cryptocurrency | `kɹˈɪptəkˌɜːɹənsi` | `kɹˈɪptoʊkˌɜːɹənsi` | `crypto-currency` |
| decorator | `dˈɛkɔːɹˌeɪɾɚ` | `dˈɛkəɹeɪɾɚ` | — |
| scaffold | `skˈæfoʊld` | `skˈæfəld` | — |
| nullptr | `nˈʌlptɚ` | `nˈʌl pˈɔɪntɚ` | `null pointer` |
| journalctl / sysctl | `dʒˈɜːnælktəl` | spelled out | `journal control` |
| inkscape | `ɪŋkskˈeɪp` | `ˈɪŋkskeɪp` | `ink-scape` |
| pulseaudio | `pʌlsˈɔːdɪˌoʊ` | `pˈʌls ˈɔːdioʊ` | `pulse audio` |
| pytorch | `pˈaɪɾɔːɹtʃ` *(flap before stress)* | `pˈaɪtɔːɹtʃ` | `py-torch` (`pˈaɪtˈɔːɹtʃ`) |
| keras | `kˈiəɹəz` | `kˈɛɹəs` | `kerass` |
| matplotlib | `mˈætplətlˌɪb` | `mˈætplɑːtlɪb` | `matplot-lib` |
| anaconda | `ˈænɐkˌɑːndə` | `ˌænəkˈɑːndə` | — |
| Ethereum | `ˌiːθɚɹˈiːəm` | `ɪθˈɪɹiəm` | `e-theerium` |
| kokoro | `kəkˈɔːɹoʊ` | `koʊkˈɔːɹoʊ` | `ko-koro` |
| nano | `nˌænoʊ` *(no primary stress)* | `nˈænoʊ` | — |
| Nvidia | `ˈɛnvˈɪdiə` *(2 primaries)* | `ɛnvˈɪdiə` | — |
| Fujitsu | `fuːdʒˈɪtsuː` | `fuːdʒˈiːtsuː` | `foo-jeetsu` |
| Heroku | `hˈiəɹoʊkˌuː` | `həɹˈoʊkuː` | `heh-rokoo` |
| Nestle | `nˈɛsəl` | `nɛslˈeɪ` | `ness-lay` |
| Loreal | `lˈɔːɹiəl` | `ˌlɔːɹiˈæl` | `lor-ay-al` |
| Adidas | `ˈædɪdəz` | `ədˈiːdəs` | `a-deedas` |
| Mercedes | `mɜːsˈeɪdiːz` | `mɚsˈeɪdiːz` | — |
| Citroen | `sˈɪtɹoʊn` | `sˈɪtɹoʊɛn` | `citro-en` |
| Fiat | `fˈiːət` | `fˈiːɑːt` | `fee-ott` |
| Toyota | `tɔɪˈoʊɾə` | `təjˈoʊɾə` | `to-yota` |
| Skoda | `skˈoʊdə` | `ʃkˈoʊdə` | `shkoda` |
| Sonos | `sˈoʊnoʊz` | `sˈoʊnoʊs` | `sonoss` |
| Nivea | `nˈaɪviə` | `nɪvˈeɪə` | `ni-vay-a` |
| Carlsberg | `kˈɑːɹlsbɜːɡ` | `kˈɑːɹlzbɚɡ` | — |
| subreddit | `sˈʌbɹɪdˌɪt` | `sʌbɹˈɛdɪt` | `sub-reddit` |
| metaverse | `mˌɛɾəvˈɜːs` | `mˈɛɾəvɜːɹs` | `meta-verse` |
| lowkey / highkey | `lˈoʊki` / `hˈaɪki` | `ˌloʊkˈiː` | `low-key` |
| telecommute | `tˈɛlᵻkˌɑːmjuːt` | `ˌtɛlɪkəmjˈuːt` | `tele-commute` |
| nootropic | `nuːtɹˈɑːpɪk` | `ˌnoʊətɹˈoʊpɪk` | `noh-oh-tropic` |
| comptroller | `kɑːmtɹˈoʊlɚ` | `kəntɹˈoʊlɚ` | `controller` |
| boatswain | `bˈoʊtsweɪn` | `bˈoʊsən` | `bosun` |
| forecastle | `fˈɔːɹkæsəl` | `fˈoʊksəl` | `foke-sul` |
| topgallant | `tˈɑːpɡælənt` | `təɡˈælənt` | `tuh-gallant` |
| liqueur | `lᵻkˈɜː` | `lɪkˈɜːɹ` | — |
| chauffeur | `ʃoʊfˈʊɹ` | `ʃˈoʊfɚ` | `showfer` |
| recipe | `ɹˈɛsᵻpˌiː` | `ɹˈɛsəpi` | — |
| Cholmondeley / Featherstonehaugh / Marjoribanks / Dalziel / Kirkcudbright / Milngavie / Wemyss / Beauchamp / Belvoir | all spelling-pronounced | famously irregular | respell (obscure; low impact) |
| gyro (food) | `dʒˈaɪɹoʊ` | `jˈɪəɹoʊ` | `yeero` |
| sake (rice wine) | `sˈeɪk` | `sˈɑːki` | `sah-kee` |
| feng shui | `fˈɛŋ ʃjˈuːi` | `fˈʌŋ ʃwˈeɪ` | `fung shway` |
| qigong | `kˈɪɡɔŋ` | `tʃˈiːɡɔːŋ` | `chee-gong` |
| samurai | `sˈæmjʊɹɹˌaɪ` | `sˈæmʊɹaɪ` | `sam-oo-rye` |
| sudoku | `sˈuːdoʊkˌuː` | `suːdˈoʊkuː` | `soo-dohkoo` |
| regime | `ɹeɪʒˈiːm` | `ɹəʒˈiːm` | — |
| Norwich / Salisbury / Islington / Edinburgh / Leominster / Beaulieu | UK-irregular, US-plausible | — | respell if UK intent |
| Reykjavik | `ɹˈeɪkdʒɐvˌɪk` | `ɹˈeɪkjəviːk` | `rayk-ya-veek` |
| Zurich | `zjˈʊɹɪk` | `zˈʊɹɪk` | `zoo-rick` |
| Lucerne | `lˈuːsɚn` | `luːsˈɜːɹn` | `loo-sern` |
| Krakow | `kɹˈækaʊ` | `kɹˈɑːkaʊ` | `krah-kow` |
| Wroclaw | `ɹˈɑːklɔː` | `vɹˈɔːtswɑːf` | `vrots-wahf` |
| Odessa | `ˈoʊdɛsə` | `oʊdˈɛsə` | `oh-dessa` |
| Kyiv | `kjˈɪv` | `kˈiːv` | `keev` |
| Chongqing | `tʃˈɑːŋkɪŋ` | `tʃʊŋtʃˈɪŋ` | `chong-ching` |
| Cebu | `sˈɛbuː` | `sɛbˈuː` | `se-boo` |
| Yangon | `jˈɑːŋɑːn` | `jɑːnɡˈoʊn` | `yan-gohn` |
| Chennai | `tʃˈɛnaɪ` | `tʃɛnˈaɪ` | `chen-nigh` |
| Xerxes | `zˈɜːksᵻz` | `zˈɜːɹksiːz` | `zerk-seez` |
| bureaux / tableaux / gateaux | singular-sounding | `-oʊz` | `boo-rohz` etc. |
| epitomes / recipes | final `-iːz` stressed | unstressed `-iz` | — |
| sopranos | `səpɹˈɑːnoʊz` | `səpɹˈænoʊz` | — |
| symposia | `sɪmpˈoʊʒə` | `sɪmpˈoʊziə` | — |
| Rogerses | `ɹˈɑːdʒɜːsᵻz` | `ɹˈɑːdʒɚzɪz` | — |
| `12:30`, `3:2`, `10-7`, `64-bit`, phone numbers | colon/hyphen literal or digit-grouped | natural readings | write out |
| `Rd.` / `hrs.` / `vs.` | spelled `R-D`, `H-R-S`, `V-S` | `ɹˈoʊd`, `ˈaʊɚz`, `vˈɜːsəs` | write the word |
| `e-mail` / `co-operate` | 2 primary stresses | 1 | `email`, `cooperate` |
| acetaminophen | `ˈæsɪtˌæmɪnˌɑːfən` | `əsˌiːtəmˈɪnəfən` | `a-seeta-minophen` |
| melatonin | `mˈɛlɐtˌɑːnɪn` | `ˌmɛlətˈoʊnɪn` | `mela-tonin` |
| oxytocin | `ˌɑːksɪtˈɑːsɪn` | `ˌɑːksɪtˈoʊsɪn` | `oxy-tohcin` |
| aneurysm | `ˈænʊɹɹˌɪzəm` | `ˈænjəɹɪzəm` | — |
| eosinophil | `ˈiːəsˌɪnəfˌɪl` | `ˌiːəsˈɪnəfɪl` | — |
| erythrocyte | `ɜːɹˈɪθɹəsˌaɪt` | `ɪɹˈɪθɹəsaɪt` | — |
| telomere / centromere | `-mɚ` | `-mɪɹ` | `-meer` |
| occipital / parietal / calcaneus / medulla / pleura / patella / ileum / jejunum | stress or vowel off by one | — | respell |
| Saturn | `sˈæɾɜːn` | `sˈætɚn` | — |
| Betelgeuse / Phobos / Deimos / Procyon / Altair / Polaris / nadir | final voicing / stress | — | respell |
| sievert / farad / lumen / lepton / hadron / baryon / gluon | unstressed-vowel or spurious-`j` slips | — | — |
| macaron / madeleine / mille-feuille / bouillabaisse / mirepoix / saute | French `-e`/stress slips | — | respell |
| fusilli / cannoli / churros / burrito / chimichurri / arepa / gazpacho / tapas | stress or vowel slips | — | respell |
| sashimi / teriyaki / donburi / tonkatsu / congee / sichuan / szechuan / char siu | stress slips | — | respell |
| muesli / spaetzle / lebkuchen / lutefisk / knackebrod / langos / kedgeree | Germanic/Nordic slips | — | respell |
| ganache / compote / coulis / plantain / kefir / seitan / mache | minor slips | — | respell |
| paratha / pakora / raita / lassi / tandoori / asafoetida / marjoram | stress/vowel slips | — | respell |
| digestive (biscuit) | `daɪdʒˈɛstɪv` | `dɪdʒˈɛstɪv` | `di-gestive` |
| Games addict the young. | `ˈædɪkt` | `ədˈɪkt` | `a-dict` |
| He will intimate his intent. | `ˈɪntᵻmət` | `ˈɪntᵻmeɪt` | `intimayt` |
| The affect was flat. | `ɐfˈɛkt` | `ˈæfɛkt` | `af-fect` |
| Storms impact travel. | `ˈɪmpækt` | `ɪmpˈækt` | `im-pact` |
| Please abstract the key points. | `ˈæbstɹækt` | `æbstɹˈækt` | `ab-stract` |
| The rerun aired. / The reject bin. / Please relay the message. | verb/noun stress swapped | — | hyphenate |
| It was an august assembly. | `ˈɔːɡəst` | `ɔːɡˈʌst` | `au-gust` |
| Mobile, Alabama | `mˈoʊbəl` | `moʊbˈiːl` | `Mo-beel` |
| He read the primer on chemistry. | `pɹˈaɪmɚ` | `pɹˈɪmɚ` | `primmer` |
| They had a terrible row. | `ɹˈoʊ` | `ɹˈaʊ` | `rau` |
