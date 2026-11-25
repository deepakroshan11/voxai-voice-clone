"""Tamil text front-end for IndicF5.

Tamil is diglossic: written (literary) Tamil and spoken Tamil differ in grammar,
not just accent.  A TTS model reads every letter it is given, so

    இன்று வானிலை மிகவும் அழகாக இருக்கிறது.      ->  sounds like a news reader / old book
    இன்னைக்கு வானிலை ரொம்ப அழகா இருக்கு.        ->  sounds like a person talking

`to_spoken()` rewrites the most common written forms into everyday spoken Tamil
(verb endings, pronouns, case endings, a small lexicon).  Rules only touch known
literary endings, so text that is already colloquial passes through unchanged.

`prepare_for_tts()` also spells out digits (the model never learned to read them,
so they were silently skipped) and makes sure the text ends with punctuation.
"""
import re

TA = "஀-௿‌‍"
_TOKEN = re.compile(f"([{TA}]+)")
_PULLI = "்"
_CONS = "கஙசஞடணதநபமயரலவழளறனஜஷஸஹ"

# ---------------------------------------------------------------- whole words
_LEX = {
    # be / exist
    "இருக்கிறது": "இருக்கு", "இருக்கின்றது": "இருக்கு", "இருக்கின்றன": "இருக்கு",
    "உள்ளது": "இருக்கு", "உள்ளன": "இருக்கு", "இருக்கிறதா": "இருக்கா",
    "இருக்கிறேன்": "இருக்கேன்", "உள்ளேன்": "இருக்கேன்",
    "இருக்கிறாய்": "இருக்க", "இருக்கிறாயா": "இருக்கியா",
    "இருக்கிறான்": "இருக்கான்", "இருக்கிறாள்": "இருக்கா",
    "இருக்கிறார்": "இருக்கார்", "உள்ளார்": "இருக்கார்",
    "இருக்கிறோம்": "இருக்கோம்", "உள்ளோம்": "இருக்கோம்",
    "இருக்கிறீர்கள்": "இருக்கீங்க", "இருக்கிறீர்களா": "இருக்கீங்களா",
    "இருக்கிறார்கள்": "இருக்காங்க", "உள்ளனர்": "இருக்காங்க",
    "இல்லை": "இல்ல",
    # time / place
    "இன்று": "இன்னைக்கு", "இன்றைக்கு": "இன்னைக்கு", "இன்றும்": "இன்னைக்கும்",
    "நேற்று": "நேத்து", "நேற்றைக்கு": "நேத்து", "நாளை": "நாளைக்கு",
    "இப்போது": "இப்போ", "இப்பொழுது": "இப்போ", "இப்போதே": "இப்பவே",
    "அப்போது": "அப்போ", "அப்பொழுது": "அப்போ",
    "எப்போது": "எப்போ", "எப்பொழுது": "எப்போ", "எப்போதும்": "எப்பவும்", "எப்பொழுதும்": "எப்பவும்",
    "எங்கே": "எங்க", "அங்கே": "அங்க", "இங்கே": "இங்க",
    "பின்னர்": "அப்புறம்", "பிறகு": "அப்புறம்", "முன்னர்": "முன்னாடி",
    # pronouns
    "அவர்கள்": "அவங்க", "இவர்கள்": "இவங்க", "எவர்கள்": "எவங்க",
    "அவர்களுக்கு": "அவங்களுக்கு", "இவர்களுக்கு": "இவங்களுக்கு",
    "அவர்களை": "அவங்கள", "அவர்களுடைய": "அவங்களோட", "அவர்களுடன்": "அவங்களோட",
    "அவள்": "அவ", "இவள்": "இவ", "எல்லோரும்": "எல்லாரும்",
    "என்னிடம்": "என்கிட்ட", "உன்னிடம்": "உன்கிட்ட", "அவனிடம்": "அவன்கிட்ட",
    "அவளிடம்": "அவகிட்ட", "அவரிடம்": "அவர்கிட்ட", "உங்களிடம்": "உங்ககிட்ட",
    "நம்மிடம்": "நம்மகிட்ட", "அவர்களிடம்": "அவங்ககிட்ட", "எங்களிடம்": "எங்ககிட்ட",
    "அதில்": "அதுல", "இதில்": "இதுல", "எதில்": "எதுல",
    "மனதில்": "மனசுல", "வயதில்": "வயசுல",
    # common words
    "மிகவும்": "ரொம்ப", "மிக": "ரொம்ப", "நன்றாக": "நல்லா", "நன்றாய்": "நல்லா",
    "ஆனால்": "ஆனா", "ஏனென்றால்": "ஏன்னா", "ஏனெனில்": "ஏன்னா",
    "வேண்டும்": "வேணும்", "வேண்டாம்": "வேணாம்", "வேண்டுமா": "வேணுமா",
    "ஒன்று": "ஒண்ணு", "ஒன்றும்": "ஒண்ணும்", "இரண்டு": "ரெண்டு",
    "மூன்று": "மூணு", "நான்கு": "நாலு", "ஐந்து": "அஞ்சு",
    "ரூபாய்": "ரூபா", "தண்ணீர்": "தண்ணி",
    # literary verbs people never say
    "சென்றேன்": "போனேன்", "சென்றான்": "போனான்", "சென்றாள்": "போனா", "சென்றார்": "போனார்",
    "சென்றோம்": "போனோம்", "சென்றீர்கள்": "போனீங்க", "சென்றார்கள்": "போனாங்க",
    "சென்றனர்": "போனாங்க", "சென்றது": "போச்சு", "சென்று": "போய்", "சென்ற": "போன",
    "செல்கிறேன்": "போறேன்", "செல்கிறோம்": "போறோம்", "செல்ல": "போக",
    "கூறினேன்": "சொன்னேன்", "கூறினான்": "சொன்னான்", "கூறினாள்": "சொன்னா",
    "கூறினார்": "சொன்னார்", "கூறினார்கள்": "சொன்னாங்க", "கூறினர்": "சொன்னாங்க",
    "அருந்தி": "குடிச்சு", "அருந்தினேன்": "குடிச்சேன்",
    "போனது": "போச்சு", "ஆனது": "ஆச்சு",
}

# Words the suffix rules must leave alone (nouns that merely look like verb forms).
_KEEP = {"கோவில்", "வில்", "அணில்", "பதில்", "ஆக", "ஆகவே", "கார்கள்", "நாள்", "ஆள்",
         "அதனால்", "இதனால்", "எதனால்", "என்னால்", "உன்னால்", "தன்னால்", "அவனால்",
         "இவனால்", "அவளால்", "அவரால்", "நம்மால்", "உங்களால்", "முன்னாள்", "செய்தி"}

_PAST_END = r"(?=(?:ு(?:ட\S*|க்கிட்டு)?|ேன்|ான்|ா|ார்|ோம்|ீங்க|ீங்களா|ாங்க|ாங்களா|து|ுது)?$)"

# (pattern, replacement) applied in order, each on the whole word
_RULES = [
    # --- literary present -கின்ற- -> -கிற- (then handled below)
    (r"கின்ற(?=[ாேோீு]|து|ன|$)", "கிற"),
    # --- -கிற- present stem -> spoken
    (r"^(வரு|தரு)கிற", lambda m: {"வரு": "வர்", "தரு": "தர்"}[m.group(1)] + "ற"),
    (r"^போகிற", "போற"),
    (r"ுகிற", "ுற"),
    (r"க்கிற", "க்குற"),
    (r"ய்கிற", "ய்யுற"),
    (r"ல்கிற", "ல்ற"),
    (r"ள்கிற", "ள்ற"),
    (r"ட்கிற", "க்குற"),
    (r"^(நி|வி)ற்கிற", r"\1க்குற"),
    (r"ாகிற", "ாகுற"),
    (r"ிகிற", "ியுற"),
    # --- person endings
    (r"றாயா$", "றியா"),
    (r"றாய்$", "ற"),
    (r"ீர்களா$", "ீங்களா"),
    (r"ீர்கள்$", "ீங்க"),
    (r"(?<=..)ார்களா$", "ாங்களா"),
    (r"(?<=..)ார்கள்$", "ாங்க"),
    (r"(?<=[தடற])னர்$", "ாங்க"),
    (r"(?<=..[தனடறசவபகய])ாள்$", "ா"),
    (r"(?<=..[தனடறசவப])ாயா$", "ியா"),
    (r"(?<=..[தனடறசவப])ாய்$", ""),
    (r"ங்கள்$", "ங்க"),
    (r"ைகள்$", "ைங்க"),
    # --- negative: வரவில்லை -> வரல
    (r"(?<=..)வில்லையா$", "லையா"),
    (r"(?<=..)வில்லை$", "ல"),
    # --- விட்டு / விடு compounds: வந்துவிட்டேன் -> வந்துட்டேன்
    (r"ுவிட", "ுட"),
    (r"ிவிட", "ிட"),
    (r"ய்விட", "யிட"),
    # --- progressive: பேசிக்கொண்டு -> பேசிக்கிட்டு
    (r"க்கொண்டு$", "க்கிட்டு"),
    # --- palatalisation of past stems: படித்து -> படிச்சு, முடிந்த -> முடிஞ்ச, செய்து -> செஞ்சு
    (r"([ிை])த்த" + _PAST_END, r"\1ச்ச"),
    (r"([ிை])த்து$", r"\1ச்சு"),
    (r"^(முடி|தெரி|புரி|பிரி|எரி|கிழி|உடை|அடை|நனை|குறை|நிறை|மறை|கரை|அமை)ந்த" + _PAST_END,
     r"\1ஞ்ச"),
    (r"^(முடி|தெரி|புரி|பிரி|எரி|கிழி|உடை|அடை|நனை|குறை|நிறை|மறை|கரை|அமை)ந்து$", r"\1ஞ்சு"),
    (r"^(செ|பெ)ய்த" + _PAST_END, r"\1ஞ்ச"),
    (r"^(செ|பெ)ய்து$", r"\1ஞ்சு"),
    # --- conditional: வந்தால் -> வந்தா
    (r"(?<=ந்த|த்த|ட்ட|ச்ச|ஞ்ச|ன்ன)ால்$", "ா"),
    (r"(?<=..ன)ால்$", "ா"),
    # --- locative -இல் -> -ல
    (r"த்தில்$", "த்துல"),
    (r"ட்டில்$", "ட்டுல"),
    (r"ற்றில்$", "த்துல"),
    (r"(க்க|ப்ப|ச்ச|ண்ண|ன்ன|ல்ல|ள்ள|ம்ம)ில்$", r"\1ுல"),
    (r"([ைீி])யில்$", r"\1ல"),
    (r"(?<=..)வில்$", "வுல"),
    (r"(?<=.)([ரலளழணம])ில்$", r"\1்ல"),
    # --- possessive / comitative: என்னுடைய -> என்னோட, அவனுடன் -> அவனோட
    (r"(?<=..)ுடைய$", "ோட"),
    (r"(?<=..)ுடன்$", "ோட"),
    # --- adverbs: வேகமாக -> வேகமா
    (r"(?<=..)ாகவும்$", "ாவும்"),
    (r"(?<=..)ாக$", "ா"),
    # --- assimilation in frequent verbs: பார்த்தேன் -> பாத்தேன், கேட்க -> கேக்க
    (r"^பார்(?=[த்க]்)", "பா"),
    (r"^(சே|தீ|வள)ர்த்", r"\1த்"),
    (r"^கேட்க", "கேக்க"),
    (r"^(நி|வி)ற்க", r"\1க்க"),
]
_RULES = [(re.compile(p), r) for p, r in _RULES]

# Finite neuter verbs (sentence-final only; mid-sentence the same form is a noun)
_FINAL_RULES = [
    (r"ிகிறதா$", "ியுதா"), (r"ிகிறது$", "ியுது"),
    (r"^(முடி|தெரி|புரி|பிரி|எரி|கிழி|உடை|அடை|நனை|குறை|நிறை|மறை|கரை|அமை)ந்தது$", r"\1ஞ்சுது"),
    (r"ுகிறதா$", "ுதா"), (r"ுகிறது$", "ுது"),
    (r"க்கிறதா$", "க்குதா"), (r"க்கிறது$", "க்குது"),
    (r"ய்கிறது$", "ய்யுது"), (r"ல்கிறது$", "ல்லுது"), (r"ள்கிறது$", "ள்ளுது"),
    (r"ட்கிறது$", "க்குது"), (r"([ாோ])கிறது$", r"\1குது"), (r"கிறது$", "குது"),
    (r"ந்தது$", "ந்துச்சு"), (r"ச்சது$", "ச்சுது"), (r"ஞ்சது$", "ஞ்சுது"),
    (r"([ுி])ட்டது$", r"\1டுச்சு"), (r"ட்டது$", "ட்டுச்சு"),
    (r"(?<=[டு])த்தது$", "த்துச்சு"),
]
_FINAL_RULES = [(re.compile(p), r) for p, r in _FINAL_RULES]

# Accusative -ை is a short -a in speech (புத்தகத்தை -> புத்தகத்த, உன்னை -> உன்ன).
_FINAL_AI = re.compile(r"(?<=.)(த்த|ட்ட|ன்ன)ை$")
_ACC = {"அவனை": "அவன", "அவளை": "அவள", "அவரை": "அவர", "இவனை": "இவன", "இவளை": "இவள",
        "நம்மை": "நம்மள", "உங்களை": "உங்கள", "எங்களை": "எங்கள", "அதை": "அத",
        "இதை": "இத", "எதை": "எத"}
# Written sandhi consonants (கடைக்குச் சென்றேன், அந்தப் பையன்) are not spoken.
_SANDHI = re.compile(r"(?:(?<=[ுை])|(?<=^[அஇஎ]ந்த))[கசதப]்$")


def _word_to_spoken(w: str, final: bool) -> str:
    if len(w) > 3:
        w = _SANDHI.sub("", w)
    if w in _LEX:
        return _LEX[w]
    if w in _ACC:
        return _ACC[w]
    if w in _KEEP:
        return w
    if final:
        for pat, rep in _FINAL_RULES:
            new = pat.sub(rep, w, count=1)
            if new != w:
                w = new
                break
    for pat, rep in _RULES:
        w = pat.sub(rep, w, count=1)
    if w in _LEX:                       # a rule may have produced a lexicon form
        return _LEX[w]
    return _FINAL_AI.sub(r"\1", w)


def _merge_particles(tokens: list) -> list:
    """Join written-apart particles the way they are spoken:
    வருவேன் என்று -> வருவேன்னு,  போக வேண்டும் -> போகணும்,  வந்து விட்டேன் -> வந்துவிட்டேன்."""
    out = []
    for t in tokens:
        prev = None
        # find the previous Tamil word, skipping one plain space
        if out and out[-1] == " " and len(out) > 1 and _TOKEN.fullmatch(out[-2]):
            prev = out[-2]
        if prev and t in ("என்று", "என்றால்"):
            tail = "னு" if t == "என்று" else "னா"
            out.pop(); out[-1] = prev + (tail if prev.endswith(_PULLI) else "ன்" + tail)
            continue
        if (prev and t in ("வேண்டும்", "வேணும்") and prev[-1] in "கசடயரலறப"
                and prev not in ("என்ன", "எவ்வளவு", "அது", "இது")):
            out.pop(); out[-1] = prev + "ணும்"
            continue
        if prev and re.match(r"விட(்ட|ு|ுவ)", t) and (prev[-1] in "ுி" or prev.endswith("ய்")):
            out.pop(); out[-1] = prev + t
            continue
        out.append(t)
    return out


def to_spoken(text: str) -> str:
    """Rewrite written (literary) Tamil into everyday spoken Tamil."""
    parts = [p for p in _TOKEN.split(text) if p != ""]
    parts = _merge_particles(parts)
    out = []
    for i, p in enumerate(parts):
        if not _TOKEN.fullmatch(p):
            out.append(p)
            continue
        rest = "".join(parts[i + 1:])
        nxt = re.match(f"[^{TA}]*", rest).group(0)
        final = bool(re.search(r"[.!?।]", nxt)) or not rest.strip()
        # a word may contain an embedded split (கொண்டிருக்கிறேன் -> கிட்டு இருக்கேன்)
        p = re.sub(r"(க்)?கொண்டிருக்", lambda m: ("க்" if m.group(1) else "") + "கிட்டு இருக்", p)
        words = p.split(" ")
        out.append(" ".join(_word_to_spoken(w, final and j == len(words) - 1)
                            for j, w in enumerate(words)))
    return "".join(out)


# ---------------------------------------------------------------- numbers
_ONES = ["பூஜ்ஜியம்", "ஒண்ணு", "ரெண்டு", "மூணு", "நாலு", "அஞ்சு", "ஆறு", "ஏழு", "எட்டு", "ஒம்பது",
         "பத்து", "பதினொண்ணு", "பன்னெண்டு", "பதிமூணு", "பதினாலு", "பதினஞ்சு", "பதினாறு",
         "பதினேழு", "பதினெட்டு", "பத்தொம்பது"]
_TENS = {2: ("இருபது", "இருபத்தி"), 3: ("முப்பது", "முப்பத்தி"), 4: ("நாப்பது", "நாப்பத்தி"),
         5: ("அம்பது", "அம்பத்தி"), 6: ("அறுபது", "அறுபத்தி"), 7: ("எழுபது", "எழுபத்தி"),
         8: ("எம்பது", "எம்பத்தி"), 9: ("தொண்ணூறு", "தொண்ணூத்தி")}
_HUND = {1: ("நூறு", "நூத்தி"), 2: ("இருநூறு", "இருநூத்தி"), 3: ("முன்னூறு", "முன்னூத்தி"),
         4: ("நானூறு", "நானூத்தி"), 5: ("ஐநூறு", "ஐநூத்தி"), 6: ("அறுநூறு", "அறுநூத்தி"),
         7: ("எழுநூறு", "எழுநூத்தி"), 8: ("எண்ணூறு", "எண்ணூத்தி"), 9: ("தொள்ளாயிரம்", "தொள்ளாயிரத்தி")}
_THOU = {1: "ஆயிரம்", 2: "ரெண்டாயிரம்", 3: "மூவாயிரம்", 4: "நாலாயிரம்", 5: "அஞ்சாயிரம்",
         6: "ஆறாயிரம்", 7: "ஏழாயிரம்", 8: "எட்டாயிரம்", 9: "ஒம்பதாயிரம்", 10: "பத்தாயிரம்"}


def _join(big: str, combining: str, rest: int) -> str:
    return big if rest == 0 else combining + " " + num_to_tamil(rest)


def num_to_tamil(n: int) -> str:
    if n < 20:
        return _ONES[n]
    if n < 100:
        t, r = divmod(n, 10)
        return _join(*_TENS[t], r)
    if n < 1000:
        h, r = divmod(n, 100)
        return _join(*_HUND[h], r)
    if n < 100000:
        t, r = divmod(n, 1000)
        big = _THOU.get(t) or (num_to_tamil(t) + " ஆயிரம்")
        return _join(big, big[:-2] + "த்தி" if big.endswith("ம்") else big, r)
    if n < 10000000:
        l, r = divmod(n, 100000)
        big = ("ஒரு" if l == 1 else num_to_tamil(l)) + " லட்சம்"
        return _join(big, big[:-2] + "த்தி", r)
    c, r = divmod(n, 10000000)
    big = ("ஒரு" if c == 1 else num_to_tamil(c)) + " கோடி"
    return _join(big, big, r)


def _spell_numbers(text: str) -> str:
    def num(m):
        whole, frac = m.group(1).replace(",", ""), m.group(2)
        s = num_to_tamil(int(whole)) if len(whole) <= 12 else " ".join(_ONES[int(d)] for d in whole)
        if frac:
            s += " புள்ளி " + " ".join(_ONES[int(d)] for d in frac)
        return s
    text = re.sub(r"(?:₹|Rs\.?|INR)\s*(\d[\d,]*(?:\.\d+)?)", r"\1 ரூபா", text)
    text = re.sub(r"(\d[\d,]*(?:\.\d+)?)\s*%", r"\1 சதவீதம்", text)
    return re.sub(r"(\d+(?:,\d+)*)(?:\.(\d+))?", num, text)


def prepare_for_tts(text: str, spoken: bool = True) -> str:
    """Everything IndicF5 should receive: digits spelled out, optional spoken-Tamil
    rewrite, tidy spacing and closing punctuation (helps the model end the sentence
    instead of clipping the last word)."""
    text = re.sub(r"\[[^\]]*\]", " ", text)   # English-engine tags like [laugh]
    text = _spell_numbers(text)
    text = text.replace("&", " மற்றும் ")
    if spoken:
        text = to_spoken(text)
    text = re.sub(r"\s+", " ", text).strip()
    if text and text[-1] not in ".!?।":
        text += "."
    return text
