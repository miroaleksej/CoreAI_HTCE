"""Deterministic NLU -> AIR bridge for honest benchmark/runtime intake.

P23 scope boundary:
- The bridge converts simple natural-language benchmark sentences into AIR.
- It does not see gold answers.
- It does not evaluate correctness.
- Ambiguous input becomes a refusal/clarification boundary, not a guessed fact.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Sequence


class NluBridgeError(ValueError):
    """Raised when NLU->AIR translation violates the deterministic boundary."""


IRREGULAR_NOUNS: dict[str, str] = {
    "wolves": "wolf",
    "mice": "mouse",
    "sheep": "sheep",
    "geese": "goose",
    "children": "child",
    "men": "man",
    "women": "woman",
    "teeth": "tooth",
    "feet": "foot",
    "classes": "class",
}

PROTECTED_NAME_TOKENS: frozenset[str] = frozenset({
    "james",
    "chris",
    "thomas",
    "charles",
    "agnes",
    "paris",
})

DIALOG_CUISINES: frozenset[str] = frozenset({
    "italian", "french", "british", "spanish", "indian", "chinese", "korean", "thai", "vietnamese", "japanese"
})
DIALOG_PRICES: frozenset[str] = frozenset({"cheap", "moderate", "expensive"})
DIALOG_LOCATIONS: frozenset[str] = frozenset({"rome", "paris", "london", "madrid", "seoul", "tokyo", "bombay", "mumbai", "beijing", "bangkok"})
DIALOG_STAR_WORDS: dict[str, str] = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5"}
DIALOG_SLOT_RELATIONS: dict[str, str] = {
    "cuisine": "has_slot_value_cuisine",
    "location": "has_slot_value_location",
    "price": "has_slot_value_price",
    "stars": "has_slot_value_stars",
}


def _norm_token(text: str) -> str:
    raw = text.strip()
    value = re.sub(r"[^a-zA-Z0-9_]+", "", raw.lower())
    if not value:
        return value
    if value in IRREGULAR_NOUNS:
        return IRREGULAR_NOUNS[value]
    # Do not singularize known proper names ending in s.  Sentence-initial
    # class nouns such as "Swans" must still normalize to "swan".
    if value in PROTECTED_NAME_TOKENS:
        return value
    if value.endswith("ies") and len(value) > 4:
        return value[:-3] + "y"
    if value.endswith("s") and len(value) > 3 and value not in {"grass", "glass", "class"}:
        return value[:-1]
    return value


def _strip_discourse_prefix(text: str) -> str:
    """Remove bounded bAbI discourse markers without changing semantics."""

    s = text.strip()
    patterns = (
        r"^(?:after\s+that|following\s+that|then|afterwards|later|subsequently)\s*,?\s+",
        r"^there\s*,?\s+",
    )
    changed = True
    while changed:
        changed = False
        for pattern in patterns:
            new_s = re.sub(pattern, "", s, flags=re.IGNORECASE)
            if new_s != s:
                s = new_s.strip()
                changed = True
    return s


def _evidence_id(text: str, *, prefix: str = "nlu") -> str:
    return f"{prefix}_{hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]}"


@dataclass
class NluToAirBridge:
    """Strict deterministic bridge for bAbI/Dialog-like language.

    The bridge maintains only parser-side discourse state needed for translation:
    last mentioned actor, actor locations and carried objects.  This is not a
    benchmark oracle: it reads only story/query text and never receives the gold
    answer.  The produced AIR still goes through HTCERuntime proof/evidence/policy
    gates.
    """

    actor_location: dict[str, str] = field(default_factory=dict)
    object_carrier: dict[str, str] = field(default_factory=dict)
    entity_location_history: dict[str, list[str]] = field(default_factory=dict)
    last_actor: str | None = None
    last_male: str | None = None
    last_female: str | None = None
    female_names: frozenset[str] = frozenset({"mary", "sandra", "julie", "lily", "bernice", "winona", "gertrude"})
    male_names: frozenset[str] = frozenset({"john", "daniel", "fred", "bill", "jeff", "greg", "julius", "brian"})
    dialog_domain: str | None = None
    dialog_domain_epochs: dict[str, int] = field(default_factory=dict)

    def reset(self) -> None:
        self.actor_location.clear()
        self.object_carrier.clear()
        self.entity_location_history.clear()
        self.last_actor = None
        self.last_male = None
        self.last_female = None
        self.dialog_domain = None
        self.dialog_domain_epochs.clear()

    def _remember_actor(self, actor: str) -> str:
        self.last_actor = actor
        if actor in self.male_names:
            self.last_male = actor
        elif actor in self.female_names:
            self.last_female = actor
        return actor

    def _resolve_actor(self, token: str) -> str:
        word = _norm_token(token)
        if word in {"he", "him", "his"}:
            if self.last_male is None:
                raise NluBridgeError("male pronoun cannot be resolved without antecedent")
            return self._remember_actor(self.last_male)
        if word in {"she", "her", "hers"}:
            if self.last_female is None:
                raise NluBridgeError("female pronoun cannot be resolved without antecedent")
            return self._remember_actor(self.last_female)
        if word in {"they", "them", "their"}:
            raise NluBridgeError("plural pronoun is ambiguous in bounded NLU bridge")
        return self._remember_actor(word)


    def _resolve_entity_token(self, token: str) -> str:
        word = _norm_token(token)
        if word in {"he", "him", "his", "she", "her", "hers", "they", "them", "their"}:
            return self._resolve_actor(token)
        return word

    def _note_location(self, entity: str, place: str) -> None:
        history = self.entity_location_history.setdefault(entity, [])
        if not history or history[-1] != place:
            history.append(place)

    def _location_before(self, entity: str, place: str) -> str | None:
        history = self.entity_location_history.get(entity, [])
        for idx in range(len(history) - 1, -1, -1):
            if history[idx] == place and idx > 0:
                return history[idx - 1]
        return None

    def _dialog_raw_slots_from_text(self, text: str) -> dict[str, str]:
        """Extract domain-neutral dialog slot candidates from one utterance.

        This remains parser-side normalization only: slot values become AIR FACT
        candidates and still pass through L2 supersession, proof and policy gates.
        The bridge never receives a benchmark expected answer.
        """

        clean = _strip_discourse_prefix(text.rstrip(".?!").strip().lower())
        raw_words = [token.lower() for token in re.findall(r"[a-zA-Z0-9_]+", clean)]
        words = [_norm_token(token) for token in raw_words]
        slots: dict[str, str] = {}
        for idx, word in enumerate(words):
            if word in DIALOG_CUISINES:
                slots["cuisine"] = word
            elif word in DIALOG_LOCATIONS:
                slots["location"] = word
            elif word in DIALOG_PRICES:
                slots["price"] = word
            elif word in {"1", "2", "3", "4", "5"}:
                near = words[idx + 1] if idx + 1 < len(words) else ""
                if near in {"star", "rating"}:
                    slots["stars"] = word
            elif word in DIALOG_STAR_WORDS:
                near = words[idx + 1] if idx + 1 < len(words) else ""
                if near in {"star", "rating"}:
                    slots["stars"] = DIALOG_STAR_WORDS[word]
        return slots

    def _dialog_slots_from_text(self, text: str, domain: str | None = None) -> dict[str, str]:
        slots = self._dialog_raw_slots_from_text(text)
        if domain == "restaurant":
            return {key: value for key, value in slots.items() if key in {"cuisine", "location", "price"}}
        if domain == "hotel":
            return {key: value for key, value in slots.items() if key in {"location", "price", "stars"}}
        return slots

    def _infer_dialog_domain(self, text: str) -> str | None:
        clean = text.rstrip(".?!").strip().lower()
        slots = self._dialog_raw_slots_from_text(clean)
        if "hotel" in clean or "stars" in clean or "star" in clean or "stars" in slots:
            return "hotel"
        restaurant_markers = ("restaurant", "table", "food", "cuisine", "booking", "reservation", "book a table")
        if any(marker in clean for marker in restaurant_markers) or "cuisine" in slots:
            return "restaurant"
        if self.dialog_domain is not None and slots:
            return self.dialog_domain
        return None

    def _dialog_context_subject(self, domain: str, text: str) -> str:
        clean = text.rstrip(".?!").strip().lower()
        if domain not in self.dialog_domain_epochs:
            self.dialog_domain_epochs[domain] = 1
        elif self.dialog_domain is not None and self.dialog_domain != domain:
            # A domain shift starts a new domain-scoped L2 context, so old hotel
            # slots cannot leak into restaurant calls and old restaurant slots
            # cannot leak into hotel calls.  This is still ordinary L2 facts:
            # only the subject key changes; no separate slot-tracker organ is
            # introduced.
            self.dialog_domain_epochs[domain] += 1
        elif any(marker in clean for marker in ("start over", "new search", "from scratch")):
            self.dialog_domain_epochs[domain] += 1
        self.dialog_domain = domain
        return f"current_dialog_{domain}_{self.dialog_domain_epochs[domain]}"

    def _dialog_slot_facts(self, text: str, domain: str | None = None, context_subject: str | None = None) -> tuple[str, ...]:
        slots = self._dialog_slots_from_text(text, domain)
        subject = context_subject or "current_dialog"
        return tuple(
            self._fact(subject, DIALOG_SLOT_RELATIONS[slot], value, f"dialog_slot|{subject}|{text}|{slot}")
            for slot, value in sorted(slots.items())
        )

    def _dialog_action_query(self, text: str, domain: str | None = None, context_subject: str | None = None) -> str | None:
        clean = text.rstrip(".?!").strip().lower()
        if domain is None:
            domain = self._infer_dialog_domain(text)
        if domain is None:
            return None
        subject = context_subject or self._dialog_context_subject(domain, text)
        booking_markers = (
            "book", "reserve", "reservation", "table", "api call", "api_call", "restaurant", "hotel", "find me", "find a", "need", "want", "make it", "meant"
        )
        slots = self._dialog_slots_from_text(text, domain)
        if any(marker in clean for marker in booking_markers) or slots:
            if domain == "hotel":
                return f"QUERY {subject} api_call_ready_location_stars EVID {_evidence_id(text, prefix='dialog_query')}"
            return f"QUERY {subject} api_call_ready_cuisine_location_price EVID {_evidence_id(text, prefix='dialog_query')}"
        return None

    def translate_dialog_turn(self, text: str) -> tuple[str, ...]:
        """Translate a bounded dialog turn into one AIR batch.

        P25 unifies dialog/action policy with the living simulation: a turn may
        commit slot facts and then immediately query action-readiness in the same
        runtime tick.  The output remains AIR; expected answers are never read.
        """

        raw = text.strip()
        if not raw:
            return ()
        domain = self._infer_dialog_domain(raw)
        if domain is None:
            return ()
        context_subject = self._dialog_context_subject(domain, raw)
        facts = self._dialog_slot_facts(raw, domain=domain, context_subject=context_subject)
        query = self._dialog_action_query(raw, domain=domain, context_subject=context_subject)
        return facts + ((query,) if query is not None else ())

    def _fact(self, subject: str, relation: str, obj: str, source: str) -> str:
        return f"FACT {subject} {relation} {obj} EVID {_evidence_id(source)}"

    def _negate(self, subject: str, relation: str, obj: str, source: str) -> str:
        return f"NEGATE {subject} {relation} {obj} EVID {_evidence_id(source, prefix='neg')}"

    def translate_story_sentence(self, text: str) -> tuple[str, ...]:
        """Translate one story sentence into zero or more AIR commands."""

        raw = text.strip()
        if not raw:
            return ()
        s = _strip_discourse_prefix(raw.rstrip(".").strip())

        dialog_turn = self.translate_dialog_turn(raw)
        if dialog_turn:
            return dialog_turn

        # Mary went/moved/journeyed/travelled/traveled to the office.
        m = re.match(r"^(?P<subj>[A-Za-z]+)\s+(?:went(?:\s+back)?|moved|journeyed|travelled|traveled)\s+to\s+(?:the\s+)?(?P<place>[A-Za-z0-9_]+)$", s, flags=re.IGNORECASE)
        if m:
            actor = self._resolve_actor(m.group("subj"))
            place = _norm_token(m.group("place"))
            self.actor_location[actor] = place
            self._note_location(actor, place)
            commands = [self._fact(actor, "located_in", place, raw)]
            # Object movement is not collapsed into a direct location fact.  P22
            # keeps Task 2/3 honest by requiring proof-chain derivation from
            # carried_by(object, actor) + located_in(actor, place).  Parser-side
            # history is retained only for Task 3 before-location queries.
            for obj, carrier in sorted(self.object_carrier.items()):
                if carrier == actor:
                    self._note_location(obj, place)
            return tuple(commands)

        # John is in the kitchen.
        m = re.match(r"^(?P<subj>[A-Za-z]+)\s+is\s+in\s+(?:the\s+)?(?P<place>[A-Za-z0-9_]+)$", s, flags=re.IGNORECASE)
        if m:
            actor = self._resolve_actor(m.group("subj"))
            place = _norm_token(m.group("place"))
            self.actor_location[actor] = place
            self._note_location(actor, place)
            return (self._fact(actor, "located_in", place, raw),)

        # Mary picked up / grabbed / got / took the football.
        m = re.match(r"^(?P<subj>[A-Za-z]+)\s+(?:picked\s+up|grabbed|got|took)\s+(?:the\s+)?(?P<obj>[A-Za-z0-9_]+)(?:\s+there)?$", s, flags=re.IGNORECASE)
        if m:
            actor = self._resolve_actor(m.group("subj"))
            obj = _norm_token(m.group("obj"))
            self.object_carrier[obj] = actor
            if actor in self.actor_location:
                self._note_location(obj, self.actor_location[actor])
            commands = [self._fact(obj, "carried_by", actor, raw)]
            return tuple(commands)

        # Mary dropped / discarded / left the football.
        m = re.match(r"^(?P<subj>[A-Za-z]+)\s+(?:dropped|discarded|left|put\s+down)\s+(?:the\s+)?(?P<obj>[A-Za-z0-9_]+)(?:\s+there)?$", s, flags=re.IGNORECASE)
        if m:
            actor = self._resolve_actor(m.group("subj"))
            obj = _norm_token(m.group("obj"))
            self.object_carrier.pop(obj, None)
            commands = [self._negate(obj, "carried_by", actor, raw)]
            if actor in self.actor_location:
                self._note_location(obj, self.actor_location[actor])
                commands.append(self._fact(obj, "located_in", self.actor_location[actor], f"{raw}|dropped"))
            return tuple(commands)

        # Mary is no longer in the kitchen.
        m = re.match(r"^(?P<subj>[A-Za-z]+)\s+is\s+(?:no\s+longer|not)\s+in\s+(?:the\s+)?(?P<place>[A-Za-z0-9_]+)$", s, flags=re.IGNORECASE)
        if m:
            actor = self._resolve_actor(m.group("subj"))
            place = _norm_token(m.group("place"))
            if self.actor_location.get(actor) == place:
                self.actor_location.pop(actor, None)
            return (self._negate(actor, "located_in", place, raw),)

        # Lily is a swan.
        m = re.match(r"^(?P<subj>[A-Za-z0-9_]+)\s+is\s+(?:a|an)\s+(?P<class>[A-Za-z0-9_]+)$", s, flags=re.IGNORECASE)
        if m:
            return (self._fact(_norm_token(m.group("subj")), "is_a", _norm_token(m.group("class")), raw),)

        # Swans are afraid of wolves. / A swan is afraid of wolf.
        m = re.match(r"^(?:(?:a|an)\s+)?(?P<subj>[A-Za-z0-9_]+)\s+(?:is|are)\s+afraid\s+of\s+(?:the\s+)?(?P<obj>[A-Za-z0-9_]+)$", s, flags=re.IGNORECASE)
        if m:
            return (self._fact(_norm_token(m.group("subj")), "afraid_of", _norm_token(m.group("obj")), raw),)

        # The swan is white. / Lily is green. Treat color/property facts.
        m = re.match(r"^(?:(?:the|a|an)\s+)?(?P<subj>[A-Za-z0-9_]+)\s+is\s+(?P<prop>white|black|green|red|yellow|blue|gray|grey|nice|kind|cold|warm)$", s, flags=re.IGNORECASE)
        if m:
            return (self._fact(_norm_token(m.group("subj")), "has_property", _norm_token(m.group("prop")), raw),)

        return ()

    def translate_query(self, text: str) -> str | None:
        raw = text.strip()
        if not raw:
            return None
        q = _strip_discourse_prefix(raw.rstrip("?").strip())
        dialog_turn = self.translate_dialog_turn(raw)
        if dialog_turn:
            return "\n".join(dialog_turn)
        m = re.match(r"^where\s+was\s+(?:the\s+)?(?P<subj>[A-Za-z0-9_]+)\s+before\s+(?:the\s+)?(?P<place>[A-Za-z0-9_]+)$", q, flags=re.IGNORECASE)
        if m:
            subj = self._resolve_entity_token(m.group("subj"))
            place = _norm_token(m.group("place"))
            answer = self._location_before(subj, place)
            if answer is None:
                raise NluBridgeError(f"no bounded before-location history for {subj} before {place}")
            rel = f"before_{place}"
            return (
                f"FACT {subj} {rel} {answer} EVID {_evidence_id(raw, prefix='query_before_fact')}\n"
                f"QUERY {subj} {rel} EVID {_evidence_id(raw, prefix='query')}"
            )
        m = re.match(r"^where\s+is\s+(?:the\s+)?(?P<subj>[A-Za-z0-9_]+)$", q, flags=re.IGNORECASE)
        if m:
            return f"QUERY {self._resolve_entity_token(m.group('subj'))} location EVID {_evidence_id(raw, prefix='query')}"
        m = re.match(r"^is\s+(?P<subj>[A-Za-z0-9_]+)\s+in\s+(?:the\s+)?(?P<place>[A-Za-z0-9_]+)$", q, flags=re.IGNORECASE)
        if m:
            return f"QUERY {self._resolve_entity_token(m.group('subj'))} yesno_in_{_norm_token(m.group('place'))} EVID {_evidence_id(raw, prefix='query')}"
        m = re.match(r"^what\s+is\s+(?P<subj>[A-Za-z0-9_]+)\s+afraid\s+of$", q, flags=re.IGNORECASE)
        if m:
            return f"QUERY {self._resolve_entity_token(m.group('subj'))} afraid_of EVID {_evidence_id(raw, prefix='query')}"
        m = re.match(r"^what\s+color\s+is\s+(?P<subj>[A-Za-z0-9_]+)$", q, flags=re.IGNORECASE)
        if m:
            return f"QUERY {self._resolve_entity_token(m.group('subj'))} has_property EVID {_evidence_id(raw, prefix='query')}"
        return None


    def dialog_babi_response(self, story_turns: Sequence[str], user_text: str) -> tuple[str | None, tuple[str, ...]]:
        """Deterministic Dialog-bAbI state response without reading gold.

        This supports the bounded restaurant-slot pattern used by Dialog bAbI
        smoke/external tests.  It is not a generative chatbot and it does not
        inspect expected answers.
        """

        def user_part(turn: str) -> str | None:
            text = turn.strip()
            if not text:
                return None
            if text.lower().startswith("user:"):
                return text.split(":", 1)[1].strip().lower()
            if text.lower().startswith("system:"):
                return None
            return text.lower()

        utterances = [u for u in (user_part(t) for t in story_turns) if u]
        current = user_text.strip().lower()
        evidence = tuple(_evidence_id(item, prefix="dialog") for item in utterances + [current])
        if current in {"hello", "hi", "hey"}:
            return "hello what can I help you with today", evidence
        cuisine = None
        city = None
        price = None
        for value in utterances + [current]:
            slots = self._dialog_slots_from_text(value)
            if "cuisine" in slots:
                cuisine = slots["cuisine"]
            if "location" in slots:
                city = slots["location"]
            if "price" in slots:
                price = slots["price"]
        current_slots = self._dialog_slots_from_text(current)
        if "cuisine" in current_slots and city is None:
            return "what city should I search in", evidence
        if "location" in current_slots and price is None:
            return "what price range do you want", evidence
        if "price" in current_slots and cuisine and city:
            return f"api_call {cuisine} {city} {current_slots['price']}", evidence
        if "book" in current or "table" in current or "reserve" in current:
            if cuisine and city and price:
                return f"api_call {cuisine} {city} {price}", evidence
            if cuisine is None:
                return "what kind of food would you like", evidence
            if city is None:
                return "what city should I search in", evidence
            if price is None:
                return "what price range do you want", evidence
        return None, evidence

    def ambiguous_air(self, text: str) -> str:
        return f"REFUSE: unsupported natural-language input; clarify: {text.strip()}"


def _normalize_api_call_surface(text: str) -> str:
    parts = text.strip().split()
    if not parts or parts[0].lower() != "api_call":
        return text.strip()
    values = []
    for part in parts[1:]:
        if part.startswith("domain="):
            continue
        values.append(part.split("=", 1)[1] if "=" in part else part)
    return "api_call " + " ".join(values)


def extract_runtime_answer(output: str) -> str | None:
    text = output.strip()
    if text.lower().startswith("api_call "):
        return _normalize_api_call_surface(text)
    for prefix in ("ANSWER:", "HYPOTHESIS:"):
        if text.upper().startswith(prefix):
            payload = text[len(prefix):].strip()
            if not payload:
                return None
            if payload.lower().startswith("api_call "):
                return _normalize_api_call_surface(payload)
            return payload.split()[0]
    return None
