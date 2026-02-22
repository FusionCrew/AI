from __future__ import annotations

import hashlib
from typing import Dict, List, Tuple

# Full motion catalog (frontend must map these IDs to actual files)
ALL_MOTIONS: List[str] = [
    "idle_default",
    "m01_nod_light",
    "m02_nod_strong_hands_together",
    "m03_reluctant_nod_arms_crossed",
    "m04_surprised_then_ack_nod",
    "m05_open_arms_surprised_nod",
    "m06_explain_shift_right",
    "m07_explain_shift_left",
    "m08_polite_bow_smile",
    "m09_mid_bow",
    "m10_fluster_forward_back",
    "m11_disagree_headshake_crossed",
    "m12_shock_double_hand_deny",
    "m13_suspicious_lean_in",
    "m14_awkward_disdain_look",
    "m15_teasing_squint_sway",
    "m16_sad_teary_blush_hands",
    "m17_flirty_side_glance_blush",
    "m18_blush_eyes_closed_smile",
    "m19_shy_blush_wiggle",
    "m20_thinking_troubled_crossed",
    "m21_happy_bounce_blush",
    "m22_happy_close_eyes_approach",
    "m23_tilt_then_calm",
    "m24_cute_angry_blush",
    "m25_small_surprise_recover",
    "m26_big_surprise_recover",
]


def _pick(seed: str, motions: List[str]) -> str:
    if not motions:
        return "idle_default"
    h = hashlib.sha1(seed.encode("utf-8", errors="ignore")).hexdigest()
    idx = int(h[:8], 16) % len(motions)
    return motions[idx]


def _contains_any(text: str, tokens: List[str]) -> bool:
    return any(tok in text for tok in tokens)


def pick_live2d_profile(intent: str, action: str, stage: str, speech: str = "") -> Tuple[str, str, str]:
    i = (intent or "").upper()
    a = (action or "").upper()
    s = (stage or "").upper()
    sp = (speech or "").strip()
    sp_norm = sp.replace(" ", "")
    seed = f"{i}|{a}|{s}|{sp_norm}"

    # High-priority semantic cues from speech text
    if _contains_any(sp_norm, ["죄송", "불편", "어려워", "못", "실패"]):
        return ("supportive", "serious", _pick(seed, ["m16_sad_teary_blush_hands", "m23_tilt_then_calm"]))
    if _contains_any(sp_norm, ["놀라", "깜짝", "어머"]):
        return ("neutral", "attentive", _pick(seed, ["m25_small_surprise_recover", "m26_big_surprise_recover"]))
    if _contains_any(sp_norm, ["안돼", "불가", "제외", "없어요"]):
        return ("serious", "serious", _pick(seed, ["m11_disagree_headshake_crossed", "m12_shock_double_hand_deny"]))
    if _contains_any(sp_norm, ["추천", "도와", "안내"]):
        return ("supportive", "smile", _pick(seed, ["m06_explain_shift_right", "m07_explain_shift_left", "m23_tilt_then_calm"]))
    if _contains_any(sp_norm, ["확인", "결제", "진행", "이동"]):
        return ("confident", "smile", _pick(seed, ["m08_polite_bow_smile", "m09_mid_bow", "m01_nod_light"]))

    # Intent-first mapping
    if i == "MENU_INFO":
        return ("neutral", "attentive", _pick(seed, ["m06_explain_shift_right", "m07_explain_shift_left", "m01_nod_light"]))
    if i == "MENU_RECOMMEND":
        return ("supportive", "smile", _pick(seed, ["m23_tilt_then_calm", "m21_happy_bounce_blush", "m06_explain_shift_right"]))
    if i == "PROACTIVE_HELP":
        return ("supportive", "soft_smile", _pick(seed, ["m20_thinking_troubled_crossed", "m23_tilt_then_calm"]))

    # Action mapping
    if a in {"ADD_MENU", "ADD_TO_CART"}:
        return ("happy", "smile", _pick(seed, ["m21_happy_bounce_blush", "m22_happy_close_eyes_approach", "m01_nod_light"]))
    if a in {"REMOVE_MENU", "REMOVE_FROM_CART"}:
        return ("neutral", "attentive", _pick(seed, ["m03_reluctant_nod_arms_crossed", "m11_disagree_headshake_crossed"]))
    if a == "CHECKOUT":
        return ("confident", "smile", _pick(seed, ["m08_polite_bow_smile", "m09_mid_bow"]))
    if a == "SELECT_PAYMENT":
        return ("confident", "smile", _pick(seed, ["m01_nod_light", "m08_polite_bow_smile"]))
    if a == "SET_DINING":
        return ("confident", "smile", _pick(seed, ["m01_nod_light", "m02_nod_strong_hands_together"]))
    if a == "CALL_STAFF":
        return ("supportive", "serious", _pick(seed, ["m06_explain_shift_right", "m09_mid_bow"]))
    if a == "CHECK_CART":
        return ("neutral", "attentive", _pick(seed, ["m20_thinking_troubled_crossed", "m01_nod_light"]))
    if a == "CONTINUE_ORDER":
        return ("happy", "smile", _pick(seed, ["m21_happy_bounce_blush", "m01_nod_light"]))

    # Stage mapping fallback
    if s == "PAYMENT":
        return ("confident", "attentive", _pick(seed, ["m08_polite_bow_smile", "m09_mid_bow"]))
    if s == "ORDER_REVIEW":
        return ("neutral", "attentive", _pick(seed, ["m20_thinking_troubled_crossed", "m23_tilt_then_calm"]))
    if s == "RECOMMENDATION":
        return ("supportive", "smile", _pick(seed, ["m06_explain_shift_right", "m07_explain_shift_left", "m23_tilt_then_calm"]))
    if s == "PROACTIVE_HELP":
        return ("supportive", "soft_smile", _pick(seed, ["m23_tilt_then_calm", "m20_thinking_troubled_crossed"]))

    return ("neutral", "attentive", "idle_default")
